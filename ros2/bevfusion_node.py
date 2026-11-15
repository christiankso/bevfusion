"""BEVFusion camera+LiDAR detector as a ROS 2 node.

Subscribes to the six cameras, the LiDAR, and the ego/sensor transforms
published by nuscenes_replay.py, synchronizes each LiDAR sweep with the
nearest image from every camera, merges the previous LiDAR sweeps, builds
the same model input as the offline test pipeline, and publishes 3D boxes.

ROS callbacks only buffer messages; a worker thread runs inference. By
default the worker always takes the newest LiDAR sweep and drops older
unprocessed ones (real-time behaviour). ``--process-all`` queues every sweep
instead, for checking accuracy against the offline evaluation.

Output:
    /bevfusion/detections   vision_msgs/Detection3DArray (frame LIDAR_TOP)
    --save-results          pickle with raw boxes (incl. velocity) and per-frame
                            timing, scored by ros2/evaluate_ros_results.py

Example (inside the container, after ``source /opt/ros/foxy/setup.bash``):
    python ros2/bevfusion_node.py --save-results results/ros/run.pkl
"""

import argparse
import collections
import io
import pickle
import sys
import threading
import time

import numpy as np
import rclpy
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, collate
from mmcv.runner import load_checkpoint, wrap_fp16_model
from PIL import Image
from pyquaternion import Quaternion
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import CameraInfo, CompressedImage, PointCloud2
from tf2_msgs.msg import TFMessage
from torchpack.utils.config import configs
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

sys.path.insert(0, "tools")
from test import apply_degradation  # noqa: E402

from mmdet3d.core.bbox import get_box_type  # noqa: E402
from mmdet3d.core.points import get_points_type  # noqa: E402
from mmdet3d.datasets.pipelines import Compose  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402
from mmdet3d.utils import recursive_eval  # noqa: E402

CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]
LIDAR = "LIDAR_TOP"
LOADER_STEPS = {
    "LoadMultiViewImageFromFiles",
    "LoadPointsFromFile",
    "LoadPointsFromMultiSweeps",
    "LoadRadarPointsMultiSweeps",
    "LoadAnnotations3D",
}
LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def stamp_us(stamp):
    return stamp.sec * 1_000_000 + stamp.nanosec // 1000


def transform_matrix(t):
    matrix = np.eye(4, dtype=np.float32)
    r = t.transform.rotation
    matrix[:3, :3] = Quaternion(r.w, r.x, r.y, r.z).rotation_matrix
    v = t.transform.translation
    matrix[:3, 3] = [v.x, v.y, v.z]
    return matrix


def remove_close(points, radius=1.0):
    close = (np.abs(points[:, 0]) < radius) & (np.abs(points[:, 1]) < radius)
    return points[~close]


class BEVFusionNode:
    def __init__(self, node, args):
        self.node = node
        self.args = args
        self.log = node.get_logger()
        self._load_model()

        self.lock = threading.Lock()
        self.pending = collections.deque()  # LiDAR sweeps waiting for inference
        self.history = collections.deque(maxlen=30)  # all recent sweeps, for merging
        self.images = {cam: collections.deque(maxlen=30) for cam in CAMERAS}
        self.poses = {}  # stamp_us -> ego2global
        self.sensor2ego = {}  # frame -> 4x4
        self.intrinsics = {}  # camera -> 3x3
        self.last_lidar_stamp = None
        self.last_lidar_wall = None
        self.records = []
        self.stats = collections.Counter()
        self.stop = threading.Event()

        node.create_subscription(PointCloud2, f"/{LIDAR}/points", self.on_lidar, 10)
        for cam in CAMERAS:
            node.create_subscription(
                CompressedImage,
                f"/{cam}/image/compressed",
                lambda msg, cam=cam: self.on_image(cam, msg),
                10,
            )
            node.create_subscription(
                CameraInfo,
                f"/{cam}/camera_info",
                lambda msg, cam=cam: self.on_camera_info(cam, msg),
                LATCHED,
            )
        node.create_subscription(TFMessage, "/tf", self.on_tf, 100)
        node.create_subscription(TFMessage, "/tf_static", self.on_tf_static, LATCHED)
        self.det_pub = node.create_publisher(Detection3DArray, "/bevfusion/detections", 10)
        self.worker = threading.Thread(target=self.run_worker, daemon=True)
        self.worker.start()
        self.log.info("BEVFusion node ready")

    # ------------------------------------------------------------------ model
    def _load_model(self):
        configs.load(self.args.config, recursive=True)
        cfg = Config(recursive_eval(configs), filename=self.args.config)
        self.classes = cfg.object_classes

        steps = [dict(s) for s in cfg.data.test.pipeline if s["type"] not in LOADER_STEPS]
        for step in steps:
            if step["type"] == "Collect3D":
                step["keys"] = [k for k in step["keys"] if k in ("img", "points")]
        if self.args.degrade:
            apply_degradation(steps, self.args.degrade)
        self.pipeline = Compose(steps)
        self.box_type_3d, self.box_mode_3d = get_box_type("LiDAR")

        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
        if cfg.get("fp16", None) is not None:
            wrap_fp16_model(model)
        load_checkpoint(model, self.args.checkpoint, map_location="cpu")
        torch.backends.cudnn.benchmark = True
        self.model = MMDataParallel(model.cuda(), device_ids=[0]).eval()

    # -------------------------------------------------------------- callbacks
    def on_lidar(self, msg):
        wall = time.monotonic()
        t = stamp_us(msg.header.stamp)
        points = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 5).copy()
        with self.lock:
            if self.last_lidar_stamp is not None and not (
                0 < t - self.last_lidar_stamp < 1_000_000
            ):
                self.history.clear()  # new scene or time jump
            self.last_lidar_stamp, self.last_lidar_wall = t, wall
            self.pending.append((t, points, wall))
            self.history.append((t, points))
            self.stats["received"] += 1

    def on_image(self, cam, msg):
        with self.lock:
            self.images[cam].append((stamp_us(msg.header.stamp), bytes(msg.data)))

    def on_camera_info(self, cam, msg):
        with self.lock:
            self.intrinsics[cam] = np.array(msg.k, dtype=np.float32).reshape(3, 3)

    def on_tf(self, msg):
        with self.lock:
            for t in msg.transforms:
                if t.header.frame_id == "map" and t.child_frame_id == "base_link":
                    self.poses[stamp_us(t.header.stamp)] = transform_matrix(t)
            if len(self.poses) > 2000:
                for key in sorted(self.poses)[:1000]:
                    del self.poses[key]

    def on_tf_static(self, msg):
        with self.lock:
            for t in msg.transforms:
                self.sensor2ego[t.child_frame_id] = transform_matrix(t)

    # ----------------------------------------------------------------- worker
    def next_frame(self):
        with self.lock:
            if not self.pending:
                return None
            if self.args.process_all:
                return self.pending.popleft()
            frame = self.pending.pop()
            self.stats["dropped"] += len(self.pending)
            self.pending.clear()
            return frame

    def match_cameras(self, t, received_wall):
        """Nearest image per camera; waits until it is known to be the nearest."""
        deadline = received_wall + self.args.sync_timeout
        matched = {}
        while not self.stop.is_set():
            with self.lock:
                matched, complete = {}, True
                for cam in CAMERAS:
                    buf = self.images[cam]
                    if buf:
                        matched[cam] = min(buf, key=lambda item: abs(item[0] - t))
                    if not buf or buf[-1][0] < t:
                        complete = False
            if complete or time.monotonic() > deadline:
                break
            time.sleep(0.002)
        tolerance = self.args.sync_tolerance_ms * 1000
        if len(matched) < len(CAMERAS) or any(
            abs(ts - t) > tolerance for ts, _ in matched.values()
        ):
            return None
        return matched

    def build_input(self, t, points, matched):
        with self.lock:
            ego2global = self.poses.get(t)
            cam_poses = {cam: self.poses.get(ts) for cam, (ts, _) in matched.items()}
            sweeps = [(ts, p) for ts, p in self.history if ts < t][-self.args.sweeps :]
            sweep_poses = [self.poses.get(ts) for ts, _ in sweeps]
            lidar2ego = self.sensor2ego.get(LIDAR)
            cam2ego = {cam: self.sensor2ego.get(cam) for cam in CAMERAS}
            intrinsics = dict(self.intrinsics)
        if (
            ego2global is None
            or lidar2ego is None
            or any(p is None for p in cam_poses.values())
            or any(p is None for p in sweep_poses)
            or any(cam2ego[c] is None or c not in intrinsics for c in CAMERAS)
        ):
            return None

        global2lidar = np.linalg.inv(lidar2ego) @ np.linalg.inv(ego2global)

        # Merge previous sweeps into the current LiDAR frame, as
        # LoadPointsFromMultiSweeps does; column 4 becomes the time lag.
        clouds = [points.copy()]
        clouds[0][:, 4] = 0
        if not sweeps:
            sweeps, sweep_poses = [(t, points)] * self.args.sweeps, [ego2global] * self.args.sweeps
        for (ts, sweep), pose in zip(sweeps, sweep_poses):
            sweep = remove_close(sweep).copy()
            sweep2lidar = global2lidar @ pose @ lidar2ego
            sweep[:, :3] = sweep[:, :3] @ sweep2lidar[:3, :3].T + sweep2lidar[:3, 3]
            sweep[:, 4] = (t - ts) / 1e6
            clouds.append(sweep)
        merged = np.concatenate(clouds).astype(np.float32)

        data = dict(
            token=str(t),
            timestamp=t,
            lidar_path=str(t),
            points=get_points_type("LIDAR")(merged, points_dim=5),
            img=[],
            image_paths=[],
            lidar2ego=lidar2ego,
            ego2global=ego2global,
            lidar2camera=[],
            lidar2image=[],
            camera2ego=[],
            camera_intrinsics=[],
            camera2lidar=[],
            box_type_3d=self.box_type_3d,
            box_mode_3d=self.box_mode_3d,
        )
        for cam in CAMERAS:
            ts, jpeg = matched[cam]
            image = Image.open(io.BytesIO(jpeg))
            camera2lidar = global2lidar @ cam_poses[cam] @ cam2ego[cam]
            lidar2camera = np.linalg.inv(camera2lidar)
            k = np.eye(4, dtype=np.float32)
            k[:3, :3] = intrinsics[cam]
            data["img"].append(image)
            data["image_paths"].append(f"ros/{cam}/{ts}.jpg")
            data["camera2lidar"].append(camera2lidar.astype(np.float32))
            data["lidar2camera"].append(lidar2camera.astype(np.float32))
            data["camera_intrinsics"].append(k)
            data["lidar2image"].append((k @ lidar2camera).astype(np.float32))
            data["camera2ego"].append(cam2ego[cam])
        size = data["img"][0].size
        data.update(ori_shape=size, img_shape=size, pad_shape=size, scale_factor=1.0)
        return collate([self.pipeline(data)], samples_per_gpu=1)

    def publish(self, t, boxes, scores, labels):
        msg = Detection3DArray()
        msg.header.stamp.sec, msg.header.stamp.nanosec = t // 1_000_000, (t % 1_000_000) * 1000
        msg.header.frame_id = LIDAR
        keep = scores >= self.args.publish_score_thr
        centers = boxes.gravity_center[keep].numpy()
        dims = boxes.dims[keep].numpy()
        yaws = boxes.yaw[keep].numpy()
        for center, dim, yaw, score, label in zip(
            centers, dims, yaws, scores[keep].tolist(), labels[keep].tolist()
        ):
            det = Detection3D()
            det.header = msg.header
            hyp = ObjectHypothesisWithPose()
            hyp.id = self.classes[label]
            hyp.score = float(score)
            det.results.append(hyp)
            det.bbox.center.position.x, det.bbox.center.position.y, det.bbox.center.position.z = (
                float(v) for v in center
            )
            q = Quaternion(axis=[0, 0, 1], angle=float(yaw))
            det.bbox.center.orientation.w, det.bbox.center.orientation.x = q.w, q.x
            det.bbox.center.orientation.y, det.bbox.center.orientation.z = q.y, q.z
            det.bbox.size.x, det.bbox.size.y, det.bbox.size.z = (float(v) for v in dim)
            msg.detections.append(det)
        self.det_pub.publish(msg)

    def run_worker(self):
        while not self.stop.is_set():
            frame = self.next_frame()
            if frame is None:
                if (
                    self.args.idle_timeout
                    and self.last_lidar_wall is not None
                    and time.monotonic() - self.last_lidar_wall > self.args.idle_timeout
                ):
                    self.stop.set()
                time.sleep(0.002)
                continue
            t, points, received = frame
            start = time.monotonic()
            matched = self.match_cameras(t, received)
            synced = time.monotonic()
            if matched is None:
                self.stats["skipped_no_sync"] += 1
                continue
            batch = self.build_input(t, points, matched)
            if batch is None:
                self.stats["skipped_no_calibration"] += 1
                continue
            prepared = time.monotonic()
            with torch.no_grad():
                result = self.model(return_loss=False, rescale=True, **batch)[0]
            torch.cuda.synchronize()
            inferred = time.monotonic()
            self.publish(t, result["boxes_3d"], result["scores_3d"], result["labels_3d"])
            published = time.monotonic()
            self.stats["processed"] += 1
            self.records.append(
                dict(
                    stamp=t,
                    boxes=result["boxes_3d"].tensor.numpy(),
                    box_dim=result["boxes_3d"].box_dim,
                    scores=result["scores_3d"].numpy(),
                    labels=result["labels_3d"].numpy(),
                    camera_offsets_ms={c: (ts - t) / 1000 for c, (ts, _) in matched.items()},
                    num_points=int(batch["points"].data[0][0].shape[0]),
                    queue_ms=(start - received) * 1000,
                    sync_ms=(synced - start) * 1000,
                    preprocess_ms=(prepared - synced) * 1000,
                    inference_ms=(inferred - prepared) * 1000,
                    publish_ms=(published - inferred) * 1000,
                    latency_ms=(published - received) * 1000,
                )
            )

    def save(self):
        if not self.args.save_results:
            return
        import os

        os.makedirs(os.path.dirname(self.args.save_results) or ".", exist_ok=True)
        with open(self.args.save_results, "wb") as f:
            pickle.dump(
                dict(args=vars(self.args), stats=dict(self.stats), records=self.records), f
            )
        self.log.info(f"saved {len(self.records)} frames to {self.args.save_results}")


def main():
    from mmcv import DictAction

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        default="configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml",
    )
    parser.add_argument("--checkpoint", default="pretrained/bevfusion-det.pth")
    parser.add_argument("--sweeps", type=int, default=9, help="previous LiDAR sweeps to merge")
    parser.add_argument("--process-all", action="store_true", help="queue every sweep, never drop")
    parser.add_argument("--sync-timeout", type=float, default=0.2, help="max wall seconds to wait for cameras")
    parser.add_argument("--sync-tolerance-ms", type=float, default=100, help="max camera-LiDAR stamp gap")
    parser.add_argument("--publish-score-thr", type=float, default=0.1)
    parser.add_argument("--degrade", nargs="+", action=DictAction, help="same options as tools/test.py")
    parser.add_argument("--save-results", help="pickle path for raw boxes and timing")
    parser.add_argument("--idle-timeout", type=float, default=0, help="exit after this many seconds without LiDAR")
    args = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node("bevfusion")
    detector = BEVFusionNode(node, args)
    try:
        while rclpy.ok() and not detector.stop.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    detector.stop.set()
    detector.worker.join(timeout=5)
    node.get_logger().info(f"stats: {dict(detector.stats)}")
    detector.save()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
