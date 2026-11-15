"""Replay nuScenes sensor data as ROS 2 topics with the original timing.

Publishes every camera frame and LiDAR sweep of the selected scenes in
timestamp order, paced against the wall clock (``--rate 2`` = twice real
time). Message stamps are the original nuScenes timestamps. Only depends on
nuscenes-devkit and ROS 2, so any detector node can subscribe.

Topics:
    /LIDAR_TOP/points              sensor_msgs/PointCloud2  x, y, z, intensity, ring (float32)
    /<CAM>/image/compressed        sensor_msgs/CompressedImage  original JPEG
    /<CAM>/camera_info             sensor_msgs/CameraInfo   intrinsics (latched)
    /tf                            map -> base_link ego pose at every sensor timestamp
    /tf_static                     base_link -> <sensor> mounting (latched, per scene)

Example (inside the container, after ``source /opt/ros/foxy/setup.bash``):
    python ros2/nuscenes_replay.py --dataroot data/nuscenes --split mini_val --rate 1
"""

import argparse
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import CameraInfo, CompressedImage, PointCloud2, PointField
from tf2_msgs.msg import TFMessage

CAMERAS = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]
LIDAR = "LIDAR_TOP"

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def to_stamp(timestamp_us):
    return Time(sec=timestamp_us // 1_000_000, nanosec=(timestamp_us % 1_000_000) * 1000)


def make_transform(stamp, parent, child, translation, rotation_wxyz):
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent
    t.child_frame_id = child
    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (
        float(v) for v in translation
    )
    w, x, y, z = (float(v) for v in rotation_wxyz)
    t.transform.rotation.w, t.transform.rotation.x = w, x
    t.transform.rotation.y, t.transform.rotation.z = y, z
    return t


def make_pointcloud(points, stamp):
    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = LIDAR
    msg.height = 1
    msg.width = len(points)
    msg.fields = [
        PointField(name=name, offset=4 * i, datatype=PointField.FLOAT32, count=1)
        for i, name in enumerate(["x", "y", "z", "intensity", "ring"])
    ]
    msg.is_bigendian = False
    msg.point_step = 20
    msg.row_step = 20 * len(points)
    msg.data = points.astype(np.float32).tobytes()
    msg.is_dense = True
    return msg


def scene_events(nusc, scene):
    """All sample_data records of a scene for our sensors, in time order."""
    first_sample = nusc.get("sample", scene["first_sample_token"])
    events = []
    for channel in CAMERAS + [LIDAR]:
        sd = nusc.get("sample_data", first_sample["data"][channel])
        # Start from the earliest sweep so the first keyframe has LiDAR history.
        while sd["prev"]:
            sd = nusc.get("sample_data", sd["prev"])
        while True:
            events.append((sd["timestamp"], channel, sd))
            if not sd["next"]:
                break
            sd = nusc.get("sample_data", sd["next"])
    events.sort(key=lambda e: e[0])
    return events


class ReplayNode:
    def __init__(self, node, nusc, rate):
        self.node = node
        self.nusc = nusc
        self.rate = rate
        self.lidar_pub = node.create_publisher(PointCloud2, f"/{LIDAR}/points", 10)
        self.image_pubs = {
            cam: node.create_publisher(CompressedImage, f"/{cam}/image/compressed", 10)
            for cam in CAMERAS
        }
        self.info_pubs = {
            cam: node.create_publisher(CameraInfo, f"/{cam}/camera_info", LATCHED)
            for cam in CAMERAS
        }
        self.tf_pub = node.create_publisher(TFMessage, "/tf", 100)
        self.tf_static_pub = node.create_publisher(TFMessage, "/tf_static", LATCHED)

    def publish_calibration(self, events, stamp):
        """Sensor mounting and camera intrinsics; these change between scenes."""
        transforms, seen = [], set()
        for _, channel, sd in events:
            if channel in seen:
                continue
            seen.add(channel)
            cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            transforms.append(
                make_transform(stamp, "base_link", channel, cs["translation"], cs["rotation"])
            )
            if channel in CAMERAS:
                info = CameraInfo()
                info.header.stamp = stamp
                info.header.frame_id = channel
                info.width, info.height = sd["width"], sd["height"]
                info.k = [float(v) for v in np.array(cs["camera_intrinsic"]).flatten()]
                self.info_pubs[channel].publish(info)
        self.tf_static_pub.publish(TFMessage(transforms=transforms))

    def play_scene(self, scene):
        events = scene_events(self.nusc, scene)
        self.publish_calibration(events, to_stamp(events[0][0]))
        t0, wall0 = events[0][0], time.monotonic()
        for timestamp, channel, sd in events:
            delay = wall0 + (timestamp - t0) / 1e6 / self.rate - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            stamp = to_stamp(timestamp)
            pose = self.nusc.get("ego_pose", sd["ego_pose_token"])
            self.tf_pub.publish(
                TFMessage(
                    transforms=[
                        make_transform(stamp, "map", "base_link", pose["translation"], pose["rotation"])
                    ]
                )
            )
            path = f"{self.nusc.dataroot}/{sd['filename']}"
            if channel == LIDAR:
                points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
                self.lidar_pub.publish(make_pointcloud(points, stamp))
            else:
                msg = CompressedImage()
                msg.header.stamp = stamp
                msg.header.frame_id = channel
                msg.format = "jpeg"
                with open(path, "rb") as f:
                    msg.data = f.read()
                self.image_pubs[channel].publish(msg)
        return len(events), time.monotonic() - wall0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataroot", default="data/nuscenes")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument(
        "--split",
        default="mini_val",
        help="nuscenes-devkit split name (e.g. mini_val, val) or comma-separated scene names",
    )
    parser.add_argument("--rate", type=float, default=1.0, help="playback speed; 2 = twice real time")
    parser.add_argument("--start-delay", type=float, default=3.0, help="seconds to wait for subscribers")
    parser.add_argument("--scene-gap", type=float, default=2.0, help="seconds of silence between scenes")
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    names = getattr(splits, args.split, None) or args.split.split(",")
    scenes = [s for s in nusc.scene if s["name"] in names]
    assert scenes, f"no scenes found for {args.split}"

    rclpy.init()
    node = rclpy.create_node("nuscenes_replay")
    replay = ReplayNode(node, nusc, args.rate)
    time.sleep(args.start_delay)
    for scene in scenes:
        count, duration = replay.play_scene(scene)
        node.get_logger().info(f"{scene['name']}: {count} messages in {duration:.1f} s")
        time.sleep(args.scene_gap)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
