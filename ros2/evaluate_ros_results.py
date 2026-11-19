"""Score BEVFusion ROS 2 node output with the official nuScenes evaluation.

Two accuracy measures:

* **Processed keyframes** — boxes computed from each keyframe's own sweep; a
  keyframe the node never processed counts as an empty prediction. With
  ``--keyframes-only`` or ``--process-all`` this measures model accuracy.
* **Real time** — for each keyframe, the newest boxes the node had already
  published when that keyframe's sweep arrived, moved into the keyframe's
  LiDAR frame to cancel the vehicle's own motion. This is what a planner
  would actually have, so it includes the cost of latency and dropped frames.
  Reported without and with moving each box forward by its predicted
  velocity (``extrapolated``).

Also summarizes latency, throughput, and CPU/GPU/memory use.

Example (inside the container):
    python ros2/evaluate_ros_results.py results/ros/run.pkl --out results/ros/run.json
"""

import argparse
import bisect
import json
import pickle

import numpy as np
import torch
from mmcv import Config
from pyquaternion import Quaternion
from torchpack.utils.config import configs

from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets import build_dataset
from mmdet3d.utils import recursive_eval

DEFAULT_CONFIG = (
    "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml"
)
TIMING_KEYS = ["latency_ms", "queue_ms", "sync_ms", "preprocess_ms", "inference_ms", "publish_ms"]
MONITOR_KEYS = ["cpu_percent", "rss_mb", "gpu_util_percent", "gpu_mem_used_mb", "torch_gpu_peak_mb"]


def percentiles(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def pose_matrix(rotation, translation):
    matrix = np.eye(4)
    matrix[:3, :3] = Quaternion(rotation).rotation_matrix
    matrix[:3, 3] = translation
    return matrix


def move_boxes(tensor, source2target, dt=None):
    """Express boxes from one LiDAR frame in another; optionally move each
    box forward by its predicted velocity for dt seconds."""
    boxes = tensor.clone().double()
    rotation = torch.from_numpy(source2target[:3, :3])
    translation = torch.from_numpy(source2target[:3, 3])
    boxes[:, :3] = boxes[:, :3] @ rotation.T + translation
    boxes[:, 6] += float(np.arctan2(source2target[1, 0], source2target[0, 0]))
    if boxes.shape[1] >= 9:
        boxes[:, 7:9] = boxes[:, 7:9] @ rotation[:2, :2].T
        if dt:
            boxes[:, :2] += boxes[:, 7:9] * dt
    return boxes.float()


def empty_output():
    return dict(
        boxes_3d=LiDARInstance3DBoxes(torch.zeros(0, 9), box_dim=9),
        scores_3d=torch.zeros(0),
        labels_3d=torch.zeros(0, dtype=torch.long),
    )


def record_output(record, tensor=None):
    tensor = torch.from_numpy(record["boxes"]) if tensor is None else tensor
    return dict(
        boxes_3d=LiDARInstance3DBoxes(tensor, box_dim=record["box_dim"]),
        scores_3d=torch.from_numpy(record["scores"]),
        labels_3d=torch.from_numpy(record["labels"]).long(),
    )


def evaluate(dataset, outputs, eval_kwargs, jsonfile_prefix=None):
    """Official nuScenes metrics; with jsonfile_prefix, keep results_nusc.json
    and metrics_summary.json / metrics_details.json in that folder."""
    if all(len(o["scores_3d"]) == 0 for o in outputs):
        return 0.0, 0.0, {}  # nuScenes eval cannot score an empty submission
    metrics = dataset.evaluate(
        outputs, metric="bbox", jsonfile_prefix=jsonfile_prefix, **eval_kwargs
    )
    return metrics["object/map"], metrics["object/nds"], metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", help="pickle written by bevfusion_node.py --save-results")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--out", help="write metrics and timing summary to this json file")
    parser.add_argument("--gpu-log", help="nvidia-smi csv (utilization %%, memory MB) sampled during the run")
    parser.add_argument(
        "--jsonfile-prefix",
        help="folder for nuScenes results_nusc.json + official metrics files, one "
        "subfolder per measure: keyframes/, realtime/, realtime_extrapolated/",
    )
    args = parser.parse_args()

    with open(args.results, "rb") as f:
        run = pickle.load(f)
    records = run["records"]

    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    eval_kwargs = cfg.get("evaluation", {}).copy()
    for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule", "pipeline"]:
        eval_kwargs.pop(key, None)
    infos = dataset.data_infos

    # Accuracy on the keyframes the node processed from their own sweep.
    by_stamp = {r["stamp"]: r for r in records}
    outputs = [
        record_output(by_stamp[i["timestamp"]]) if i["timestamp"] in by_stamp else empty_output()
        for i in infos
    ]
    processed = sum(i["timestamp"] in by_stamp for i in infos)
    def prefix(name):
        return f"{args.jsonfile_prefix}/{name}" if args.jsonfile_prefix else None

    mAP, NDS, metrics = evaluate(dataset, outputs, eval_kwargs, prefix("keyframes"))
    summary = dict(
        results=args.results,
        node_args=run["args"],
        node_stats=run["stats"],
        keyframes=len(infos),
        keyframes_processed=processed,
        mAP=mAP,
        NDS=NDS,
    )

    # Real-time accuracy: newest published output when each keyframe arrived.
    arrivals = run.get("arrivals")
    if arrivals and records and "published_wall" in records[0]:
        ordered = sorted(records, key=lambda r: r["published_wall"])
        published = [r["published_wall"] for r in ordered]
        realtime, extrapolated, ages = [], [], []
        for info in infos:
            k = info["timestamp"]
            i = bisect.bisect_right(published, arrivals.get(k, -1.0)) - 1
            if k not in arrivals or i < 0:
                realtime.append(empty_output())
                extrapolated.append(empty_output())
                continue
            record = ordered[i]
            if not 0 <= k - record["stamp"] <= 2_000_000:  # other scene or > 2 s old
                realtime.append(empty_output())
                extrapolated.append(empty_output())
                continue
            keyframe2global = pose_matrix(
                info["ego2global_rotation"], info["ego2global_translation"]
            ) @ pose_matrix(info["lidar2ego_rotation"], info["lidar2ego_translation"])
            source2global = record["ego2global"].astype(np.float64) @ record["lidar2ego"].astype(
                np.float64
            )
            source2keyframe = np.linalg.inv(keyframe2global) @ source2global
            dt = (k - record["stamp"]) / 1e6
            ages.append(dt * 1000)
            tensor = torch.from_numpy(record["boxes"])
            realtime.append(record_output(record, move_boxes(tensor, source2keyframe)))
            extrapolated.append(record_output(record, move_boxes(tensor, source2keyframe, dt)))
        rt_map, rt_nds, _ = evaluate(dataset, realtime, eval_kwargs, prefix("realtime"))
        ex_map, ex_nds, _ = evaluate(
            dataset, extrapolated, eval_kwargs, prefix("realtime_extrapolated")
        )
        summary.update(
            realtime_mAP=rt_map,
            realtime_NDS=rt_nds,
            realtime_extrapolated_mAP=ex_map,
            realtime_extrapolated_NDS=ex_nds,
            realtime_keyframes_covered=len(ages),
            realtime_box_age_ms=percentiles(ages) if ages else None,
        )

    if arrivals:
        # Time the sensor was actually streaming: sum of per-scene spans,
        # split wherever LiDAR went quiet for more than a second.
        walls = sorted(arrivals.values())
        active = sum(b - a for a, b in zip(walls, walls[1:]) if b - a < 1.0)
        summary["lidar_input_hz"] = len(walls) / active if active > 0 else None
        summary["throughput_hz"] = len(records) / active if active > 0 else None
    if records:
        summary["timing_ms"] = {k: percentiles([r[k] for r in records]) for k in TIMING_KEYS}
        keyframe_stamps = {i["timestamp"] for i in infos}
        offsets = [
            abs(v)
            for r in records
            if r["stamp"] in keyframe_stamps
            for v in r["camera_offsets_ms"].values()
        ]
        summary["max_camera_offset_ms"] = max(offsets) if offsets else None
    monitor = run.get("monitor") or []
    if args.gpu_log:
        gpu = np.loadtxt(args.gpu_log, delimiter=",", ndmin=2)
        monitor = monitor + [
            dict(gpu_util_percent=u, gpu_mem_used_mb=m) for u, m in gpu[:, :2]
        ]
    if monitor:
        summary["resources"] = {
            k: percentiles([s[k] for s in monitor if k in s])
            for k in MONITOR_KEYS
            if any(k in s for s in monitor)
        }
    summary["metrics"] = metrics

    print(json.dumps({k: v for k, v in summary.items() if k != "metrics"}, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
