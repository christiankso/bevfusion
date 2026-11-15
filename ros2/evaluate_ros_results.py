"""Score BEVFusion ROS 2 node output with the official nuScenes evaluation.

Matches each saved frame to a validation keyframe by LiDAR timestamp. A
keyframe the node never processed (dropped, unsynchronized) counts as an
empty prediction, so mAP/NDS reflect what the running system delivered.
Also summarizes latency and throughput from the per-frame timing.

Example (inside the container):
    python ros2/evaluate_ros_results.py results/ros/run.pkl --out results/ros/run.json
"""

import argparse
import json
import pickle

import numpy as np
import torch
from mmcv import Config
from torchpack.utils.config import configs

from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets import build_dataset
from mmdet3d.utils import recursive_eval

DEFAULT_CONFIG = (
    "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml"
)
TIMING_KEYS = ["latency_ms", "queue_ms", "sync_ms", "preprocess_ms", "inference_ms", "publish_ms"]


def percentiles(values):
    values = np.asarray(values)
    return {
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", help="pickle written by bevfusion_node.py --save-results")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--out", help="write metrics and timing summary to this json file")
    args = parser.parse_args()

    with open(args.results, "rb") as f:
        run = pickle.load(f)
    records = run["records"]

    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)

    by_stamp = {r["stamp"]: r for r in records}
    outputs, matched = [], 0
    for info in dataset.data_infos:
        record = by_stamp.get(info["timestamp"])
        if record is None:
            boxes = LiDARInstance3DBoxes(torch.zeros(0, 9), box_dim=9)
            scores, labels = torch.zeros(0), torch.zeros(0, dtype=torch.long)
        else:
            matched += 1
            boxes = LiDARInstance3DBoxes(
                torch.from_numpy(record["boxes"]), box_dim=record["box_dim"]
            )
            scores = torch.from_numpy(record["scores"])
            labels = torch.from_numpy(record["labels"]).long()
        outputs.append(dict(boxes_3d=boxes, scores_3d=scores, labels_3d=labels))

    eval_kwargs = cfg.get("evaluation", {}).copy()
    for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule", "pipeline"]:
        eval_kwargs.pop(key, None)
    metrics = dataset.evaluate(outputs, metric="bbox", **eval_kwargs)

    keyframe_records = [r for r in records if r["stamp"] in {i["timestamp"] for i in dataset.data_infos}]
    summary = dict(
        results=args.results,
        node_args=run["args"],
        node_stats=run["stats"],
        keyframes=len(dataset),
        keyframes_processed=matched,
        mAP=metrics["object/map"],
        NDS=metrics["object/nds"],
        timing_all_frames={k: percentiles([r[k] for r in records]) for k in TIMING_KEYS}
        if records
        else {},
        max_camera_offset_ms=float(
            max(abs(v) for r in keyframe_records for v in r["camera_offsets_ms"].values())
        )
        if keyframe_records
        else None,
        metrics=metrics,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "metrics"}, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
