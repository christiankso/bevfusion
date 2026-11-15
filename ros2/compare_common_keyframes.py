"""Score several node runs using only the keyframes that every run processed.

Runs that skip different keyframes (e.g. no camera image yet at the start of
a scene) otherwise differ partly because of the skipped frames, which count
as empty predictions. Here any keyframe not processed by all runs is left
empty for every run (the nuScenes evaluator needs all samples), so all runs
carry the same penalty and differences come only from the shared keyframes.
Absolute values are therefore lower than a full evaluation; compare runs
with each other.

Example (inside the container):
    python ros2/compare_common_keyframes.py results/ros/stress/kf_clock0.pkl \
        results/ros/stress/kf_clock500.pkl --out results/ros/stress/kf_common.json
"""

import argparse
import json
import os
import pickle

import torch
from mmcv import Config
from torchpack.utils.config import configs

from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets import build_dataset
from mmdet3d.utils import recursive_eval

DEFAULT_CONFIG = (
    "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", nargs="+", help="pickles written by bevfusion_node.py")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--out")
    args = parser.parse_args()

    runs = {}
    for path in args.results:
        with open(path, "rb") as f:
            records = pickle.load(f)["records"]
        runs[os.path.basename(path)[: -len(".pkl")]] = {r["stamp"]: r for r in records}

    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    eval_kwargs = cfg.get("evaluation", {}).copy()
    for key in ["interval", "tmpdir", "start", "gpu_collect", "save_best", "rule", "pipeline"]:
        eval_kwargs.pop(key, None)

    common = set.intersection(*(set(r) for r in runs.values()))
    summary = {
        "common_keyframes": sum(i["timestamp"] in common for i in dataset.data_infos),
        "keyframes": len(dataset.data_infos),
        "runs": {},
    }
    for name, by_stamp in runs.items():
        outputs = []
        for info in dataset.data_infos:
            if info["timestamp"] not in common:
                outputs.append(
                    dict(
                        boxes_3d=LiDARInstance3DBoxes(torch.zeros(0, 9), box_dim=9),
                        scores_3d=torch.zeros(0),
                        labels_3d=torch.zeros(0, dtype=torch.long),
                    )
                )
                continue
            r = by_stamp[info["timestamp"]]
            outputs.append(
                dict(
                    boxes_3d=LiDARInstance3DBoxes(torch.from_numpy(r["boxes"]), box_dim=r["box_dim"]),
                    scores_3d=torch.from_numpy(r["scores"]),
                    labels_3d=torch.from_numpy(r["labels"]).long(),
                )
            )
        metrics = dataset.evaluate(outputs, metric="bbox", **eval_kwargs)
        summary["runs"][name] = {
            k.split("/")[1]: metrics[k]
            for k in ["object/map", "object/nds", "object/mATE", "object/mAOE", "object/mAVE"]
        }
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
