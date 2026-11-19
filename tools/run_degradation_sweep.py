"""Run BEVFusion evaluation under each sensor-degradation condition.

Each condition calls tools/test.py with --degrade and writes
<out-dir>/<group>__<name>.json; finished conditions are skipped, so the
sweep can be resumed. A summary CSV is written to <out-dir>/summary.csv.

Example (inside the container, from /workspace/bevfusion):
    python tools/run_degradation_sweep.py --groups lidar_point_drop cam_drop
"""

import argparse
import csv
import json
import os
import subprocess

FUSION_CONFIG = (
    "configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml"
)
FUSION_CHECKPOINT = "pretrained/bevfusion-det.pth"

ALL_CAMS = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

# group -> list of (name, degrade options)
CONDITIONS = {
    "baseline": [("clean", {})],
    "lidar_point_drop": [(f"drop{p}", {"point_drop": p}) for p in [0.25, 0.5, 0.75, 0.9]],
    "lidar_beams": [(f"beams{b}", {"reduce_beams": b}) for b in [16, 4, 1]],
    "lidar_sweeps": [(f"sweeps{s}", {"sweeps_num": s}) for s in [4, 1, 0]],
    "lidar_range": [(f"range{r}m", {"max_range": r}) for r in [40, 30, 20, 10]],
    "cam_brightness": [(f"bright{b}", {"brightness": b}) for b in [0.5, 0.25, 0.1]],
    "cam_blur": [(f"blur{r}px", {"blur": r}) for r in [2, 5, 10]],
    "cam_drop": [
        ("front", {"drop_cams": ["CAM_FRONT"]}),
        ("front_back", {"drop_cams": ["CAM_FRONT", "CAM_BACK"]}),
        ("all", {"drop_cams": ALL_CAMS}),
    ],
    # Both sensors degraded at once, approximating weather and hardware faults.
    "combined": [
        ("fog_light", {"contrast": 0.6, "blur": 2, "max_range": 40, "point_drop": 0.25}),
        ("fog_heavy", {"contrast": 0.3, "blur": 5, "max_range": 25, "point_drop": 0.5}),
        ("rain_light", {"brightness": 0.8, "blur": 2, "point_drop": 0.25}),
        ("rain_heavy", {"brightness": 0.6, "blur": 5, "point_drop": 0.5, "max_range": 35}),
        ("night_rain", {"brightness": 0.2, "blur": 3, "point_drop": 0.4}),
        ("hw_failure", {"drop_cams": ["CAM_FRONT", "CAM_FRONT_LEFT"], "reduce_beams": 16}),
    ],
    # LiDAR stressors repeated with all cameras off; comparing against the
    # matching lidar_* run (cameras on) shows how much the camera rescues.
    "lidar_stress_no_cams": [
        (name, dict(options, drop_cams=ALL_CAMS))
        for name, options in [
            ("drop0.75", {"point_drop": 0.75}),
            ("drop0.9", {"point_drop": 0.9}),
            ("beams16", {"reduce_beams": 16}),
            ("beams4", {"reduce_beams": 4}),
            ("beams1", {"reduce_beams": 1}),
            ("sweeps0", {"sweeps_num": 0}),
            ("range30m", {"max_range": 30}),
            ("range20m", {"max_range": 20}),
        ]
    ],
}

CLASSES = [
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
    "pedestrian",
    "motorcycle",
    "bicycle",
    "traffic_cone",
    "barrier",
]


def degrade_args(options):
    args = []
    for key, value in options.items():
        if isinstance(value, list):
            value = "[" + ",".join(str(v) for v in value) + "]"
        args.append(f"{key}={value}")
    return args


def run_condition(config, checkpoint, out_path, options, jsonfile_prefix=None):
    cmd = [
        "torchpack", "dist-run", "-np", "1",
        "python", "tools/test.py", config, checkpoint,
        "--eval", "bbox",
        "--metrics-out", out_path,
    ]
    if options:
        cmd += ["--degrade"] + degrade_args(options)
    if jsonfile_prefix:
        # Keep the nuScenes submission (results_nusc.json) and the official
        # metrics_summary.json / metrics_details.json in this folder.
        cmd += ["--eval-options", f"jsonfile_prefix={jsonfile_prefix}"]
    log_path = out_path[: -len(".json")] + ".log"
    print(" ".join(cmd), flush=True)
    with open(log_path, "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)


def class_ap(metrics, name):
    keys = [k for k in metrics if k.startswith(f"object/{name}_ap_dist_")]
    return sum(metrics[k] for k in keys) / len(keys) if keys else float("nan")


def write_summary(out_dir):
    rows = []
    for group in CONDITIONS:
        for name, options in CONDITIONS[group]:
            path = os.path.join(out_dir, f"{group}__{name}.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                result = json.load(f)
            metrics = result["metrics"]
            row = {
                "group": group,
                "condition": name,
                "degrade": json.dumps(options),
                "mAP": metrics["object/map"],
                "NDS": metrics["object/nds"],
                "wall_time_s": result["wall_time_s"],
                "fps": result["num_frames"] / result["wall_time_s"],
            }
            for cls in CLASSES:
                row[f"AP_{cls}"] = class_ap(metrics, cls)
            rows.append(row)
    summary_path = os.path.join(out_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=FUSION_CONFIG)
    parser.add_argument("--checkpoint", default=FUSION_CHECKPOINT)
    parser.add_argument("--out-dir", default="results/degradation/fusion")
    parser.add_argument(
        "--groups",
        nargs="+",
        default=list(CONDITIONS),
        choices=list(CONDITIONS),
        help="condition groups to run (default: all)",
    )
    parser.add_argument(
        "--save-nusc-json",
        action="store_true",
        help="also keep results_nusc.json and the official metrics files in "
        "<out-dir>/nusc/<group>__<name>/ (reruns conditions that lack them)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    groups = ["baseline"] + [g for g in args.groups if g != "baseline"]
    for group in groups:
        for name, options in CONDITIONS[group]:
            out_path = os.path.join(args.out_dir, f"{group}__{name}.json")
            prefix = None
            if args.save_nusc_json:
                prefix = os.path.join(args.out_dir, "nusc", f"{group}__{name}")
            done = os.path.exists(out_path) and (
                prefix is None or os.path.exists(os.path.join(prefix, "results_nusc.json"))
            )
            if done:
                print(f"skip {group}/{name} (done)")
                continue
            run_condition(args.config, args.checkpoint, out_path, options, prefix)
    write_summary(args.out_dir)


if __name__ == "__main__":
    main()
