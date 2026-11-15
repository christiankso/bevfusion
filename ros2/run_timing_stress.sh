#!/bin/bash
# Timing stress tests for the BEVFusion ROS 2 node (run inside the container
# from /workspace/bevfusion). Each run starts the node, replays the mini-val
# scenes, then scores the saved output. Finished runs are skipped.
#
# Usage: bash ros2/run_timing_stress.sh [name ...]   (default: all runs)

source /opt/ros/foxy/setup.bash
OUT=results/ros/stress
mkdir -p "$OUT"

# name | node options | replay options
RUNS=(
  "rate1|--idle-timeout 10|--rate 1"
  "rate1_r2|--idle-timeout 10|--rate 1"
  "rate1_r3|--idle-timeout 10|--rate 1"
  "rate2|--idle-timeout 10|--rate 2"
  "rate4|--idle-timeout 10|--rate 4"
  "kf_clock0|--keyframes-only --idle-timeout 10|--rate 1"
  "kf_clock50|--keyframes-only --idle-timeout 10|--rate 1 --camera-clock-offset-ms 50"
  "kf_clock100|--keyframes-only --idle-timeout 10|--rate 1 --camera-clock-offset-ms 100"
  "kf_clock200|--keyframes-only --idle-timeout 10|--rate 1 --camera-clock-offset-ms 200"
  "kf_clock500|--keyframes-only --idle-timeout 10|--rate 1 --camera-clock-offset-ms 500"
  "kf_nocams|--keyframes-only --degrade drop_cams=[CAM_FRONT,CAM_FRONT_RIGHT,CAM_FRONT_LEFT,CAM_BACK,CAM_BACK_LEFT,CAM_BACK_RIGHT] --idle-timeout 10|--rate 1"
  "delay100|--idle-timeout 10|--rate 1 --camera-delay-ms 100"
  "delay300|--idle-timeout 10|--rate 1 --camera-delay-ms 300"
  "delay300_wait500|--sync-timeout 0.5 --idle-timeout 10|--rate 1 --camera-delay-ms 300"
  "sweeps0|--sweeps 0 --idle-timeout 10|--rate 1"
  "pointdrop75|--degrade point_drop=0.75 --idle-timeout 10|--rate 1"
  "dark25|--degrade brightness=0.25 --idle-timeout 10|--rate 1"
)

for run in "${RUNS[@]}"; do
  IFS="|" read -r name node_opts replay_opts <<< "$run"
  if [ $# -gt 0 ] && [[ ! " $* " =~ " $name " ]]; then continue; fi
  if [ -f "$OUT/$name.json" ]; then echo "skip $name (done)"; continue; fi
  echo "=== $name"
  python ros2/bevfusion_node.py $node_opts --save-results "$OUT/$name.pkl" > "$OUT/$name.node.log" 2>&1 &
  node_pid=$!
  # Wait until the model is loaded.
  for _ in $(seq 120); do
    grep -q "BEVFusion node ready" "$OUT/$name.node.log" && break
    sleep 1
  done
  # GPU use, sampled by a separate process so it does not slow the node.
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits \
    -lms 500 > "$OUT/$name.gpu.csv" &
  smi_pid=$!
  python ros2/nuscenes_replay.py $replay_opts > "$OUT/$name.replay.log" 2>&1
  kill $smi_pid
  wait $node_pid
  python ros2/evaluate_ros_results.py "$OUT/$name.pkl" --out "$OUT/$name.json" \
    --gpu-log "$OUT/$name.gpu.csv" > "$OUT/$name.eval.log" 2>&1 \
    || { echo "evaluation failed for $name"; tail -5 "$OUT/$name.eval.log"; }
  grep -h "stats" "$OUT/$name.node.log"
done
