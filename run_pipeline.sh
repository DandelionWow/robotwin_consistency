#!/usr/bin/env bash
set -euo pipefail

# 运行 cube_center_x0p00_yneg0p15_z1p00_edge* 这一整组空桌面立方体实验.
# 覆盖范围: 左/右臂 * X/Y/Z 三轴 * 5/10/15cm 三种边长 * 每组 8 个 episode.
#
# 用法:
#   ./run_pipeline.sh all
#   ./run_pipeline.sh robotwin
#   ./run_pipeline.sh convert
#   ./run_pipeline.sh bwm
#   ./run_pipeline.sh diagnostics
#
# 常用变量:
#   GPU_ROBOTWIN=0 GPU_BWM=1 ./run_pipeline.sh all
#   ROBOTWIN_ENV=robotwin_cons BWM_ENV=bwm_cons ./run_pipeline.sh all

STAGE="${1:-all}"

ROBOTWIN_ENV="${ROBOTWIN_ENV:-robotwin_cons}"
BWM_ENV="${BWM_ENV:-bwm_cons}"
GPU_ROBOTWIN="${GPU_ROBOTWIN:-0}"
GPU_BWM="${GPU_BWM:-0}"

CENTER_ARGS=(0 -0.15 1.0)
RUN_PREFIX="cube_center_x0p00_yneg0p15_z1p00"

BWM_MODEL_PATH="${BWM_MODEL_PATH:-/data1/sunyang/models/modelscope/models/Wan-AI/Wan2.2-TI2V-5B}"
BWM_CKPT_PATH="${BWM_CKPT_PATH:-/data1/sunyang/dev/world-model/wm_for_vla_rl/ckpt/BLM/step-12000.safetensors}"

ARMS=(left right)
AXES=(x y z)
EDGES=("0p05:0.05" "0p10:0.10" "0p15:0.15")

run_ids() {
  for ARM in "${ARMS[@]}"; do
    for AXIS in "${AXES[@]}"; do
      for EDGE in "${EDGES[@]}"; do
        EDGE_TAG="${EDGE%%:*}"
        EDGE_LEN="${EDGE##*:}"
        RUN_ID="${RUN_PREFIX}_edge${EDGE_TAG}_axis_${AXIS}_arm_${ARM}"
        printf '%s %s %s %s\n' "${RUN_ID}" "${ARM}" "${AXIS}" "${EDGE_LEN}"
      done
    done
  done
}

run_robotwin() {
  run_ids | while read -r RUN_ID ARM AXIS EDGE_LEN; do
    echo "[robotwin] ${RUN_ID}"
    env CUDA_VISIBLE_DEVICES="${GPU_ROBOTWIN}" conda run --no-capture-output -n "${ROBOTWIN_ENV}" \
      python -u run_cube_motion.py \
      --run_id "${RUN_ID}" \
      --arm "${ARM}" \
      --axis "${AXIS}" \
      --cube_center "${CENTER_ARGS[@]}" \
      --edge_length "${EDGE_LEN}" \
      --seed 0
  done
}

run_convert() {
  run_ids | while read -r RUN_ID _ARM _AXIS _EDGE_LEN; do
    echo "[convert/full] ${RUN_ID}"
    conda run --no-capture-output -n "${BWM_ENV}" \
      python -u robotwin_to_bwm.py \
      --robotwin_dir "outputs/robotwin/${RUN_ID}" \
      --output_dir "outputs/wm/${RUN_ID}/input"

    echo "[convert/start] ${RUN_ID}"
    conda run --no-capture-output -n "${BWM_ENV}" \
      python -u robotwin_to_bwm.py \
      --robotwin_dir "outputs/robotwin/${RUN_ID}" \
      --output_dir "outputs/wm/${RUN_ID}/input_from_experiment_start" \
      --crop_to_experiment
  done
}

run_bwm() {
  run_ids | while read -r RUN_ID ARM _AXIS _EDGE_LEN; do
    echo "[bwm/full] ${RUN_ID}"
    env CUDA_VISIBLE_DEVICES="${GPU_BWM}" TORCHINDUCTOR_COMPILE_THREADS=1 USE_TF=0 USE_FLAX=0 \
      conda run --no-capture-output -n "${BWM_ENV}" \
      python -u bwm_infer.py \
      --input_dir "outputs/wm/${RUN_ID}/input" \
      --output_dir "outputs/wm/${RUN_ID}/video" \
      --robotwin_dir "outputs/robotwin/${RUN_ID}" \
      --overlay_camera head_camera \
      --overlay_arm "${ARM}" \
      --model_paths "${BWM_MODEL_PATH}" \
      --ckpt_path "${BWM_CKPT_PATH}" \
      --height 480 \
      --width 640 \
      --num_frames 81 \
      --num_history_frames 9 \
      --num_inference_steps 50 \
      --fps 24

    echo "[bwm/start] ${RUN_ID}"
    env CUDA_VISIBLE_DEVICES="${GPU_BWM}" TORCHINDUCTOR_COMPILE_THREADS=1 USE_TF=0 USE_FLAX=0 \
      conda run --no-capture-output -n "${BWM_ENV}" \
      python -u bwm_infer.py \
      --input_dir "outputs/wm/${RUN_ID}/input_from_experiment_start" \
      --output_dir "outputs/wm/${RUN_ID}/video_from_experiment_start" \
      --robotwin_dir "outputs/robotwin/${RUN_ID}" \
      --overlay_camera head_camera \
      --overlay_arm "${ARM}" \
      --overlay_start_frame_from_scene \
      --model_paths "${BWM_MODEL_PATH}" \
      --ckpt_path "${BWM_CKPT_PATH}" \
      --height 480 \
      --width 640 \
      --num_frames 81 \
      --num_history_frames 9 \
      --num_inference_steps 50 \
      --fps 24
  done
}

run_diagnostics() {
  run_ids | while read -r RUN_ID _ARM _AXIS _EDGE_LEN; do
    echo "[diagnostics] ${RUN_ID}"
    conda run --no-capture-output -n "${BWM_ENV}" \
      python -u draw_cube_vertices.py \
      --robotwin_dir "outputs/robotwin/${RUN_ID}" \
      --output_path "outputs/robotwin/${RUN_ID}/diagnostics/cuboid_vertices_head_camera.png" \
      --camera head_camera
  done
}

case "${STAGE}" in
  all)
    run_robotwin
    run_convert
    run_bwm
    run_diagnostics
    ;;
  robotwin)
    run_robotwin
    ;;
  convert)
    run_convert
    ;;
  bwm)
    run_bwm
    ;;
  diagnostics)
    run_diagnostics
    ;;
  *)
    echo "Unknown stage: ${STAGE}" >&2
    echo "Usage: $0 {all|robotwin|convert|bwm|diagnostics}" >&2
    exit 2
    ;;
esac
