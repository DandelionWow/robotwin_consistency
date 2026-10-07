#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  COLLECTION_PYTHON=/abs/path/to/python \
  CUDA_ROOT=/abs/path/to/cuda \
  CUROBO_SOURCE=/abs/path/to/clean/curobo \
  bash experiments/policy_shift/run_remote_matched_collection.sh preflight

  COLLECTION_PYTHON=/abs/path/to/python \
  CUDA_ROOT=/abs/path/to/cuda \
  CUROBO_SOURCE=/abs/path/to/clean/curobo \
  bash experiments/policy_shift/run_remote_matched_collection.sh collect

Optional environment variables:
  GPU_ID=0                 Physical GPU index; GPU 5 is the authorized remote exception.
  DENOISER=optix           One of oidn, optix, or none.
  XLA_MEMORY_FRACTION=0.4
  OUTPUT_DIR=<repo>/outputs/policy_shift/matched_raw
EOF
}

if [[ $# -ne 1 || ( "$1" != "preflight" && "$1" != "collect" ) ]]; then
  usage >&2
  exit 2
fi
mode=$1

: "${COLLECTION_PYTHON:?Set COLLECTION_PYTHON to the exact environment Python executable}"
: "${CUDA_ROOT:?Set CUDA_ROOT to a CUDA toolkit containing bin/ptxas}"
: "${CUROBO_SOURCE:?Set CUROBO_SOURCE to the clean CuRobo checkout used by the editable install}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "$script_dir/../.." && pwd)
robotwin_root="$project_root/third_party/robotwin"
provenance_dir="$project_root/experiments/policy_shift/provenance"
task_config="$robotwin_root/task_config/demo_clean.yml"
output_dir=${OUTPUT_DIR:-"$project_root/outputs/policy_shift/matched_raw"}
gpu_id=${GPU_ID:-0}
denoiser=${DENOISER:-optix}
xla_memory_fraction=${XLA_MEMORY_FRACTION:-0.4}

case "$gpu_id" in
  0|1|2|3|5) ;;
  *) echo "GPU_ID must be 0, 1, 2, 3, or the authorized remote exception 5" >&2; exit 2 ;;
esac
case "$denoiser" in
  oidn|optix|none) ;;
  *) echo "DENOISER must be oidn, optix, or none" >&2; exit 2 ;;
esac

required_robotwin_commit=ce63ccb13e7b3e891ed6209b9b374c07e73c9311
actual_robotwin_commit=$(git -C "$robotwin_root" rev-parse HEAD)
if [[ "$actual_robotwin_commit" != "$required_robotwin_commit" ]]; then
  echo "RoboTwin commit mismatch" >&2
  echo "required: $required_robotwin_commit" >&2
  echo "actual:   $actual_robotwin_commit" >&2
  exit 1
fi
if [[ -n "$(git -C "$robotwin_root" status --porcelain=v1 --untracked-files=no)" ]]; then
  echo "RoboTwin has tracked modifications; remote formal collection requires a clean checkout" >&2
  git -C "$robotwin_root" status --short >&2
  exit 1
fi

for path in "$COLLECTION_PYTHON" "$CUDA_ROOT/bin/ptxas" "$task_config"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path does not exist: $path" >&2
    exit 1
  fi
done
if [[ ! -d "$CUROBO_SOURCE/.git" ]]; then
  echo "CUROBO_SOURCE is not a git checkout: $CUROBO_SOURCE" >&2
  exit 1
fi

code_provenance="$provenance_dir/code_provenance.json"
if [[ -f "$code_provenance" ]]; then
  echo "Reusing existing code provenance: $code_provenance"
  echo "The collector will reject it if the root or RoboTwin commit is stale."
else
  "$COLLECTION_PYTHON" "$project_root/experiments/policy_shift/capture_code_provenance.py" \
    --root "$project_root" \
    --output-dir "$provenance_dir"
fi

tasks=(
  blocks_ranking_size
  blocks_ranking_size
  hanging_mug
  hanging_mug
  stamp_seal
  stamp_seal
)
seeds=(200001 200002 200001 200002 200003 200004)

common_args=(
  --task-config "$task_config"
  --output-dir "$output_dir"
  --gpu-id "$gpu_id"
  --cuda-root "$CUDA_ROOT"
  --curobo-source "$CUROBO_SOURCE"
  --xla-memory-fraction "$xla_memory_fraction"
  --physics-timestep 0.004
  --physics-steps-per-sample 25
  --bwm-sampling-stride 1
  --instruction-type unseen
  --denoiser "$denoiser"
)

for index in "${!tasks[@]}"; do
  task=${tasks[$index]}
  seed=${seeds[$index]}
  policy_config="$project_root/experiments/policy_shift/configs/pi05_${task}.json"
  echo "PREFLIGHT task=$task seed=$seed gpu=$gpu_id"
  PYTHONNOUSERSITE=1 "$COLLECTION_PYTHON" \
    "$project_root/experiments/policy_shift/collect_matched_pair.py" \
    --task "$task" \
    --env-seed "$seed" \
    --policy-config "$policy_config" \
    "${common_args[@]}" \
    --preflight-only
done

if [[ "$mode" == "preflight" ]]; then
  echo "All six preflights passed; no trajectory was collected."
  exit 0
fi

echo "Formal collection is paused: three candidate seeds failed the Expert gate." >&2
echo "Complete the documented Expert-only sequential seed screen and commit the final seed plan first." >&2
exit 1

for index in "${!tasks[@]}"; do
  task=${tasks[$index]}
  seed=${seeds[$index]}
  policy_config="$project_root/experiments/policy_shift/configs/pi05_${task}.json"
  echo "COLLECT task=$task seed=$seed gpu=$gpu_id"
  PYTHONNOUSERSITE=1 "$COLLECTION_PYTHON" \
    "$project_root/experiments/policy_shift/collect_matched_pair.py" \
    --task "$task" \
    --env-seed "$seed" \
    --policy-config "$policy_config" \
    "${common_args[@]}"
done

manifest_count=$(find "$output_dir" -mindepth 2 -maxdepth 2 -type f -name manifest.json | wc -l)
if [[ "$manifest_count" -ne 6 ]]; then
  echo "Expected exactly 6 pair manifests, found $manifest_count" >&2
  exit 1
fi

(
  cd -- "$output_dir"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS
)
echo "Collection complete: $output_dir"
echo "Transfer the entire directory together with $output_dir/SHA256SUMS"
