#!/usr/bin/env bash
set -euo pipefail

RLINF="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${PI05_LIBERO_CHECKPOINT:?Set PI05_LIBERO_CHECKPOINT to the pinned checkpoint directory}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "No active Conda environment." >&2
  echo "Activate residual-vla-pi05 before running this script." >&2
  exit 2
fi

python_bin="$(command -v python || true)"
if [[ "${python_bin}" != "${CONDA_PREFIX}/bin/python" ]]; then
  echo "python does not belong to the active Conda environment:" >&2
  echo "  python=${python_bin}" >&2
  echo "  CONDA_PREFIX=${CONDA_PREFIX}" >&2
  exit 2
fi

cd "${RLINF}"

export REPO_PATH="${RLINF}"
export EMBODIED_PATH="${RLINF}/examples/embodiment"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

export RESIDUAL_BASE_PLACEMENT="${RESIDUAL_BASE_PLACEMENT:-0}"

if [[ $# -gt 0 ]]; then
  variant="$1"
  shift
else
  variant="standard"
fi

case "${variant}" in
  standard)
    export LIBERO_TYPE=standard
    unset LIBERO_SUFFIX || true
    ;;
  plus)
    export LIBERO_TYPE=plus
    export LIBERO_SUFFIX="${LIBERO_SUFFIX:-all}"
    ;;
  *)
    echo "Usage: $0 [standard|plus] [hydra overrides...]" >&2
    exit 2
    ;;
esac

# Do not use CUDA_VISIBLE_DEVICES as an RLinf placement mechanism.
unset CUDA_VISIBLE_DEVICES || true

episodes="${BASE_SCREEN_EPISODES:-20}"
task_ids="${BASE_SCREEN_TASK_IDS:-0 1 2 3 4 5 6 7 8 9}"

echo "Base screening"
echo "  python:       $(command -v python)"
echo "  variant:      ${LIBERO_TYPE}"
echo "  placement:    ${RESIDUAL_BASE_PLACEMENT}"
echo "  episodes/task:${episodes}"
echo "  tasks:        ${task_ids}"

for task_id in ${task_ids}; do
  echo
  echo "============================================================"
  echo "BASE SCREEN | ${LIBERO_TYPE} | TASK ${task_id}"
  echo "============================================================"

  bash evaluations/run_eval.sh \
    libero \
    libero_10_openpi_pi05_residual_base_eval \
    runner.logger.experiment_name="pi05_base_${LIBERO_TYPE}_task${task_id}" \
    env.eval.total_num_envs="${episodes}" \
    env.eval.task_id_filter="[${task_id}]" \
    env.eval.video_cfg.save_video=false \
    "$@"
done
