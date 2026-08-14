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

# Logical RLinf accelerator rank, not CUDA_VISIBLE_DEVICES.
export RESIDUAL_BASE_PLACEMENT="${RESIDUAL_BASE_PLACEMENT:-0}"

export LIBERO_TYPE="${LIBERO_TYPE:-standard}"

# Avoid an inherited shell mask disagreeing with Ray's accelerator rank map.
unset CUDA_VISIBLE_DEVICES || true

task_id="${BASE_SMOKE_TASK_ID:-0}"
num_envs="${BASE_SMOKE_ENVS:-4}"

echo "Residual base episode smoke"
echo "  python:     $(command -v python)"
echo "  placement: ${RESIDUAL_BASE_PLACEMENT}"
echo "  variant:   ${LIBERO_TYPE}"
echo "  task:      ${task_id}"
echo "  episodes:  ${num_envs}"

bash evaluations/run_eval.sh \
  libero \
  libero_10_openpi_pi05_residual_base_eval \
  runner.logger.experiment_name="pi05_base_smoke_task${task_id}" \
  env.eval.total_num_envs="${num_envs}" \
  env.eval.task_id_filter="[${task_id}]" \
  env.eval.video_cfg.save_video=true \
  "$@"
