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
export PI05_LIBERO_CHECKPOINT

export RESIDUAL_VLA_OUTPUT_ROOT="${RESIDUAL_VLA_OUTPUT_ROOT:-${RLINF}/results/residual_vla/logs}"
export RESIDUAL_VLA_TRACE_ROOT="${RESIDUAL_VLA_TRACE_ROOT:-${RLINF}/results/residual_vla/traces}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

export LIBERO_TYPE="${LIBERO_TYPE:-standard}"
export LIBERO_SUFFIX="${LIBERO_SUFFIX:-all}"

export RESIDUAL_TRAIN_PLACEMENT="${RESIDUAL_TRAIN_PLACEMENT:-0-1}"
unset CUDA_VISIBLE_DEVICES || true

python examples/embodiment/train_embodied_agent.py \
  --config-name libero_pi05_residual_ac_formal \
  "$@"
