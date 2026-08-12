#!/usr/bin/env bash
set -euo pipefail

RLINF="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${PI05_LIBERO_CHECKPOINT:?Set PI05_LIBERO_CHECKPOINT to the pinned checkpoint directory}"

cd "${RLINF}"
. .venv/bin/activate
export REPO_PATH="${RLINF}"
export PI05_LIBERO_CHECKPOINT
export RESIDUAL_VLA_OUTPUT_ROOT="${RLINF}/results/residual_vla/logs"
export RESIDUAL_VLA_TRACE_ROOT="${RLINF}/results/residual_vla/traces"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export LIBERO_TYPE="${LIBERO_TYPE:-plus}"
export LIBERO_SUFFIX="${LIBERO_SUFFIX:-all}"

IFS=',' read -r -a residual_visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#residual_visible_gpus[@]}" -ne 2 ]]; then
  echo "Expected exactly two CUDA_VISIBLE_DEVICES entries, got: ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

python examples/embodiment/train_embodied_agent.py \
  --config-name libero_pi05_residual_ac_smoke \
  runner.logger.experiment_name=pi05_libero_residual_v0_formal \
  env.train.total_num_envs=64 \
  actor.micro_batch_size=256 \
  actor.global_batch_size=512 \
  "$@"
