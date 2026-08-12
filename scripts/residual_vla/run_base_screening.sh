#!/usr/bin/env bash
set -euo pipefail

RLINF="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${PI05_LIBERO_CHECKPOINT:?Set PI05_LIBERO_CHECKPOINT to the pinned checkpoint directory}"

cd "${RLINF}"
. .venv/bin/activate
export REPO_PATH="${RLINF}"
export EMBODIED_PATH="${RLINF}/examples/embodiment"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

export LIBERO_TYPE=standard
bash evaluations/run_eval.sh libero libero_10_openpi_pi05_eval \
  rollout.model.model_path="${PI05_LIBERO_CHECKPOINT}" \
  env.eval.total_num_envs=8 env.eval.video_cfg.save_video=false

export LIBERO_TYPE=plus
export LIBERO_SUFFIX="${LIBERO_SUFFIX:-all}"
bash evaluations/run_eval.sh libero libero_10_openpi_pi05_eval \
  rollout.model.model_path="${PI05_LIBERO_CHECKPOINT}" \
  env.eval.total_num_envs=8 env.eval.video_cfg.save_video=false

echo "Parse fixed task/reset/seed results into ${RLINF}/results/residual_vla/base_screening.csv before task selection."
