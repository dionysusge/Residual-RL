#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Conservative first-pass throughput sweep. These are total env counts; the
# existing EnvWorker divides them across the two env ranks. Evaluation remains
# disabled so no second simulator pool is created during profiling.
for total_envs in 32 64; do
  run_id="pi05_residual_profile_env${total_envs}"
  "${SCRIPT_DIR}/run_residual_smoke.sh" \
    runner.logger.experiment_name="${run_id}" \
    runner.max_epochs=1 \
    runner.val_check_interval=-1 \
    algorithm.update_epoch=0 \
    env.train.total_num_envs="${total_envs}" \
    actor.micro_batch_size=128 \
    actor.global_batch_size=256
done

echo "Review throughput, CPU RAM, simulator stability, and H200 memory before trying 96 total envs."
