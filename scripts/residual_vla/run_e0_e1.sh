#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLINF="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# E0: collect a short scale-zero trace. Do not allow learner updates.
"${SCRIPT_DIR}/run_residual_smoke.sh" \
  runner.logger.experiment_name=pi05_residual_e0 \
  runner.max_epochs=1 \
  algorithm.update_epoch=0 \
  actor.model.residual.scale=0.0
python "${SCRIPT_DIR}/validate_e0_trace.py" \
  "${RLINF}/results/residual_vla/traces/pi05_residual_e0"

# E1: three controlled amplitudes. Inspect the generated summaries for action
# change, finite values, and measurable saturation before opening E2.
for scale in 0.0 0.25 0.5; do
  run_id="pi05_residual_e1_${scale//./p}"
  "${SCRIPT_DIR}/run_residual_smoke.sh" \
    runner.logger.experiment_name="${run_id}" \
    runner.max_epochs=1 \
    algorithm.update_epoch=0 \
    actor.model.residual.scale="${scale}"
  python "${SCRIPT_DIR}/summarize_residual_traces.py" \
    "${RLINF}/results/residual_vla/traces/${run_id}"
done

echo "E1_RANDOM_RESIDUAL_PASS may be recorded only after the environment response and gripper encoding checks are reviewed."
