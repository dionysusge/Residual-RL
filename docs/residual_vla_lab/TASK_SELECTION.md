# Task selection

Status: **PENDING base screening on the target server**.

`results/base_screening.csv` contains only its schema; no success rates were
fabricated on the CPU-only workstation. After fixed reset IDs, inference seeds,
and task IDs are evaluated, rank candidates by:

1. high standard-LIBERO success;
2. LIBERO-Plus success in the 50–90% target interval;
3. repeatable shifted failures;
4. non-extreme episode length.

The top-ranked task/perturbation may be used for the smoke run. Record the
selected task and both standard/Plus rates here before any formal training.

