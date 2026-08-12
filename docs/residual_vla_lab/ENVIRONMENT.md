# Environment

## Locked source

- RLinf: `e42803e1cecc5719c0ff39b8f7c1cfee9d90b942`
- RLinf branch: `agent/pi05-explicit-residual-v0`
- OpenPI reference snapshot: `15a9616a00943ada6c20a0f158e3adb39df2ccac`

## Local workstation audit (2026-08-12)

- OS/shell: Windows PowerShell 5.1
- Python: 3.12.5
- PyTorch: 2.4.0+cpu in the project test environment
- CUDA visible to project PyTorch: no
- MuJoCo/LIBERO/OpenPI runtime: not installed in the lightweight test venv

The local workstation cannot execute the required two-H200 pi0.5 rollout. The
project-local `.venv` is used only for CPU tests; global Python, CUDA, pip,
shell-profile, and conda configuration are not modified.

## Target server install and launch

From the repository root:

```bash
bash scripts/residual_vla/setup_server_env.sh
export PI05_LIBERO_CHECKPOINT=/absolute/path/to/pinned/pi05_libero
bash scripts/residual_vla/run_residual_smoke.sh
```

The smoke config expects exactly two visible GPUs by default and places
actor/env/rollout on both. It uses two full frozen pi0.5 replicas, two full
residual actor/twin-Q learners, FSDP `no_shard`, local CPU replay, and no eval
envs (`val_check_interval=-1`).

After the 4-env wiring smoke is stable, profile conservative total env counts:

```bash
bash scripts/residual_vla/profile_env_counts.sh
```

This runs 32 then 64 total envs, divided by the existing EnvWorkers. Review
throughput, host RAM, simulator stability, and GPU memory using throughput batch
`micro=128, global=256` before optionally trying 96. Do not open 128 envs in the
first profiling pass.

After all research-readiness gates pass, the first formal candidate is:

```bash
bash scripts/residual_vla/run_residual_formal.sh
```

It uses 64 total envs and `micro_batch_size=256,
global_batch_size=512`. With two learner ranks this is 256 replay transitions
per GPU per optimizer update, processed as one microbatch. The larger global
batch does not put all 512 transitions on either GPU at once; the microbatch is
the primary batch-related activation-memory control. Batch size remains an RL
hyperparameter, so retain learner throughput, Q stability, and sample/update
ratio metrics when comparing 256 against 512.

The current explicit-residual v0 replay observation is only `proprio` plus
`ref_chunk`; it intentionally does not store RLT `z_rl`/hidden tokens. If a
later experiment adds image observations or hidden-token context, keep the
formal global batch at 512 initially and lower `micro_batch_size` to 64 or 32
to control peak memory (which increases accumulation steps per rank).

Record torch/CUDA/MuJoCo/LIBERO/LIBERO-Plus/OpenPI versions and the checkpoint
artifact hash before Phase A is marked PASS.
