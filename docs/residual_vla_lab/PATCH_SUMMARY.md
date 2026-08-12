# Patch summary

## New implementation

- `rlinf/algorithms/residual/`: composer, context, rollout, transition, metrics,
  atomic traces, and frozen-model validation.
- `rlinf/envs/residual/`: benchmark base and shared standard/Plus LIBERO adapter.
- `residual_mlp_policy.py`: 3x256 MLP residual actor and independent critic
  context feeding the existing twin-Q heads.
- `fsdp_residual_ac_policy_worker.py`: Stage-2 learner reuse plus residual metrics.
- `libero_pi05_residual_ac_smoke.yaml`: synchronous two-GPU replicated learner,
  rollout, and env placement with CPU replay and evaluation disabled.
- `run_residual_formal.sh`: first formal candidate override with 64 envs,
  `micro=256`, and `global=512` across the two learner ranks.
- `profile_env_counts.sh`: conservative 32/64 total-env profiling sweep.
- CPU unit and integration tests for residual contracts and replay residency.

## Narrow existing-file hooks

- register the residual model and select the residual actor worker;
- add OpenPI `extract_residual_obs()` using existing inference/output transforms;
- let rollout hold a frozen base model beside the small synchronized actor;
- route the two-field residual transition in EnvWorker and write local traces;
- expose LIBERO task/reset/episode-step identities in info.

The adjustment does not add GPU replay, central replay, a sampler, manual
collectives, asynchronous scheduling, or a new communication path.
