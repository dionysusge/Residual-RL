# Communication boundary

## CPU replay ownership

Replay remains RLinf's existing rank-local CPU storage. The trajectory builder
moves actions, rewards, done fields, forward inputs, `curr_obs`, and `next_obs`
to contiguous CPU tensors. Replay cache insertion clones tensors to CPU again.
Only a sampled microbatch is moved to the learner device immediately before the
critic/actor passes.

There is no GPU-resident replay, central replay process, replay mirroring, or
new replay transport. Pinned-memory/non-blocking transfer is deliberately not
introduced in v0; it may be profiled later if host-to-device copies are proven
to matter.

The replay payload is the existing trajectory schema:

- `curr_obs={proprio, ref_chunk}`
- executed `action`
- reward and done fields
- `next_obs={proprio, ref_chunk}`

No `z_rl` is created or stored.

## Distributed learner

Both GPUs are actor ranks under the existing FSDP backend with
`sharding_strategy: no_shard`. Every rank has a complete Residual Actor, Q1/Q2,
Target Q1/Q2, and optimizer state. Each rank samples its own local CPU replay;
FSDP performs the existing gradient synchronization. No manual `all_reduce` or
new process group is added.

The frozen pi0.5 replicas are rollout-local auxiliary models. They have no
gradients and are absent from actor optimizers and gradient synchronization.
The existing residual-policy WeightSyncer refresh is retained; its rank-0
source is safe because learner parameters are synchronized across actor ranks.

## Environment and routing

Both hardware ranks host EnvWorkers. RLinf divides `total_num_envs` across env
ranks and uses the existing Channel split/routing and distributed readiness
reductions. Fixed evaluation reset partitioning is unchanged. No simulator
sampler, multiprocessing layer, or async queue is added.

## Rollout-local traces

Residual trace shards contain base/raw/bounded/scaled/preclip/executed actions,
proprio used, paired identity, rewards/dones/success when available, version,
seed, residual standard deviation, and lambda. They are atomically written
under `traces/<run>/rank_<n>/part_*.pt`; they are not synchronized per step.

## Do-not-touch audit

- Channel modified: NO
- ProcessGroup modified: NO
- scheduler core modified: NO
- WeightSyncer core modified: NO
- FSDP communication topology modified: NO
- distributed collectives modified: NO
- global Trajectory schema modified: NO
- generic replay-buffer schema modified: NO
- pi0.5 flow-matching internals modified: NO

