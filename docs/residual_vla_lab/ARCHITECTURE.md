# Architecture

The v0 run is synchronous and off-policy. `pipeline_stage_num=1`; it does not
add an asynchronous producer/consumer pipeline.

```text
rank 0 (H200)                                      rank 1 (H200)
----------------------------                       ----------------------------
frozen pi0.5 rollout replica                       frozen pi0.5 rollout replica
complete residual actor + Q1/Q2                    complete residual actor + Q1/Q2
complete target Q1/Q2 + optimizers                 complete target Q1/Q2 + optimizers
rank-local CPU replay                              rank-local CPU replay
CPU EnvWorker share                                CPU EnvWorker share
             |                                                  |
             +-- FSDP no_shard gradient synchronization -------+
             +-- existing Channel trajectory routing ----------+
```

Each rollout rank computes a base action chunk with its own frozen pi0.5
replica, constructs residual context from current proprioception and that base
chunk, and composes

```text
A_exec = clip(A_base + lambda * bounded_delta_action)
```

The learner reuses RLinf's twin-Q, target-Q, optimizer, checkpoint, Channel,
trajectory, replay, readiness-reduction, and WeightSyncer plumbing. There is no
`z_rl`, visual residual encoder, central replay service, or GPU replay.

`actor.fsdp_config.sharding_strategy=no_shard` gives standard replicated data
parallel semantics: each actor rank samples a different minibatch from its
local CPU replay, then FSDP synchronizes actor and critic gradients. Target
networks are updated locally by the same deterministic Polyak rule. The frozen
pi0.5 replicas never participate in backward, optimizers, or gradient sync.

