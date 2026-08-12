# Static audit

The relevant RLinf paths were reviewed at the locked commit.

- RLT rollout extracts feature/reference state locally before the small policy.
- RLT transition plumbing uses existing `curr_obs/next_obs` trajectory slots.
- Stage-2 already provides twin Q, target Q, off-policy replay, BC/reference
  regularization, checkpointing, and small-policy synchronization.
- `EmbodiedTrajectoryBuilder.to_trajectory()` makes trajectory tensors
  contiguous on CPU.
- `TrajectoryReplayBuffer` clones cached tensors to CPU.
- SAC moves only sampled microbatches to the learner device.
- Existing EnvWorkers split total envs across env ranks and route trajectories
  through Channel to actor-local replay buffers.
- FSDP validates global batch divisibility by microbatch times actor world size;
  `no_shard` provides replicated data-parallel gradient synchronization.
- OpenPI already produces transformed environment action chunks.
- HuggingFace rollout supports a frozen auxiliary model beside the synchronized
  rollout model.
- LIBERO standard and Plus share `LiberoEnv`, fixed resets, and task filters.

Conclusion: explicit residual does not require replay or communication
infrastructure changes. The implementation uses new modules and narrow
call-site hooks.

