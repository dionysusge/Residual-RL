# Research readiness report

Status: **NOT YET `PI05_RESIDUAL_V0_RESEARCH_READY`**.

## Passed locally

- [x] RLinf locked commit and branch recorded
- [x] OpenPI reference revision and official checkpoint location recorded
- [x] ResidualPolicy/ActionComposer/BenchmarkAdapter CPU contracts
- [x] E0 composer identity (`torch.equal`)
- [x] bounded random residual interface and metrics on CPU tensors
- [x] base frozen/optimizer exclusion helper
- [x] residual checkpoint save/load
- [x] atomic trace output and offline metrics
- [x] model registry construction
- [x] replay cache produces independent contiguous CPU tensors
- [x] two-rank global batch arithmetic (`32 = 16 x 2`)
- [x] Channel/ProcessGroup/WeightSyncer/schema boundary audit

## Synthetic CPU contract smoke

`results/residual_vla/cpu_contract_smoke.json` records 100 learner updates:
replay ingestion, critic update, actor update, target update, small weight copy,
checkpoint roundtrip, and base checksum all passed. Final residual RMS was
`8.574269e-4`, with residual/base RMS ratio `1.816385e-3`. This is an interface
preflight only and is explicitly **not** the required LIBERO E2.

## Requires the target two-H200 Linux server

- [ ] environment versions and checkpoint artifact hash
- [ ] pi0.5 checkpoint load and CUDA inference on both rollout ranks
- [ ] standard LIBERO reset/base rollout
- [ ] LIBERO-Plus reset/base rollout
- [ ] fixed-seed base screening and shifted task selection
- [ ] real-environment E0 paired reward/done/success equality
- [ ] real-environment E1 response and gripper contract
- [ ] two-rank rollout -> local CPU replay -> critic -> actor -> target E2
- [ ] FSDP no-shard actor and twin-Q gradient synchronization
- [ ] real residual actor WeightSyncer application
- [ ] both frozen pi0.5 replicas retain before/after checksums across E2
- [ ] distributed checkpoint reload and new rollout
- [ ] 32/64 total-env throughput and memory profile with eval disabled

The hard gates are intentionally not inferred from synthetic CPU tests. Run
`scripts/residual_vla/run_base_screening.sh`, then
`scripts/residual_vla/run_residual_smoke.sh` on the server. Do not begin formal
training if base reproduction or E0 fails.

