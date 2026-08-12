# Action contract

- Base and residual chunks: `[B, H, D]`; smoke config uses `H=5`, `D=7`.
- Actor emits raw residual mean and samples only in residual space.
- Train: `raw ~ Normal(mean, residual.std)`; eval: `raw = mean`.
- `bounded = bound * tanh(raw) * dim_mask`.
- `scaled = scale * bounded`.
- `preclip = base + scaled`; execution clips to the configured LIBERO bounds.
- Default mask enables every dimension, including gripper. No dimension has a
  special rule.

For `scale=0`, composer returns cloned base actions without an added clip so the
E0 path is bitwise identical to the existing base rollout. Every trace retains
base, raw/bounded/scaled residual, preclip, and executed actions.

