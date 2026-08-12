# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import torch

from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACFSDPPolicy


class ResidualACFSDPPolicy(RLTACFSDPPolicy):
    """Explicit-residual Stage-2 learner on existing SAC/RLT plumbing.

    This subclass deliberately reuses replay ingestion, twin Q, target Q,
    optimizer, checkpoint, and small-model weight-sync behavior. It does not
    use RLT Stage 1, ``z_rl``, or any VLA parameters.
    """

    def _bc_metrics(
        self,
        pi: torch.Tensor,
        actions: torch.Tensor,
        ref_chunk: torch.Tensor,
        intervene_flags: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        bc_loss, metrics = super()._bc_metrics(pi, actions, ref_chunk, intervene_flags)
        chunk_len, action_dim = self._chunk_shape()
        residual = (self._flatten_chunk(pi) - self._flatten_chunk(ref_chunk)).reshape(
            -1, chunk_len, action_dim
        )
        residual = residual.detach().float()
        residual_rms = torch.sqrt(torch.mean(residual.square()))
        base_rms = torch.sqrt(
            torch.mean(self._flatten_chunk(ref_chunk).detach().float().square())
        )
        metrics.update(
            {
                "residual_abs_mean": residual.abs().mean().item(),
                "residual_rms": residual_rms.item(),
                "residual_max": residual.abs().max().item(),
                "base_action_rms": base_rms.item(),
                "residual_base_rms_ratio": (
                    residual_rms / torch.clamp(base_rms, min=1e-12)
                ).item(),
            }
        )
        per_dim = torch.sqrt(residual.square().mean(dim=(0, 1)))
        per_chunk = torch.sqrt(residual.square().mean(dim=(0, 2)))
        metrics.update(
            {
                f"residual_per_dim_rms/{i}": value.item()
                for i, value in enumerate(per_dim)
            }
        )
        metrics.update(
            {
                f"residual_per_chunk_rms/{i}": value.item()
                for i, value in enumerate(per_chunk)
            }
        )
        return bc_loss, metrics
