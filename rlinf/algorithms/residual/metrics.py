# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import torch

from rlinf.algorithms.residual.action import ResidualComposition


def residual_metrics(
    composition: ResidualComposition,
    action_low: float | torch.Tensor = -1.0,
    action_high: float | torch.Tensor = 1.0,
) -> dict[str, float | list[float]]:
    """Compute mandatory smoke metrics over a composed action chunk."""
    residual = composition.scaled_residual.detach().float()
    base = composition.base_action.detach().float()
    preclip = composition.preclip_action.detach().float()
    executed = composition.executed_action.detach().float()
    base_rms = torch.sqrt(torch.mean(base.square()))
    residual_rms = torch.sqrt(torch.mean(residual.square()))
    low = torch.as_tensor(action_low, device=preclip.device, dtype=preclip.dtype)
    high = torch.as_tensor(action_high, device=preclip.device, dtype=preclip.dtype)
    preclip_saturated = (preclip < low) | (preclip > high)
    executed_saturated = torch.isclose(executed, low) | torch.isclose(executed, high)
    return {
        "residual_abs_mean": residual.abs().mean().item(),
        "residual_rms": residual_rms.item(),
        "residual_max": residual.abs().max().item(),
        "per_action_dim_rms": torch.sqrt(residual.square().mean(dim=(0, 1))).tolist(),
        "per_chunk_position_rms": torch.sqrt(
            residual.square().mean(dim=(0, 2))
        ).tolist(),
        "preclip_saturation_rate": preclip_saturated.float().mean().item(),
        "executed_saturation_rate": executed_saturated.float().mean().item(),
        "base_action_rms": base_rms.item(),
        "residual_base_rms_ratio": (
            residual_rms / torch.clamp(base_rms, min=1e-12)
        ).item(),
    }
