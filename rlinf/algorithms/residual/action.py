# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class ActionSpec:
    """Environment action bounds for one action step."""

    low: float | Sequence[float] | torch.Tensor
    high: float | Sequence[float] | torch.Tensor


@dataclass(frozen=True)
class ResidualComposition:
    """All action-path values retained for audit and offline analysis."""

    base_action: torch.Tensor
    raw_residual: torch.Tensor
    bounded_residual: torch.Tensor
    scaled_residual: torch.Tensor
    preclip_action: torch.Tensor
    executed_action: torch.Tensor


def _per_dim_tensor(
    value: float | Sequence[float] | torch.Tensor,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    if tensor.numel() == 1:
        return tensor
    if tensor.ndim != 1 or tensor.shape[0] != reference.shape[-1]:
        raise ValueError(
            f"{name} must be scalar or have one value per action dimension; "
            f"got {tuple(tensor.shape)} for action shape {tuple(reference.shape)}."
        )
    return tensor


class ResidualActionComposer:
    """Bound, scale, mask, add, and clip an explicit action residual."""

    def __init__(
        self,
        bound: float | Sequence[float] | torch.Tensor,
        dim_mask: float | Sequence[float] | torch.Tensor,
    ) -> None:
        self.bound = bound
        self.dim_mask = dim_mask

    def compose(
        self,
        base_action: torch.Tensor,
        residual: torch.Tensor,
        scale: float,
        action_spec: ActionSpec,
    ) -> ResidualComposition:
        """Compose a full ``[B, H, D]`` residual action chunk."""
        if base_action.shape != residual.shape:
            raise ValueError(
                "base_action and residual must have identical [B, H, D] shapes, "
                f"got {tuple(base_action.shape)} and {tuple(residual.shape)}."
            )
        if base_action.ndim != 3:
            raise ValueError(
                f"Residual composition requires [B, H, D], got {base_action.shape}."
            )

        bound = _per_dim_tensor(self.bound, residual, "bound")
        mask = _per_dim_tensor(self.dim_mask, residual, "dim_mask")
        if torch.any(bound < 0):
            raise ValueError("Residual bounds must be non-negative.")
        if torch.any((mask != 0) & (mask != 1)):
            raise ValueError("dim_mask values must be 0 or 1.")

        bounded = bound * torch.tanh(residual) * mask
        scaled = float(scale) * bounded

        # Preserve the exact base path for the E0 hard gate. The normal base
        # rollout does not add another clip, so scale=0 must not add one here.
        if float(scale) == 0.0:
            preclip = base_action.clone()
            executed = base_action.clone()
        else:
            preclip = base_action + scaled
            low = _per_dim_tensor(action_spec.low, preclip, "action_spec.low")
            high = _per_dim_tensor(action_spec.high, preclip, "action_spec.high")
            if torch.any(low > high):
                raise ValueError("Every action lower bound must be <= its upper bound.")
            executed = torch.maximum(torch.minimum(preclip, high), low)

        return ResidualComposition(
            base_action=base_action,
            raw_residual=residual,
            bounded_residual=bounded,
            scaled_residual=scaled,
            preclip_action=preclip,
            executed_action=executed,
        )
