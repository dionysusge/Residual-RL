# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class ResidualContext:
    """Actor context with an explicit future-history extension point."""

    current_proprio: torch.Tensor
    history: object | None = None


class CurrentProprioEncoder(nn.Module):
    """Flatten configured robot-observable fields without fabricating values."""

    def __init__(self, fields: Sequence[str]) -> None:
        super().__init__()
        if not fields:
            raise ValueError("At least one proprioception field must be configured.")
        self.fields = tuple(fields)

    def forward(self, proprio: Mapping[str, torch.Tensor]) -> torch.Tensor:
        missing = [field for field in self.fields if field not in proprio]
        if missing:
            raise ValueError(
                f"Missing configured proprioception fields: {missing}. "
                "Unavailable robot state must not be fabricated."
            )
        flattened = []
        batch_size = None
        for field in self.fields:
            value = torch.as_tensor(proprio[field])
            if value.ndim == 1:
                value = value.unsqueeze(0)
            value = value.reshape(value.shape[0], -1)
            if batch_size is None:
                batch_size = value.shape[0]
            elif value.shape[0] != batch_size:
                raise ValueError("All proprioception fields must share a batch size.")
            flattened.append(value)
        return torch.cat(flattened, dim=-1)


class ResidualContextEncoder(nn.Module):
    """Current-only encoder; temporal encoders are intentionally absent in v0."""

    def __init__(self, current_proprio: CurrentProprioEncoder) -> None:
        super().__init__()
        self.current_proprio = current_proprio

    def forward(self, proprio: Mapping[str, torch.Tensor]) -> ResidualContext:
        return ResidualContext(current_proprio=self.current_proprio(proprio))
