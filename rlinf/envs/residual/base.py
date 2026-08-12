# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from abc import ABC, abstractmethod
from typing import Any, Mapping

import torch

from rlinf.algorithms.residual.action import ActionSpec, ResidualComposition


class BenchmarkAdapter(ABC):
    """Benchmark-specific observation and action contract for residual RL."""

    @abstractmethod
    def get_proprio(self, env_obs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Return only robot-observable proprioception fields."""

    @abstractmethod
    def get_action_spec(self, action: torch.Tensor | None = None) -> ActionSpec:
        """Return per-dimension execution bounds."""

    def compose_action(self, composition: ResidualComposition) -> torch.Tensor:
        """Convert composed actions to the benchmark execution contract."""
        return composition.executed_action

    @abstractmethod
    def get_success(self, info: Mapping[str, Any]) -> Any:
        """Extract success without exposing it to the actor context."""

    @abstractmethod
    def get_task_id(self, info: Mapping[str, Any]) -> Any:
        """Extract a paired-evaluation task identity."""

    @abstractmethod
    def get_reset_id(self, info: Mapping[str, Any]) -> Any:
        """Extract a paired-evaluation reset identity."""
