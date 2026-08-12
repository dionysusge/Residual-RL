# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from typing import Any, Mapping

import torch

from rlinf.algorithms.residual.action import ActionSpec
from rlinf.envs.residual.base import BenchmarkAdapter


class LiberoResidualAdapter(BenchmarkAdapter):
    """Shared residual contract for standard LIBERO and LIBERO-Plus."""

    def __init__(self, action_low: float = -1.0, action_high: float = 1.0):
        self.action_low = float(action_low)
        self.action_high = float(action_high)

    def get_proprio(self, env_obs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        if "states" not in env_obs:
            raise ValueError("RLinf LIBERO observation is missing `states`.")
        states = torch.as_tensor(env_obs["states"])
        if states.ndim == 1:
            states = states.unsqueeze(0)
        if states.shape[-1] < 7:
            raise ValueError(
                "LIBERO state must contain EEF position, EEF orientation, and "
                f"gripper state; got shape {tuple(states.shape)}."
            )
        return {
            "eef_pos": states[..., :3],
            # LiberoEnv converts the quaternion to a 3D axis-angle vector.
            "eef_quat_or_rot": states[..., 3:6],
            "gripper": states[..., 6:],
        }

    def flatten_proprio(self, env_obs: Mapping[str, Any]) -> torch.Tensor:
        fields = self.get_proprio(env_obs)
        return torch.cat(
            [fields["eef_pos"], fields["eef_quat_or_rot"], fields["gripper"]],
            dim=-1,
        )

    def get_action_spec(self, action: torch.Tensor | None = None) -> ActionSpec:
        del action
        return ActionSpec(low=self.action_low, high=self.action_high)

    @staticmethod
    def _first_present(info: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
        for key in keys:
            if key in info:
                return info[key]
        return None

    def get_success(self, info: Mapping[str, Any]) -> Any:
        episode = info.get("episode", {})
        if isinstance(episode, Mapping):
            value = self._first_present(
                episode, ("success_at_end", "success_once", "success")
            )
            if value is not None:
                return value
        return self._first_present(info, ("success", "is_success"))

    def get_task_id(self, info: Mapping[str, Any]) -> Any:
        return self._first_present(info, ("task_id", "task_ids"))

    def get_reset_id(self, info: Mapping[str, Any]) -> Any:
        return self._first_present(
            info, ("reset_state_id", "reset_state_ids", "trial_id", "trial_ids")
        )
