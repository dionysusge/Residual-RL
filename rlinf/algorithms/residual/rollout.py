# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from typing import Any, Literal

import numpy as np
import torch

from rlinf.algorithms.residual.action import ResidualActionComposer
from rlinf.algorithms.residual.context import CurrentProprioEncoder
from rlinf.algorithms.residual.transition import (
    RESIDUAL_OBS_KEYS,
    RESIDUAL_TRANSITION_PREFIX,
)


class BaseVLAAdapter:
    """Minimal frozen-VLA interface backed by RLinf's existing model wrapper."""

    def __init__(self, model: Any) -> None:
        self.model = model

    @torch.no_grad()
    def predict_base_action(self, env_obs: dict[str, Any]) -> torch.Tensor:
        actions, _ = self.model.predict_action_batch(
            env_obs=env_obs, mode="eval", compute_values=False
        )
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        if actions.ndim != 3:
            raise ValueError(f"Base VLA must return [B, H, D], got {actions.shape}.")
        return actions

    @torch.no_grad()
    def extract_residual_obs(self, env_obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        return self.model.extract_residual_obs(env_obs)


class ResidualRollout:
    """Explicit residual rollout composition independent of simulator routing."""

    def __init__(
        self,
        base_model: BaseVLAAdapter,
        benchmark_adapter: Any,
        composer: ResidualActionComposer,
    ) -> None:
        self.base_model = base_model
        self.benchmark_adapter = benchmark_adapter
        self.composer = composer

    def predict(
        self,
        policy_model: Any,
        env_obs: dict[str, Any],
        final_obs: dict[str, Any] | None,
        mode: Literal["train", "eval"],
        version: int = 0,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Predict and compose one batch of full action chunks."""
        return predict_residual_actions(
            policy_model=policy_model,
            base_model=self.base_model,
            benchmark_adapter=self.benchmark_adapter,
            composer=self.composer,
            env_obs=env_obs,
            final_obs=final_obs,
            mode=mode,
            version=version,
        )


def _append_transition_obs(
    base_model: BaseVLAAdapter,
    result: dict[str, Any],
    current_obs: dict[str, torch.Tensor],
    final_obs: dict[str, Any] | None,
) -> None:
    transition_obs = current_obs
    if final_obs is not None:
        transition_obs = base_model.extract_residual_obs(final_obs)
    for key in RESIDUAL_OBS_KEYS:
        result["forward_inputs"][f"{RESIDUAL_TRANSITION_PREFIX}{key}"] = transition_obs[
            key
        ]


def predict_residual_actions(
    *,
    policy_model: Any,
    base_model: BaseVLAAdapter,
    benchmark_adapter: Any,
    composer: ResidualActionComposer,
    env_obs: dict[str, Any],
    final_obs: dict[str, Any] | None,
    mode: Literal["train", "eval"],
    version: int = 0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run frozen base inference, residual exploration, and local composition."""
    del version
    with torch.no_grad():
        residual_obs = base_model.extract_residual_obs(env_obs)
        proprio_fields = benchmark_adapter.get_proprio(env_obs)
        residual_obs["proprio"] = CurrentProprioEncoder(tuple(proprio_fields.keys()))(
            proprio_fields
        ).to(
            device=residual_obs["ref_chunk"].device,
            dtype=torch.float32,
        )
        raw_residual, result = policy_model.predict_residual_batch(
            residual_obs, mode=mode
        )
        base_action = residual_obs["ref_chunk"]

        # The frozen base VLA and the lightweight residual policy may reside
        # on different devices. Composition is defined in the base action
        # space, so explicitly cross the device/dtype boundary here.
        raw_residual = raw_residual.to(
            device=base_action.device,
            dtype=base_action.dtype,
        )

        composition = composer.compose(
            base_action,
            raw_residual,
            policy_model.residual_scale,
            benchmark_adapter.get_action_spec(base_action),
        )
        actions = composition.executed_action.contiguous()
        forward_inputs = result["forward_inputs"]
        forward_inputs.update(
            {
                "action": actions.reshape(actions.shape[0], -1).contiguous(),
                "model_action": actions.reshape(actions.shape[0], -1).contiguous(),
                "base_action_chunk": composition.base_action,
                "raw_residual": composition.raw_residual,
                "bounded_residual": composition.bounded_residual,
                "scaled_residual": composition.scaled_residual,
                "preclip_action": composition.preclip_action,
                "executed_action": composition.executed_action,
            }
        )
        _append_transition_obs(base_model, result, residual_obs, final_obs)
    return actions, result
