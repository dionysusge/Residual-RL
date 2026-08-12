# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from typing import Any

from rlinf.utils.nested_dict_process import copy_dict_tensor

RESIDUAL_OBS_KEYS = ("proprio", "ref_chunk")
RESIDUAL_TRANSITION_PREFIX = "residual_transition_"


def extract_residual_obs_from_forward_inputs(
    forward_inputs: dict[str, Any], *, transition: bool = False
) -> dict[str, Any]:
    """Read the two-field residual observation contract from rollout output."""
    prefix = RESIDUAL_TRANSITION_PREFIX if transition else ""
    missing = [
        f"{prefix}{key}"
        for key in RESIDUAL_OBS_KEYS
        if f"{prefix}{key}" not in forward_inputs
    ]
    if missing:
        raise ValueError(f"Missing residual forward_inputs keys: {missing}.")
    return copy_dict_tensor(
        {key: forward_inputs[f"{prefix}{key}"] for key in RESIDUAL_OBS_KEYS}
    )


def update_residual_transitions(
    stage_id: int,
    pending_obs: list[dict[str, Any] | None],
    trajectory_builders: list[Any],
    policy_output: Any,
    *,
    cache_current: bool,
) -> None:
    """Populate existing trajectory curr/next obs without changing its schema."""
    if pending_obs[stage_id] is not None:
        next_obs = extract_residual_obs_from_forward_inputs(
            policy_output.forward_inputs, transition=True
        )
        trajectory_builders[stage_id].append_transitions(
            pending_obs[stage_id], next_obs
        )
        pending_obs[stage_id] = None
    if cache_current:
        pending_obs[stage_id] = extract_residual_obs_from_forward_inputs(
            policy_output.forward_inputs
        )
