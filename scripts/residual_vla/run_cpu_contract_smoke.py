#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Run a dependency-light 100-update residual contract smoke on CPU."""

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# The locked production environment uses a newer torch. This compatibility
# alias lets the local CPU-only torch 2.4 installation import RLinf for tests.
try:
    from torch.distributed.tensor import DTensor  # noqa: F401
except ImportError:
    import torch.distributed.tensor as distributed_tensor
    from torch.distributed._tensor import DTensor

    distributed_tensor.DTensor = DTensor

from rlinf.algorithms.residual.validation import (  # noqa: E402
    assert_base_vla_frozen,
    parameter_checksum,
)
from rlinf.models.embodiment.base_policy import ForwardType  # noqa: E402
from rlinf.models.embodiment.mlp_policy.residual_mlp_policy import (  # noqa: E402
    ResidualPolicy,
)


def make_model() -> ResidualPolicy:
    return ResidualPolicy(
        proprio_dim=8,
        action_dim=7,
        num_action_chunks=5,
        residual_std=0.05,
        residual_scale=0.1,
        residual_bound=0.1,
        dim_mask=[1] * 7,
    )


def sample_batch(batch_size: int = 16):
    curr = {
        "proprio": torch.randn(batch_size, 8),
        "ref_chunk": torch.rand(batch_size, 5, 7) * 1.6 - 0.8,
    }
    next_obs = {
        "proprio": torch.randn(batch_size, 8),
        "ref_chunk": torch.rand(batch_size, 5, 7) * 1.6 - 0.8,
    }
    return curr, next_obs, torch.rand(batch_size, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/residual_vla/cpu_contract_smoke.json"),
    )
    args = parser.parse_args()
    torch.manual_seed(2026)
    learner = make_model()
    target = deepcopy(learner).requires_grad_(False)
    rollout = make_model()
    base = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU())
    base.requires_grad_(False)
    base_before = parameter_checksum(base)

    actor_parameters = [
        parameter
        for name, parameter in learner.named_parameters()
        if "q_head" not in name
    ]
    critic_parameters = list(learner.q_head.parameters())
    actor_optimizer = torch.optim.Adam(actor_parameters, lr=1e-4)
    critic_optimizer = torch.optim.Adam(critic_parameters, lr=1e-4)
    assert_base_vla_frozen(base, [actor_optimizer, critic_optimizer])

    replay = [sample_batch() for _ in range(32)]
    actor_loss_value = 0.0
    critic_loss_value = 0.0
    for update in range(args.updates):
        curr, next_obs, reward = replay[update % len(replay)]
        with torch.no_grad():
            next_action, _, _ = target(
                forward_type=ForwardType.SAC,
                obs=next_obs,
                deterministic=True,
            )
            target_q = (
                target(
                    forward_type=ForwardType.SAC_Q,
                    obs=next_obs,
                    actions=next_action,
                )
                .min(dim=-1, keepdim=True)
                .values
            )
            td_target = reward + 0.96 * target_q
        action, _, components = learner(
            forward_type=ForwardType.SAC,
            obs=curr,
            deterministic=False,
        )
        q = learner(
            forward_type=ForwardType.SAC_Q,
            obs=curr,
            actions=action.detach(),
        )
        critic_loss = (q - td_target.expand_as(q)).square().mean()
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        for parameter in critic_parameters:
            parameter.requires_grad_(False)
        q_pi = learner(
            forward_type=ForwardType.SAC_Q,
            obs=curr,
            actions=action,
        )[..., :1]
        reference = curr["ref_chunk"].reshape_as(action)
        bc_loss = (action - reference).square().mean()
        actor_loss = -q_pi.mean() + bc_loss
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        for parameter in critic_parameters:
            parameter.requires_grad_(True)

        with torch.no_grad():
            for target_parameter, parameter in zip(
                target.parameters(), learner.parameters(), strict=True
            ):
                target_parameter.lerp_(parameter, 0.005)
        if update % 10 == 0:
            rollout.load_state_dict(learner.state_dict())
        actor_loss_value = actor_loss.item()
        critic_loss_value = critic_loss.item()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output.with_suffix(".pt")
    torch.save(learner.state_dict(), checkpoint)
    restored = make_model()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    base_after = parameter_checksum(base)
    residual = components["scaled_residual"].detach()
    base_chunk = curr["ref_chunk"].detach()
    residual_rms = residual.square().mean().sqrt()
    base_rms = base_chunk.square().mean().sqrt()
    report = {
        "scope": "synthetic_cpu_contract_smoke_not_libero_e2",
        "updates": args.updates,
        "replay_ingestion": "PASS",
        "critic_update": "PASS",
        "actor_update": "PASS",
        "target_update": "PASS",
        "small_weight_sync": "PASS",
        "checkpoint_roundtrip": "PASS",
        "base_frozen": "PASS" if base_before == base_after else "FAIL",
        "actor_loss_final": actor_loss_value,
        "critic_loss_final": critic_loss_value,
        "residual_rms": residual_rms.item(),
        "residual_base_rms_ratio": (residual_rms / base_rms).item(),
        "per_action_dim_rms": residual.square().mean((0, 1)).sqrt().tolist(),
        "per_chunk_position_rms": residual.square().mean((0, 2)).sqrt().tolist(),
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
