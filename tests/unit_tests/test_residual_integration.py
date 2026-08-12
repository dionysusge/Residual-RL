# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from copy import deepcopy
from pathlib import Path

import torch

from rlinf.algorithms.residual.action import ActionSpec, ResidualActionComposer
from rlinf.algorithms.residual.rollout import BaseVLAAdapter, ResidualRollout
from rlinf.algorithms.residual.validation import (
    assert_base_vla_frozen,
    parameter_checksum,
)
from rlinf.data.storage.replay.buffer import clone_dict_of_tensors
from rlinf.envs.residual.libero_adapter import LiberoResidualAdapter
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.mlp_policy.residual_mlp_policy import ResidualPolicy


def _model() -> ResidualPolicy:
    return ResidualPolicy(
        proprio_dim=8,
        action_dim=7,
        num_action_chunks=5,
        residual_std=0.05,
        residual_scale=0.1,
        residual_bound=0.1,
        dim_mask=[1] * 7,
    )


def _transition_batch(batch_size: int = 8):
    curr_obs = {
        "proprio": torch.randn(batch_size, 8),
        "ref_chunk": torch.rand(batch_size, 5, 7) * 1.6 - 0.8,
    }
    next_obs = {
        "proprio": torch.randn(batch_size, 8),
        "ref_chunk": torch.rand(batch_size, 5, 7) * 1.6 - 0.8,
    }
    return {
        "curr_obs": curr_obs,
        "next_obs": next_obs,
        "actions": curr_obs["ref_chunk"].clone(),
        "rewards": torch.rand(batch_size, 1),
        "dones": torch.zeros(batch_size, 1, dtype=torch.bool),
    }


def test_replay_cache_payload_is_cpu_contiguous_and_independent_copy():
    source = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).transpose(1, 2)
    payload = clone_dict_of_tensors({"curr_obs": {"proprio": source}})
    replay_tensor = payload["curr_obs"]["proprio"]

    assert replay_tensor.device.type == "cpu"
    assert replay_tensor.is_contiguous()
    assert replay_tensor.data_ptr() != source.data_ptr()
    source.zero_()
    assert torch.count_nonzero(replay_tensor) > 0


class _DeterministicChunkEnv:
    def __init__(self):
        self.state = torch.zeros(1, 7)

    def step(self, chunk: torch.Tensor):
        self.state = self.state + chunk.sum(dim=1)
        reward = -self.state.square().sum(dim=-1)
        done = self.state.abs().max(dim=-1).values > 2.0
        success = self.state.norm(dim=-1) < 0.1
        return self.state.clone(), reward, done, success


def test_e0_paired_environment_equivalence():
    torch.manual_seed(11)
    base_chunks = [torch.rand(1, 5, 7) * 0.2 - 0.1 for _ in range(4)]
    residuals = [torch.randn_like(chunk) for chunk in base_chunks]
    base_env = _DeterministicChunkEnv()
    zero_env = _DeterministicChunkEnv()
    composer = ResidualActionComposer(0.2, 1.0)
    for base, residual in zip(base_chunks, residuals, strict=True):
        composed = composer.compose(base, residual, 0.0, ActionSpec(-1.0, 1.0))
        assert torch.equal(base, composed.preclip_action)
        assert torch.equal(base, composed.executed_action)
        base_result = base_env.step(base)
        zero_result = zero_env.step(composed.executed_action)
        assert all(
            torch.equal(left, right)
            for left, right in zip(base_result, zero_result, strict=True)
        )


def test_e1_random_bounded_residual_changes_environment():
    base = torch.zeros(1, 5, 7)
    residual = torch.randn_like(base)
    composer = ResidualActionComposer(0.05, 1.0)
    composed = composer.compose(base, residual, 0.5, ActionSpec(-1.0, 1.0))
    assert not torch.equal(base, composed.executed_action)
    assert torch.isfinite(composed.executed_action).all()
    base_state, *_ = _DeterministicChunkEnv().step(base)
    residual_state, *_ = _DeterministicChunkEnv().step(composed.executed_action)
    assert not torch.equal(base_state, residual_state)


def test_base_chunk_to_actor_to_composer_contract():
    model = _model()
    batch = _transition_batch(2)
    actions, result = model.predict_action_batch(batch["curr_obs"], mode="eval")
    assert actions.shape == (2, 5, 7)
    assert set(result["forward_inputs"]) >= {
        "proprio",
        "ref_chunk",
        "action",
    }


def test_residual_rollout_uses_benchmark_proprio_contract():
    class FakeBase(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

        def extract_residual_obs(self, env_obs):
            return {
                "proprio": torch.full((env_obs["states"].shape[0], 8), 999.0),
                "ref_chunk": torch.zeros(env_obs["states"].shape[0], 5, 7),
            }

    policy = _model()
    adapter = LiberoResidualAdapter()
    rollout = ResidualRollout(
        BaseVLAAdapter(FakeBase()),
        adapter,
        ResidualActionComposer(0.1, 1.0),
    )
    states = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    _, result = rollout.predict(policy, {"states": states}, final_obs=None, mode="eval")
    assert torch.equal(result["forward_inputs"]["proprio"], states)


def test_transition_replay_sample_to_learner_update():
    torch.manual_seed(7)
    model = _model()
    target = deepcopy(model).requires_grad_(False)
    replay = [_transition_batch(4) for _ in range(3)]
    batch = replay[1]
    actor_parameters = list(model.backbone.parameters()) + list(
        model.actor_mean.parameters()
    )
    critic_parameters = list(model.q_head.parameters())
    actor_optimizer = torch.optim.Adam(actor_parameters, lr=1e-3)
    critic_optimizer = torch.optim.Adam(critic_parameters, lr=1e-3)

    with torch.no_grad():
        next_actions, _, _ = target(
            forward_type=ForwardType.SAC,
            obs=batch["next_obs"],
            deterministic=True,
        )
        target_q = (
            target(
                forward_type=ForwardType.SAC_Q,
                obs=batch["next_obs"],
                actions=next_actions,
            )
            .min(dim=-1, keepdim=True)
            .values
        )
        td_target = batch["rewards"] + 0.96 * target_q
    q_values = model(
        forward_type=ForwardType.SAC_Q,
        obs=batch["curr_obs"],
        actions=batch["actions"],
    )
    critic_loss = (q_values - td_target.expand_as(q_values)).square().mean()
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    for parameter in critic_parameters:
        parameter.requires_grad_(False)
    pi, _, _ = model(
        forward_type=ForwardType.SAC,
        obs=batch["curr_obs"],
        deterministic=True,
    )
    q_pi = model(
        forward_type=ForwardType.SAC_Q,
        obs=batch["curr_obs"],
        actions=pi,
    )[..., :1]
    reference = batch["curr_obs"]["ref_chunk"].reshape_as(pi)
    actor_loss = -q_pi.mean() + (pi - reference).square().mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    for parameter in critic_parameters:
        parameter.requires_grad_(True)

    tau = 0.005
    with torch.no_grad():
        for target_parameter, parameter in zip(
            target.parameters(), model.parameters(), strict=True
        ):
            target_parameter.lerp_(parameter, tau)
    assert torch.isfinite(critic_loss)
    assert torch.isfinite(actor_loss)


def test_small_weight_sync_and_checkpoint_with_frozen_base(tmp_path: Path):
    learner = _model()
    rollout = _model()
    rollout.load_state_dict(learner.state_dict())
    for left, right in zip(learner.parameters(), rollout.parameters(), strict=True):
        assert torch.equal(left, right)

    base = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU())
    base.requires_grad_(False)
    before = parameter_checksum(base)
    optimizer = torch.optim.Adam(learner.parameters())
    assert_base_vla_frozen(base, [optimizer])

    checkpoint = tmp_path / "residual_integration.pt"
    torch.save(
        {
            "actor_critic": learner.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        checkpoint,
    )
    restored = _model()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True)["actor_critic"])
    assert parameter_checksum(base) == before
    for left, right in zip(learner.parameters(), restored.parameters(), strict=True):
        assert torch.equal(left, right)
