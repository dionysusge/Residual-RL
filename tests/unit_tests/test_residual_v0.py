# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from pathlib import Path

import pytest
import torch

from rlinf.algorithms.residual.action import ActionSpec, ResidualActionComposer
from rlinf.algorithms.residual.context import CurrentProprioEncoder
from rlinf.algorithms.residual.metrics import residual_metrics
from rlinf.algorithms.residual.trace import AtomicResidualTraceWriter
from rlinf.algorithms.residual.validation import (
    assert_base_vla_frozen,
    parameter_checksum,
)
from rlinf.envs.residual.libero_adapter import LiberoResidualAdapter
from rlinf.models.embodiment.mlp_policy.residual_mlp_policy import ResidualPolicy


def _policy(scale: float = 1.0, dim_mask=1.0) -> ResidualPolicy:
    return ResidualPolicy(
        proprio_dim=8,
        action_dim=7,
        num_action_chunks=5,
        residual_std=0.1,
        residual_scale=scale,
        residual_bound=[0.1] * 7,
        dim_mask=dim_mask,
    )


def _obs(batch: int = 3) -> dict[str, torch.Tensor]:
    return {
        "proprio": torch.randn(batch, 8),
        "ref_chunk": torch.rand(batch, 5, 7) * 1.8 - 0.9,
    }


def test_zero_residual_identity():
    base = torch.rand(2, 5, 7) * 1.8 - 0.9
    raw = torch.randn_like(base)
    result = ResidualActionComposer(0.2, 1.0).compose(
        base, raw, 0.0, ActionSpec(-1.0, 1.0)
    )
    assert torch.equal(result.base_action, result.preclip_action)
    assert torch.equal(result.base_action, result.executed_action)


def test_residual_shape():
    policy = _policy()
    raw, _ = policy.predict_residual_batch(_obs(), mode="train")
    assert raw.shape == (3, 5, 7)


def test_residual_bound():
    base = torch.zeros(2, 5, 7)
    result = ResidualActionComposer([0.1] * 7, 1.0).compose(
        base, torch.full_like(base, 100.0), 1.0, ActionSpec(-1.0, 1.0)
    )
    assert torch.all(result.bounded_residual <= 0.1)
    assert torch.all(result.bounded_residual >= -0.1)


def test_residual_dim_mask():
    base = torch.zeros(1, 5, 7)
    mask = [1, 0, 1, 0, 1, 0, 1]
    result = ResidualActionComposer(0.2, mask).compose(
        base, torch.ones_like(base), 1.0, ActionSpec(-1.0, 1.0)
    )
    assert torch.count_nonzero(result.scaled_residual[..., 1::2]) == 0


def test_action_composer():
    base = torch.full((1, 2, 3), 0.95)
    result = ResidualActionComposer(0.2, 1.0).compose(
        base, torch.ones_like(base), 1.0, ActionSpec(-1.0, 1.0)
    )
    assert torch.all(result.preclip_action > result.base_action)
    assert torch.all(result.executed_action <= 1.0)


def test_eval_deterministic():
    policy = _policy()
    obs = _obs()
    first, _ = policy.predict_residual_batch(obs, mode="eval")
    second, _ = policy.predict_residual_batch(obs, mode="eval")
    assert torch.equal(first, second)


def test_train_stochastic():
    policy = _policy()
    obs = _obs()
    first, _ = policy.predict_residual_batch(obs, mode="train")
    second, _ = policy.predict_residual_batch(obs, mode="train")
    assert not torch.equal(first, second)


def test_proprio_adapter():
    states = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    adapter = LiberoResidualAdapter()
    fields = adapter.get_proprio({"states": states})
    encoder = CurrentProprioEncoder(["eef_pos", "eef_quat_or_rot", "gripper"])
    assert torch.equal(encoder(fields), states)
    with pytest.raises(ValueError, match="must not be fabricated"):
        CurrentProprioEncoder(["joint_pos"])(fields)


def test_base_vla_frozen():
    base = torch.nn.Linear(3, 2)
    base.requires_grad_(False)
    before = parameter_checksum(base)
    actor = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(actor.parameters())
    assert_base_vla_frozen(base, [optimizer])
    assert parameter_checksum(base) == before


def test_residual_checkpoint_roundtrip(tmp_path: Path):
    policy = _policy()
    checkpoint = tmp_path / "residual.pt"
    torch.save(policy.state_dict(), checkpoint)
    restored = _policy()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    obs = _obs()
    first, _ = policy.predict_action_batch(obs, mode="eval")
    second, _ = restored.predict_action_batch(obs, mode="eval")
    assert torch.equal(first, second)


def test_trace_output_and_metrics(tmp_path: Path):
    base = torch.zeros(1, 5, 7)
    composition = ResidualActionComposer(0.1, 1.0).compose(
        base, torch.ones_like(base), 1.0, ActionSpec(-1.0, 1.0)
    )
    metrics = residual_metrics(composition)
    assert len(metrics["per_action_dim_rms"]) == 7
    assert len(metrics["per_chunk_position_rms"]) == 5
    path = AtomicResidualTraceWriter(tmp_path, "run_test", 0).write(
        [{"base_action_chunk": base, "metrics": metrics}]
    )
    assert path.exists()
    assert len(torch.load(path, weights_only=False)) == 1
