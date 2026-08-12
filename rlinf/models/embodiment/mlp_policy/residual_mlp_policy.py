# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy
from rlinf.models.embodiment.modules.q_head import MultiCrossQHead, MultiQHead


def _config_vector(
    value: float | Sequence[float], action_dim: int, name: str
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).flatten()
    if tensor.numel() == 1:
        tensor = tensor.repeat(action_dim)
    if tensor.numel() != action_dim:
        raise ValueError(
            f"{name} must be scalar or length {action_dim}, got {tensor.numel()}."
        )
    return tensor


class ResidualCritic(nn.Module):
    """Independent twin-Q critic over context and executed action chunks."""

    def __init__(
        self,
        context_dim: int,
        action_feature_dim: int,
        q_head_type: str = "default",
    ) -> None:
        super().__init__()
        head_cls = {
            "default": MultiQHead,
            "crossq": MultiCrossQHead,
        }.get(q_head_type)
        if head_cls is None:
            raise ValueError(f"Invalid q_head_type: {q_head_type}")
        self.network = head_cls(
            hidden_size=context_dim,
            hidden_dims=[256, 256, 256],
            num_q_heads=2,
            output_dim=1,
            action_feature_dim=action_feature_dim,
        )

    def forward(self, context, actions, **kwargs):
        """Evaluate environment-executed actions, never latent residuals."""
        return self.network(context, actions, **kwargs)


class ResidualPolicy(MLPPolicy):
    """Explicit full-chunk residual actor with an independent twin-Q critic.

    The actor consumes ``flatten(base_action_chunk) + proprio``. The critic
    consumes the same state context but evaluates the executed action chunk.
    No VLA feature, history, image, or privileged simulator state enters here.
    """

    def __init__(
        self,
        proprio_dim: int,
        action_dim: int,
        num_action_chunks: int,
        ref_num_action_chunks: int | None = None,
        add_q_head: bool = True,
        q_head_type: str = "default",
        residual_std: float = 0.05,
        residual_scale: float = 1.0,
        residual_bound: float | Sequence[float] = 0.1,
        dim_mask: float | Sequence[float] = 1.0,
        action_low: float | Sequence[float] = -1.0,
        action_high: float | Sequence[float] = 1.0,
    ) -> None:
        if not add_q_head:
            raise ValueError("ResidualPolicy requires twin Q heads.")
        self.proprio_dim = int(proprio_dim)
        self.step_action_dim = int(action_dim)
        self.chunk_len = int(num_action_chunks)
        self.ref_chunk_len = (
            self.chunk_len
            if ref_num_action_chunks is None
            else int(ref_num_action_chunks)
        )
        if self.ref_chunk_len < self.chunk_len:
            raise ValueError("ref_num_action_chunks must cover the executed horizon.")
        self.flat_action_dim = self.chunk_len * self.step_action_dim
        actor_obs_dim = self.proprio_dim + self.flat_action_dim
        critic_obs_dim = self.proprio_dim + self.flat_action_dim
        super().__init__(
            obs_dim=actor_obs_dim,
            action_dim=self.step_action_dim,
            num_action_chunks=self.chunk_len,
            add_value_head=False,
            add_q_head=True,
            q_head_type=q_head_type,
            critic_obs_dim=critic_obs_dim,
        )
        self.q_head = ResidualCritic(
            context_dim=critic_obs_dim,
            action_feature_dim=self.flat_action_dim,
            q_head_type=q_head_type,
        )
        # Exploration std is an explicit scalar config in v0, not a learned
        # action head. Remove the generic SAC log-std module inherited above.
        self.actor_logstd = None
        self.residual_std = float(residual_std)
        self.residual_scale = float(residual_scale)
        if self.residual_std <= 0:
            raise ValueError("residual_std must be positive.")
        bound = _config_vector(residual_bound, self.step_action_dim, "residual_bound")
        mask = _config_vector(dim_mask, self.step_action_dim, "dim_mask")
        if torch.any(bound < 0):
            raise ValueError("residual_bound values must be non-negative.")
        if torch.any((mask != 0) & (mask != 1)):
            raise ValueError("dim_mask values must be 0 or 1.")
        self.register_buffer("residual_bound", bound)
        self.register_buffer("residual_dim_mask", mask)
        self.register_buffer(
            "execution_low",
            _config_vector(action_low, self.step_action_dim, "action_low"),
        )
        self.register_buffer(
            "execution_high",
            _config_vector(action_high, self.step_action_dim, "action_high"),
        )

    def preprocess_env_obs(self, env_obs):
        device = next(self.parameters()).device
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in env_obs.items()
        }

    @staticmethod
    def _flatten_batch(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(tensor.shape[0], -1)

    def _base_chunk(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        ref = obs["ref_chunk"].reshape(
            obs["ref_chunk"].shape[0], -1, self.step_action_dim
        )
        return ref[:, : self.chunk_len]

    def _actor_state(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [
                self._flatten_batch(self._base_chunk(obs)),
                self._flatten_batch(obs["proprio"]),
            ],
            dim=-1,
        )

    def _critic_state(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        # Actor and critic have deliberately separate methods/extension points.
        return torch.cat(
            [
                self._flatten_batch(obs["proprio"]),
                self._flatten_batch(self._base_chunk(obs)),
            ],
            dim=-1,
        )

    def _residual_distribution(self, obs: dict[str, torch.Tensor]) -> Normal:
        mean = self.actor_mean(self.backbone(self._actor_state(obs)))
        return Normal(mean, torch.full_like(mean, self.residual_std))

    def residual_components(
        self, obs: dict[str, torch.Tensor], deterministic: bool
    ) -> dict[str, torch.Tensor]:
        probs = self._residual_distribution(obs)
        raw = probs.mean if deterministic else probs.rsample()
        raw_chunk = raw.reshape(-1, self.chunk_len, self.step_action_dim)
        bounded = torch.tanh(raw_chunk) * self.residual_bound * self.residual_dim_mask
        scaled = self.residual_scale * bounded
        base = self._base_chunk(obs).to(dtype=scaled.dtype)
        if self.residual_scale == 0.0:
            preclip = base.clone()
            executed = base.clone()
        else:
            preclip = base + scaled
            executed = torch.maximum(
                torch.minimum(preclip, self.execution_high), self.execution_low
            )
        return {
            "raw_residual": raw_chunk,
            "bounded_residual": bounded,
            "scaled_residual": scaled,
            "preclip_action": preclip,
            "executed_action": executed,
            "logprobs": probs.log_prob(raw),
        }

    def sac_forward(self, obs, deterministic: bool = False, **kwargs):
        del kwargs
        components = self.residual_components(obs, deterministic=deterministic)
        return (
            self._flatten_batch(components["executed_action"]),
            components["logprobs"],
            components,
        )

    def sac_q_forward(self, obs, actions, shared_feature=None, detach_encoder=False):
        del shared_feature
        critic_state = self._critic_state(obs)
        if detach_encoder:
            critic_state = critic_state.detach()
        return self.q_head(critic_state, self._flatten_batch(actions))

    def crossq_q_forward(
        self,
        obs,
        actions,
        next_obs=None,
        next_actions=None,
        shared_feature=None,
        detach_encoder=False,
    ):
        del shared_feature
        critic_state = self._critic_state(obs)
        next_state = self._critic_state(next_obs) if next_obs is not None else None
        if detach_encoder:
            critic_state = critic_state.detach()
            if next_state is not None:
                next_state = next_state.detach()
        return self.q_head(
            critic_state,
            self._flatten_batch(actions),
            next_state_features=next_state,
            next_action_features=(
                self._flatten_batch(next_actions) if next_actions is not None else None
            ),
        )

    def crossq_forward(self, obs, **kwargs):
        return self.sac_forward(obs, **kwargs)

    def sft_forward(self, data, **kwargs):
        del kwargs
        obs = data["obs"] if "obs" in data else data
        target = data.get("action", data.get("actions"))
        if target is None:
            raise ValueError("Residual SFT compatibility path requires action targets.")
        mean = self._residual_distribution(obs).mean
        base = self._flatten_batch(self._base_chunk(obs))
        target_delta = self._flatten_batch(target) - base
        return F.mse_loss(mean, target_delta, reduction="none")

    @torch.inference_mode()
    def predict_residual_batch(self, env_obs, mode="train"):
        obs = self.preprocess_env_obs(env_obs)
        components = self.residual_components(obs, deterministic=(mode == "eval"))
        forward_inputs = {
            "proprio": obs["proprio"],
            "ref_chunk": obs["ref_chunk"],
        }
        result = {
            "prev_logprobs": components["logprobs"],
            "prev_values": torch.zeros_like(components["logprobs"][..., :1]),
            "forward_inputs": forward_inputs,
        }
        return components["raw_residual"], result

    @torch.inference_mode()
    def predict_action_batch(self, env_obs, mode="train", return_obs=True, **kwargs):
        del kwargs
        obs = self.preprocess_env_obs(env_obs)
        components = self.residual_components(obs, deterministic=(mode == "eval"))
        executed = components["executed_action"]
        forward_inputs = {
            "action": self._flatten_batch(executed),
            "model_action": self._flatten_batch(executed),
        }
        if return_obs:
            forward_inputs.update(
                {"proprio": obs["proprio"], "ref_chunk": obs["ref_chunk"]}
            )
        result = {
            "prev_logprobs": components["logprobs"],
            "prev_values": torch.zeros_like(components["logprobs"][..., :1]),
            "forward_inputs": forward_inputs,
        }
        return executed, result
