"""State-based tanh-squashed Gaussian action-chunk policy."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .base import ActionEvaluation, PolicyOutput

LOG_PROB_EPSILON = 1e-6


def _mlp(input_size: int, hidden_sizes: Sequence[int]) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    previous = input_size
    for hidden in hidden_sizes:
        if hidden <= 0:
            raise ValueError("hidden layer sizes must be positive")
        layers.extend((nn.Linear(previous, hidden), nn.Tanh()))
        previous = hidden
    return nn.Sequential(*layers), previous


class MLPChunkPolicy(nn.Module):
    """Independent actor/critic with a continuous distribution over each chunk."""

    def __init__(
        self,
        observation_size: int,
        action_chunk_size: int,
        action_size: int,
        *,
        hidden_sizes: Sequence[int] = (256, 256),
        initial_action: Sequence[float] | None = None,
        initial_log_std: float = -3.0,
        has_value: bool = True,
    ) -> None:
        super().__init__()
        if observation_size <= 0 or action_chunk_size <= 0 or action_size <= 0:
            raise ValueError("policy dimensions must be positive")
        self.observation_size = observation_size
        self.action_chunk_size = action_chunk_size
        self.action_size = action_size
        self.has_value = has_value

        self.actor, actor_size = _mlp(observation_size, hidden_sizes)
        self.actor_mean = nn.Linear(actor_size, action_chunk_size * action_size)
        nn.init.zeros_(self.actor_mean.weight)
        if initial_action is None:
            initial = np.zeros(action_size, dtype=np.float32)
        else:
            initial = np.asarray(initial_action, dtype=np.float32)
            if initial.shape != (action_size,):
                raise ValueError(f"initial_action must have shape {(action_size,)}")
        initial = np.clip(initial, -0.999, 0.999)
        initial_pre_tanh = np.arctanh(initial)
        bias = np.tile(initial_pre_tanh, action_chunk_size)
        with torch.no_grad():
            self.actor_mean.bias.copy_(torch.as_tensor(bias))
        self.log_std = nn.Parameter(
            torch.full((action_chunk_size, action_size), float(initial_log_std))
        )

        if has_value:
            self.critic, critic_size = _mlp(observation_size, hidden_sizes)
            self.critic_value: nn.Linear | None = nn.Linear(critic_size, 1)
        else:
            self.critic = None
            self.critic_value = None

    def _distribution(self, observations: torch.Tensor) -> Normal:
        mean = self.actor_mean(self.actor(observations)).reshape(
            -1, self.action_chunk_size, self.action_size
        )
        log_std = self.log_std.clamp(-5.0, 2.0).expand_as(mean)
        return Normal(mean, log_std.exp())

    @staticmethod
    def _squashed_log_prob(
        distribution: Normal,
        pre_tanh: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        correction = torch.log(1.0 - actions.square() + LOG_PROB_EPSILON)
        return (distribution.log_prob(pre_tanh) - correction).sum(dim=-1)

    def sample(self, observations: torch.Tensor) -> PolicyOutput:
        distribution = self._distribution(observations)
        pre_tanh = distribution.rsample()
        actions = torch.tanh(pre_tanh)
        return PolicyOutput(
            action_chunk=actions,
            log_prob=self._squashed_log_prob(distribution, pre_tanh, actions),
            value=self.value(observations),
        )

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> ActionEvaluation:
        actions = actions.clamp(-1.0 + LOG_PROB_EPSILON, 1.0 - LOG_PROB_EPSILON)
        pre_tanh = torch.atanh(actions)
        distribution = self._distribution(observations)
        log_prob = self._squashed_log_prob(distribution, pre_tanh, actions)
        return ActionEvaluation(
            log_prob=log_prob,
            entropy=-log_prob,
            value=self.value(observations),
        )

    def predict(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self._distribution(observations).mean)

    def value(self, observations: torch.Tensor) -> torch.Tensor | None:
        if not self.has_value or self.critic is None or self.critic_value is None:
            return None
        return self.critic_value(self.critic(observations)).squeeze(-1)

    def extra_repr(self) -> str:
        return (
            f"observation_size={self.observation_size}, "
            f"action_chunk_size={self.action_chunk_size}, "
            f"action_size={self.action_size}, has_value={self.has_value}"
        )


def assert_finite_policy(policy: nn.Module) -> None:
    for name, parameter in policy.named_parameters():
        if not torch.isfinite(parameter).all():
            raise RuntimeError(f"policy parameter became non-finite: {name}")


def global_gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    squared_norm = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared_norm += float(parameter.grad.detach().square().sum())
    return math.sqrt(squared_norm)
