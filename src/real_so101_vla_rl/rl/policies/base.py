"""Policy interface shared by MLP and future VLA implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True, slots=True)
class PolicyOutput:
    action_chunk: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class ActionEvaluation:
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor | None


class ChunkPolicy(Protocol):
    action_chunk_size: int
    action_size: int
    has_value: bool

    def sample(self, observations: torch.Tensor) -> PolicyOutput:
        """Sample normalized continuous action chunks."""

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> ActionEvaluation:
        """Evaluate actions sampled by an older copy of this policy."""

    def predict(self, observations: torch.Tensor) -> torch.Tensor:
        """Return deterministic normalized action chunks."""

    def value(self, observations: torch.Tensor) -> torch.Tensor | None:
        """Return state values when the policy has a critic."""
