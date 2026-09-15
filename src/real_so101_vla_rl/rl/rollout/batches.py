"""Typed trajectory data shared by collectors and algorithms."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True, slots=True)
class TrajectoryBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    action_masks: torch.Tensor
    old_values: torch.Tensor | None
    next_values: torch.Tensor | None
    advantages: torch.Tensor | None
    returns: torch.Tensor | None
    episode_ids: tuple[int, ...]
    group_ids: tuple[int, ...]
    reset_ids: tuple[str, ...]
    sequence_shape: tuple[int, int] | None = None

    @property
    def size(self) -> int:
        return int(self.observations.shape[0])

    def to(self, device: torch.device | str) -> TrajectoryBatch:
        tensor_fields = {
            "observations": self.observations.to(device),
            "actions": self.actions.to(device),
            "old_log_probs": self.old_log_probs.to(device),
            "rewards": self.rewards.to(device),
            "terminated": self.terminated.to(device),
            "truncated": self.truncated.to(device),
            "action_masks": self.action_masks.to(device),
            "old_values": None if self.old_values is None else self.old_values.to(device),
            "next_values": None if self.next_values is None else self.next_values.to(device),
            "advantages": None if self.advantages is None else self.advantages.to(device),
            "returns": None if self.returns is None else self.returns.to(device),
        }
        return replace(self, **tensor_fields)

    def with_targets(
        self,
        *,
        advantages: torch.Tensor,
        returns: torch.Tensor | None,
    ) -> TrajectoryBatch:
        return replace(self, advantages=advantages, returns=returns)


@dataclass(frozen=True, slots=True)
class CollectionResult:
    batch: TrajectoryBatch
    metrics: dict[str, float]
