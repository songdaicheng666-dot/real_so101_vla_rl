"""Full-trajectory group-relative policy optimization."""

from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn

from real_so101_vla_rl.rl.config import AlgorithmConfig
from real_so101_vla_rl.rl.policies import MLPChunkPolicy
from real_so101_vla_rl.rl.rollout import TrajectoryBatch

from .advantages import group_relative_advantages
from .base import RLAlgorithm, register_algorithm


@register_algorithm("grpo")
class GRPOAlgorithm(RLAlgorithm):
    collection_strategy = "grouped_full_episode"
    requires_value = False

    def __init__(self, policy: MLPChunkPolicy, config: AlgorithmConfig) -> None:
        if policy.has_value:
            raise ValueError("the baseline GRPO policy must not create a critic")
        super().__init__(policy)
        self.config = config
        self._optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    def prepare_batch(self, batch: TrajectoryBatch) -> TrajectoryBatch:
        advantages, returns = group_relative_advantages(batch)
        return batch.with_targets(advantages=advantages, returns=returns)

    def update(self, batch: TrajectoryBatch) -> dict[str, float]:
        if batch.advantages is None:
            raise ValueError("GRPO batch has not been prepared")
        metric_sums: dict[str, float] = defaultdict(float)
        updates = 0
        for _ in range(self.config.epochs):
            permutation = torch.randperm(batch.size, device=batch.observations.device)
            for start in range(0, batch.size, self.config.minibatch_size):
                indices = permutation[start : start + self.config.minibatch_size]
                evaluation = self.policy.evaluate_actions(
                    batch.observations[indices], batch.actions[indices]
                )
                mask = batch.action_masks[indices]
                denominator = mask.sum().clamp_min(1.0)
                log_ratio = evaluation.log_prob - batch.old_log_probs[indices]
                ratio = log_ratio.clamp(-20.0, 20.0).exp()
                action_advantages = batch.advantages[indices, None]
                unclipped = ratio * action_advantages
                clipped = ratio.clamp(
                    1.0 - self.config.clip_coefficient,
                    1.0 + self.config.clip_coefficient,
                ) * action_advantages
                policy_loss = -(torch.minimum(unclipped, clipped) * mask).sum() / denominator
                entropy = (evaluation.entropy * mask).sum() / denominator
                approx_kl = (((ratio - 1.0) - log_ratio) * mask).sum() / denominator
                loss = (
                    policy_loss
                    - self.config.entropy_coefficient * entropy
                    + self.config.kl_coefficient * approx_kl
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.max_gradient_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    clip_fraction = (
                        ((ratio - 1.0).abs() > self.config.clip_coefficient) * mask
                    ).sum() / denominator
                metric_sums["loss/policy"] += float(policy_loss.detach())
                metric_sums["policy/entropy"] += float(entropy.detach())
                metric_sums["policy/approx_kl"] += float(approx_kl.detach())
                metric_sums["policy/clip_fraction"] += float(clip_fraction)
                metric_sums["policy/gradient_norm"] += float(gradient_norm)
                updates += 1
        return {name: value / updates for name, value in metric_sums.items()}
