"""Clipped PPO for continuous action chunks."""

from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn

from real_so101_vla_rl.rl.config import AlgorithmConfig
from real_so101_vla_rl.rl.policies import MLPChunkPolicy
from real_so101_vla_rl.rl.rollout import TrajectoryBatch

from .advantages import generalized_advantage_estimate
from .base import RLAlgorithm, register_algorithm


@register_algorithm("ppo")
class PPOAlgorithm(RLAlgorithm):
    collection_strategy = "fixed_horizon"
    requires_value = True

    def __init__(self, policy: MLPChunkPolicy, config: AlgorithmConfig) -> None:
        if not policy.has_value:
            raise ValueError("PPO requires a policy with a critic")
        super().__init__(policy)
        self.config = config
        self._optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    def prepare_batch(self, batch: TrajectoryBatch) -> TrajectoryBatch:
        advantages, returns = generalized_advantage_estimate(
            batch,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        return batch.with_targets(advantages=advantages, returns=returns)

    def update(self, batch: TrajectoryBatch) -> dict[str, float]:
        if batch.advantages is None or batch.returns is None or batch.old_values is None:
            raise ValueError("PPO batch has not been prepared")
        advantages = batch.advantages
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
        metric_sums: dict[str, float] = defaultdict(float)
        updates = 0
        for _ in range(self.config.epochs):
            permutation = torch.randperm(batch.size, device=batch.observations.device)
            for start in range(0, batch.size, self.config.minibatch_size):
                indices = permutation[start : start + self.config.minibatch_size]
                evaluation = self.policy.evaluate_actions(
                    batch.observations[indices], batch.actions[indices]
                )
                if evaluation.value is None:
                    raise RuntimeError("PPO policy stopped returning values")
                mask = batch.action_masks[indices]
                denominator = mask.sum().clamp_min(1.0)
                log_ratio = evaluation.log_prob - batch.old_log_probs[indices]
                ratio = log_ratio.clamp(-20.0, 20.0).exp()
                action_advantages = advantages[indices, None]
                unclipped = ratio * action_advantages
                clipped = ratio.clamp(
                    1.0 - self.config.clip_coefficient,
                    1.0 + self.config.clip_coefficient,
                ) * action_advantages
                policy_loss = -(torch.minimum(unclipped, clipped) * mask).sum() / denominator

                old_value = batch.old_values[indices]
                value_delta = evaluation.value - old_value
                clipped_value = old_value + value_delta.clamp(
                    -self.config.value_clip_coefficient,
                    self.config.value_clip_coefficient,
                )
                value_loss = 0.5 * torch.maximum(
                    (evaluation.value - batch.returns[indices]).square(),
                    (clipped_value - batch.returns[indices]).square(),
                ).mean()
                entropy = (evaluation.entropy * mask).sum() / denominator
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.max_gradient_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (((ratio - 1.0) - log_ratio) * mask).sum() / denominator
                    clip_fraction = (
                        ((ratio - 1.0).abs() > self.config.clip_coefficient) * mask
                    ).sum() / denominator
                metric_sums["loss/policy"] += float(policy_loss.detach())
                metric_sums["loss/value"] += float(value_loss.detach())
                metric_sums["policy/entropy"] += float(entropy.detach())
                metric_sums["policy/approx_kl"] += float(approx_kl)
                metric_sums["policy/clip_fraction"] += float(clip_fraction)
                metric_sums["policy/gradient_norm"] += float(gradient_norm)
                updates += 1
        return {name: value / updates for name, value in metric_sums.items()}
