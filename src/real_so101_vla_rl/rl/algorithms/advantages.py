"""Advantage estimators shared by algorithm implementations."""

from __future__ import annotations

import torch

from real_so101_vla_rl.rl.rollout import TrajectoryBatch


def generalized_advantage_estimate(
    batch: TrajectoryBatch,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch.sequence_shape is None:
        raise ValueError("GAE requires a [time, environment] collection shape")
    if batch.old_values is None or batch.next_values is None:
        raise ValueError("GAE requires old_values and next_values")
    time_steps, num_envs = batch.sequence_shape
    shape = (time_steps, num_envs)
    rewards = batch.rewards.reshape(shape)
    values = batch.old_values.reshape(shape)
    next_values = batch.next_values.reshape(shape)
    terminated = batch.terminated.reshape(shape)
    boundaries = torch.logical_or(terminated, batch.truncated.reshape(shape))
    advantages = torch.zeros_like(rewards)
    next_advantage = torch.zeros(num_envs, dtype=rewards.dtype, device=rewards.device)
    for time_index in range(time_steps - 1, -1, -1):
        bootstrap = (~terminated[time_index]).to(rewards.dtype)
        delta = (
            rewards[time_index]
            + gamma * next_values[time_index] * bootstrap
            - values[time_index]
        )
        continuation = (~boundaries[time_index]).to(rewards.dtype)
        next_advantage = delta + gamma * gae_lambda * continuation * next_advantage
        advantages[time_index] = next_advantage
    advantages = advantages.reshape(-1)
    return advantages, advantages + batch.old_values


def group_relative_advantages(
    batch: TrajectoryBatch,
    *,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(batch.episode_ids) != batch.size or len(batch.group_ids) != batch.size:
        raise ValueError("trajectory metadata does not match the batch size")
    episode_returns: dict[int, torch.Tensor] = {}
    episode_groups: dict[int, int] = {}
    episode_reset_ids: dict[int, set[str]] = {}
    for index, (episode_id, group_id, reset_id) in enumerate(
        zip(batch.episode_ids, batch.group_ids, batch.reset_ids, strict=True)
    ):
        episode_returns[episode_id] = episode_returns.get(
            episode_id, torch.zeros((), device=batch.rewards.device)
        ) + batch.rewards[index]
        episode_groups[episode_id] = group_id
        episode_reset_ids.setdefault(episode_id, set()).add(reset_id)
    if any(len(reset_ids) != 1 for reset_ids in episode_reset_ids.values()):
        raise ValueError("one trajectory contains multiple reset IDs")

    grouped_episodes: dict[int, list[int]] = {}
    for episode_id, group_id in episode_groups.items():
        grouped_episodes.setdefault(group_id, []).append(episode_id)

    episode_advantages: dict[int, torch.Tensor] = {}
    for group_id, episode_ids in grouped_episodes.items():
        if len(episode_ids) < 2:
            raise ValueError(f"GRPO group {group_id} has fewer than two trajectories")
        reset_ids = {
            next(iter(episode_reset_ids[episode_id])) for episode_id in episode_ids
        }
        if len(reset_ids) != 1:
            raise ValueError(f"GRPO group {group_id} does not share one reset snapshot")
        returns = torch.stack([episode_returns[episode_id] for episode_id in episode_ids])
        standard_deviation = returns.std(unbiased=False)
        if float(standard_deviation) <= epsilon:
            normalized = torch.zeros_like(returns)
        else:
            normalized = (returns - returns.mean()) / (standard_deviation + epsilon)
        for episode_id, advantage in zip(episode_ids, normalized, strict=True):
            episode_advantages[episode_id] = advantage

    advantages = torch.stack(
        [episode_advantages[episode_id] for episode_id in batch.episode_ids]
    )
    returns = torch.stack([episode_returns[episode_id] for episode_id in batch.episode_ids])
    return advantages, returns
