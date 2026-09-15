"""Synchronous single-machine trajectory collection."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from real_so101_vla_rl.envs.base import TaskEnvironment
from real_so101_vla_rl.rl.policies import ChunkPolicy

from .batches import CollectionResult, TrajectoryBatch


def _tensor(array: np.ndarray | list, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.asarray(array), dtype=dtype)


class PPOCollector:
    """Collect fixed-length windows while preserving episode boundaries."""

    def __init__(
        self,
        environments: list[TaskEnvironment],
        *,
        steps_per_env: int,
        device: torch.device,
    ) -> None:
        if not environments or steps_per_env <= 0:
            raise ValueError("PPO collection requires environments and positive steps")
        self.environments = environments
        self.steps_per_env = steps_per_env
        self.device = device
        self.observations: list[np.ndarray] = []
        self.infos: list[dict[str, Any]] = []
        self._episode_returns = [0.0] * len(environments)
        self._next_episode_id = len(environments)
        self._episode_ids = list(range(len(environments)))
        for environment in environments:
            observation, info = environment.reset()
            self.observations.append(observation)
            self.infos.append(info)

    def collect(self, policy: ChunkPolicy) -> CollectionResult:
        fields: dict[str, list] = defaultdict(list)
        metric_sums: dict[str, float] = defaultdict(float)
        completed_returns: list[float] = []
        completed_lengths: list[int] = []

        for _ in range(self.steps_per_env):
            observation_array = np.stack(self.observations)
            observation_tensor = _tensor(observation_array).to(self.device)
            with torch.no_grad():
                output = policy.sample(observation_tensor)
            if output.value is None:
                raise RuntimeError("PPO collection requires a critic value")
            action_array = output.action_chunk.detach().cpu().numpy()

            next_observations: list[np.ndarray] = []
            transition_infos: list[dict[str, Any]] = []
            terminal_flags: list[bool] = []
            truncation_flags: list[bool] = []
            rewards: list[float] = []
            episode_ids: list[int] = []
            reset_ids: list[str] = []
            action_masks: list[np.ndarray] = []
            for index, environment in enumerate(self.environments):
                episode_ids.append(self._episode_ids[index])
                reset_ids.append(str(self.infos[index]["reset_id"]))
                observation, reward, terminated, truncated, info = environment.step(
                    action_array[index]
                )
                next_observations.append(observation)
                transition_infos.append(info)
                terminal_flags.append(terminated)
                truncation_flags.append(truncated)
                rewards.append(reward)
                action_masks.append(np.asarray(info["action_mask"], dtype=np.float32))
                self._episode_returns[index] += reward
                for name, value in info["reward_components"].items():
                    metric_sums[f"reward/{name}"] += float(value)
                for event in info["reward_events"]:
                    metric_sums[f"events/{event}"] += 1.0
                if terminated or truncated:
                    completed_returns.append(self._episode_returns[index])
                    completed_lengths.append(int(info["episode_chunks"]))
                    self._episode_returns[index] = 0.0
                    metric_sums["episodes"] += 1.0
                    metric_sums["successes"] += float(info["success"])
                    if info["failure_type"]:
                        metric_sums[f"failures/{info['failure_type']}"] += 1.0

            next_tensor = _tensor(np.stack(next_observations)).to(self.device)
            with torch.no_grad():
                next_value = policy.value(next_tensor)
            if next_value is None:
                raise RuntimeError("PPO collection requires next-state values")

            fields["observations"].append(observation_array)
            fields["actions"].append(action_array)
            fields["old_log_probs"].append(output.log_prob.detach().cpu().numpy())
            fields["rewards"].append(rewards)
            fields["terminated"].append(terminal_flags)
            fields["truncated"].append(truncation_flags)
            fields["action_masks"].append(action_masks)
            fields["old_values"].append(output.value.detach().cpu().numpy())
            fields["next_values"].append(next_value.detach().cpu().numpy())
            fields["episode_ids"].extend(episode_ids)
            fields["group_ids"].extend([-1] * len(self.environments))
            fields["reset_ids"].extend(reset_ids)

            for index, environment in enumerate(self.environments):
                if terminal_flags[index] or truncation_flags[index]:
                    reset_observation, reset_info = environment.reset()
                    self._episode_ids[index] = self._next_episode_id
                    self._next_episode_id += 1
                    self.observations[index] = reset_observation
                    self.infos[index] = reset_info
                else:
                    self.observations[index] = next_observations[index]
                    self.infos[index] = transition_infos[index]

        time_steps = self.steps_per_env
        num_envs = len(self.environments)

        def flatten(name: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
            values = np.asarray(fields[name])
            return _tensor(values.reshape((-1, *values.shape[2:])), dtype=dtype)

        batch = TrajectoryBatch(
            observations=flatten("observations"),
            actions=flatten("actions"),
            old_log_probs=flatten("old_log_probs"),
            rewards=flatten("rewards"),
            terminated=flatten("terminated", torch.bool),
            truncated=flatten("truncated", torch.bool),
            action_masks=flatten("action_masks"),
            old_values=flatten("old_values"),
            next_values=flatten("next_values"),
            advantages=None,
            returns=None,
            episode_ids=tuple(fields["episode_ids"]),
            group_ids=tuple(fields["group_ids"]),
            reset_ids=tuple(fields["reset_ids"]),
            sequence_shape=(time_steps, num_envs),
        )
        metrics = dict(metric_sums)
        metrics["samples/chunks"] = float(batch.size)
        metrics["samples/control_steps"] = float(batch.action_masks.sum())
        if completed_returns:
            metrics["episodes/return_mean"] = float(np.mean(completed_returns))
            metrics["episodes/chunks_mean"] = float(np.mean(completed_lengths))
        return CollectionResult(batch=batch, metrics=metrics)


class GRPOCollector:
    """Collect G complete trajectories from each identical reset snapshot."""

    def __init__(
        self,
        environments: list[TaskEnvironment],
        *,
        group_size: int,
        device: torch.device,
    ) -> None:
        if not environments or group_size <= 1 or len(environments) % group_size:
            raise ValueError("GRPO environments must form complete groups")
        self.environments = environments
        self.group_size = group_size
        self.device = device
        self._trajectory_serial = 0

    def collect(self, policy: ChunkPolicy) -> CollectionResult:
        trajectory_records: list[list[dict[str, Any]]] = [
            [] for _ in self.environments
        ]
        observations: list[np.ndarray] = [np.empty(0)] * len(self.environments)
        infos: list[dict[str, Any]] = [{} for _ in self.environments]
        group_ids = [index // self.group_size for index in range(len(self.environments))]
        trajectory_ids = []
        for group_start in range(0, len(self.environments), self.group_size):
            snapshot = self.environments[group_start].sample_reset_snapshot()
            for index in range(group_start, group_start + self.group_size):
                observations[index], infos[index] = self.environments[index].reset(
                    options={"snapshot": snapshot}
                )
                trajectory_ids.append(self._trajectory_serial)
                self._trajectory_serial += 1

        active = set(range(len(self.environments)))
        metric_sums: dict[str, float] = defaultdict(float)
        episode_returns = [0.0] * len(self.environments)
        while active:
            active_indices = sorted(active)
            observation_tensor = _tensor(
                np.stack([observations[index] for index in active_indices])
            ).to(self.device)
            with torch.no_grad():
                output = policy.sample(observation_tensor)
            action_array = output.action_chunk.detach().cpu().numpy()
            log_prob_array = output.log_prob.detach().cpu().numpy()
            for batch_index, env_index in enumerate(active_indices):
                observation = observations[env_index]
                next_observation, reward, terminated, truncated, info = self.environments[
                    env_index
                ].step(action_array[batch_index])
                trajectory_records[env_index].append(
                    {
                        "observation": observation,
                        "action": action_array[batch_index],
                        "old_log_prob": log_prob_array[batch_index],
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "action_mask": np.asarray(info["action_mask"], dtype=np.float32),
                        "episode_id": trajectory_ids[env_index],
                        "group_id": group_ids[env_index],
                        "reset_id": str(info["reset_id"]),
                    }
                )
                observations[env_index] = next_observation
                infos[env_index] = info
                episode_returns[env_index] += reward
                for name, value in info["reward_components"].items():
                    metric_sums[f"reward/{name}"] += float(value)
                for event in info["reward_events"]:
                    metric_sums[f"events/{event}"] += 1.0
                if terminated or truncated:
                    active.remove(env_index)
                    metric_sums["episodes"] += 1.0
                    metric_sums["successes"] += float(info["success"])
                    if info["failure_type"]:
                        metric_sums[f"failures/{info['failure_type']}"] += 1.0

        records = [record for trajectory in trajectory_records for record in trajectory]
        if not records:
            raise RuntimeError("GRPO collection produced no transitions")
        batch = TrajectoryBatch(
            observations=_tensor([record["observation"] for record in records]),
            actions=_tensor([record["action"] for record in records]),
            old_log_probs=_tensor([record["old_log_prob"] for record in records]),
            rewards=_tensor([record["reward"] for record in records]),
            terminated=_tensor(
                [record["terminated"] for record in records], dtype=torch.bool
            ),
            truncated=_tensor(
                [record["truncated"] for record in records], dtype=torch.bool
            ),
            action_masks=_tensor([record["action_mask"] for record in records]),
            old_values=None,
            next_values=None,
            advantages=None,
            returns=None,
            episode_ids=tuple(record["episode_id"] for record in records),
            group_ids=tuple(record["group_id"] for record in records),
            reset_ids=tuple(record["reset_id"] for record in records),
            sequence_shape=None,
        )
        metrics = dict(metric_sums)
        metrics["samples/chunks"] = float(batch.size)
        metrics["samples/control_steps"] = float(batch.action_masks.sum())
        metrics["episodes/return_mean"] = float(np.mean(episode_returns))
        metrics["episodes/chunks_mean"] = float(
            np.mean([len(trajectory) for trajectory in trajectory_records])
        )
        metrics["groups"] = float(len(self.environments) // self.group_size)
        return CollectionResult(batch=batch, metrics=metrics)
