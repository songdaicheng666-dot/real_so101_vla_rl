"""Construct environments and policies from the typed RL configuration."""

from __future__ import annotations

import math
from dataclasses import asdict

import torch

from real_so101_vla_rl.envs import TaskEnvironment, make_environment
from real_so101_vla_rl.rewards import T0RewardConfig
from real_so101_vla_rl.rl.config import RLConfig
from real_so101_vla_rl.rl.policies import MLPChunkPolicy


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def make_task_environment(
    config: RLConfig,
    *,
    seed: int,
    render_mode: str | None = None,
) -> TaskEnvironment:
    reward_config = T0RewardConfig(**asdict(config.reward))
    return make_environment(
        config.environment.name,
        action_chunk_size=config.environment.action_chunk_size,
        control_hz=config.environment.control_hz,
        max_episode_chunks=config.environment.max_episode_chunks,
        position_jitter_m=config.environment.position_jitter_m,
        yaw_jitter_rad=math.radians(config.environment.yaw_jitter_deg),
        reward_config=reward_config,
        render_mode=render_mode,
        seed=seed,
    )


def make_mlp_policy(
    config: RLConfig,
    environment: TaskEnvironment,
    *,
    has_value: bool,
    device: torch.device,
) -> MLPChunkPolicy:
    policy = MLPChunkPolicy(
        observation_size=environment.observation_size,
        action_chunk_size=environment.action_chunk_size,
        action_size=environment.action_size,
        hidden_sizes=config.policy.hidden_sizes,
        initial_action=environment.initial_normalized_action,
        initial_log_std=config.policy.initial_log_std,
        has_value=has_value,
    )
    return policy.to(device)
