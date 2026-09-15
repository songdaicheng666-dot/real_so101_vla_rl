"""Typed configuration for state-based MuJoCo RL baselines."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    name: str = "basic_t0"
    action_chunk_size: int = 8
    control_hz: int = 30
    max_episode_chunks: int = 128
    position_jitter_m: float = 0.005
    yaw_jitter_deg: float = 10.0


@dataclass(frozen=True, slots=True)
class RewardConfig:
    approach_scale: float = 10.0
    grasp_bonus: float = 1.0
    lift_scale: float = 20.0
    lift_bonus: float = 2.0
    transport_scale: float = 10.0
    inside_bonus: float = 4.0
    release_bonus: float = 2.0
    success_bonus: float = 10.0
    time_penalty: float = -0.001
    failure_penalty: float = -10.0
    lift_threshold_m: float = 0.010
    stable_seconds: float = 3.0
    max_linear_speed_m_s: float = 0.010
    max_angular_speed_rad_s: float = 0.20


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    kind: str = "mlp_squashed_gaussian"
    hidden_sizes: tuple[int, ...] = (256, 256)
    initial_log_std: float = -3.0


@dataclass(frozen=True, slots=True)
class CollectionConfig:
    num_envs: int = 16
    steps_per_env_per_update: int = 32
    group_size: int = 4


@dataclass(frozen=True, slots=True)
class AlgorithmConfig:
    name: str = "ppo"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coefficient: float = 0.20
    value_clip_coefficient: float = 0.20
    value_coefficient: float = 0.50
    entropy_coefficient: float = 0.01
    learning_rate: float = 3e-4
    epochs: int = 4
    minibatch_size: int = 128
    max_gradient_norm: float = 0.50
    kl_coefficient: float = 0.0


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int = 42
    device: str = "auto"
    total_updates: int = 1000
    evaluation_interval: int = 50
    evaluation_episodes: int = 16
    checkpoint_interval: int = 50
    output_root: str = "runs/rl/hybrid"
    route: str = "hybrid"
    source: str = "sim"
    run_name: str | None = None


@dataclass(frozen=True, slots=True)
class RLConfig:
    environment: EnvironmentConfig
    reward: RewardConfig
    policy: PolicyConfig
    collection: CollectionConfig
    algorithm: AlgorithmConfig
    training: TrainingConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_section[ConfigSection](
    raw: dict[str, Any],
    key: str,
    section_type: type[ConfigSection],
) -> ConfigSection:
    values = raw.get(key, {})
    if not isinstance(values, dict):
        raise TypeError(f"{key} must be a mapping")
    expected = set(section_type.__dataclass_fields__)
    unknown = set(values) - expected
    if unknown:
        raise ValueError(f"unknown {key} keys: {sorted(unknown)}")
    if section_type is PolicyConfig and "hidden_sizes" in values:
        values = dict(values)
        values["hidden_sizes"] = tuple(values["hidden_sizes"])
    return section_type(**values)


def validate_rl_config(config: RLConfig) -> None:
    env = config.environment
    collection = config.collection
    algorithm = config.algorithm
    training = config.training
    if env.name != "basic_t0":
        raise ValueError("only the basic_t0 task is implemented")
    if env.action_chunk_size <= 0 or env.control_hz <= 0 or env.max_episode_chunks <= 0:
        raise ValueError("environment timing values must be positive")
    if env.position_jitter_m < 0 or env.yaw_jitter_deg < 0:
        raise ValueError("environment reset jitter must be non-negative")
    if config.policy.kind != "mlp_squashed_gaussian":
        raise ValueError("only mlp_squashed_gaussian is implemented")
    if not config.policy.hidden_sizes or any(size <= 0 for size in config.policy.hidden_sizes):
        raise ValueError("policy.hidden_sizes must contain positive integers")
    if collection.num_envs <= 0 or collection.steps_per_env_per_update <= 0:
        raise ValueError("collection sizes must be positive")
    if collection.group_size <= 1:
        raise ValueError("collection.group_size must be greater than one")
    if algorithm.name not in {"ppo", "grpo"}:
        raise ValueError("algorithm.name must be ppo or grpo")
    if algorithm.name == "grpo" and collection.num_envs % collection.group_size:
        raise ValueError("GRPO num_envs must be divisible by group_size")
    if not 0 < algorithm.gamma <= 1 or not 0 <= algorithm.gae_lambda <= 1:
        raise ValueError("gamma and gae_lambda are outside their valid ranges")
    positive_algorithm_values = (
        algorithm.clip_coefficient,
        algorithm.value_clip_coefficient,
        algorithm.learning_rate,
        algorithm.epochs,
        algorithm.minibatch_size,
        algorithm.max_gradient_norm,
    )
    if any(value <= 0 for value in positive_algorithm_values):
        raise ValueError("algorithm optimization sizes must be positive")
    if not math.isfinite(algorithm.kl_coefficient) or algorithm.kl_coefficient < 0:
        raise ValueError("algorithm.kl_coefficient must be finite and non-negative")
    if training.total_updates <= 0 or training.evaluation_episodes <= 0:
        raise ValueError("training update and evaluation counts must be positive")
    if training.evaluation_interval <= 0 or training.checkpoint_interval <= 0:
        raise ValueError("training intervals must be positive")
    if training.route not in {"hybrid", "real_only"}:
        raise ValueError("training.route must be hybrid or real_only")
    if training.source not in {"sim", "real"}:
        raise ValueError("training.source must be sim or real")


def rl_config_from_dict(raw: dict[str, Any]) -> RLConfig:
    """Parse a configuration mapping, including mappings saved in checkpoints."""

    if not isinstance(raw, dict):
        raise TypeError("RL configuration must be a mapping")
    expected = {"environment", "reward", "policy", "collection", "algorithm", "training"}
    unknown = set(raw) - expected
    if unknown:
        raise ValueError(f"unknown top-level RL config keys: {sorted(unknown)}")
    config = RLConfig(
        environment=_parse_section(raw, "environment", EnvironmentConfig),
        reward=_parse_section(raw, "reward", RewardConfig),
        policy=_parse_section(raw, "policy", PolicyConfig),
        collection=_parse_section(raw, "collection", CollectionConfig),
        algorithm=_parse_section(raw, "algorithm", AlgorithmConfig),
        training=_parse_section(raw, "training", TrainingConfig),
    )
    validate_rl_config(config)
    return config


def load_rl_config(path: str | Path) -> RLConfig:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    return rl_config_from_dict(raw)


def smoke_config(config: RLConfig) -> RLConfig:
    """Return a fast configuration that still performs two real updates."""

    num_envs = 4 if config.algorithm.name == "grpo" else 2
    smoke = replace(
        config,
        environment=replace(config.environment, max_episode_chunks=2),
        collection=replace(
            config.collection,
            num_envs=num_envs,
            steps_per_env_per_update=2,
            group_size=4,
        ),
        algorithm=replace(config.algorithm, epochs=1, minibatch_size=8),
        training=replace(
            config.training,
            total_updates=2,
            evaluation_interval=2,
            evaluation_episodes=4,
            checkpoint_interval=2,
        ),
    )
    validate_rl_config(smoke)
    return smoke
