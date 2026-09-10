"""Train-split-only statistics for OpenVLA-OFT BOUNDS_Q99 normalization."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .robot_profile import _atomic_write
from .schema import (
    ACTION_DIM,
    ACTION_KEY,
    SCHEMA_VERSION,
    STATE_KEY,
    validate_joint_vector,
)


@dataclass(frozen=True, slots=True)
class FeatureStats:
    minimum: tuple[float, ...]
    maximum: tuple[float, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    q01: tuple[float, ...]
    q99: tuple[float, ...]
    mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        fields = (self.minimum, self.maximum, self.mean, self.std, self.q01, self.q99, self.mask)
        if any(len(field) != ACTION_DIM for field in fields):
            raise ValueError(f"Every statistics field must contain {ACTION_DIM} values")

    def to_openvla_dict(self) -> dict[str, list[float] | list[bool]]:
        return {
            "min": list(self.minimum),
            "max": list(self.maximum),
            "mean": list(self.mean),
            "std": list(self.std),
            "q01": list(self.q01),
            "q99": list(self.q99),
            "mask": list(self.mask),
        }


@dataclass(frozen=True, slots=True)
class NormalizationStats:
    schema_version: int
    method: str
    unnorm_key: str
    train_episode_sha256: str
    train_episode_count: int
    observation_state: FeatureStats
    action: FeatureStats

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.method != "bounds_q99":
            raise ValueError("Only schema v1 BOUNDS_Q99 statistics are supported")
        if not self.unnorm_key.strip():
            raise ValueError("unnorm_key must not be empty")
        if len(self.train_episode_sha256) != 64 or self.train_episode_count <= 0:
            raise ValueError("Invalid training episode provenance")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_openvla_dict(self) -> dict[str, dict[str, dict[str, list]]]:
        return {
            self.unnorm_key: {
                STATE_KEY: self.observation_state.to_openvla_dict(),
                ACTION_KEY: self.action.to_openvla_dict(),
            }
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> NormalizationStats:
        raw = dict(raw)
        for key in ("observation_state", "action"):
            stats = dict(raw[key])
            for stats_key in ("minimum", "maximum", "mean", "std", "q01", "q99", "mask"):
                stats[stats_key] = tuple(stats[stats_key])
            raw[key] = FeatureStats(**stats)
        return cls(**raw)


def training_episode_digest(episode_indices: Iterable[int]) -> tuple[str, int]:
    raw_indices = list(episode_indices)
    indices = sorted(raw_indices)
    if (
        not indices
        or len(indices) != len(set(indices))
        or any(type(index) is not int or index < 0 for index in indices)
    ):
        raise ValueError("Training episode indices must be non-empty, unique non-negative integers")
    payload = ",".join(str(index) for index in indices).encode("ascii")
    return hashlib.sha256(payload).hexdigest(), len(indices)


def _quantile(sorted_values: Sequence[float], quantile: float) -> float:
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _compute_feature_stats(vectors: Sequence[tuple[float, ...]]) -> FeatureStats:
    if not vectors:
        raise ValueError("Cannot compute normalization statistics from no samples")
    columns = tuple(tuple(vector[index] for vector in vectors) for index in range(ACTION_DIM))
    minimum = tuple(min(column) for column in columns)
    maximum = tuple(max(column) for column in columns)
    mean = tuple(sum(column) / len(column) for column in columns)
    std = tuple(
        math.sqrt(sum((value - column_mean) ** 2 for value in column) / len(column))
        for column, column_mean in zip(columns, mean, strict=True)
    )
    sorted_columns = tuple(tuple(sorted(column)) for column in columns)
    q01 = tuple(_quantile(column, 0.01) for column in sorted_columns)
    q99 = tuple(_quantile(column, 0.99) for column in sorted_columns)
    return FeatureStats(minimum, maximum, mean, std, q01, q99, (True,) * ACTION_DIM)


def compute_normalization_stats(
    samples: Iterable[Mapping[str, Sequence[float]]],
    *,
    train_episode_indices: Iterable[int],
    unnorm_key: str,
) -> NormalizationStats:
    """Compute state/action statistics after the caller selects successful train episodes."""

    states = []
    actions = []
    for sample_index, sample in enumerate(samples):
        if STATE_KEY not in sample or ACTION_KEY not in sample:
            raise ValueError(f"Sample {sample_index} is missing {STATE_KEY!r} or {ACTION_KEY!r}")
        states.append(validate_joint_vector(sample[STATE_KEY], field_name=f"sample[{sample_index}].state"))
        actions.append(validate_joint_vector(sample[ACTION_KEY], field_name=f"sample[{sample_index}].action"))
    digest, episode_count = training_episode_digest(train_episode_indices)
    return NormalizationStats(
        schema_version=SCHEMA_VERSION,
        method="bounds_q99",
        unnorm_key=unnorm_key,
        train_episode_sha256=digest,
        train_episode_count=episode_count,
        observation_state=_compute_feature_stats(states),
        action=_compute_feature_stats(actions),
    )


def normalize_q99(values: Sequence[float], stats: FeatureStats) -> tuple[float, ...]:
    values = validate_joint_vector(values, field_name="values")
    normalized = []
    for value, low, high in zip(values, stats.q01, stats.q99, strict=True):
        if high < low:
            raise ValueError("q99 must be greater than or equal to q01")
        clipped = min(high, max(low, value))
        normalized.append(2 * (clipped - low) / (high - low + 1e-8) - 1)
    return tuple(normalized)


def unnormalize_q99(values: Sequence[float], stats: FeatureStats) -> tuple[float, ...]:
    values = validate_joint_vector(values, field_name="normalized values")
    return tuple(
        0.5 * (value + 1) * (high - low + 1e-8) + low
        for value, low, high in zip(values, stats.q01, stats.q99, strict=True)
    )


def write_normalization_stats(path: str | Path, stats: NormalizationStats) -> None:
    payload = json.dumps(stats.to_dict(), ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    _atomic_write(Path(path), payload)


def load_normalization_stats(path: str | Path) -> NormalizationStats:
    return NormalizationStats.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
