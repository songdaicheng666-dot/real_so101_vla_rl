"""Shared task-environment contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


@dataclass(frozen=True, slots=True)
class BatteryReset:
    """Randomization applied to one battery in a reset snapshot."""

    color: str
    x_jitter_m: float
    y_jitter_m: float
    yaw_jitter_rad: float


@dataclass(frozen=True, slots=True)
class ResetSnapshot:
    """Complete, reusable initial state for one task episode."""

    reset_id: str
    target_color: str
    qpos: tuple[float, ...]
    qvel: tuple[float, ...]
    battery_randomization: tuple[BatteryReset, ...]


class TaskEnvironment(Protocol):
    """Minimum interface consumed by trajectory collectors."""

    observation_size: int
    action_chunk_size: int
    action_size: int

    def sample_reset_snapshot(self, target_color: str | None = None) -> ResetSnapshot:
        """Sample, but do not apply, an episode initial state."""

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[FloatArray, dict]:
        """Reset the environment and return an observation and metadata."""

    def step(self, action: FloatArray) -> tuple[FloatArray, float, bool, bool, dict]:
        """Execute one action chunk."""

    def close(self) -> None:
        """Release environment resources."""
