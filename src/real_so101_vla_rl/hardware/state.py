"""Timestamped SO-101 joint-state acquisition."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from real_so101_vla_rl.data.schema import ordered_joint_vector

from .capture_types import CapturedSample


class SO101StateAdapter:
    """Wrap a follower observation call and retain only canonical joint state."""

    def __init__(
        self,
        robot: Any,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        max_consecutive_failures: int = 3,
    ) -> None:
        if max_consecutive_failures <= 0:
            raise ValueError("max_consecutive_failures must be positive")
        self._robot = robot
        self._clock_ns = clock_ns
        self._max_failures = max_consecutive_failures
        self._failures = 0
        self._sequence_id = 0

    @property
    def is_connected(self) -> bool:
        return bool(getattr(self._robot, "is_connected", False))

    @property
    def is_broken(self) -> bool:
        return self._failures >= self._max_failures

    def connect(self) -> None:
        if not self.is_connected:
            self._robot.connect()

    def read(self, timeout_ms: int) -> CapturedSample[np.ndarray]:
        del timeout_ms  # The follower bus owns its retry/timeout policy.
        sequence_id = self._sequence_id
        self._sequence_id += 1
        try:
            observation = self._robot.get_observation()
            timestamp_ns = self._clock_ns()
            if not isinstance(observation, Mapping):
                raise TypeError("robot observation must be a mapping")
            values = ordered_joint_vector(observation)
            state = np.asarray(values, dtype=np.float32)
        except Exception as exc:  # noqa: BLE001 - bus failures are capture results
            self._failures += 1
            return CapturedSample.failure(
                timestamp_ns=self._clock_ns(),
                sequence_id=sequence_id,
                error=f"{type(exc).__name__}: {exc}",
            )
        self._failures = 0
        return CapturedSample.success(
            state, timestamp_ns=timestamp_ns, sequence_id=sequence_id
        )

    def disconnect(self) -> None:
        # Robot lifecycle is owned by the recorder, not this view over its state.
        return None
