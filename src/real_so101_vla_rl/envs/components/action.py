"""Continuous action-chunk conversion and control timing."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class ActionChunkController:
    """Map normalized actions to actuators and schedule physics steps exactly."""

    def __init__(
        self,
        ctrlrange: NDArray[np.floating],
        *,
        action_chunk_size: int,
        control_hz: int,
        physics_timestep: float,
    ) -> None:
        ctrlrange = np.asarray(ctrlrange, dtype=np.float64)
        if ctrlrange.ndim != 2 or ctrlrange.shape[1] != 2:
            raise ValueError("ctrlrange must have shape [action_size, 2]")
        if action_chunk_size <= 0 or control_hz <= 0 or physics_timestep <= 0:
            raise ValueError("action timing values must be positive")
        physics_hz = round(1.0 / physics_timestep)
        if not np.isclose(physics_hz * physics_timestep, 1.0, atol=1e-9):
            raise ValueError("physics_timestep must be the reciprocal of an integer Hz")
        if physics_hz < control_hz:
            raise ValueError("physics rate must be at least the control rate")

        self.ctrlrange = ctrlrange
        self.action_chunk_size = action_chunk_size
        self.action_size = ctrlrange.shape[0]
        self.control_hz = control_hz
        self.physics_hz = physics_hz
        self._phase = 0

    def reset_timing(self) -> None:
        self._phase = 0

    def validate_chunk(self, action: NDArray[np.floating]) -> NDArray[np.float64]:
        action = np.asarray(action, dtype=np.float64)
        expected = (self.action_chunk_size, self.action_size)
        if action.shape != expected:
            raise ValueError(f"action must have shape {expected}, got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError("action contains non-finite values")
        return np.clip(action, -1.0, 1.0)

    def denormalize(self, action: NDArray[np.floating]) -> NDArray[np.float64]:
        action = np.asarray(action, dtype=np.float64)
        lower = self.ctrlrange[:, 0]
        upper = self.ctrlrange[:, 1]
        return lower + 0.5 * (action + 1.0) * (upper - lower)

    def normalize(self, controls: NDArray[np.floating]) -> NDArray[np.float64]:
        controls = np.asarray(controls, dtype=np.float64)
        lower = self.ctrlrange[:, 0]
        upper = self.ctrlrange[:, 1]
        return 2.0 * (controls - lower) / (upper - lower) - 1.0

    def next_physics_steps(self) -> int:
        """Return a deterministic 16/17-style schedule without time drift."""

        self._phase += self.physics_hz
        steps, self._phase = divmod(self._phase, self.control_hz)
        return steps
