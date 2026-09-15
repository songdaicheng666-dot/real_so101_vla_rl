"""Observation boundary reserved for future RGB VLA policies."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class RGBProprioInstructionWrapper(gym.ObservationWrapper):
    """Expose RGB, robot proprioception, and canonical task text without task truth."""

    def __init__(self, environment: gym.Env) -> None:
        if getattr(environment, "render_mode", None) != "rgb_array":
            raise ValueError("visual observations require render_mode='rgb_array'")
        super().__init__(environment)
        height = int(environment.render_height)
        width = int(environment.render_width)
        self.observation_space = gym.spaces.Dict(
            {
                "rgb": gym.spaces.Box(0, 255, (height, width, 3), dtype=np.uint8),
                "proprioception": gym.spaces.Box(
                    -np.inf, np.inf, (12,), dtype=np.float32
                ),
                "instruction": gym.spaces.Text(max_length=128),
            }
        )

    def observation(self, observation: np.ndarray) -> dict[str, Any]:
        target_color = str(self.env.unwrapped.target_color)
        return {
            "rgb": self.env.render(),
            "proprioception": np.asarray(observation[:12], dtype=np.float32),
            "instruction": f"Pick up the {target_color} battery and place it in T0.",
        }
