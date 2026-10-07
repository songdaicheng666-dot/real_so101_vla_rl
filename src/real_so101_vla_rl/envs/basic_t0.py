"""Gymnasium environment for the Competition 2026 Basic T0 task."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import mujoco
import numpy as np
import yaml
from gymnasium.utils import seeding

from real_so101_vla_rl.alignment import (
    OverviewSensorModel,
    home_qpos,
    load_alignment,
    reset_to_home_keyframe,
    validate_mujoco_joint_refs,
)
from real_so101_vla_rl.envs.base import BatteryReset, ResetSnapshot
from real_so101_vla_rl.envs.components import (
    BATTERY_COLORS,
    ActionChunkController,
    PrivilegedStateBuilder,
)
from real_so101_vla_rl.joint_angle_mapping import JointAngleMapping
from real_so101_vla_rl.rewards import REWARD_COMPONENTS, T0Reward, T0RewardConfig

ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets" / "mujoco" / "competition_2026"
SCENE_PATH = ASSET_ROOT / "scene_basic_t0.xml"
LAYOUT_PATH = ASSET_ROOT / "layouts.yaml"
ACTUATOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def _quaternion_multiply(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.asarray(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dtype=np.float64,
    )


class SO101BasicT0Env(gym.Env[np.ndarray, np.ndarray]):
    """State-based SO101 task with RGB rendering available only on demand."""

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["rgb_array"],
        "render_fps": 30,
    }
    task_name = "basic_t0"

    def __init__(
        self,
        *,
        action_chunk_size: int = 8,
        control_hz: int = 30,
        max_episode_chunks: int = 128,
        position_jitter_m: float = 0.005,
        yaw_jitter_rad: float = math.radians(10.0),
        reward_config: T0RewardConfig | None = None,
        render_mode: str | None = None,
        render_width: int = 640,
        render_height: int = 480,
        overview_mode: str = "sensor",
        sensor_noise: bool = True,
        seed: int | None = None,
    ) -> None:
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"unsupported render_mode: {render_mode}")
        if overview_mode not in ("sensor", "raw"):
            raise ValueError("overview_mode must be 'sensor' or 'raw'")
        if overview_mode == "sensor" and (render_width, render_height) != (640, 480):
            raise ValueError("sensor overview rendering requires 640x480")
        if max_episode_chunks <= 0:
            raise ValueError("max_episode_chunks must be positive")
        if position_jitter_m < 0 or yaw_jitter_rad < 0:
            raise ValueError("reset jitter must be non-negative")
        if not SCENE_PATH.is_file() or not LAYOUT_PATH.is_file():
            raise FileNotFoundError("Competition 2026 T0 assets are incomplete")

        self.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        self._validate_model()
        self.alignment = load_alignment()
        validate_mujoco_joint_refs(self.model, alignment=self.alignment)
        self.joint_mapping = JointAngleMapping(self.alignment)
        self.joint_mapping.validate_mujoco_ranges(self.model)
        self.alignment_id = str(self.alignment["alignment_id"])
        self._home_key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "home"
        )
        if self._home_key_id < 0 or not np.allclose(
            self.model.key_qpos[self._home_key_id, :6],
            home_qpos(alignment=self.alignment),
            rtol=0.0,
            atol=1e-9,
        ):
            raise RuntimeError("scene home keyframe differs from recorded real home")
        if not np.allclose(
            self.model.key_ctrl[self._home_key_id, :6],
            home_qpos(alignment=self.alignment),
            rtol=0.0,
            atol=1e-9,
        ):
            raise RuntimeError("scene home control differs from recorded real home")
        reset_to_home_keyframe(self.model, self.data)
        self.action_chunk_size = action_chunk_size
        self.action_size = self.model.nu
        self.control_hz = control_hz
        self.max_episode_chunks = max_episode_chunks
        self.position_jitter_m = position_jitter_m
        self.yaw_jitter_rad = yaw_jitter_rad
        self.render_mode = render_mode
        self.render_width = render_width
        self.render_height = render_height
        self.overview_mode = overview_mode
        self.sensor_noise = sensor_noise
        self._renderer: mujoco.Renderer | None = None
        self._sensor_model = (
            OverviewSensorModel(self.alignment) if overview_mode == "sensor" else None
        )
        self._render_frame_index = 0
        self._np_random, self._np_random_seed = seeding.np_random(seed)

        with LAYOUT_PATH.open(encoding="utf-8") as stream:
            layout = yaml.safe_load(stream)
        target_layout = layout["scenes"]["basic_t0"]["target"]
        battery_layout = layout["common"]["battery_cubes"]
        board_bounds = layout["board"]["bounds_m"]
        self.target_center_xy = tuple(float(v) for v in target_layout["center_m"])
        self.target_size_xy = tuple(float(v) for v in target_layout["size_m"])
        self.board_bounds_xy = (
            tuple(float(v) for v in board_bounds["x"]),
            tuple(float(v) for v in board_bounds["y"]),
        )
        self.state_builder = PrivilegedStateBuilder(
            self.model,
            target_center_xy=self.target_center_xy,
            target_size_xy=self.target_size_xy,
            battery_half_size=0.5 * float(battery_layout["edge_length_m"]),
        )
        self.observation_size = self.state_builder.observation_size
        self.controller = ActionChunkController(
            self.model.actuator_ctrlrange,
            action_chunk_size=action_chunk_size,
            control_hz=control_hz,
            physics_timestep=float(self.model.opt.timestep),
        )
        self.reward = T0Reward(
            reward_config or T0RewardConfig(),
            control_hz=control_hz,
            board_bounds_xy=self.board_bounds_xy,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_chunk_size, self.action_size),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_size,),
            dtype=np.float32,
        )
        robot_qpos = self.model.key_qpos[self._home_key_id, : self.model.nu]
        self.initial_normalized_action = self.controller.normalize(robot_qpos).astype(
            np.float32
        )
        self.current_snapshot: ResetSnapshot | None = None
        self.target_color = BATTERY_COLORS[0]
        self.episode_chunks = 0
        self.episode_control_steps = 0

    def _validate_model(self) -> None:
        if (self.model.nq, self.model.nv, self.model.nu) != (34, 30, 6):
            raise RuntimeError("basic_t0 model must have nq=34, nv=30, nu=6")
        names = tuple(
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
            for index in range(self.model.nu)
        )
        if names != ACTUATOR_NAMES:
            raise RuntimeError(f"unexpected actuator order: {names}")

    @property
    def np_random(self) -> np.random.Generator:
        return self._np_random

    def sample_reset_snapshot(self, target_color: str | None = None) -> ResetSnapshot:
        if target_color is None:
            target_color = str(self.np_random.choice(BATTERY_COLORS))
        if target_color not in BATTERY_COLORS:
            raise ValueError(f"unknown battery color: {target_color}")

        qpos = self.model.key_qpos[self._home_key_id].astype(np.float64, copy=True)
        qvel = np.zeros(self.model.nv, dtype=np.float64)
        randomization = []
        for color in BATTERY_COLORS:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"battery_{color}"
            )
            joint_id = int(self.model.body_jntadr[body_id])
            qpos_address = int(self.model.jnt_qposadr[joint_id])
            x_jitter = float(
                self.np_random.uniform(-self.position_jitter_m, self.position_jitter_m)
            )
            y_jitter = float(
                self.np_random.uniform(-self.position_jitter_m, self.position_jitter_m)
            )
            yaw_jitter = float(
                self.np_random.uniform(-self.yaw_jitter_rad, self.yaw_jitter_rad)
            )
            qpos[qpos_address] += x_jitter
            qpos[qpos_address + 1] += y_jitter
            yaw_quaternion = np.asarray(
                (math.cos(yaw_jitter / 2.0), 0.0, 0.0, math.sin(yaw_jitter / 2.0)),
                dtype=np.float64,
            )
            base_quaternion = qpos[qpos_address + 3 : qpos_address + 7].copy()
            quaternion = _quaternion_multiply(yaw_quaternion, base_quaternion)
            qpos[qpos_address + 3 : qpos_address + 7] = quaternion / np.linalg.norm(
                quaternion
            )
            randomization.append(
                BatteryReset(color, x_jitter, y_jitter, yaw_jitter)
            )

        digest = hashlib.sha256()
        digest.update(target_color.encode("ascii"))
        digest.update(qpos.tobytes())
        reset_id = digest.hexdigest()[:16]
        return ResetSnapshot(
            reset_id=reset_id,
            target_color=target_color,
            qpos=tuple(float(value) for value in qpos),
            qvel=tuple(float(value) for value in qvel),
            battery_randomization=tuple(randomization),
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        snapshot = options.get("snapshot")
        if snapshot is None:
            snapshot = self.sample_reset_snapshot(options.get("target_color"))
        if not isinstance(snapshot, ResetSnapshot):
            raise TypeError("options['snapshot'] must be a ResetSnapshot")
        if len(snapshot.qpos) != self.model.nq or len(snapshot.qvel) != self.model.nv:
            raise ValueError("reset snapshot does not match the MuJoCo model")

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = snapshot.qpos
        self.data.qvel[:] = snapshot.qvel
        self.data.ctrl[:] = self.data.qpos[: self.model.nu]
        self.controller.reset_timing()
        mujoco.mj_forward(self.model, self.data)

        self.current_snapshot = snapshot
        self.target_color = snapshot.target_color
        self.episode_chunks = 0
        self.episode_control_steps = 0
        self._render_frame_index = 0
        state = self.state_builder.extract(self.data, target_color=self.target_color)
        self.reward.reset(state)
        return self.state_builder.observation(state), self._base_info()

    def _base_info(self) -> dict[str, Any]:
        if self.current_snapshot is None:
            raise RuntimeError("environment has not been reset")
        return {
            "task_name": self.task_name,
            "target_color": self.target_color,
            "instruction": (
                f"Pick up the {self.target_color} battery and place it in T0."
            ),
            "reset_id": self.current_snapshot.reset_id,
            "episode_id": self.current_snapshot.reset_id,
            "snapshot": self.current_snapshot,
            "alignment_id": self.alignment_id,
            "overview_mode": self.overview_mode,
        }

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        normalized_chunk = self.controller.validate_chunk(action)
        control_chunk = self.controller.denormalize(normalized_chunk)
        return self._step_control_chunk(control_chunk)

    def step_real_action_chunk(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute LeRobot absolute degrees/percent through the joint mapping."""

        real_chunk = np.asarray(action, dtype=np.float64)
        expected = (self.action_chunk_size, self.action_size)
        if real_chunk.shape != expected or not np.all(np.isfinite(real_chunk)):
            raise ValueError(f"real action must be finite with shape {expected}")
        requested = self.joint_mapping.real_to_mujoco_qpos(real_chunk)
        effective = np.clip(
            requested,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )
        result = self._step_control_chunk(requested)
        info = result[4]
        info.update(
            {
                "requested_real_action_deg_or_percent": real_chunk.copy(),
                "requested_sim_ctrl_rad": requested.copy(),
                "effective_sim_ctrl_rad": effective,
                "control_limited": ~np.isclose(
                    requested, effective, rtol=0, atol=1e-12
                ),
            }
        )
        return result

    def real_joint_state(self) -> np.ndarray:
        """Read the six simulated joints in LeRobot degrees/percent."""

        if self.current_snapshot is None:
            raise RuntimeError("reset() must be called before reading robot state")
        qpos = self.data.qpos[self.state_builder.robot_qpos_addresses]
        return self.joint_mapping.mujoco_qpos_to_real_state(qpos)

    def _step_control_chunk(
        self,
        control_chunk: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.current_snapshot is None:
            raise RuntimeError("reset() must be called before step()")
        action_mask = np.zeros(self.action_chunk_size, dtype=np.float32)
        reward_components = {name: 0.0 for name in REWARD_COMPONENTS}
        events: list[str] = []
        total_reward = 0.0
        terminated = False
        success = False
        failure_type = None

        for action_index, controls in enumerate(control_chunk):
            self.data.ctrl[:] = controls
            for _ in range(self.controller.next_physics_steps()):
                mujoco.mj_step(self.model, self.data)
            self.episode_control_steps += 1
            action_mask[action_index] = 1.0
            state = self.state_builder.extract(self.data, target_color=self.target_color)
            result = self.reward.step(state)
            total_reward += result.total
            for name, value in result.components.items():
                reward_components[name] += value
            events.extend(result.events)
            terminated = result.terminated
            success = result.success
            failure_type = result.failure_type
            if terminated:
                break

        self.episode_chunks += 1
        truncated = not terminated and self.episode_chunks >= self.max_episode_chunks
        final_state = self.state_builder.extract(self.data, target_color=self.target_color)
        info = self._base_info()
        info.update(
            {
                "action_mask": action_mask,
                "reward_components": reward_components,
                "reward_events": tuple(events),
                "success": success,
                "failure_type": failure_type,
                "episode_chunks": self.episode_chunks,
                "episode_control_steps": self.episode_control_steps,
            }
        )
        return (
            self.state_builder.observation(final_state),
            float(total_reward),
            terminated,
            truncated,
            info,
        )

    def render(self) -> np.ndarray:
        if self.render_mode != "rgb_array":
            raise RuntimeError("construct the environment with render_mode='rgb_array'")
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model,
                height=self.render_height,
                width=self.render_width,
            )
        self._renderer.update_scene(self.data, camera="overview")
        raw = self._renderer.render().copy()
        if self._sensor_model is None:
            return raw
        seed = int(self._np_random_seed) + self._render_frame_index
        self._render_frame_index += 1
        return self._sensor_model.apply(
            raw, seed=seed, noise=self.sensor_noise
        )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
