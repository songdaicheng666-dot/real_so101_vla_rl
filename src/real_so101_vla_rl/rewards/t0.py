"""Stage-aware reward and termination logic for the Basic T0 task."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from real_so101_vla_rl.envs.components.state import TaskState

REWARD_COMPONENTS = (
    "approach",
    "grasp",
    "lift_progress",
    "lift_event",
    "transport",
    "inside",
    "release",
    "success",
    "time",
    "failure",
)


@dataclass(frozen=True, slots=True)
class T0RewardConfig:
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
class RewardResult:
    total: float
    components: dict[str, float]
    events: tuple[str, ...]
    terminated: bool
    success: bool
    failure_type: str | None


class T0Reward:
    """Maintain episode phase state and score one 30 Hz control transition."""

    def __init__(
        self,
        config: T0RewardConfig,
        *,
        control_hz: int,
        board_bounds_xy: tuple[tuple[float, float], tuple[float, float]],
    ) -> None:
        self.config = config
        self.control_hz = control_hz
        self.board_bounds_xy = board_bounds_xy
        self._initialized = False

    def reset(self, state: TaskState) -> None:
        self.initial_z = {battery.color: float(battery.position[2]) for battery in state.batteries}
        self.previous_tcp_distance = self._tcp_distance(state)
        self.previous_target_z = float(state.target.position[2])
        self.previous_target_distance_xy = self._target_distance_xy(state)
        self.grasped_once = False
        self.lifted_once = False
        self.inside_once = False
        self.released_once = False
        self.stable_steps = 0
        self._initialized = True

    @staticmethod
    def _tcp_distance(state: TaskState) -> float:
        return float(np.linalg.norm(state.tcp_position - state.target.position))

    @staticmethod
    def _target_distance_xy(state: TaskState) -> float:
        return float(np.linalg.norm(state.target.position[:2] - state.target_center_xy))

    def _out_of_bounds(self, state: TaskState) -> bool:
        (x_min, x_max), (y_min, y_max) = self.board_bounds_xy
        x, y, z = state.target.position
        return bool(x < x_min or x > x_max or y < y_min or y > y_max or z < -0.02)

    def step(self, state: TaskState) -> RewardResult:
        if not self._initialized:
            raise RuntimeError("T0Reward.reset() must be called before step()")

        components = {name: 0.0 for name in REWARD_COMPONENTS}
        events: list[str] = []
        target = state.target
        double_contact = target.fixed_jaw_contact and target.moving_jaw_contact

        tcp_distance = self._tcp_distance(state)
        target_z = float(target.position[2])
        target_distance_xy = self._target_distance_xy(state)

        if not self.grasped_once:
            components["approach"] = self.config.approach_scale * (
                self.previous_tcp_distance - tcp_distance
            )
        if double_contact and not self.grasped_once:
            self.grasped_once = True
            components["grasp"] = self.config.grasp_bonus
            events.append("target_grasped")

        if self.grasped_once and not self.inside_once:
            components["lift_progress"] = self.config.lift_scale * (
                target_z - self.previous_target_z
            )
        if (
            target_z >= self.initial_z[state.target_color] + self.config.lift_threshold_m
            and not self.lifted_once
        ):
            self.lifted_once = True
            components["lift_event"] = self.config.lift_bonus
            events.append("target_lifted")

        if self.lifted_once and not self.inside_once:
            components["transport"] = self.config.transport_scale * (
                self.previous_target_distance_xy - target_distance_xy
            )
        if state.target_inside and not self.inside_once:
            self.inside_once = True
            components["inside"] = self.config.inside_bonus
            events.append("target_inside_t0")

        jaw_contact = target.fixed_jaw_contact or target.moving_jaw_contact
        if self.grasped_once and state.target_inside and not jaw_contact and not self.released_once:
            self.released_once = True
            components["release"] = self.config.release_bonus
            events.append("target_released_in_t0")

        linear_speed = float(np.linalg.norm(target.linear_velocity))
        angular_speed = float(np.linalg.norm(target.angular_velocity))
        stable_now = (
            state.target_inside
            and target.board_contact
            and not jaw_contact
            and linear_speed < self.config.max_linear_speed_m_s
            and angular_speed < self.config.max_angular_speed_rad_s
        )
        self.stable_steps = self.stable_steps + 1 if stable_now else 0
        stable_steps_required = round(self.config.stable_seconds * self.control_hz)
        success = self.stable_steps >= stable_steps_required
        if success:
            components["success"] = self.config.success_bonus
            events.append("task_success")

        failure_type = None
        for battery in state.batteries:
            if battery.color == state.target_color:
                continue
            if battery.position[2] >= self.initial_z[battery.color] + self.config.lift_threshold_m:
                failure_type = "wrong_battery_lifted"
                break
        if failure_type is None and self._out_of_bounds(state):
            failure_type = "target_out_of_bounds"
        if failure_type is not None:
            components["failure"] = self.config.failure_penalty
            events.append(failure_type)

        components["time"] = self.config.time_penalty
        self.previous_tcp_distance = tcp_distance
        self.previous_target_z = target_z
        self.previous_target_distance_xy = target_distance_xy
        return RewardResult(
            total=float(sum(components.values())),
            components=components,
            events=tuple(events),
            terminated=success or failure_type is not None,
            success=success,
            failure_type=failure_type,
        )
