from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("mujoco")

from real_so101_vla_rl.envs import SO101BasicT0Env
from real_so101_vla_rl.envs.components.action import ActionChunkController
from real_so101_vla_rl.envs.components.state import (
    BATTERY_COLORS,
    BatteryState,
    TaskState,
    cylinder_projection_extent,
)
from real_so101_vla_rl.rewards import T0Reward, T0RewardConfig


def _battery(
    color: str,
    *,
    position: tuple[float, float, float],
    fixed: bool = False,
    moving: bool = False,
    board: bool = True,
) -> BatteryState:
    return BatteryState(
        color=color,
        position=np.asarray(position, dtype=np.float64),
        quaternion=np.asarray((1.0, 0.0, 0.0, 0.0)),
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
        axis=np.asarray((0.0, 1.0, 0.0)),
        fixed_jaw_contact=fixed,
        moving_jaw_contact=moving,
        board_contact=board,
    )


def _state(
    *,
    target_position: tuple[float, float, float] = (-0.2, 0.3, 0.00525),
    fixed: bool = False,
    moving: bool = False,
    inside: bool = False,
) -> TaskState:
    batteries = tuple(
        _battery(
            color,
            position=target_position if color == "red" else (-0.1, 0.2, 0.00525),
            fixed=fixed if color == "red" else False,
            moving=moving if color == "red" else False,
        )
        for color in BATTERY_COLORS
    )
    return TaskState(
        robot_qpos=np.zeros(6),
        robot_qvel=np.zeros(6),
        tcp_position=np.asarray(target_position, dtype=np.float64),
        tcp_orientation_6d=np.asarray((1.0, 0.0, 0.0, 1.0, 0.0, 0.0)),
        batteries=batteries,
        target_color="red",
        target_center_xy=np.asarray((0.115, 0.3)),
        target_inside=inside,
    )


def test_basic_t0_reset_snapshot_is_exactly_reusable() -> None:
    first = SO101BasicT0Env(seed=7)
    second = SO101BasicT0Env(seed=8)
    try:
        snapshot = first.sample_reset_snapshot("blue")
        observation_1, info_1 = first.reset(options={"snapshot": snapshot})
        observation_2, info_2 = second.reset(options={"snapshot": snapshot})

        assert observation_1.shape == (92,)
        assert observation_1.dtype == np.float32
        assert np.array_equal(observation_1, observation_2)
        assert np.array_equal(first.data.qpos, second.data.qpos)
        assert info_1["reset_id"] == info_2["reset_id"] == snapshot.reset_id
        assert info_1["target_color"] == "blue"
    finally:
        first.close()
        second.close()


def test_reset_randomization_bounds_and_seed_reproducibility() -> None:
    environment = SO101BasicT0Env(seed=1)
    try:
        observation_1, info_1 = environment.reset(
            seed=123, options={"target_color": "green"}
        )
        observation_2, info_2 = environment.reset(
            seed=123, options={"target_color": "green"}
        )
        assert np.array_equal(observation_1, observation_2)
        assert info_1["reset_id"] == info_2["reset_id"]
        for randomization in info_1["snapshot"].battery_randomization:
            assert abs(randomization.x_jitter_m) <= 0.005
            assert abs(randomization.y_jitter_m) <= 0.005
            assert abs(randomization.yaw_jitter_rad) <= np.deg2rad(10.0)
    finally:
        environment.close()


def test_action_conversion_and_control_clock_have_no_drift() -> None:
    controller = ActionChunkController(
        np.asarray(((-2.0, 2.0), (-1.0, 3.0))),
        action_chunk_size=2,
        control_hz=30,
        physics_timestep=0.002,
    )
    assert controller.denormalize(np.asarray((-1.0, 1.0))) == pytest.approx(
        (-2.0, 3.0)
    )
    assert [controller.next_physics_steps() for _ in range(6)] == [16, 17, 17, 16, 17, 17]
    controller.reset_timing()
    assert sum(controller.next_physics_steps() for _ in range(30)) == 500


def test_episode_horizon_and_action_mask_are_reported() -> None:
    environment = SO101BasicT0Env(seed=3, max_episode_chunks=1)
    try:
        environment.reset(options={"target_color": "yellow"})
        action = np.tile(environment.initial_normalized_action, (8, 1))
        _, reward, terminated, truncated, info = environment.step(action)
        assert np.isfinite(reward)
        assert not terminated
        assert truncated
        assert info["episode_control_steps"] == 8
        assert info["action_mask"].tolist() == [1.0] * 8
        assert set(info["reward_components"]) == {
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
        }
    finally:
        environment.close()


def test_cylinder_containment_uses_full_projected_footprint() -> None:
    horizontal = cylinder_projection_extent(
        np.asarray((0.0, 1.0, 0.0)), radius=0.00525, half_length=0.02225
    )
    vertical = cylinder_projection_extent(
        np.asarray((0.0, 0.0, 1.0)), radius=0.00525, half_length=0.02225
    )
    assert horizontal == pytest.approx((0.00525, 0.02225))
    assert vertical == pytest.approx((0.00525, 0.00525))


def test_reward_emits_grasp_lift_release_and_stable_success() -> None:
    reward = T0Reward(
        T0RewardConfig(stable_seconds=0.2),
        control_hz=10,
        board_bounds_xy=((-0.4, 0.3), (-0.1, 0.45)),
    )
    initial = _state()
    reward.reset(initial)

    grasp = reward.step(_state(fixed=True, moving=True))
    assert grasp.components["grasp"] == 1.0
    assert "target_grasped" in grasp.events

    lifted_state = _state(
        target_position=(-0.2, 0.3, 0.020), fixed=True, moving=True
    )
    lifted = reward.step(lifted_state)
    assert lifted.components["lift_event"] == 2.0
    assert "target_lifted" in lifted.events

    released_state = _state(target_position=(0.115, 0.3, 0.00525), inside=True)
    released = reward.step(released_state)
    assert released.components["inside"] == 4.0
    assert released.components["release"] == 2.0
    assert not released.terminated

    success = reward.step(released_state)
    assert success.success
    assert success.terminated
    assert success.components["success"] == 10.0


def test_reward_terminates_when_a_wrong_battery_is_lifted() -> None:
    reward = T0Reward(
        T0RewardConfig(),
        control_hz=30,
        board_bounds_xy=((-0.4, 0.3), (-0.1, 0.45)),
    )
    state = _state()
    reward.reset(state)
    batteries = list(state.batteries)
    batteries[1] = replace(
        batteries[1], position=np.asarray((-0.1, 0.2, 0.020), dtype=np.float64)
    )
    failed = reward.step(replace(state, batteries=tuple(batteries)))
    assert failed.terminated
    assert failed.failure_type == "wrong_battery_lifted"
    assert failed.components["failure"] == -10.0
