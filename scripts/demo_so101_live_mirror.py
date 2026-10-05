"""Mirror hand-moved SO-101 follower joints using the MuJoCo alignment.

The first five axes use the original LeRobot degree readings as MuJoCo angles.
The gripper maps its calibrated 0..100 percent range to the MJCF -10..100
degree range. Only the real arm's torque setting changes, after confirmation.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import yaml
from mujoco.glfw import glfw

from real_so101_vla_rl.alignment import (
    load_alignment,
    real_state_to_mujoco_qpos,
    validate_mujoco_joint_refs,
)

if __package__:
    from .demo_mujoco_so101_unlimited import (
        JOINT_NAMES,
        ROBOT_PATH,
        SERVO_SELECTION_KEYS,
        reset_robot,
        restore_viewer_shortcut_side_effect,
        validate_robot,
    )
else:
    from demo_mujoco_so101_unlimited import (
        JOINT_NAMES,
        ROBOT_PATH,
        SERVO_SELECTION_KEYS,
        reset_robot,
        restore_viewer_shortcut_side_effect,
        validate_robot,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_PATH = (
    PROJECT_ROOT / "calibration/lerobot/robots/so_follower/my_follower_arm.json"
)
RECORDING_CONFIG_PATH = PROJECT_ROOT / "configs/recording/so101_t0_100_lowlight_v1.yaml"
ROBOT_ID = "my_follower_arm"
ENCODER_MAX = 4095
HOMING_POSITION = ENCODER_MAX // 2
STATUS_PERIOD_S = 1.0


def load_homing_zero_deg(path: Path = CALIBRATION_PATH) -> tuple[float, ...]:
    """Return the five LeRobot degree readings at the saved half-turn pose."""

    calibration = json.loads(path.read_text(encoding="utf-8"))
    if set(calibration) != set(JOINT_NAMES):
        raise ValueError(
            "follower calibration must contain exactly the six SO-101 joints"
        )
    zeros = []
    for name in JOINT_NAMES[:5]:
        motor = calibration[name]
        if motor["drive_mode"] != 0 or motor["range_min"] >= motor["range_max"]:
            raise ValueError(f"invalid calibration for {name}")
        midpoint = (motor["range_min"] + motor["range_max"]) / 2
        zeros.append((HOMING_POSITION - midpoint) * 360 / ENCODER_MAX)
    return tuple(zeros)


def load_calibrated_ranges_deg(
    path: Path = CALIBRATION_PATH,
) -> tuple[tuple[float, float], ...]:
    """Derive the LeRobot ranges represented by the saved calibration."""

    calibration = json.loads(path.read_text(encoding="utf-8"))
    if set(calibration) != set(JOINT_NAMES):
        raise ValueError(
            "follower calibration must contain exactly the six SO-101 joints"
        )
    ranges = []
    for name in JOINT_NAMES[:5]:
        motor = calibration[name]
        if motor["drive_mode"] != 0 or motor["range_min"] >= motor["range_max"]:
            raise ValueError(f"invalid calibration for {name}")
        half_range = (motor["range_max"] - motor["range_min"]) * 180 / ENCODER_MAX
        ranges.append((-half_range, half_range))
    gripper = calibration["gripper"]
    if gripper["drive_mode"] != 0 or gripper["range_min"] >= gripper["range_max"]:
        raise ValueError("invalid calibration for gripper")
    ranges.append((0.0, 100.0))
    return tuple(ranges)


def load_calibrated_robot() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load the project robot with its authored calibration-derived limits."""

    if not ROBOT_PATH.is_file():
        raise FileNotFoundError(f"SO-101 robot model not found: {ROBOT_PATH}")
    model = mujoco.MjModel.from_xml_path(str(ROBOT_PATH))
    validate_robot(model)

    alignment_robot = load_alignment()["robot"]
    calibrated_ranges = load_calibrated_ranges_deg()
    physical_ranges = alignment_robot["physical_ranges_deg_or_percent"]
    if len(physical_ranges) != len(JOINT_NAMES) or any(
        not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-9)
        for actual_range, expected_range in zip(
            physical_ranges, calibrated_ranges, strict=True
        )
        for actual, expected in zip(actual_range, expected_range, strict=True)
    ):
        raise RuntimeError("alignment physical ranges differ from follower calibration")

    alignment_ranges = alignment_robot["mujoco_ranges_deg"]
    if len(alignment_ranges) != len(JOINT_NAMES):
        raise ValueError("alignment must contain six MuJoCo joint ranges")
    for index, (name, expected_deg) in enumerate(
        zip(JOINT_NAMES, alignment_ranges, strict=True)
    ):
        if not bool(model.jnt_limited[index]):
            raise RuntimeError(f"MuJoCo joint limit is disabled: {name}")
        if not bool(model.actuator_ctrllimited[index]):
            raise RuntimeError(f"MuJoCo actuator control limit is disabled: {name}")
        expected_rad = tuple(math.radians(float(value)) for value in expected_deg)
        if not all(
            math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-7)
            for actual, expected in zip(
                model.jnt_range[index], expected_rad, strict=True
            )
        ):
            raise RuntimeError(f"MuJoCo joint range differs from alignment: {name}")
        if not all(
            math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-7)
            for actual, expected in zip(
                model.actuator_ctrlrange[index], expected_rad, strict=True
            )
        ):
            raise RuntimeError(f"MuJoCo control range differs from alignment: {name}")

    data = mujoco.MjData(model)
    reset_robot(model, data)
    return model, data


def map_real_state_deg(
    state: Mapping[str, float],
    zero_deg: Sequence[float],
    signs: Sequence[int],
) -> tuple[float, ...]:
    """Map real readings to MuJoCo angles; sign flips are visual diagnostics."""

    if (
        len(zero_deg) != 5
        or len(signs) != 5
        or any(sign not in (-1, 1) for sign in signs)
    ):
        raise ValueError("expected five zero readings and five +/-1 signs")
    try:
        values = tuple(float(state[name]) for name in JOINT_NAMES)
    except KeyError as exc:
        raise ValueError(f"missing follower joint: {exc.args[0]}") from exc
    if not all(math.isfinite(value) for value in (*values, *zero_deg)):
        raise ValueError("joint readings and zero readings must be finite")
    alignment = load_alignment()
    formal_deg = tuple(
        math.degrees(value)
        for value in real_state_to_mujoco_qpos(values, alignment=alignment)
    )
    arm_deg = tuple(
        zero_deg[index] + signs[index] * (formal_deg[index] - zero_deg[index])
        for index in range(5)
    )
    real_min, real_max = (
        float(value)
        for value in alignment["robot"]["physical_ranges_deg_or_percent"][5]
    )
    sim_min, sim_max = (
        float(value) for value in alignment["robot"]["mujoco_ranges_deg"][5]
    )
    gripper_deg = sim_min + (values[5] - real_min) * (sim_max - sim_min) / (
        real_max - real_min
    )
    return arm_deg + (gripper_deg,)


def apply_pose_deg(
    model: mujoco.MjModel, data: mujoco.MjData, angles: Sequence[float]
) -> tuple[float, ...]:
    """Set the pose directly while respecting joint and control limits."""

    if len(angles) != len(JOINT_NAMES):
        raise ValueError("expected six MuJoCo joint angles")
    applied = []
    for joint_id, angle in enumerate(angles):
        value = math.radians(angle)
        lower = -math.inf
        upper = math.inf
        if model.jnt_limited[joint_id]:
            lower = max(lower, float(model.jnt_range[joint_id, 0]))
            upper = min(upper, float(model.jnt_range[joint_id, 1]))
        if model.actuator_ctrllimited[joint_id]:
            lower = max(lower, float(model.actuator_ctrlrange[joint_id, 0]))
            upper = min(upper, float(model.actuator_ctrlrange[joint_id, 1]))
        if lower > upper:
            raise RuntimeError(f"inconsistent MuJoCo limits for {JOINT_NAMES[joint_id]}")
        limited_value = min(max(value, lower), upper)
        data.qpos[model.jnt_qposadr[joint_id]] = limited_value
        data.ctrl[joint_id] = limited_value
        applied.append(math.degrees(limited_value))
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    return tuple(applied)


@contextmanager
def manual_follower_bus(follower: Any, *, confirm: Callable[[str], str] = input):
    """Check calibration, ask for arm support, then release torque for hand motion."""

    bus = follower.bus
    try:
        bus.connect()
        if not follower.is_calibrated:
            raise RuntimeError("舵机内标定与 my_follower_arm.json 不一致；已停止")
        reply = confirm(
            "请托住机械臂；准备好后按 Enter 关闭六轴扭矩（输入其他内容取消）："
        )
        if reply.strip():
            raise RuntimeError("已取消关闭扭矩")
        bus.disable_torque(num_retry=2)
        enabled = {
            name: bus.read("Torque_Enable", name, normalize=False, num_retry=2)
            for name in JOINT_NAMES
        }
        if any(value != 0 for value in enabled.values()):
            raise RuntimeError(f"无法确认全部舵机扭矩已关闭：{enabled}")
        print("[真机] 六轴扭矩已关闭；可托住机械臂逐轴缓慢移动。")
        yield bus
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)


def _print_state(
    real: Mapping[str, float], sim_deg: Sequence[float], signs: Sequence[int]
) -> None:
    direction = " ".join(
        f"{name}={'+' if sign > 0 else '-'}"
        for name, sign in zip(JOINT_NAMES[:5], signs, strict=True)
    )
    print(f"[方向] {direction}")
    for name, angle in zip(JOINT_NAMES, sim_deg, strict=True):
        unit = "%" if name == "gripper" else "°"
        print(f"[镜像] {name:14s} 真机={real[name]:+8.2f}{unit}  仿真={angle:+8.2f}°")


def read_real_state(bus: Any) -> Mapping[str, float]:
    try:
        return bus.sync_read("Present_Position", num_retry=2)
    except Exception as exc:
        raise RuntimeError("真机关节读取失败，镜像已停止") from exc


def run_mirror(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    bus: Any,
    zero_deg: Sequence[float],
    *,
    poll_hz: float,
) -> None:
    key_events: queue.SimpleQueue[int] = queue.SimpleQueue()
    signs = [1] * 5
    selected = 0
    last_status = -math.inf
    print(
        "按 1～5 选择关节，F 翻转所选关节方向，P 打印全部数值；关闭窗口退出。\n"
        "前五轴真机度数与仿真角度同值；夹爪按 0%→-10°、100%→100° 跟随。"
    )
    print(
        "[零位] "
        + ", ".join(
            f"{name}={value:+.4f}°"
            for name, value in zip(JOINT_NAMES[:5], zero_deg, strict=True)
        )
    )

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_events.put,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = model.stat.center
            viewer.cam.distance = 0.85
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25
        geomgroup_before = tuple(int(value) for value in viewer.opt.geomgroup)
        flags_before = tuple(int(value) for value in viewer.opt.flags)
        while viewer.is_running():
            started = time.monotonic()
            real = read_real_state(bus)
            print_now = False
            with viewer.lock():
                while True:
                    try:
                        keycode = key_events.get_nowait()
                    except queue.Empty:
                        break
                    restore_viewer_shortcut_side_effect(
                        viewer.opt,
                        keycode,
                        geomgroup_before=geomgroup_before,
                        flags_before=flags_before,
                    )
                    joint_index = SERVO_SELECTION_KEYS.get(keycode)
                    if joint_index is not None and joint_index < 5:
                        selected = joint_index
                        print(f"[选择] {JOINT_NAMES[selected]}")
                        print_now = True
                    elif keycode == glfw.KEY_F:
                        signs[selected] *= -1
                        print(
                            f"[方向] {JOINT_NAMES[selected]} 改为 {'+' if signs[selected] > 0 else '-'}1"
                        )
                        print_now = True
                    elif keycode == glfw.KEY_P:
                        print_now = True
                geomgroup_before = tuple(int(value) for value in viewer.opt.geomgroup)
                flags_before = tuple(int(value) for value in viewer.opt.flags)
                requested_deg = map_real_state_deg(real, zero_deg, signs)
                sim_deg = apply_pose_deg(model, data, requested_deg)
            viewer.sync()
            if print_now or started - last_status >= STATUS_PERIOD_S:
                _print_state(real, sim_deg, signs)
                last_status = started
            remaining = 1 / poll_hz - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
    print(
        "[结束] 最终方向："
        + " ".join(
            f"{name}={'+' if sign > 0 else '-'}"
            for name, sign in zip(JOINT_NAMES[:5], signs, strict=True)
        )
    )


def _positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be a finite value greater than zero")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port", help="override the configured SO-101 follower serial port"
    )
    parser.add_argument("--poll-hz", type=_positive_float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    zero_deg = load_homing_zero_deg()
    configured_ref = load_alignment()["robot"]["joint_ref_deg"]
    if any(
        not math.isclose(actual, expected, abs_tol=1e-8)
        for actual, expected in zip(zero_deg, configured_ref, strict=True)
    ):
        raise RuntimeError("真机标定中位与正式 alignment.yaml 的关节 ref 不一致")
    recording = yaml.safe_load(RECORDING_CONFIG_PATH.read_text(encoding="utf-8"))
    if recording["follower"]["id"] != ROBOT_ID:
        raise ValueError("recording configuration uses a different follower ID")
    port = args.port or recording["follower"]["port"]
    model, data = load_calibrated_robot()
    validate_mujoco_joint_refs(model)

    try:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ImportError as exc:
        raise SystemExit(
            "需要在安装了 LeRobot 和 MuJoCo 的 lerobot 环境中运行"
        ) from exc
    follower = SO101Follower(
        SO101FollowerConfig(
            port=port,
            id=ROBOT_ID,
            calibration_dir=CALIBRATION_PATH.parent,
            cameras={},
            use_degrees=True,
        )
    )
    print(f"[真机] 串口：{port}\n[真机] 标定：{CALIBRATION_PATH}")
    try:
        with manual_follower_bus(follower) as bus:
            run_mirror(model, data, bus, zero_deg, poll_hz=args.poll_hz)
    except KeyboardInterrupt:
        print("\n[结束] 已停止镜像。")
    print("[真机] 串口已断开；程序未重新开启扭矩。")


if __name__ == "__main__":
    main()
