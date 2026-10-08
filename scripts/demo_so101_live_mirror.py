"""Mirror a hand-moved SO-101 in the standalone robot or formal T0 scene.

Joints 1--4 and the gripper linearly map their complete follower calibration
ranges onto the independently measured MuJoCo hard-stop ranges.  Wrist roll
keeps direct angle mirroring because no MuJoCo mechanical stop was found.
Only the real arm's torque setting changes, after confirmation. The formal
MJCF already contains the calibrated joint and actuator ranges.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
import yaml
from mujoco.glfw import glfw

from real_so101_vla_rl.alignment import (
    home_qpos,
    load_alignment,
    reset_to_home_keyframe,
    validate_mujoco_joint_refs,
)
from real_so101_vla_rl.joint_angle_mapping import JointAngleMapping

if __package__:
    from .demo_mujoco_so101_unlimited import (
        JOINT_NAMES,
        ROBOT_PATH,
        SERVO_SELECTION_KEYS,
        reset_robot,
        restore_viewer_shortcut_side_effect,
        validate_robot,
    )
    from .demo_mujoco_t0_grasp import load_t0_scene
else:
    from demo_mujoco_so101_unlimited import (
        JOINT_NAMES,
        ROBOT_PATH,
        SERVO_SELECTION_KEYS,
        reset_robot,
        restore_viewer_shortcut_side_effect,
        validate_robot,
    )
    from demo_mujoco_t0_grasp import load_t0_scene


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_PATH = (
    PROJECT_ROOT / "calibration/lerobot/robots/so_follower/my_follower_arm.json"
)
RECORDING_CONFIG_PATH = PROJECT_ROOT / "configs/recording/so101_t0_100_lowlight_v1.yaml"
ROBOT_ID = "my_follower_arm"
ENCODER_MAX = 4095
HOMING_POSITION = ENCODER_MAX // 2
STATUS_PERIOD_S = 1.0
SCENE_CHOICES = ("robot", "basic_t0")


@dataclass(frozen=True)
class MappedJointState:
    """One real reading and its endpoint-linear MuJoCo mapping."""

    real_value: float
    normalized: float
    mujoco_deg: float
    outside_calibration: bool


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


def mapped_mujoco_ranges_deg(
    joint_refs_deg: Sequence[float],
) -> tuple[tuple[float, float] | None, ...]:
    """Return the formal qpos endpoints, or ``None`` for direct wrist roll."""

    if len(joint_refs_deg) != 5 or not all(
        math.isfinite(float(value)) for value in joint_refs_deg
    ):
        raise ValueError("expected five finite MuJoCo joint references")
    expected_refs = load_alignment()["robot"]["joint_ref_deg"]
    if not np.allclose(joint_refs_deg, expected_refs, rtol=0, atol=1e-8):
        raise ValueError("joint references differ from the formal alignment")
    ranges = JointAngleMapping().mujoco_ranges_deg()
    return tuple(
        None if index == 4 else tuple(float(value) for value in ranges[index])
        for index in range(6)
    )


def load_calibrated_robot(
    real_ranges: Sequence[tuple[float, float]] | None = None,
    *,
    scene: str = "robot",
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load a mirror model and check its formal mapped limits and home."""

    if scene == "robot":
        if not ROBOT_PATH.is_file():
            raise FileNotFoundError(f"SO-101 robot model not found: {ROBOT_PATH}")
        model = mujoco.MjModel.from_xml_path(str(ROBOT_PATH))
        validate_robot(model)
        data = mujoco.MjData(model)
        reset_robot(model, data)
    elif scene == "basic_t0":
        model, data = load_t0_scene()
        robot_joint_names = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
            for index in range(len(JOINT_NAMES))
        )
        if robot_joint_names != JOINT_NAMES:
            raise RuntimeError(f"unexpected robot joint order: {robot_joint_names}")
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        expected_home = home_qpos()
        if (
            key_id < 0
            or not np.allclose(
                model.key_qpos[key_id, :6], expected_home, rtol=0, atol=1e-9
            )
            or not np.allclose(
                model.key_ctrl[key_id, :6], expected_home, rtol=0, atol=1e-9
            )
        ):
            raise RuntimeError("T0 home differs from the formal real-home mapping")
        reset_to_home_keyframe(model, data)
    else:
        raise ValueError(f"unknown mirror scene: {scene}")

    alignment_robot = load_alignment()["robot"]
    calibrated_ranges = tuple(
        real_ranges if real_ranges is not None else load_calibrated_ranges_deg()
    )
    physical_ranges = alignment_robot["physical_ranges_deg_or_percent"]
    if len(physical_ranges) != len(JOINT_NAMES) or any(
        not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-9)
        for actual_range, expected_range in zip(
            physical_ranges, calibrated_ranges, strict=True
        )
        for actual, expected in zip(actual_range, expected_range, strict=True)
    ):
        raise RuntimeError("alignment physical ranges differ from follower calibration")

    JointAngleMapping().validate_mujoco_ranges(model)
    validate_mujoco_joint_refs(model)
    return model, data


def map_real_state(
    state: Mapping[str, float],
    real_ranges: Sequence[tuple[float, float]],
    joint_refs_deg: Sequence[float],
    signs: Sequence[int],
) -> tuple[MappedJointState, ...]:
    """Map follower readings through the formal converter for display."""

    if (
        len(real_ranges) != len(JOINT_NAMES)
        or len(joint_refs_deg) != 5
        or len(signs) != 5
        or any(sign not in (-1, 1) for sign in signs)
    ):
        raise ValueError(
            "expected six real ranges, five joint references and five +/-1 signs"
        )
    try:
        values = tuple(float(state[name]) for name in JOINT_NAMES)
    except KeyError as exc:
        raise ValueError(f"missing follower joint: {exc.args[0]}") from exc
    refs = tuple(float(value) for value in joint_refs_deg)
    if not all(math.isfinite(value) for value in (*values, *refs)):
        raise ValueError("joint readings and references must be finite")
    mapper = JointAngleMapping()
    if not np.allclose(real_ranges, mapper.real_ranges, rtol=0, atol=1e-8):
        raise ValueError("real ranges differ from the formal alignment")
    if not np.allclose(
        refs, load_alignment()["robot"]["joint_ref_deg"], rtol=0, atol=1e-8
    ):
        raise ValueError("joint references differ from the formal alignment")
    displayed_values = np.asarray(values, dtype=np.float64).copy()
    for index, sign in enumerate(signs):
        if sign < 0:
            if index == 4:
                displayed_values[index] = 2 * refs[index] - values[index]
            else:
                lower, upper = mapper.real_ranges[index]
                displayed_values[index] = lower + upper - values[index]
    mapped_deg = np.rad2deg(mapper.real_to_mujoco_qpos(displayed_values))
    return tuple(
        MappedJointState(
            real_value=value,
            normalized=(value - mapper.real_ranges[index, 0])
            / (mapper.real_ranges[index, 1] - mapper.real_ranges[index, 0]),
            mujoco_deg=float(mapped_deg[index]),
            outside_calibration=bool(
                value < mapper.real_ranges[index, 0] - 1e-8
                or value > mapper.real_ranges[index, 1] + 1e-8
            ),
        )
        for index, value in enumerate(values)
    )


def map_real_state_deg(
    state: Mapping[str, float],
    zero_deg: Sequence[float],
    signs: Sequence[int],
) -> tuple[float, ...]:
    """Compatibility wrapper returning only mapped formal qpos degrees."""

    return tuple(
        item.mujoco_deg
        for item in map_real_state(
            state,
            load_calibrated_ranges_deg(),
            zero_deg,
            signs,
        )
    )


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
            raise RuntimeError(
                f"inconsistent MuJoCo limits for {JOINT_NAMES[joint_id]}"
            )
        limited_value = min(max(value, lower), upper)
        data.qpos[model.jnt_qposadr[joint_id]] = limited_value
        data.ctrl[joint_id] = limited_value
        applied.append(math.degrees(limited_value))
    for joint_id in range(len(JOINT_NAMES)):
        data.qvel[model.jnt_dofadr[joint_id]] = 0
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


def print_mapping_ranges(
    model: mujoco.MjModel,
    real_ranges: Sequence[tuple[float, float]],
    joint_refs_deg: Sequence[float],
) -> None:
    """Print the complete endpoint mapping used by this demo."""

    formal_ranges = mapped_mujoco_ranges_deg(joint_refs_deg)
    print("[范围映射] 真机端点直接线性映射到正式 MuJoCo qpos 端点")
    for index, (name, real_range, formal_range) in enumerate(
        zip(JOINT_NAMES, real_ranges, formal_ranges, strict=True)
    ):
        real_unit = "%" if name == "gripper" else "deg"
        if formal_range is None:
            model_range = tuple(math.degrees(value) for value in model.jnt_range[index])
            qpos_text = f"direct [{model_range[0]:+.6f},{model_range[1]:+.6f}] deg"
        else:
            qpos_text = f"[{formal_range[0]:+.6f},{formal_range[1]:+.6f}] deg"
        print(
            f"  {index + 1} {name:14s} "
            f"real=[{real_range[0]:+.6f},{real_range[1]:+.6f}] {real_unit} "
            f"qpos={qpos_text}"
        )


def _print_state(
    mapped: Sequence[MappedJointState],
    applied_deg: Sequence[float],
    signs: Sequence[int],
) -> None:
    direction = " ".join(
        f"{name}={'+' if sign > 0 else '-'}"
        for name, sign in zip(JOINT_NAMES[:5], signs, strict=True)
    )
    print(f"[方向] {direction}")
    for name, item, applied in zip(JOINT_NAMES, mapped, applied_deg, strict=True):
        unit = "%" if name == "gripper" else "°"
        clipping = " INPUT_OUTSIDE_CALIBRATION" if item.outside_calibration else ""
        if not math.isclose(applied, item.mujoco_deg, rel_tol=0.0, abs_tol=1e-9):
            clipping += " MODEL_CLIPPED"
        print(
            f"[镜像] {name:14s} "
            f"真机={item.real_value:+9.4f}{unit} "
            f"t={item.normalized:.6f} "
            f"请求={item.mujoco_deg:+10.6f}° "
            f"应用={applied:+10.6f}°{clipping}"
        )


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
    real_ranges: Sequence[tuple[float, float]],
    *,
    poll_hz: float,
    scene: str = "robot",
) -> None:
    key_events: queue.SimpleQueue[int] = queue.SimpleQueue()
    signs = [1] * 5
    selected = 0
    last_status = -math.inf
    print(
        "按 1～5 选择关节，F 翻转所选关节方向，P 打印全部数值；关闭窗口退出。\n"
        "1～4 号和夹爪按两端标定范围线性映射；5 号保持直接角度镜像。"
    )
    print(
        "[零位] "
        + ", ".join(
            f"{name}={value:+.4f}°"
            for name, value in zip(JOINT_NAMES[:5], zero_deg, strict=True)
        )
    )
    print_mapping_ranges(model, real_ranges, zero_deg)
    if scene == "basic_t0":
        print(
            "[场景] basic_t0：方块保持 home 位置；Viewer 左侧 Camera 可切换 Free/overview。"
        )

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_events.put,
        show_left_ui=scene == "basic_t0",
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
                    if keycode == glfw.KEY_F:
                        viewer.opt.flags[:] = flags_before
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
                mapped = map_real_state(real, real_ranges, zero_deg, signs)
                requested_deg = tuple(item.mujoco_deg for item in mapped)
                sim_deg = apply_pose_deg(model, data, requested_deg)
            viewer.sync()
            if print_now or started - last_status >= STATUS_PERIOD_S:
                _print_state(mapped, sim_deg, signs)
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
        "--scene",
        choices=SCENE_CHOICES,
        default="robot",
        help="mirror in the standalone robot (default) or formal basic_t0 scene",
    )
    parser.add_argument(
        "--port", help="override the configured SO-101 follower serial port"
    )
    parser.add_argument("--poll-hz", type=_positive_float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    zero_deg = load_homing_zero_deg()
    real_ranges = load_calibrated_ranges_deg()
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
    model, data = load_calibrated_robot(real_ranges, scene=args.scene)

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
            run_mirror(
                model,
                data,
                bus,
                zero_deg,
                real_ranges,
                poll_hz=args.poll_hz,
                scene=args.scene,
            )
    except KeyboardInterrupt:
        print("\n[结束] 已停止镜像。")
    print("[真机] 串口已断开；程序未重新开启扭矩。")


if __name__ == "__main__":
    main()
