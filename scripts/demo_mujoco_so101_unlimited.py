"""Inspect SO-101 motion without the XML joint and control-angle limits.

This experiment changes only the in-memory model. It does not model the real
servo's internal mechanical stops, and it does not alter the task scenes.
"""

from __future__ import annotations

import argparse
import math
import queue
import time
from collections.abc import Sequence
from pathlib import Path

try:
    import mujoco
    import mujoco.viewer
    from mujoco.glfw import glfw
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("mujoco"):
        raise SystemExit(
            "MuJoCo is required. Install it with: pip install -e '.[sim]'"
        ) from None
    raise


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_PATH = (
    PROJECT_ROOT
    / "src"
    / "real_so101_vla_rl"
    / "assets"
    / "mujoco"
    / "competition_2026"
    / "so101_competition.xml"
)
JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
SERVO_SELECTION_KEYS = {
    glfw.KEY_1: 0,
    glfw.KEY_2: 1,
    glfw.KEY_3: 2,
    glfw.KEY_4: 3,
    glfw.KEY_5: 4,
    glfw.KEY_6: 5,
}
ADJUSTMENT_KEYS = {glfw.KEY_A: -1, glfw.KEY_D: 1}
PAUSE_KEY = glfw.KEY_SPACE
RESET_KEY = glfw.KEY_BACKSPACE
STATUS_KEY = glfw.KEY_P
HEADLESS_CHECK_STEPS = 500
STATUS_PERIOD_S = 0.5

# Passive Viewer callbacks cannot consume native shortcuts. Restore their
# display effects so selecting a joint never makes robot geoms disappear.
GEOM_GROUP_SHORTCUTS = {
    glfw.KEY_1: 1,
    glfw.KEY_2: 2,
    glfw.KEY_3: 3,
    glfw.KEY_4: 4,
    glfw.KEY_5: 5,
}
VIS_FLAG_SHORTCUTS = {
    glfw.KEY_A: int(mujoco.mjtVisFlag.mjVIS_AUTOCONNECT),
    glfw.KEY_D: int(mujoco.mjtVisFlag.mjVIS_STATIC),
}


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Control only the SO-101 robot without XML angle limits.",
    )
    parser.add_argument(
        "--joint-step-deg",
        type=_positive_float,
        default=2.0,
        help="keyboard step for the five arm joints in degrees (default: 2)",
    )
    parser.add_argument(
        "--gripper-step-deg",
        type=_positive_float,
        default=1.0,
        help="keyboard step for the gripper joint in degrees (default: 1)",
    )
    parser.add_argument(
        "--headless-check",
        action="store_true",
        help="load and step the limit-free robot without opening the Viewer",
    )
    return parser.parse_args(argv)


def _object_id(model: mujoco.MjModel, object_type: int, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"required MuJoCo object is missing: {name}")
    return object_id


def validate_robot(model: mujoco.MjModel) -> None:
    if (model.nq, model.nv, model.nu, model.njnt) != (6, 6, 6, 6):
        raise RuntimeError("expected only the six SO-101 hinge joints and actuators")
    for index, name in enumerate(JOINT_NAMES):
        joint_id = _object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator_id = _object_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if joint_id != index or actuator_id != index:
            raise RuntimeError(f"unexpected joint or actuator order: {name}")
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise RuntimeError(f"expected a hinge joint: {name}")
        if int(model.actuator_trnid[actuator_id, 0]) != joint_id:
            raise RuntimeError(f"actuator does not drive its named joint: {name}")


def load_unlimited_robot() -> tuple[mujoco.MjModel, mujoco.MjData]:
    if not ROBOT_PATH.is_file():
        raise FileNotFoundError(f"SO-101 robot model not found: {ROBOT_PATH}")
    model = mujoco.MjModel.from_xml_path(str(ROBOT_PATH))
    validate_robot(model)
    # Joint stops and actuator command limits are two independent authored
    # bounds. Disable both, but leave contacts and actuator force limits intact.
    model.jnt_limited[:] = False
    model.actuator_ctrllimited[:] = False
    data = mujoco.MjData(model)
    reset_robot(model, data)
    return model, data


def reset_robot(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Restore the standalone model's reference pose and hold every joint there."""
    mujoco.mj_resetData(model, data)
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        data.ctrl[actuator_id] = data.qpos[model.jnt_qposadr[joint_id]]
    mujoco.mj_forward(model, data)


def adjust_control(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    servo_id: int,
    direction: int,
    *,
    joint_step_rad: float,
    gripper_step_rad: float,
) -> float:
    if servo_id not in range(1, 7):
        raise ValueError(f"servo ID must be in 1..6, got {servo_id}")
    if direction not in (-1, 1):
        raise ValueError(f"direction must be -1 or 1, got {direction}")
    actuator_id = servo_id - 1
    step = gripper_step_rad if servo_id == 6 else joint_step_rad
    data.ctrl[actuator_id] += direction * step
    return float(data.ctrl[actuator_id])


def restore_viewer_shortcut_side_effect(
    option: mujoco.MjvOption,
    keycode: int,
    *,
    geomgroup_before: Sequence[int],
    flags_before: Sequence[int],
) -> None:
    geom_group = GEOM_GROUP_SHORTCUTS.get(keycode)
    if geom_group is not None:
        option.geomgroup[geom_group] = geomgroup_before[geom_group]
    vis_flag = VIS_FLAG_SHORTCUTS.get(keycode)
    if vis_flag is not None:
        option.flags[vis_flag] = flags_before[vis_flag]


def contact_summary(model: mujoco.MjModel, data: mujoco.MjData) -> str:
    if data.ncon == 0:
        return "0"
    body_pairs = []
    for contact in data.contact[: min(data.ncon, 3)]:
        body_ids = (
            int(model.geom_bodyid[contact.geom1]),
            int(model.geom_bodyid[contact.geom2]),
        )
        names = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            or f"body#{body_id}"
            for body_id in body_ids
        )
        body_pairs.append("/".join(names))
    suffix = ", ..." if data.ncon > len(body_pairs) else ""
    return f"{data.ncon} ({', '.join(body_pairs)}{suffix})"


def status_line(model: mujoco.MjModel, data: mujoco.MjData, servo_id: int) -> str:
    actuator_id = servo_id - 1
    joint_id = int(model.actuator_trnid[actuator_id, 0])
    actual = float(data.qpos[model.jnt_qposadr[joint_id]])
    target = float(data.ctrl[actuator_id])
    velocity = float(data.qvel[model.jnt_dofadr[joint_id]])
    force = float(data.actuator_force[actuator_id])
    force_status = "off"
    if model.actuator_forcelimited[actuator_id]:
        lower, upper = model.actuator_forcerange[actuator_id]
        tolerance = 0.01 * float(upper - lower)
        saturated = force <= lower + tolerance or force >= upper - tolerance
        force_status = "YES" if saturated else "no"
    return (
        f"[状态] {servo_id} {JOINT_NAMES[actuator_id]} "
        f"目标={math.degrees(target):+.1f}° "
        f"实际={math.degrees(actual):+.1f}° "
        f"差值={math.degrees(target - actual):+.1f}° "
        f"速度={math.degrees(velocity):+.1f}°/s "
        f"驱动力={force:+.2f} 饱和={force_status} "
        f"接触={contact_summary(model, data)}"
    )


def run_headless_check(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    steps: int = HEADLESS_CHECK_STEPS,
) -> None:
    reset_robot(model, data)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    if not all(math.isfinite(float(value)) for value in data.qpos):
        raise RuntimeError("headless check failed: qpos contains a non-finite value")
    if not all(math.isfinite(float(value)) for value in data.qvel):
        raise RuntimeError("headless check failed: qvel contains a non-finite value")
    print(
        f"Headless check passed: robot only, nq={model.nq}, "
        f"nu={model.nu}, steps={steps}"
    )


def print_controls(joint_step_deg: float, gripper_step_deg: float) -> None:
    print(
        """
SO-101 standalone limit-free observation demo

  1  shoulder_pan (base)         4  wrist_flex
  2  shoulder_lift               5  wrist_roll
  3  elbow_flex                  6  gripper

  A / D      decrease / increase selected target
  P          print status immediately
  Space      pause / resume
  Backspace  reset robot to zero pose
  Close the Viewer window to exit

Both XML joint-angle and actuator-target limits are disabled in memory.
Collision geoms and actuator force limits remain. A stop is NOT evidence of
the real servo's internal mechanical stop; inspect target, actual, contacts
and force saturation together. The native Control sliders are hidden because
they may still display the old XML ranges.
""".strip()
    )
    print(
        f"Keyboard steps: arm={joint_step_deg:g} deg, "
        f"gripper={gripper_step_deg:g} deg"
    )


def run_interactive(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    joint_step_deg: float,
    gripper_step_deg: float,
) -> None:
    reset_robot(model, data)
    key_events: queue.SimpleQueue[int] = queue.SimpleQueue()
    joint_step_rad = math.radians(joint_step_deg)
    gripper_step_rad = math.radians(gripper_step_deg)
    selected_servo_id = 1
    paused = False
    last_status = -math.inf
    print_controls(joint_step_deg, gripper_step_deg)
    print(f"Loaded: {ROBOT_PATH}")

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
            loop_start = time.monotonic()
            print_status_now = False
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
                    if keycode in SERVO_SELECTION_KEYS:
                        selected_servo_id = SERVO_SELECTION_KEYS[keycode] + 1
                        print_status_now = True
                    elif keycode in ADJUSTMENT_KEYS:
                        target = adjust_control(
                            model,
                            data,
                            selected_servo_id,
                            ADJUSTMENT_KEYS[keycode],
                            joint_step_rad=joint_step_rad,
                            gripper_step_rad=gripper_step_rad,
                        )
                        print(
                            f"[控制] 舵机 {selected_servo_id} "
                            f"目标={math.degrees(target):+.1f}°"
                        )
                        print_status_now = True
                    elif keycode == PAUSE_KEY:
                        paused = not paused
                        print("[仿真] 已暂停" if paused else "[仿真] 已继续")
                    elif keycode == RESET_KEY:
                        reset_robot(model, data)
                        print("[复位] 六轴、速度和控制目标已恢复参考姿态")
                        print_status_now = True
                    elif keycode == STATUS_KEY:
                        print_status_now = True
                geomgroup_before = tuple(int(value) for value in viewer.opt.geomgroup)
                flags_before = tuple(int(value) for value in viewer.opt.flags)
                if not paused:
                    mujoco.mj_step(model, data)
                if print_status_now or loop_start - last_status >= STATUS_PERIOD_S:
                    status = status_line(model, data, selected_servo_id)
                    last_status = loop_start
                else:
                    status = None
            if status is not None:
                print(status)
            viewer.sync()
            target_period = 0.01 if paused else float(model.opt.timestep)
            remaining = target_period - (time.monotonic() - loop_start)
            if remaining > 0:
                time.sleep(remaining)


def main() -> None:
    args = parse_args()
    model, data = load_unlimited_robot()
    if args.headless_check:
        run_headless_check(model, data)
    else:
        run_interactive(
            model,
            data,
            joint_step_deg=args.joint_step_deg,
            gripper_step_deg=args.gripper_step_deg,
        )


if __name__ == "__main__":
    main()
