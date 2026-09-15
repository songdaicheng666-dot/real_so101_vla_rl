"""Interactively control SO101 and test AAA battery grasping in the T0 scene."""

from __future__ import annotations

import argparse
import math
import queue
import time
from collections.abc import Iterable, Sequence
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
SCENE_PATH = (
    PROJECT_ROOT
    / "src"
    / "real_so101_vla_rl"
    / "assets"
    / "mujoco"
    / "competition_2026"
    / "scene_basic_t0.xml"
)

ACTUATOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
SERVO_ID_TO_ACTUATOR = {
    1: "shoulder_pan",
    2: "shoulder_lift",
    3: "elbow_flex",
    4: "wrist_flex",
    5: "wrist_roll",
    6: "gripper",
}
BATTERY_BODY_NAMES = (
    "battery_red",
    "battery_yellow",
    "battery_blue",
    "battery_green",
)
EXPECTED_MODEL_SIZE = (34, 30, 6)
LIFT_DELTA_M = 0.010
HEADLESS_CHECK_STEPS = 500

# Top-row number keys select the matching physical SO101 servo ID.
SERVO_SELECTION_KEYS = {
    glfw.KEY_1: 1,
    glfw.KEY_2: 2,
    glfw.KEY_3: 3,
    glfw.KEY_4: 4,
    glfw.KEY_5: 5,
    glfw.KEY_6: 6,
}
ADJUSTMENT_KEYS = {glfw.KEY_A: -1, glfw.KEY_D: +1}
PAUSE_KEY = glfw.KEY_SPACE
RESET_KEY = glfw.KEY_BACKSPACE

# Simulate also binds top-row 0-5 to geom visibility groups and A/D to
# visualization flags. The passive key callback cannot consume those events,
# so the demo restores only the affected display value after handling a key.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Control SO101 in the Competition 2026 T0 MuJoCo scene.",
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
        help="load and step the scene without opening the Viewer",
    )
    return parser.parse_args()


def _object_id(model: mujoco.MjModel, object_type: int, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"required MuJoCo object is missing: {name}")
    return object_id


def actuator_names(model: mujoco.MjModel) -> tuple[str, ...]:
    return tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(model.nu)
    )


def validate_model(model: mujoco.MjModel) -> None:
    size = (model.nq, model.nv, model.nu)
    if size != EXPECTED_MODEL_SIZE:
        raise RuntimeError(
            f"unexpected T0 model size {size}; expected {EXPECTED_MODEL_SIZE}"
        )
    names = actuator_names(model)
    if names != ACTUATOR_NAMES:
        raise RuntimeError(f"unexpected actuator order: {names}")
    if tuple(SERVO_ID_TO_ACTUATOR.values()) != names:
        raise RuntimeError(
            "physical servo ID mapping does not match the MuJoCo actuator order"
        )
    for body_name in BATTERY_BODY_NAMES:
        _object_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)


def load_t0_scene() -> tuple[mujoco.MjModel, mujoco.MjData]:
    if not SCENE_PATH.is_file():
        raise FileNotFoundError(f"T0 scene not found: {SCENE_PATH}")
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    validate_model(model)
    return model, mujoco.MjData(model)


def initialize_controls(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Hold every position actuator at its corresponding joint position."""
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        qpos_address = int(model.jnt_qposadr[joint_id])
        lower, upper = model.actuator_ctrlrange[actuator_id]
        data.ctrl[actuator_id] = min(
            max(float(data.qpos[qpos_address]), float(lower)),
            float(upper),
        )


def reset_scene(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_resetData(model, data)
    initialize_controls(model, data)
    mujoco.mj_forward(model, data)


def servo_id_from_keycode(keycode: int) -> int | None:
    return SERVO_SELECTION_KEYS.get(keycode)


def apply_selected_control(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    servo_id: int,
    direction: int,
    *,
    joint_step_rad: float,
    gripper_step_rad: float,
) -> tuple[str, float]:
    if servo_id not in SERVO_ID_TO_ACTUATOR:
        raise ValueError(f"servo ID must be in 1..6, got {servo_id}")
    if direction not in (-1, 1):
        raise ValueError(f"direction must be -1 or 1, got {direction}")

    actuator_name = SERVO_ID_TO_ACTUATOR[servo_id]
    actuator_id = _object_id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        actuator_name,
    )
    step = gripper_step_rad if actuator_name == "gripper" else joint_step_rad
    lower, upper = model.actuator_ctrlrange[actuator_id]
    target = float(data.ctrl[actuator_id]) + direction * step
    data.ctrl[actuator_id] = min(max(target, float(lower)), float(upper))
    return actuator_name, float(data.ctrl[actuator_id])


def toggle_pause(paused: bool, keycode: int) -> bool:
    return not paused if keycode == PAUSE_KEY else paused


def restore_viewer_shortcut_side_effect(
    option: mujoco.MjvOption,
    keycode: int,
    *,
    geomgroup_before: Sequence[int],
    flags_before: Sequence[int],
) -> None:
    """Undo only the native Viewer display toggle reused by a demo key."""
    geom_group = GEOM_GROUP_SHORTCUTS.get(keycode)
    if geom_group is not None:
        option.geomgroup[geom_group] = geomgroup_before[geom_group]

    vis_flag = VIS_FLAG_SHORTCUTS.get(keycode)
    if vis_flag is not None:
        option.flags[vis_flag] = flags_before[vis_flag]


def contact_metadata(
    model: mujoco.MjModel,
) -> tuple[dict[int, str], dict[int, str], dict[str, float]]:
    battery_names_by_body_id = {
        _object_id(model, mujoco.mjtObj.mjOBJ_BODY, name): name
        for name in BATTERY_BODY_NAMES
    }
    jaw_sides_by_body_id = {
        _object_id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper"): "fixed",
        _object_id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "moving_jaw_so101_v1",
        ): "moving",
    }
    initial_z_by_name = {}
    for body_id, name in battery_names_by_body_id.items():
        joint_id = int(model.body_jntadr[body_id])
        qpos_address = int(model.jnt_qposadr[joint_id])
        initial_z_by_name[name] = float(model.qpos0[qpos_address + 2])
    return battery_names_by_body_id, jaw_sides_by_body_id, initial_z_by_name


def classify_contact_body_pairs(
    body_pairs: Iterable[tuple[int, int]],
    battery_names_by_body_id: dict[int, str],
    jaw_sides_by_body_id: dict[int, str],
) -> dict[str, frozenset[str]]:
    contacts: dict[str, set[str]] = {
        name: set() for name in battery_names_by_body_id.values()
    }
    for body_1, body_2 in body_pairs:
        if body_1 in battery_names_by_body_id and body_2 in jaw_sides_by_body_id:
            contacts[battery_names_by_body_id[body_1]].add(
                jaw_sides_by_body_id[body_2]
            )
        elif body_2 in battery_names_by_body_id and body_1 in jaw_sides_by_body_id:
            contacts[battery_names_by_body_id[body_2]].add(
                jaw_sides_by_body_id[body_1]
            )
    return {name: frozenset(sides) for name, sides in contacts.items()}


def observe_batteries(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    battery_names_by_body_id: dict[int, str],
    jaw_sides_by_body_id: dict[int, str],
    initial_z_by_name: dict[str, float],
) -> dict[str, tuple[frozenset[str], bool, bool]]:
    body_pairs = []
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        body_pairs.append(
            (
                int(model.geom_bodyid[contact.geom1]),
                int(model.geom_bodyid[contact.geom2]),
            )
        )
    contact_sides = classify_contact_body_pairs(
        body_pairs,
        battery_names_by_body_id,
        jaw_sides_by_body_id,
    )

    states = {}
    for body_id, name in battery_names_by_body_id.items():
        sides = contact_sides[name]
        lifted = float(data.xpos[body_id, 2]) >= initial_z_by_name[name] + LIFT_DELTA_M
        states[name] = (sides, lifted, lifted and sides == {"fixed", "moving"})
    return states


def feedback_messages(
    previous: dict[str, tuple[frozenset[str], bool, bool]],
    current: dict[str, tuple[frozenset[str], bool, bool]],
) -> list[str]:
    messages = []
    side_labels = {
        frozenset({"fixed"}): "固定夹爪",
        frozenset({"moving"}): "活动夹爪",
        frozenset({"fixed", "moving"}): "固定+活动夹爪（双侧）",
    }
    for name in BATTERY_BODY_NAMES:
        old_sides, old_lifted, old_success = previous[name]
        new_sides, new_lifted, new_success = current[name]
        if new_sides != old_sides:
            if new_sides:
                messages.append(f"[接触] {name}: {side_labels[new_sides]}")
            else:
                messages.append(f"[接触] {name}: 已离开夹爪")
        if new_lifted != old_lifted:
            if new_lifted:
                messages.append(f"[抬升] {name}: 已高于初始位置 10 mm")
            else:
                messages.append(f"[抬升] {name}: 已回到抬升阈值以下")
        if new_success and not old_success:
            messages.append(f"[成功] {name}: 双侧接触并抬升")
    return messages


def run_headless_check(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    steps: int = HEADLESS_CHECK_STEPS,
) -> None:
    reset_scene(model, data)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    if not all(math.isfinite(float(value)) for value in data.qpos):
        raise RuntimeError("headless check failed: qpos contains a non-finite value")
    if not all(math.isfinite(float(value)) for value in data.qvel):
        raise RuntimeError("headless check failed: qvel contains a non-finite value")
    print(
        "Headless check passed: "
        f"scene={SCENE_PATH.name}, nq={model.nq}, nv={model.nv}, "
        f"nu={model.nu}, steps={steps}"
    )


def print_controls(joint_step_deg: float, gripper_step_deg: float) -> None:
    print(
        """
SO101 T0 interactive grasp demo

  1  shoulder_pan (base)         4  wrist_flex
  2  shoulder_lift               5  wrist_roll
  3  elbow_flex                  6  gripper

  A  decrease selected target / close gripper
  D  increase selected target / open gripper
  Space      pause/resume
  Backspace  reset robot and batteries
  Close the Viewer window to exit

The Viewer right-side Control panel provides precise actuator sliders.
Mouse drag/scroll keeps the normal Free-camera rotate, pan, and zoom controls.
""".strip()
    )
    print(
        f"Keyboard steps: arm={joint_step_deg:g} deg, "
        f"gripper={gripper_step_deg:g} deg"
    )


def print_selected_servo(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    servo_id: int,
) -> None:
    actuator_name = SERVO_ID_TO_ACTUATOR[servo_id]
    actuator_id = _object_id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        actuator_name,
    )
    target = float(data.ctrl[actuator_id])
    print(
        f"[选择] 舵机 {servo_id}: {actuator_name}, "
        f"当前目标 {target:+.4f} rad ({math.degrees(target):+.1f}°)"
    )


def run_interactive(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    joint_step_deg: float,
    gripper_step_deg: float,
) -> None:
    reset_scene(model, data)
    key_events: queue.SimpleQueue[int] = queue.SimpleQueue()
    metadata = contact_metadata(model)
    previous_state = observe_batteries(model, data, *metadata)
    joint_step_rad = math.radians(joint_step_deg)
    gripper_step_rad = math.radians(gripper_step_deg)
    paused = False
    selected_servo_id = 1

    print_controls(joint_step_deg, gripper_step_deg)
    print(f"Loaded: {SCENE_PATH}")
    print_selected_servo(model, data, selected_servo_id)

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_events.put,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:
        geomgroup_before = tuple(int(value) for value in viewer.opt.geomgroup)
        flags_before = tuple(int(value) for value in viewer.opt.flags)
        while viewer.is_running():
            loop_start = time.monotonic()
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

                    servo_id = servo_id_from_keycode(keycode)
                    if servo_id is not None:
                        selected_servo_id = servo_id
                        print_selected_servo(model, data, selected_servo_id)
                    elif keycode in ADJUSTMENT_KEYS:
                        name, target = apply_selected_control(
                            model,
                            data,
                            selected_servo_id,
                            ADJUSTMENT_KEYS[keycode],
                            joint_step_rad=joint_step_rad,
                            gripper_step_rad=gripper_step_rad,
                        )
                        print(
                            f"[控制] 舵机 {selected_servo_id} {name}: "
                            f"{target:+.4f} rad ({math.degrees(target):+.1f}°)"
                        )
                    elif keycode == PAUSE_KEY:
                        paused = toggle_pause(paused, keycode)
                        print("[仿真] 已暂停" if paused else "[仿真] 已继续")
                    elif keycode == RESET_KEY:
                        reset_scene(model, data)
                        previous_state = observe_batteries(model, data, *metadata)
                        print("[复位] 机械臂、电池、速度和控制目标已恢复")

                geomgroup_before = tuple(
                    int(value) for value in viewer.opt.geomgroup
                )
                flags_before = tuple(int(value) for value in viewer.opt.flags)

                if not paused:
                    mujoco.mj_step(model, data)
                current_state = observe_batteries(model, data, *metadata)

            for message in feedback_messages(previous_state, current_state):
                print(message)
            previous_state = current_state
            viewer.sync()

            target_period = 0.01 if paused else float(model.opt.timestep)
            remaining = target_period - (time.monotonic() - loop_start)
            if remaining > 0:
                time.sleep(remaining)


def main() -> None:
    args = parse_args()
    model, data = load_t0_scene()
    if args.headless_check:
        run_headless_check(model, data)
        return
    run_interactive(
        model,
        data,
        joint_step_deg=args.joint_step_deg,
        gripper_step_deg=args.gripper_step_deg,
    )


if __name__ == "__main__":
    main()
