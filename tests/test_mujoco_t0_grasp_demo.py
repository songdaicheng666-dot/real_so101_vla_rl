from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "demo_mujoco_t0_grasp.py"
SPEC = importlib.util.spec_from_file_location("demo_mujoco_t0_grasp", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
demo = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = demo
SPEC.loader.exec_module(demo)


def _actuator_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def test_demo_loads_only_the_t0_scene() -> None:
    model, _ = demo.load_t0_scene()

    assert demo.SCENE_PATH.name == "scene_basic_t0.xml"
    assert (model.nq, model.nv, model.nu) == demo.EXPECTED_MODEL_SIZE
    assert demo.actuator_names(model) == demo.ACTUATOR_NAMES
    assert demo.SERVO_ID_TO_ACTUATOR == {
        1: "shoulder_pan",
        2: "shoulder_lift",
        3: "elbow_flex",
        4: "wrist_flex",
        5: "wrist_roll",
        6: "gripper",
    }


def test_initialize_controls_holds_current_joint_positions() -> None:
    model, data = demo.load_t0_scene()
    expected = (0.2, -0.2, 0.3, -0.3, 0.4, 0.5)
    for actuator_id, value in enumerate(expected):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        data.qpos[model.jnt_qposadr[joint_id]] = value

    demo.initialize_controls(model, data)

    assert tuple(data.ctrl) == pytest.approx(expected)


def test_number_keys_select_without_changing_controls() -> None:
    model, data = demo.load_t0_scene()
    demo.reset_scene(model, data)
    controls_before = tuple(data.ctrl)

    assert demo.servo_id_from_keycode(demo.glfw.KEY_1) == 1
    assert demo.servo_id_from_keycode(demo.glfw.KEY_4) == 4
    assert demo.servo_id_from_keycode(demo.glfw.KEY_6) == 6
    assert demo.servo_id_from_keycode(demo.glfw.KEY_7) is None
    assert tuple(data.ctrl) == controls_before


def test_ad_keys_adjust_selected_servo_and_clamp_target() -> None:
    model, data = demo.load_t0_scene()
    demo.reset_scene(model, data)
    joint_step = math.radians(2.0)
    gripper_step = math.radians(1.0)

    shoulder_name, shoulder_target = demo.apply_selected_control(
        model,
        data,
        servo_id=1,
        direction=demo.ADJUSTMENT_KEYS[demo.glfw.KEY_D],
        joint_step_rad=joint_step,
        gripper_step_rad=gripper_step,
    )
    shoulder_id = _actuator_id(model, "shoulder_pan")
    assert shoulder_name == "shoulder_pan"
    assert shoulder_target == pytest.approx(joint_step)
    assert data.ctrl[shoulder_id] == pytest.approx(joint_step)

    demo.apply_selected_control(
        model,
        data,
        servo_id=1,
        direction=demo.ADJUSTMENT_KEYS[demo.glfw.KEY_A],
        joint_step_rad=joint_step,
        gripper_step_rad=gripper_step,
    )
    assert data.ctrl[shoulder_id] == pytest.approx(0.0)

    shoulder_upper = model.actuator_ctrlrange[shoulder_id, 1]
    data.ctrl[shoulder_id] = shoulder_upper
    demo.apply_selected_control(
        model,
        data,
        servo_id=1,
        direction=+1,
        joint_step_rad=joint_step,
        gripper_step_rad=gripper_step,
    )
    assert data.ctrl[shoulder_id] == shoulder_upper

    gripper_id = _actuator_id(model, "gripper")
    gripper_name, gripper_target = demo.apply_selected_control(
        model,
        data,
        servo_id=6,
        direction=+1,
        joint_step_rad=joint_step,
        gripper_step_rad=gripper_step,
    )
    assert gripper_name == "gripper"
    assert gripper_target == pytest.approx(gripper_step)
    assert data.ctrl[gripper_id] == pytest.approx(gripper_step)


def test_reused_keys_restore_only_their_viewer_display_side_effects() -> None:
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    geomgroup_before = tuple(int(value) for value in option.geomgroup)
    flags_before = tuple(int(value) for value in option.flags)

    option.geomgroup[2] = 1 - option.geomgroup[2]
    unrelated_group = int(option.geomgroup[3])
    demo.restore_viewer_shortcut_side_effect(
        option,
        demo.glfw.KEY_2,
        geomgroup_before=geomgroup_before,
        flags_before=flags_before,
    )
    assert tuple(int(value) for value in option.geomgroup) == geomgroup_before
    assert int(option.geomgroup[3]) == unrelated_group

    auto_connect = int(mujoco.mjtVisFlag.mjVIS_AUTOCONNECT)
    static_body = int(mujoco.mjtVisFlag.mjVIS_STATIC)
    option.flags[auto_connect] = 1 - option.flags[auto_connect]
    static_before = int(option.flags[static_body])
    demo.restore_viewer_shortcut_side_effect(
        option,
        demo.glfw.KEY_A,
        geomgroup_before=geomgroup_before,
        flags_before=flags_before,
    )
    assert int(option.flags[auto_connect]) == flags_before[auto_connect]
    assert int(option.flags[static_body]) == static_before

    option.flags[static_body] = 1 - option.flags[static_body]
    auto_connect_before = int(option.flags[auto_connect])
    demo.restore_viewer_shortcut_side_effect(
        option,
        demo.glfw.KEY_D,
        geomgroup_before=geomgroup_before,
        flags_before=flags_before,
    )
    assert int(option.flags[static_body]) == flags_before[static_body]
    assert int(option.flags[auto_connect]) == auto_connect_before


def test_space_toggles_pause_and_other_keys_do_not() -> None:
    assert demo.toggle_pause(False, demo.glfw.KEY_SPACE) is True
    assert demo.toggle_pause(True, demo.glfw.KEY_SPACE) is False
    assert demo.toggle_pause(False, demo.glfw.KEY_A) is False
    assert demo.RESET_KEY == demo.glfw.KEY_BACKSPACE


def test_reset_restores_robot_batteries_velocities_and_controls() -> None:
    model, data = demo.load_t0_scene()
    data.qpos[:] = 0.123
    data.qvel[:] = 0.456
    data.ctrl[:] = 0.789

    demo.reset_scene(model, data)

    assert tuple(data.qpos) == pytest.approx(tuple(model.qpos0))
    assert tuple(data.qvel) == pytest.approx((0.0,) * model.nv)
    assert tuple(data.ctrl) == pytest.approx((0.0,) * model.nu)


def test_contact_classification_distinguishes_both_jaws() -> None:
    model, _ = demo.load_t0_scene()
    batteries, jaws, _ = demo.contact_metadata(model)
    body_ids = {name: body_id for body_id, name in batteries.items()}
    jaw_ids = {side: body_id for body_id, side in jaws.items()}

    contacts = demo.classify_contact_body_pairs(
        [
            (body_ids["battery_red"], jaw_ids["fixed"]),
            (jaw_ids["moving"], body_ids["battery_red"]),
            (body_ids["battery_blue"], jaw_ids["fixed"]),
        ],
        batteries,
        jaws,
    )

    assert contacts["battery_red"] == {"fixed", "moving"}
    assert contacts["battery_blue"] == {"fixed"}
    assert contacts["battery_yellow"] == set()
    assert contacts["battery_green"] == set()


def test_lift_and_success_feedback_are_state_based() -> None:
    empty = {
        name: (frozenset(), False, False) for name in demo.BATTERY_BODY_NAMES
    }
    current = dict(empty)
    current["battery_red"] = (frozenset({"fixed", "moving"}), True, True)

    messages = demo.feedback_messages(empty, current)

    assert messages == [
        "[接触] battery_red: 固定+活动夹爪（双侧）",
        "[抬升] battery_red: 已高于初始位置 10 mm",
        "[成功] battery_red: 双侧接触并抬升",
    ]


def test_headless_check_runs_finite_dynamics(capsys: pytest.CaptureFixture[str]) -> None:
    model, data = demo.load_t0_scene()

    demo.run_headless_check(model, data, steps=50)

    output = capsys.readouterr().out
    assert "Headless check passed" in output
    assert all(math.isfinite(float(value)) for value in data.qpos)
    assert all(math.isfinite(float(value)) for value in data.qvel)
