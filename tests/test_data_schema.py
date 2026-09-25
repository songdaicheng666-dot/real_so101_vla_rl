from __future__ import annotations

import numpy as np
import pytest

from real_so101_vla_rl.data import (
    MODEL_INPUT_KEYS,
    OVERVIEW_DEPTH_KEY,
    OVERVIEW_IMAGE_KEY,
    SCHEMA_VERSION,
    SENSOR_TIMESTAMP_KEYS,
    SENSOR_VALID_KEYS,
    WRIST_IMAGE_KEY,
    AtomicTask,
    CubeColor,
    TargetSlot,
    TaskType,
    action_delta_timestamps,
    build_action_chunk,
    ordered_joint_vector,
    project_model_inputs,
    validate_frame,
    validate_frame_sequence,
)


def _valid_frame(*, timestamp_ns: int = 1_000_000_000) -> dict:
    return {
        OVERVIEW_IMAGE_KEY: np.zeros((480, 640, 3), dtype=np.uint8),
        WRIST_IMAGE_KEY: np.ones((480, 640, 3), dtype=np.uint8),
        OVERVIEW_DEPTH_KEY: np.zeros((480, 640, 1), dtype=np.uint16),
        "observation.state": np.arange(6, dtype=np.float32),
        "action": np.arange(6, dtype=np.float32),
        "task": "Pick up the red cube and place it in T0.",
        **{
            key: np.asarray([timestamp_ns + index], dtype=np.int64)
            for index, key in enumerate(SENSOR_TIMESTAMP_KEYS)
        },
        **{
            key: np.asarray([True], dtype=np.bool_)
            for key in SENSOR_VALID_KEYS
        },
    }


def test_atomic_task_generates_and_parses_canonical_instruction() -> None:
    task = AtomicTask(TaskType.SEQUENCE_STEP, CubeColor.BLUE, TargetSlot.P2, 2)

    assert task.instruction == "Pick up the blue cube and place it in P2."
    assert AtomicTask.from_instruction(task.instruction) == task
    with pytest.raises(ValueError, match="canonical SO-101 instruction"):
        AtomicTask.from_instruction("Pick up the blue battery and place it in P2.")


def test_atomic_task_rejects_slot_step_mismatch() -> None:
    with pytest.raises(ValueError, match="requires sequence_step=2"):
        AtomicTask(TaskType.SEQUENCE_STEP, CubeColor.BLUE, TargetSlot.P2, 1)


def test_joint_mapping_uses_the_canonical_order() -> None:
    values = {
        "gripper.pos": 60,
        "wrist_roll.pos": 50,
        "wrist_flex.pos": 40,
        "elbow_flex.pos": 30,
        "shoulder_lift.pos": 20,
        "shoulder_pan.pos": 10,
    }

    assert ordered_joint_vector(values) == (10, 20, 30, 40, 50, 60)


def test_frame_schema_v2_accepts_only_the_recording_fields() -> None:
    frame = _valid_frame()

    validate_frame(frame)
    assert SCHEMA_VERSION == 2

    frame["task_index"] = 0
    with pytest.raises(ValueError, match=r"extra=\['task_index'\]"):
        validate_frame(frame)


def test_frame_schema_rejects_v1_front_camera_frame() -> None:
    frame = _valid_frame()
    frame["observation.images.front"] = frame.pop(OVERVIEW_IMAGE_KEY)

    with pytest.raises(ValueError, match="observation.images.overview"):
        validate_frame(frame)


def test_frame_schema_rejects_wrong_visual_shape_and_dtype() -> None:
    frame = _valid_frame()
    frame[WRIST_IMAGE_KEY] = np.zeros((480, 640, 1), dtype=np.uint8)

    with pytest.raises(ValueError, match=r"wrist.*\(480, 640, 3\)"):
        validate_frame(frame)

    frame = _valid_frame()
    frame[OVERVIEW_DEPTH_KEY] = np.zeros((480, 640, 1), dtype=np.float32)
    with pytest.raises(TypeError, match=r"overview_depth.*uint16"):
        validate_frame(frame)


def test_frame_schema_rejects_wrong_action_shape_and_noncanonical_task() -> None:
    frame = _valid_frame()
    frame["action"] = np.arange(3, dtype=np.float32)

    with pytest.raises(ValueError, match=r"shape \(6,\)"):
        validate_frame(frame)

    frame = _valid_frame()
    frame["task"] = "把红色方块放到 T0"
    with pytest.raises(ValueError, match="canonical SO-101 instruction"):
        validate_frame(frame)


def test_frame_schema_rejects_bad_timestamp_and_invalid_sensor() -> None:
    frame = _valid_frame()
    frame[SENSOR_TIMESTAMP_KEYS[0]] = np.asarray([-1], dtype=np.int64)
    with pytest.raises(ValueError, match="non-negative"):
        validate_frame(frame)

    frame = _valid_frame()
    frame[SENSOR_TIMESTAMP_KEYS[0]] = np.asarray([1], dtype=np.float64)
    with pytest.raises(TypeError, match="int64"):
        validate_frame(frame)

    frame = _valid_frame()
    frame[SENSOR_VALID_KEYS[2]] = np.asarray([False], dtype=np.bool_)
    with pytest.raises(ValueError, match="must be true"):
        validate_frame(frame)


def test_frame_sequence_requires_monotonic_sensor_timestamps() -> None:
    validate_frame_sequence((_valid_frame(timestamp_ns=10), _valid_frame(timestamp_ns=20)))

    regressed = _valid_frame(timestamp_ns=20)
    regressed[SENSOR_TIMESTAMP_KEYS[1]] = np.asarray([9], dtype=np.int64)
    with pytest.raises(ValueError, match="regressed"):
        validate_frame_sequence((_valid_frame(timestamp_ns=10), regressed))


def test_project_model_inputs_excludes_auxiliary_fields_and_target() -> None:
    frame = _valid_frame()
    model_inputs = project_model_inputs(frame)

    assert tuple(model_inputs) == MODEL_INPUT_KEYS
    assert model_inputs[OVERVIEW_IMAGE_KEY] is frame[OVERVIEW_IMAGE_KEY]
    assert model_inputs[WRIST_IMAGE_KEY] is frame[WRIST_IMAGE_KEY]
    assert OVERVIEW_DEPTH_KEY not in model_inputs
    assert "action" not in model_inputs


def test_action_chunk_repeats_episode_tail_and_marks_padding() -> None:
    actions = [
        [0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2],
    ]

    chunk, is_pad = build_action_chunk(actions, 1, chunk_size=4)

    assert chunk == (
        (1, 1, 1, 1, 1, 1),
        (2, 2, 2, 2, 2, 2),
        (2, 2, 2, 2, 2, 2),
        (2, 2, 2, 2, 2, 2),
    )
    assert is_pad == (False, False, True, True)
    assert action_delta_timestamps(fps=30, chunk_size=3) == {
        "action": [0.0, 1 / 30, 2 / 30]
    }
