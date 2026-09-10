from __future__ import annotations

import pytest

from real_so101_vla_rl.data import (
    AtomicTask,
    BatteryColor,
    TargetSlot,
    TaskType,
    action_delta_timestamps,
    build_action_chunk,
    ordered_joint_vector,
    validate_frame,
)


class FakeRGBImage:
    shape = (480, 640, 3)
    dtype = "uint8"


def test_atomic_task_generates_and_parses_canonical_instruction() -> None:
    task = AtomicTask(TaskType.SEQUENCE_STEP, BatteryColor.BLUE, TargetSlot.P2, 2)

    assert task.instruction == "Pick up the blue battery and place it in P2."
    assert AtomicTask.from_instruction(task.instruction) == task


def test_atomic_task_rejects_slot_step_mismatch() -> None:
    with pytest.raises(ValueError, match="requires sequence_step=2"):
        AtomicTask(TaskType.SEQUENCE_STEP, BatteryColor.BLUE, TargetSlot.P2, 1)


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


def test_frame_schema_accepts_only_the_four_caller_supplied_fields() -> None:
    frame = {
        "observation.images.front": FakeRGBImage(),
        "observation.state": [0, 1, 2, 3, 4, 50],
        "action": [1, 2, 3, 4, 5, 60],
        "task": "Pick up the red battery and place it in T0.",
    }

    validate_frame(frame)

    frame["task_index"] = 0
    with pytest.raises(ValueError, match=r"extra=\['task_index'\]"):
        validate_frame(frame)


def test_frame_schema_rejects_wrong_action_shape_and_noncanonical_task() -> None:
    frame = {
        "observation.images.front": FakeRGBImage(),
        "observation.state": [0, 1, 2, 3, 4, 50],
        "action": [1, 2, 3],
        "task": "把红色电池放到 T0",
    }

    with pytest.raises(ValueError, match="must contain 6"):
        validate_frame(frame)

    frame["action"] = [1, 2, 3, 4, 5, 60]
    with pytest.raises(ValueError, match="canonical SO-101 instruction"):
        validate_frame(frame)


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
