"""Canonical SO-101 task and frame schemas used by SFT data."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real
from typing import Any

SCHEMA_VERSION = 1
ACTION_DIM = 6
ACTION_CHUNK_SIZE = 8
DEFAULT_FPS = 30
IMAGE_KEY = "observation.images.front"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
TASK_KEY = "task"
ACTION_SOURCE = "robot_send_action_return"

JOINT_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
JOINT_UNITS = (
    "degree",
    "degree",
    "degree",
    "degree",
    "degree",
    "lerobot_range_0_100",
)

_INSTRUCTION_PATTERN = re.compile(
    r"^Pick up the (red|blue|yellow|green) battery and place it in (T0|P1|P2|P3)\.$"
)


class TaskType(StrEnum):
    """Atomic task families used by the competition scheduler."""

    SINGLE_T0 = "single_t0"
    SEQUENCE_STEP = "sequence_step"


class BatteryColor(StrEnum):
    RED = "red"
    BLUE = "blue"
    YELLOW = "yellow"
    GREEN = "green"


class TargetSlot(StrEnum):
    T0 = "T0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


_SLOT_TO_STEP = {
    TargetSlot.T0: 0,
    TargetSlot.P1: 1,
    TargetSlot.P2: 2,
    TargetSlot.P3: 3,
}


@dataclass(frozen=True, slots=True)
class AtomicTask:
    """One battery transfer with a deterministic OpenVLA instruction."""

    task_type: TaskType
    target_color: BatteryColor
    target_slot: TargetSlot
    sequence_step: int

    def __post_init__(self) -> None:
        expected_step = _SLOT_TO_STEP[self.target_slot]
        if self.sequence_step != expected_step:
            raise ValueError(
                f"target_slot={self.target_slot.value} requires sequence_step={expected_step}, "
                f"got {self.sequence_step}"
            )

        if self.task_type is TaskType.SINGLE_T0 and self.target_slot is not TargetSlot.T0:
            raise ValueError("single_t0 tasks must target T0")
        if self.task_type is TaskType.SEQUENCE_STEP and self.target_slot is TargetSlot.T0:
            raise ValueError("sequence_step tasks must target P1, P2, or P3")

    @property
    def instruction(self) -> str:
        return (
            f"Pick up the {self.target_color.value} battery "
            f"and place it in {self.target_slot.value}."
        )

    @classmethod
    def from_instruction(cls, instruction: str) -> AtomicTask:
        match = _INSTRUCTION_PATTERN.fullmatch(instruction)
        if match is None:
            raise ValueError(f"Task is not a canonical SO-101 instruction: {instruction!r}")

        color = BatteryColor(match.group(1))
        slot = TargetSlot(match.group(2))
        task_type = TaskType.SINGLE_T0 if slot is TargetSlot.T0 else TaskType.SEQUENCE_STEP
        return cls(task_type, color, slot, _SLOT_TO_STEP[slot])


@dataclass(frozen=True, slots=True)
class SO101DataSpec:
    """Model-facing field names and dimensions for the first SO-101 dataset."""

    schema_version: int = SCHEMA_VERSION
    image_key: str = IMAGE_KEY
    state_key: str = STATE_KEY
    action_key: str = ACTION_KEY
    task_key: str = TASK_KEY
    joint_names: tuple[str, ...] = JOINT_NAMES
    joint_units: tuple[str, ...] = JOINT_UNITS
    action_dim: int = ACTION_DIM
    action_chunk_size: int = ACTION_CHUNK_SIZE
    fps: int = DEFAULT_FPS
    action_representation: str = "absolute_joint_target"
    action_source: str = ACTION_SOURCE


def ordered_joint_vector(values: Mapping[str, Real]) -> tuple[float, ...]:
    """Convert a named LeRobot joint mapping to the canonical six-element order."""

    missing = [name for name in JOINT_NAMES if name not in values]
    if missing:
        raise ValueError(f"Missing SO-101 joints: {missing}")
    return validate_joint_vector([values[name] for name in JOINT_NAMES], field_name="joint mapping")


def validate_joint_vector(values: Any, *, field_name: str) -> tuple[float, ...]:
    """Validate and normalize a state or action vector without importing NumPy."""

    shape = getattr(values, "shape", None)
    if shape is not None and tuple(shape) != (ACTION_DIM,):
        raise ValueError(f"{field_name} must have shape ({ACTION_DIM},), got {tuple(shape)}")

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        if hasattr(values, "tolist"):
            values = values.tolist()
        else:
            raise TypeError(f"{field_name} must be a six-element numeric sequence")

    if len(values) != ACTION_DIM:
        raise ValueError(f"{field_name} must contain {ACTION_DIM} values, got {len(values)}")

    normalized = []
    for index, value in enumerate(values):
        if not isinstance(value, Real):
            raise TypeError(f"{field_name}[{index}] must be numeric, got {type(value).__name__}")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{field_name}[{index}] must be finite, got {value}")
        normalized.append(value)
    return tuple(normalized)


def validate_frame(frame: Mapping[str, Any]) -> None:
    """Validate the four fields supplied by the project recording loop.

    LeRobot's timestamp and integer index fields are intentionally absent: its
    writer creates them while saving an episode.
    """

    expected = {IMAGE_KEY, STATE_KEY, ACTION_KEY, TASK_KEY}
    actual = set(frame)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"Frame fields do not match the SO-101 schema; missing={missing}, extra={extra}")

    image = frame[IMAGE_KEY]
    image_shape = getattr(image, "shape", None)
    if image_shape is None or len(image_shape) != 3 or image_shape[-1] != 3:
        raise ValueError(f"{IMAGE_KEY} must be an HxWx3 RGB array, got shape={image_shape}")
    if image_shape[0] <= 0 or image_shape[1] <= 0:
        raise ValueError(f"{IMAGE_KEY} must have positive height and width")
    image_dtype = getattr(image, "dtype", None)
    if image_dtype is not None and str(image_dtype) != "uint8":
        raise TypeError(f"{IMAGE_KEY} must use uint8 pixels, got {image_dtype}")

    validate_joint_vector(frame[STATE_KEY], field_name=STATE_KEY)
    validate_joint_vector(frame[ACTION_KEY], field_name=ACTION_KEY)

    task = frame[TASK_KEY]
    if not isinstance(task, str):
        raise TypeError(f"{TASK_KEY} must be a string")
    AtomicTask.from_instruction(task)
