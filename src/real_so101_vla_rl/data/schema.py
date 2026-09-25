"""Canonical SO-101 task and frame schemas used by SFT data."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real
from typing import Any

SCHEMA_VERSION = 2
ACTION_DIM = 6
ACTION_CHUNK_SIZE = 8
DEFAULT_FPS = 30
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
OVERVIEW_IMAGE_KEY = "observation.images.overview"
WRIST_IMAGE_KEY = "observation.images.wrist"
OVERVIEW_DEPTH_KEY = "observation.images.overview_depth"
STATE_KEY = "observation.state"
ACTION_KEY = "action"
TASK_KEY = "task"
ACTION_SOURCE = "robot_send_action_return"

RGB_IMAGE_KEYS = (OVERVIEW_IMAGE_KEY, WRIST_IMAGE_KEY)
DEPTH_IMAGE_KEYS = (OVERVIEW_DEPTH_KEY,)
SENSOR_TIMESTAMP_KEYS = (
    "observation.timestamps.overview_ns",
    "observation.timestamps.wrist_ns",
    "observation.timestamps.overview_depth_ns",
    "observation.timestamps.state_ns",
)
SENSOR_VALID_KEYS = (
    "observation.valid.overview",
    "observation.valid.wrist",
    "observation.valid.overview_depth",
    "observation.valid.state",
)
MODEL_INPUT_KEYS = (*RGB_IMAGE_KEYS, STATE_KEY, TASK_KEY)
TRAINING_TARGET_KEYS = (ACTION_KEY,)
AUXILIARY_OBSERVATION_KEYS = (
    *DEPTH_IMAGE_KEYS,
    *SENSOR_TIMESTAMP_KEYS,
    *SENSOR_VALID_KEYS,
)
RECORDING_FRAME_KEYS = (
    *RGB_IMAGE_KEYS,
    *DEPTH_IMAGE_KEYS,
    STATE_KEY,
    ACTION_KEY,
    TASK_KEY,
    *SENSOR_TIMESTAMP_KEYS,
    *SENSOR_VALID_KEYS,
)

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
    r"^Pick up the (red|blue|yellow|green) cube and place it in (T0|P1|P2|P3)\.$"
)


class TaskType(StrEnum):
    """Atomic task families used by the competition scheduler."""

    SINGLE_T0 = "single_t0"
    SEQUENCE_STEP = "sequence_step"


class CubeColor(StrEnum):
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
    """One colored-cube transfer with a deterministic OpenVLA instruction."""

    task_type: TaskType
    target_color: CubeColor
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
            f"Pick up the {self.target_color.value} cube "
            f"and place it in {self.target_slot.value}."
        )

    @classmethod
    def from_instruction(cls, instruction: str) -> AtomicTask:
        match = _INSTRUCTION_PATTERN.fullmatch(instruction)
        if match is None:
            raise ValueError(f"Task is not a canonical SO-101 instruction: {instruction!r}")

        color = CubeColor(match.group(1))
        slot = TargetSlot(match.group(2))
        task_type = TaskType.SINGLE_T0 if slot is TargetSlot.T0 else TaskType.SEQUENCE_STEP
        return cls(task_type, color, slot, _SLOT_TO_STEP[slot])


@dataclass(frozen=True, slots=True)
class SO101DataSpec:
    """Canonical recording and model-facing fields for SO-101 schema v2."""

    schema_version: int = SCHEMA_VERSION
    rgb_image_keys: tuple[str, ...] = RGB_IMAGE_KEYS
    depth_image_keys: tuple[str, ...] = DEPTH_IMAGE_KEYS
    sensor_timestamp_keys: tuple[str, ...] = SENSOR_TIMESTAMP_KEYS
    sensor_valid_keys: tuple[str, ...] = SENSOR_VALID_KEYS
    recording_frame_keys: tuple[str, ...] = RECORDING_FRAME_KEYS
    model_input_keys: tuple[str, ...] = MODEL_INPUT_KEYS
    training_target_keys: tuple[str, ...] = TRAINING_TARGET_KEYS
    auxiliary_observation_keys: tuple[str, ...] = AUXILIARY_OBSERVATION_KEYS
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


def _validate_array(
    value: Any,
    *,
    field_name: str,
    shape: tuple[int, ...],
    dtype: str,
) -> None:
    actual_shape = getattr(value, "shape", None)
    if actual_shape is None or tuple(actual_shape) != shape:
        raise ValueError(f"{field_name} must have shape {shape}, got {actual_shape}")
    actual_dtype = getattr(value, "dtype", None)
    if actual_dtype is None or str(actual_dtype) != dtype:
        raise TypeError(f"{field_name} must use {dtype}, got {actual_dtype}")


def validate_frame(frame: Mapping[str, Any]) -> None:
    """Validate one caller-supplied schema-v2 recording frame.

    LeRobot's timestamp and integer index fields are intentionally absent: its
    writer creates them while saving an episode.
    """

    expected = set(RECORDING_FRAME_KEYS)
    actual = set(frame)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"Frame fields do not match the SO-101 schema; missing={missing}, extra={extra}")

    for key in RGB_IMAGE_KEYS:
        _validate_array(
            frame[key],
            field_name=key,
            shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
            dtype="uint8",
        )
    _validate_array(
        frame[OVERVIEW_DEPTH_KEY],
        field_name=OVERVIEW_DEPTH_KEY,
        shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1),
        dtype="uint16",
    )

    for key in (STATE_KEY, ACTION_KEY):
        _validate_array(
            frame[key],
            field_name=key,
            shape=(ACTION_DIM,),
            dtype="float32",
        )
        validate_joint_vector(frame[key], field_name=key)

    for key in SENSOR_TIMESTAMP_KEYS:
        value = frame[key]
        _validate_array(value, field_name=key, shape=(1,), dtype="int64")
        if int(value[0]) < 0:
            raise ValueError(f"{key} must be a non-negative host monotonic timestamp")

    for key in SENSOR_VALID_KEYS:
        value = frame[key]
        _validate_array(value, field_name=key, shape=(1,), dtype="bool")
        if not bool(value[0]):
            raise ValueError(f"{key} must be true before a frame can be persisted")

    task = frame[TASK_KEY]
    if not isinstance(task, str):
        raise TypeError(f"{TASK_KEY} must be a string")
    AtomicTask.from_instruction(task)


def validate_frame_sequence(frames: Sequence[Mapping[str, Any]]) -> None:
    """Validate frames and require non-decreasing per-sensor host timestamps."""

    previous: dict[str, int] = {}
    for frame_index, frame in enumerate(frames):
        validate_frame(frame)
        for key in SENSOR_TIMESTAMP_KEYS:
            timestamp = int(frame[key][0])
            if key in previous and timestamp < previous[key]:
                raise ValueError(
                    f"{key} regressed at frame {frame_index}: "
                    f"previous={previous[key]}, current={timestamp}"
                )
            previous[key] = timestamp


def project_model_inputs(frame: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a recording frame and return only the ordered policy inputs."""

    validate_frame(frame)
    return {key: frame[key] for key in MODEL_INPUT_KEYS}
