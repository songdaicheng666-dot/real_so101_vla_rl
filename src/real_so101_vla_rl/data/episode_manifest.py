"""Project-side metadata for LeRobot episodes."""

from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .schema import SCHEMA_VERSION, AtomicTask, BatteryColor, TargetSlot, TaskType


class FailureType(StrEnum):
    WRONG_TARGET = "wrong_target"
    GRASP_FAILED = "grasp_failed"
    DROPPED = "dropped"
    PLACEMENT_FAILED = "placement_failed"
    COLLISION = "collision"
    TIMEOUT = "timeout"
    HARDWARE_ERROR = "hardware_error"
    OPERATOR_ABORT = "operator_abort"


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    schema_version: int
    episode_index: int
    trial_id: str
    task_type: TaskType
    task: str
    target_color: BatteryColor
    target_slot: TargetSlot
    sequence_step: int
    layout_id: str
    success: bool
    failure_type: FailureType | None
    num_frames: int
    duration_s: float

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported episode schema_version={self.schema_version}")
        if self.episode_index < 0:
            raise ValueError("episode_index must be non-negative")
        if not self.trial_id.strip():
            raise ValueError("trial_id must not be empty")
        if not self.layout_id.strip():
            raise ValueError("layout_id must not be empty")
        if self.num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if self.duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if type(self.success) is not bool:
            raise TypeError("success must be a boolean")

        atomic_task = AtomicTask(
            self.task_type,
            self.target_color,
            self.target_slot,
            self.sequence_step,
        )
        if self.task != atomic_task.instruction:
            raise ValueError(
                f"task does not match structured fields: expected {atomic_task.instruction!r}, got {self.task!r}"
            )
        if self.success and self.failure_type is not None:
            raise ValueError("successful episodes must not have a failure_type")
        if not self.success and self.failure_type is None:
            raise ValueError("failed episodes must have a failure_type")

    @classmethod
    def from_task(
        cls,
        *,
        episode_index: int,
        trial_id: str,
        atomic_task: AtomicTask,
        layout_id: str,
        success: bool,
        failure_type: FailureType | None,
        num_frames: int,
        duration_s: float,
    ) -> EpisodeRecord:
        return cls(
            schema_version=SCHEMA_VERSION,
            episode_index=episode_index,
            trial_id=trial_id,
            task_type=atomic_task.task_type,
            task=atomic_task.instruction,
            target_color=atomic_task.target_color,
            target_slot=atomic_task.target_slot,
            sequence_step=atomic_task.sequence_step,
            layout_id=layout_id,
            success=success,
            failure_type=failure_type,
            num_frames=num_frames,
            duration_s=duration_s,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> EpisodeRecord:
        expected_fields = set(cls.__dataclass_fields__)
        actual_fields = set(raw)
        if actual_fields != expected_fields:
            raise ValueError(
                "Episode record fields do not match the schema; "
                f"missing={sorted(expected_fields - actual_fields)}, "
                f"extra={sorted(actual_fields - expected_fields)}"
            )
        return cls(
            schema_version=int(raw["schema_version"]),
            episode_index=int(raw["episode_index"]),
            trial_id=str(raw["trial_id"]),
            task_type=TaskType(raw["task_type"]),
            task=str(raw["task"]),
            target_color=BatteryColor(raw["target_color"]),
            target_slot=TargetSlot(raw["target_slot"]),
            sequence_step=int(raw["sequence_step"]),
            layout_id=str(raw["layout_id"]),
            success=raw["success"],
            failure_type=FailureType(raw["failure_type"]) if raw["failure_type"] is not None else None,
            num_frames=int(raw["num_frames"]),
            duration_s=float(raw["duration_s"]),
        )


def validate_episode_manifest(records: Iterable[EpisodeRecord]) -> tuple[EpisodeRecord, ...]:
    """Validate uniqueness and consistency across an episode manifest."""

    records = tuple(records)
    indices = [record.episode_index for record in records]
    if len(indices) != len(set(indices)):
        raise ValueError("episode_index values must be unique")

    trials: dict[str, list[EpisodeRecord]] = defaultdict(list)
    for record in records:
        trials[record.trial_id].append(record)

    for trial_id, trial_records in trials.items():
        layouts = {record.layout_id for record in trial_records}
        if len(layouts) != 1:
            raise ValueError(f"trial_id={trial_id!r} spans multiple layouts: {sorted(layouts)}")
        steps = [record.sequence_step for record in trial_records]
        if len(steps) != len(set(steps)):
            raise ValueError(f"trial_id={trial_id!r} contains duplicate sequence steps")
        if any(record.task_type is TaskType.SINGLE_T0 for record in trial_records) and len(trial_records) != 1:
            raise ValueError(f"single_t0 trial_id={trial_id!r} must contain exactly one episode")
    return records


def load_episode_manifest(path: str | Path) -> tuple[EpisodeRecord, ...]:
    path = Path(path)
    records = []
    with path.open(encoding="utf-8") as manifest_file:
        for line_number, line in enumerate(manifest_file, start=1):
            if not line.strip():
                continue
            try:
                records.append(EpisodeRecord.from_dict(json.loads(line)))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid episode manifest line {line_number}: {exc}") from exc
    return validate_episode_manifest(records)


def write_episode_manifest(path: str | Path, records: Iterable[EpisodeRecord]) -> None:
    """Atomically replace an episode JSONL manifest."""

    path = Path(path)
    records = validate_episode_manifest(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as manifest_file:
            for record in sorted(records, key=lambda item: item.episode_index):
                json.dump(record.to_dict(), manifest_file, ensure_ascii=False, separators=(",", ":"))
                manifest_file.write("\n")
            manifest_file.flush()
            os.fsync(manifest_file.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def append_episode_record(path: str | Path, record: EpisodeRecord) -> None:
    path = Path(path)
    records = load_episode_manifest(path) if path.exists() else ()
    write_episode_manifest(path, (*records, record))
