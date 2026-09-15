# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# This file derives its episode orchestration from LeRobot's lerobot_record.py
# at commit 4aaff99be4a1d81568c08c8f0296b41b40c99ec4. It has been modified to
# enforce this project's schema-v2 synchronized observation and final-action contract.
"""Schema-v2 SO-101 recording on top of LeRobot's hardware and dataset APIs."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

import numpy as np

from real_so101_vla_rl.data.episode_manifest import EpisodeRecord, append_episode_record
from real_so101_vla_rl.data.schema import (
    ACTION_KEY,
    JOINT_NAMES,
    AtomicTask,
    ordered_joint_vector,
    validate_frame,
)
from real_so101_vla_rl.hardware.capture_types import (
    SynchronizationResult,
    SynchronizedObservation,
)

from .config import SO101RecordingConfig

logger = logging.getLogger(__name__)


class DatasetWriter(Protocol):
    num_episodes: int

    def add_frame(self, frame: dict[str, Any]) -> None: ...

    def save_episode(self) -> None: ...

    def clear_episode_buffer(self, delete_images: bool = True) -> None: ...

    def has_pending_frames(self) -> bool: ...


class Synchronizer(Protocol):
    @property
    def is_broken(self) -> bool: ...

    def capture(self) -> SynchronizationResult: ...


class CaptureAbort(RuntimeError):
    """Raised before action dispatch when one synchronized observation is invalid."""

    def __init__(self, result: SynchronizationResult):
        super().__init__(result.error or "synchronized capture failed")
        self.result = result


@dataclass(frozen=True, slots=True)
class EpisodeCaptureSummary:
    accepted: bool
    aborted: bool
    num_frames: int
    duration_s: float
    reason: str | None = None


def _joint_mapping(state: np.ndarray) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in zip(JOINT_NAMES, state, strict=True)
    }


def build_recording_frame(
    observation: SynchronizedObservation,
    sent_action: Mapping[str, float],
    task: str,
) -> dict[str, Any]:
    """Attach the actual follower action and validate a complete recording frame."""

    AtomicTask.from_instruction(task)
    frame = {
        **observation.to_schema_fields(),
        ACTION_KEY: np.asarray(ordered_joint_vector(sent_action), dtype=np.float32),
        "task": task,
    }
    validate_frame(frame)
    return frame


class SchemaV2EpisodeRecorder:
    """One-tick recorder kept independent from UI and physical device factories."""

    def __init__(
        self,
        *,
        robot: Any,
        teleop: Any,
        synchronizer: Synchronizer,
        dataset: DatasetWriter,
        task: str,
        teleop_action_processor: Callable[[tuple[Mapping[str, float], Mapping[str, float]]], Mapping[str, float]],
        robot_action_processor: Callable[[tuple[Mapping[str, float], Mapping[str, float]]], Mapping[str, float]],
    ) -> None:
        AtomicTask.from_instruction(task)
        self.robot = robot
        self.teleop = teleop
        self.synchronizer = synchronizer
        self.dataset = dataset
        self.task = task
        self.teleop_action_processor = teleop_action_processor
        self.robot_action_processor = robot_action_processor

    def record_tick(self) -> dict[str, float]:
        synchronized = self.synchronizer.capture()
        if not synchronized.valid:
            raise CaptureAbort(synchronized)
        observation = synchronized.observation
        assert observation is not None
        robot_observation = _joint_mapping(observation.state)
        raw_action = self.teleop.get_action()
        teleop_action = self.teleop_action_processor((raw_action, robot_observation))
        action_to_send = self.robot_action_processor((teleop_action, robot_observation))
        sent_action = self.robot.send_action(action_to_send)
        frame = build_recording_frame(observation, sent_action, self.task)
        self.dataset.add_frame(frame)
        return dict(sent_action)


class RecordingControls:
    """Thread-safe phase-aware controls shared by X11 and terminal listeners."""

    def __init__(self) -> None:
        self._lock = Lock()
        self.phase = "idle"
        self.end_requested = False
        self.accept_requested = False
        self.discard_requested = False
        self.stop_requested = False

    def enter_phase(self, phase: str) -> None:
        if phase not in {"recording", "confirm", "reset"}:
            raise ValueError(f"Unknown recording phase: {phase}")
        with self._lock:
            self.phase = phase
            self.end_requested = False
            self.accept_requested = False
            self.discard_requested = False

    def dispatch(self, name: str) -> None:
        key = name.lower()
        with self._lock:
            if key in {"esc", "q"}:
                self.stop_requested = True
                self.discard_requested = True
                self.end_requested = True
            elif self.phase == "confirm" and key in {"enter", "s"}:
                self.accept_requested = True
            elif key in {"left", "r"}:
                self.discard_requested = True
                self.end_requested = True
            elif key in {"right", "n"} and self.phase in {"recording", "reset"}:
                self.end_requested = True

    def snapshot(self) -> tuple[bool, bool, bool, bool]:
        with self._lock:
            return (
                self.end_requested,
                self.accept_requested,
                self.discard_requested,
                self.stop_requested,
            )


def append_capture_failure(
    path: str | Path,
    *,
    episode_index: int,
    num_frames: int,
    result: SynchronizationResult,
) -> None:
    """Append diagnostics for a discarded capture without creating an episode record."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "episode_index_candidate": episode_index,
        "num_frames_discarded": num_frames,
        "reason": result.error,
        "timestamps_ns": dict(result.diagnostics.timestamps_ns),
        "span_ns": result.diagnostics.span_ns,
        "errors": dict(result.diagnostics.errors),
    }
    with path.open("a", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, separators=(",", ":"))
        output.write("\n")


def collect_episode(
    recorder: SchemaV2EpisodeRecorder,
    controls: RecordingControls,
    *,
    duration_s: float,
    fps: int,
    failure_log_path: Path,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> EpisodeCaptureSummary:
    controls.enter_phase("recording")
    start = clock()
    num_frames = 0
    period_s = 1.0 / fps
    while clock() - start < duration_s:
        end, _, discard, stop = controls.snapshot()
        if end or discard or stop:
            break
        tick_start = clock()
        try:
            recorder.record_tick()
        except CaptureAbort as exc:
            recorder.dataset.clear_episode_buffer()
            append_capture_failure(
                failure_log_path,
                episode_index=recorder.dataset.num_episodes,
                num_frames=num_frames,
                result=exc.result,
            )
            return EpisodeCaptureSummary(
                False, True, num_frames, clock() - start, str(exc)
            )
        num_frames += 1
        remaining = period_s - (clock() - tick_start)
        if remaining > 0:
            sleep(remaining)
    if controls.snapshot()[2] or controls.snapshot()[3]:
        recorder.dataset.clear_episode_buffer()
        return EpisodeCaptureSummary(
            False, False, num_frames, clock() - start, "operator discarded"
        )
    if num_frames == 0:
        recorder.dataset.clear_episode_buffer()
        return EpisodeCaptureSummary(
            False, False, 0, clock() - start, "no frames captured"
        )
    return EpisodeCaptureSummary(False, False, num_frames, clock() - start)


def _wait_for_confirmation(
    controls: RecordingControls,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    controls.enter_phase("confirm")
    logger.info("Episode complete: Enter/s=save, Left/r=discard, Esc/q=stop")
    while True:
        _, accept, discard, stop = controls.snapshot()
        if stop or discard:
            return False
        if accept:
            return True
        sleep(0.05)


def _run_manual_reset(
    *,
    robot: Any,
    teleop: Any,
    state_adapter: Any,
    controls: RecordingControls,
    teleop_action_processor: Callable[..., Mapping[str, float]],
    robot_action_processor: Callable[..., Mapping[str, float]],
    duration_s: float,
    fps: int,
) -> None:
    controls.enter_phase("reset")
    start = time.perf_counter()
    period_s = 1.0 / fps
    while time.perf_counter() - start < duration_s:
        end, _, _, stop = controls.snapshot()
        if end or stop:
            return
        tick_start = time.perf_counter()
        state = state_adapter.read(timeout_ms=200)
        if not state.valid:
            if state_adapter.is_broken:
                raise RuntimeError(f"SO-101 state adapter failed repeatedly: {state.error}")
            time.sleep(period_s)
            continue
        observation = _joint_mapping(state.value)
        raw_action = teleop.get_action()
        action = teleop_action_processor((raw_action, observation))
        action = robot_action_processor((action, observation))
        robot.send_action(action)
        remaining = period_s - (time.perf_counter() - tick_start)
        if remaining > 0:
            time.sleep(remaining)


def run_recording_session(
    config: SO101RecordingConfig,
    *,
    robot: Any,
    teleop: Any,
    synchronizer: Any,
    state_adapter: Any,
    dataset: Any,
    controls: RecordingControls,
    teleop_action_processor: Callable[..., Mapping[str, float]],
    robot_action_processor: Callable[..., Mapping[str, float]],
) -> tuple[EpisodeCaptureSummary, ...]:
    """Record only explicitly accepted episodes into an existing writer."""

    summaries: list[EpisodeCaptureSummary] = []
    metadata_root = Path(config.dataset.root) / "project_meta"
    failure_path = metadata_root / "capture_failures.jsonl"
    manifest_path = metadata_root / "episodes.jsonl"
    recorder = SchemaV2EpisodeRecorder(
        robot=robot,
        teleop=teleop,
        synchronizer=synchronizer,
        dataset=dataset,
        task=config.dataset.task,
        teleop_action_processor=teleop_action_processor,
        robot_action_processor=robot_action_processor,
    )
    while dataset.num_episodes < config.dataset.num_episodes:
        if controls.snapshot()[3]:
            break
        summary = collect_episode(
            recorder,
            controls,
            duration_s=config.dataset.episode_time_s,
            fps=config.dataset.fps,
            failure_log_path=failure_path,
        )
        if summary.aborted:
            summaries.append(summary)
            if synchronizer.is_broken:
                raise RuntimeError("A capture adapter failed repeatedly; stopping recording")
            _run_manual_reset(
                robot=robot,
                teleop=teleop,
                state_adapter=state_adapter,
                controls=controls,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                duration_s=config.dataset.reset_time_s,
                fps=config.dataset.fps,
            )
            continue
        if summary.reason is not None or not _wait_for_confirmation(controls):
            dataset.clear_episode_buffer()
            summaries.append(summary)
            if controls.snapshot()[3]:
                break
        else:
            episode_index = dataset.num_episodes
            dataset.save_episode()
            atomic_task = AtomicTask.from_instruction(config.dataset.task)
            record = EpisodeRecord.from_task(
                episode_index=episode_index,
                trial_id=f"{config.dataset.trial_id_prefix}-{episode_index:06d}",
                atomic_task=atomic_task,
                layout_id=config.dataset.layout_id,
                success=True,
                failure_type=None,
                num_frames=summary.num_frames,
                duration_s=summary.duration_s,
            )
            append_episode_record(manifest_path, record)
            accepted = EpisodeCaptureSummary(
                True, False, summary.num_frames, summary.duration_s
            )
            summaries.append(accepted)
        if controls.snapshot()[3]:
            break
        _run_manual_reset(
            robot=robot,
            teleop=teleop,
            state_adapter=state_adapter,
            controls=controls,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            duration_s=config.dataset.reset_time_s,
            fps=config.dataset.fps,
        )
    return tuple(summaries)
