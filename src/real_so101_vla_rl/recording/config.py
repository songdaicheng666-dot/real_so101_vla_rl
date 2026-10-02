"""Configuration contract for the schema-v2 real-hardware recorder."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from real_so101_vla_rl.data.schema import (
    DEFAULT_FPS,
    AtomicTask,
    CubeColor,
    TargetSlot,
    TaskType,
)
from real_so101_vla_rl.data.splits import SplitPolicy
from real_so101_vla_rl.hardware.cameras.profile import (
    LoadedCameraRigProfile,
    load_camera_rig_profile,
)


@dataclass(frozen=True, slots=True)
class ArmEndpointConfig:
    port: str | None
    robot_id: str
    calibration_dir: str | None = None

    def validate(self, *, name: str, require_hardware_ready: bool) -> None:
        if not self.robot_id.strip():
            raise ValueError(f"{name}.id must not be empty")
        if require_hardware_ready and not self.port:
            raise ValueError(f"{name}.port must be configured before real recording")


@dataclass(frozen=True, slots=True)
class RecordingPlanEntry:
    """One deterministic accepted-episode slot in a recording plan."""

    episode_index: int
    task: str
    layout_id: str


@dataclass(frozen=True, slots=True)
class TaskPlanConfig:
    """Balanced randomized T0 task scheduling for a multi-task dataset."""

    target_slot: TargetSlot
    colors: tuple[CubeColor, ...]
    episodes_per_color: int
    schedule: str
    seed: int
    scene_id_prefix: str

    def __post_init__(self) -> None:
        if self.target_slot is not TargetSlot.T0:
            raise ValueError("task_plan.target_slot currently supports only T0")
        if not self.colors or len(set(self.colors)) != len(self.colors):
            raise ValueError("task_plan.colors must contain unique colors")
        if set(self.colors) != set(CubeColor):
            raise ValueError(
                "task_plan.colors must contain red, blue, yellow, and green"
            )
        if self.episodes_per_color <= 0:
            raise ValueError("task_plan.episodes_per_color must be positive")
        if self.schedule != "per_session_stratified_shuffle":
            raise ValueError(
                "task_plan.schedule must be 'per_session_stratified_shuffle'"
            )
        if not isinstance(self.seed, int):
            raise TypeError("task_plan.seed must be an integer")
        if not self.scene_id_prefix.strip():
            raise ValueError("task_plan.scene_id_prefix must not be empty")

    def build_entries(
        self,
        *,
        num_episodes: int,
        session_size: int,
    ) -> tuple[RecordingPlanEntry, ...]:
        color_count = len(self.colors)
        if session_size <= 0 or session_size % color_count:
            raise ValueError(
                "dataset.session_size must be positive and divisible by the number of task-plan colors"
            )
        if num_episodes % session_size:
            raise ValueError(
                "dataset.num_episodes must be divisible by dataset.session_size"
            )
        if num_episodes != color_count * self.episodes_per_color:
            raise ValueError(
                "dataset.num_episodes must equal task_plan colors times episodes_per_color"
            )
        sessions = num_episodes // session_size
        per_color_per_session = session_size // color_count
        if sessions * per_color_per_session != self.episodes_per_color:
            raise ValueError(
                "task_plan cannot distribute every color evenly across recording sessions"
            )

        scheduled_colors: list[CubeColor] = []
        for session_index in range(sessions):
            if session_index == 0:
                # The first four episodes are a one-per-color canary. The rest of
                # the first session still preserves exact per-color balance.
                block = list(self.colors)
                remainder = [
                    color
                    for color in self.colors
                    for _ in range(per_color_per_session - 1)
                ]
                random.Random(self.seed + session_index).shuffle(remainder)
                block.extend(remainder)
            else:
                block = [
                    color for color in self.colors for _ in range(per_color_per_session)
                ]
                random.Random(self.seed + session_index).shuffle(block)
            scheduled_colors.extend(block)

        entries = []
        for episode_index, color in enumerate(scheduled_colors):
            task = AtomicTask(
                TaskType.SINGLE_T0,
                color,
                TargetSlot.T0,
                0,
            ).instruction
            entries.append(
                RecordingPlanEntry(
                    episode_index=episode_index,
                    task=task,
                    layout_id=f"{self.scene_id_prefix}-{episode_index + 1:03d}",
                )
            )
        return tuple(entries)


@dataclass(frozen=True, slots=True)
class DatasetRecordingConfig:
    repo_id: str
    root: str
    task: str | None
    layout_id: str | None
    trial_id_prefix: str
    num_episodes: int = 10
    episode_time_s: float = 60.0
    reset_time_s: float = 30.0
    fps: int = DEFAULT_FPS
    rgb_use_videos: bool = True
    purpose: str = "demonstration"
    manual_confirmation: bool = True
    layout_ids: tuple[str, ...] = ()
    split_policy: SplitPolicy = SplitPolicy.FULL_TASK_LAYOUT_V1
    normalization_key: str = "so101_cube_dual_rgb_v2"
    task_plan: TaskPlanConfig | None = None
    session_size: int | None = None
    min_free_gib: float = 0.0

    def __post_init__(self) -> None:
        if not self.repo_id.strip() or not self.root.strip():
            raise ValueError("dataset.repo_id and dataset.root must not be empty")
        fixed_task = isinstance(self.task, str) and bool(self.task.strip())
        planned_tasks = self.task_plan is not None
        if fixed_task == planned_tasks:
            raise ValueError("dataset must configure exactly one of task or task_plan")
        if self.task is not None and not fixed_task:
            raise ValueError("dataset.task must be null or a non-empty string")
        if fixed_task:
            assert self.task is not None
            AtomicTask.from_instruction(self.task)
        if not self.trial_id_prefix.strip():
            raise ValueError("dataset.trial_id_prefix must not be empty")
        if self.num_episodes <= 0:
            raise ValueError("dataset.num_episodes must be positive")
        if self.episode_time_s <= 0 or self.reset_time_s < 0:
            raise ValueError(
                "episode_time_s must be positive and reset_time_s non-negative"
            )
        if self.fps != DEFAULT_FPS:
            raise ValueError(f"Schema v2 recording fps is fixed at {DEFAULT_FPS}")
        if type(self.rgb_use_videos) is not bool:
            raise TypeError("dataset.rgb_use_videos must be a boolean")
        if self.purpose not in {"demonstration", "static_validation"}:
            raise ValueError(
                "dataset.purpose must be 'demonstration' or 'static_validation'"
            )
        if type(self.manual_confirmation) is not bool:
            raise TypeError("dataset.manual_confirmation must be a boolean")
        if (
            not isinstance(self.normalization_key, str)
            or not self.normalization_key.strip()
        ):
            raise ValueError("dataset.normalization_key must be a non-empty string")
        if self.session_size is not None and (
            self.session_size <= 0 or self.session_size > self.num_episodes
        ):
            raise ValueError(
                "dataset.session_size must be positive and no greater than num_episodes"
            )
        if self.min_free_gib < 0:
            raise ValueError("dataset.min_free_gib must be non-negative")

        fixed_layout = isinstance(self.layout_id, str) and bool(self.layout_id.strip())
        planned_layouts = bool(self.layout_ids)
        if planned_tasks:
            if fixed_layout or planned_layouts or self.layout_id is not None:
                raise ValueError(
                    "dataset.task_plan generates scene IDs; layout_id and layout_ids must be empty"
                )
            if self.session_size is None:
                raise ValueError("dataset.session_size is required with task_plan")
            assert self.task_plan is not None
            self.task_plan.build_entries(
                num_episodes=self.num_episodes,
                session_size=self.session_size,
            )
        else:
            if fixed_layout == planned_layouts:
                raise ValueError(
                    "dataset must configure exactly one of layout_id or layout_ids"
                )
            if self.layout_id is not None and not fixed_layout:
                raise ValueError("dataset.layout_id must be null or a non-empty string")
            if planned_layouts:
                if len(self.layout_ids) != self.num_episodes:
                    raise ValueError(
                        "dataset.layout_ids must contain exactly num_episodes entries"
                    )
                if any(
                    not isinstance(value, str) or not value.strip()
                    for value in self.layout_ids
                ):
                    raise ValueError(
                        "dataset.layout_ids entries must be non-empty strings"
                    )
                if len(set(self.layout_ids)) != len(self.layout_ids):
                    raise ValueError("dataset.layout_ids entries must be unique")
        SplitPolicy(self.split_policy)

    def plan_entries(self) -> tuple[RecordingPlanEntry, ...]:
        if self.task_plan is not None:
            assert self.session_size is not None
            return self.task_plan.build_entries(
                num_episodes=self.num_episodes,
                session_size=self.session_size,
            )
        assert self.task is not None
        return tuple(
            RecordingPlanEntry(
                episode_index=index,
                task=self.task,
                layout_id=(
                    self.layout_ids[index] if self.layout_ids else str(self.layout_id)
                ),
            )
            for index in range(self.num_episodes)
        )

    def plan_entry_for_episode(self, episode_index: int) -> RecordingPlanEntry:
        if not 0 <= episode_index < self.num_episodes:
            raise IndexError(
                f"episode_index must be in [0, {self.num_episodes}), got {episode_index}"
            )
        return self.plan_entries()[episode_index]

    def task_for_episode(self, episode_index: int) -> str:
        return self.plan_entry_for_episode(episode_index).task

    def session_end_for_episode(self, episode_index: int) -> int:
        if not 0 <= episode_index < self.num_episodes:
            raise IndexError(
                f"episode_index must be in [0, {self.num_episodes}), got {episode_index}"
            )
        if self.session_size is None:
            return self.num_episodes
        return min(
            ((episode_index // self.session_size) + 1) * self.session_size,
            self.num_episodes,
        )

    @property
    def plan_sha256(self) -> str:
        payload = {
            "schema_version": 1,
            "repo_id": self.repo_id,
            "num_episodes": self.num_episodes,
            "session_size": self.session_size,
            "entries": [
                {
                    "episode_index": entry.episode_index,
                    "task": entry.task,
                    "layout_id": entry.layout_id,
                }
                for entry in self.plan_entries()
            ],
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def layout_id_for_episode(self, episode_index: int) -> str:
        return self.plan_entry_for_episode(episode_index).layout_id

    @property
    def requires_layout_setup(self) -> bool:
        return bool(self.layout_ids) or self.task_plan is not None


@dataclass(frozen=True, slots=True)
class CameraRecordingConfig:
    profile_path: Path
    rig: LoadedCameraRigProfile

    def validate(self, *, require_hardware_ready: bool) -> None:
        del require_hardware_ready  # Live identity is checked by camera preflight.
        if self.rig.path != self.profile_path:
            raise ValueError("loaded camera profile path differs from recording config")


@dataclass(frozen=True, slots=True)
class SynchronizerConfig:
    tolerance_ms: float = 25.0
    timeout_ms: int = 200
    max_consecutive_failures: int = 3

    def __post_init__(self) -> None:
        if self.tolerance_ms < 0 or self.timeout_ms <= 0:
            raise ValueError("sync tolerance must be non-negative and timeout positive")
        if self.max_consecutive_failures <= 0:
            raise ValueError("sync.max_consecutive_failures must be positive")


@dataclass(frozen=True, slots=True)
class SO101RecordingConfig:
    dataset: DatasetRecordingConfig
    follower: ArmEndpointConfig
    leader: ArmEndpointConfig
    cameras: CameraRecordingConfig
    sync: SynchronizerConfig = SynchronizerConfig()
    max_relative_target: float | None = None

    def validate(self, *, require_hardware_ready: bool = True) -> None:
        self.follower.validate(
            name="follower", require_hardware_ready=require_hardware_ready
        )
        self.leader.validate(
            name="leader", require_hardware_ready=require_hardware_ready
        )
        self.cameras.validate(require_hardware_ready=require_hardware_ready)
        if self.max_relative_target is not None and self.max_relative_target <= 0:
            raise ValueError("max_relative_target must be positive when set")


def _mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping")
    return value


def load_recording_config(
    path: str | Path, *, require_hardware_ready: bool = True
) -> SO101RecordingConfig:
    """Load the intentionally small project recorder YAML contract."""

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError("Recording config root must be a mapping")
    dataset_raw = dict(_mapping(raw, "dataset"))
    if "use_videos" in dataset_raw:
        raise ValueError(
            "dataset.use_videos was removed; use dataset.rgb_use_videos. "
            "Aligned depth always uses lossless TIFF storage."
        )
    layout_ids_raw = dataset_raw.get("layout_ids", ())
    if not isinstance(layout_ids_raw, (list, tuple)):
        raise TypeError("dataset.layout_ids must be a sequence")
    dataset_raw["layout_ids"] = tuple(layout_ids_raw)
    task_plan_raw = dataset_raw.get("task_plan")
    if task_plan_raw is not None:
        if not isinstance(task_plan_raw, Mapping):
            raise TypeError("dataset.task_plan must be a mapping")
        task_plan_values = dict(task_plan_raw)
        colors_raw = task_plan_values.get("colors")
        if not isinstance(colors_raw, (list, tuple)):
            raise TypeError("dataset.task_plan.colors must be a sequence")
        try:
            task_plan_values["colors"] = tuple(CubeColor(value) for value in colors_raw)
            task_plan_values["target_slot"] = TargetSlot(
                task_plan_values.get("target_slot")
            )
        except ValueError as exc:
            raise ValueError(
                "dataset.task_plan contains an unsupported color or target slot"
            ) from exc
        dataset_raw["task_plan"] = TaskPlanConfig(**task_plan_values)
    try:
        dataset_raw["split_policy"] = SplitPolicy(
            dataset_raw.get("split_policy", SplitPolicy.FULL_TASK_LAYOUT_V1)
        )
    except ValueError as exc:
        raise ValueError("dataset.split_policy is unsupported") from exc
    dataset = DatasetRecordingConfig(**dataset_raw)
    follower_raw = dict(_mapping(raw, "follower"))
    leader_raw = dict(_mapping(raw, "leader"))
    follower = ArmEndpointConfig(
        port=follower_raw.get("port"),
        robot_id=str(follower_raw.get("id", "so101_follower")),
        calibration_dir=follower_raw.get("calibration_dir"),
    )
    leader = ArmEndpointConfig(
        port=leader_raw.get("port"),
        robot_id=str(leader_raw.get("id", "so101_leader")),
        calibration_dir=leader_raw.get("calibration_dir"),
    )
    cameras_raw = dict(_mapping(raw, "cameras"))
    removed_camera_keys = {"orbbec_serial", "wrist_device"} & set(cameras_raw)
    if removed_camera_keys:
        raise ValueError(
            "cameras.orbbec_serial and cameras.wrist_device were removed; "
            "configure only cameras.profile"
        )
    if set(cameras_raw) != {"profile"}:
        raise ValueError("cameras must contain exactly one field: profile")
    profile_value = cameras_raw["profile"]
    if not isinstance(profile_value, str) or not profile_value.strip():
        raise ValueError("cameras.profile must be a non-empty path")
    profile_path = (Path(path).resolve().parent / profile_value).resolve()
    camera_rig = load_camera_rig_profile(profile_path)
    config = SO101RecordingConfig(
        dataset=dataset,
        follower=follower,
        leader=leader,
        cameras=CameraRecordingConfig(profile_path=profile_path, rig=camera_rig),
        sync=SynchronizerConfig(**dict(raw.get("sync", {}))),
        max_relative_target=raw.get("max_relative_target"),
    )
    config.validate(require_hardware_ready=require_hardware_ready)
    return config
