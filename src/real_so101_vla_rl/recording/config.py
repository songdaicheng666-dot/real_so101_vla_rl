"""Configuration contract for the schema-v2 real-hardware recorder."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from real_so101_vla_rl.data.schema import DEFAULT_FPS, AtomicTask


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
class DatasetRecordingConfig:
    repo_id: str
    root: str
    task: str
    layout_id: str
    trial_id_prefix: str
    num_episodes: int = 10
    episode_time_s: float = 60.0
    reset_time_s: float = 30.0
    fps: int = DEFAULT_FPS
    rgb_use_videos: bool = True

    def __post_init__(self) -> None:
        if not self.repo_id.strip() or not self.root.strip():
            raise ValueError("dataset.repo_id and dataset.root must not be empty")
        AtomicTask.from_instruction(self.task)
        if not self.layout_id.strip() or not self.trial_id_prefix.strip():
            raise ValueError("dataset layout_id and trial_id_prefix must not be empty")
        if self.num_episodes <= 0:
            raise ValueError("dataset.num_episodes must be positive")
        if self.episode_time_s <= 0 or self.reset_time_s < 0:
            raise ValueError("episode_time_s must be positive and reset_time_s non-negative")
        if self.fps != DEFAULT_FPS:
            raise ValueError(f"Schema v2 recording fps is fixed at {DEFAULT_FPS}")
        if type(self.rgb_use_videos) is not bool:
            raise TypeError("dataset.rgb_use_videos must be a boolean")


@dataclass(frozen=True, slots=True)
class CameraRecordingConfig:
    orbbec_serial: str | None
    wrist_device: str | int | None

    def validate(self, *, require_hardware_ready: bool) -> None:
        if require_hardware_ready and not self.orbbec_serial:
            raise ValueError("cameras.orbbec_serial must be configured")
        if require_hardware_ready and self.wrist_device is None:
            raise ValueError(
                "cameras.wrist_device must be configured; /dev/video4 is deliberately not a default"
            )


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
        self.follower.validate(name="follower", require_hardware_ready=require_hardware_ready)
        self.leader.validate(name="leader", require_hardware_ready=require_hardware_ready)
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
    config = SO101RecordingConfig(
        dataset=dataset,
        follower=follower,
        leader=leader,
        cameras=CameraRecordingConfig(**dict(_mapping(raw, "cameras"))),
        sync=SynchronizerConfig(**dict(raw.get("sync", {}))),
        max_relative_target=raw.get("max_relative_target"),
    )
    config.validate(require_hardware_ready=require_hardware_ready)
    return config
