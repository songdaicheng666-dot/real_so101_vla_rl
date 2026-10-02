"""Project-owned real-hardware recording orchestration."""

from .camera_metadata import write_camera_setup_metadata
from .config import (
    RecordingPlanEntry,
    SO101RecordingConfig,
    TaskPlanConfig,
    load_recording_config,
)
from .dataset_lifecycle import (
    validate_recording_plan_metadata,
    validate_recording_root_mode,
    validate_recording_storage,
    validate_resumed_recording_dataset,
    write_recording_plan_metadata,
    write_recording_robot_profile,
)
from .lerobot_v2 import (
    CaptureAbort,
    EpisodeCaptureSummary,
    SchemaV2EpisodeRecorder,
    build_recording_frame,
    collect_episode,
    run_recording_session,
)

__all__ = [
    "CaptureAbort",
    "EpisodeCaptureSummary",
    "RecordingPlanEntry",
    "SO101RecordingConfig",
    "SchemaV2EpisodeRecorder",
    "TaskPlanConfig",
    "build_recording_frame",
    "collect_episode",
    "load_recording_config",
    "run_recording_session",
    "validate_recording_plan_metadata",
    "validate_recording_root_mode",
    "validate_recording_storage",
    "validate_resumed_recording_dataset",
    "write_camera_setup_metadata",
    "write_recording_plan_metadata",
    "write_recording_robot_profile",
]
