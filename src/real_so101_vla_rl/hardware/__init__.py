"""Timestamped hardware capture interfaces for SO-101 recording."""

from .capture_types import (
    CapturedSample,
    RGBDFrame,
    SyncDiagnostics,
    SynchronizationResult,
    SynchronizedObservation,
)
from .so101_connection import (
    FollowerConnectionSnapshot,
    SO101CalibrationMismatchError,
    connect_calibrated_so101_follower,
    connect_calibrated_so101_leader,
)
from .state import SO101StateAdapter
from .synchronizer import SO101ObservationSynchronizer

__all__ = [
    "CapturedSample",
    "FollowerConnectionSnapshot",
    "RGBDFrame",
    "SO101CalibrationMismatchError",
    "SO101ObservationSynchronizer",
    "SO101StateAdapter",
    "SyncDiagnostics",
    "SynchronizationResult",
    "SynchronizedObservation",
    "connect_calibrated_so101_follower",
    "connect_calibrated_so101_leader",
]
