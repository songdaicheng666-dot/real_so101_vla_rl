"""Timestamped hardware capture interfaces for SO-101 recording."""

from .capture_types import (
    CapturedSample,
    RGBDFrame,
    SyncDiagnostics,
    SynchronizationResult,
    SynchronizedObservation,
)
from .state import SO101StateAdapter
from .synchronizer import SO101ObservationSynchronizer

__all__ = [
    "CapturedSample",
    "RGBDFrame",
    "SO101ObservationSynchronizer",
    "SO101StateAdapter",
    "SyncDiagnostics",
    "SynchronizationResult",
    "SynchronizedObservation",
]
