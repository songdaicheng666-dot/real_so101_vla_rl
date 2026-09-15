"""Small dependency-light value types shared by hardware adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from real_so101_vla_rl.data.schema import (
    OVERVIEW_DEPTH_KEY,
    OVERVIEW_IMAGE_KEY,
    SENSOR_TIMESTAMP_KEYS,
    SENSOR_VALID_KEYS,
    STATE_KEY,
    WRIST_IMAGE_KEY,
)


@dataclass(frozen=True, slots=True)
class CapturedSample[T]:
    """One timestamped adapter output or an explicit acquisition failure."""

    value: T | None
    timestamp_ns: int
    valid: bool
    sequence_id: int
    error: str | None = None

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if self.sequence_id < 0:
            raise ValueError("sequence_id must be non-negative")
        if self.valid and (self.value is None or self.error is not None):
            raise ValueError("A valid sample requires a value and no error")
        if not self.valid and not self.error:
            raise ValueError("An invalid sample requires an error message")

    @classmethod
    def success(cls, value: T, *, timestamp_ns: int, sequence_id: int) -> CapturedSample[T]:
        return cls(value, timestamp_ns, True, sequence_id)

    @classmethod
    def failure(
        cls, *, timestamp_ns: int, sequence_id: int, error: str
    ) -> CapturedSample[T]:
        return cls(None, timestamp_ns, False, sequence_id, error)


@dataclass(frozen=True, slots=True)
class RGBDFrame:
    """Color and aligned depth returned from one Orbbec frameset."""

    overview: CapturedSample[np.ndarray]
    depth: CapturedSample[np.ndarray]


@dataclass(frozen=True, slots=True)
class SynchronizedObservation:
    """A complete schema-v2 observation before task and action are attached."""

    overview: np.ndarray
    wrist: np.ndarray
    overview_depth: np.ndarray
    state: np.ndarray
    overview_timestamp_ns: int
    wrist_timestamp_ns: int
    overview_depth_timestamp_ns: int
    state_timestamp_ns: int

    def to_schema_fields(self) -> dict[str, object]:
        timestamps = (
            self.overview_timestamp_ns,
            self.wrist_timestamp_ns,
            self.overview_depth_timestamp_ns,
            self.state_timestamp_ns,
        )
        return {
            OVERVIEW_IMAGE_KEY: self.overview,
            WRIST_IMAGE_KEY: self.wrist,
            OVERVIEW_DEPTH_KEY: self.overview_depth,
            STATE_KEY: self.state,
            **{
                key: np.asarray([timestamp], dtype=np.int64)
                for key, timestamp in zip(SENSOR_TIMESTAMP_KEYS, timestamps, strict=True)
            },
            **{
                key: np.asarray([True], dtype=np.bool_)
                for key in SENSOR_VALID_KEYS
            },
        }


@dataclass(frozen=True, slots=True)
class SyncDiagnostics:
    """Diagnostics retained even when a candidate observation is rejected."""

    timestamps_ns: Mapping[str, int]
    span_ns: int | None
    errors: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamps_ns", MappingProxyType(dict(self.timestamps_ns)))
        object.__setattr__(self, "errors", MappingProxyType(dict(self.errors)))


@dataclass(frozen=True, slots=True)
class SynchronizationResult:
    """Success/failure result returned by the synchronizer without partial fallback."""

    observation: SynchronizedObservation | None
    diagnostics: SyncDiagnostics
    error: str | None = None

    def __post_init__(self) -> None:
        if (self.observation is None) == (self.error is None):
            raise ValueError("A synchronization result must contain observation xor error")

    @property
    def valid(self) -> bool:
        return self.observation is not None
