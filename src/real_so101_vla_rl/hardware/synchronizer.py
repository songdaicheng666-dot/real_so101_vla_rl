"""Concurrent timestamp synchronization for schema-v2 observations."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Protocol, Self

import numpy as np

from real_so101_vla_rl.data.schema import SENSOR_TIMESTAMP_KEYS

from .cameras.base import RGBAdapter, RGBDAdapter
from .capture_types import (
    CapturedSample,
    SyncDiagnostics,
    SynchronizationResult,
    SynchronizedObservation,
)


class StateAdapter(Protocol):
    @property
    def is_connected(self) -> bool: ...

    @property
    def is_broken(self) -> bool: ...

    def connect(self) -> None: ...

    def read(self, timeout_ms: int) -> CapturedSample[np.ndarray]: ...

    def disconnect(self) -> None: ...


class SO101ObservationSynchronizer:
    """Acquire the three hardware sources concurrently and enforce host-time skew."""

    def __init__(
        self,
        overview: RGBDAdapter,
        wrist: RGBAdapter,
        state: StateAdapter,
        *,
        tolerance_ms: float = 25.0,
        timeout_ms: int = 200,
    ) -> None:
        if tolerance_ms < 0:
            raise ValueError("tolerance_ms must be non-negative")
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        self.overview = overview
        self.wrist = wrist
        self.state = state
        self.tolerance_ns = int(tolerance_ms * 1_000_000)
        self.timeout_ms = timeout_ms
        self._last_sequences: dict[str, int] = {}
        self._executor: ThreadPoolExecutor | None = None
        self._worker_failed = False

    @property
    def is_broken(self) -> bool:
        return self._worker_failed or any(
            adapter.is_broken for adapter in (self.overview, self.wrist, self.state)
        )

    def connect(self) -> None:
        if self._executor is not None:
            raise RuntimeError("Synchronizer is already connected")
        connected: list[Any] = []
        try:
            for adapter in (self.overview, self.wrist, self.state):
                adapter.connect()
                connected.append(adapter)
        except BaseException:
            for adapter in reversed(connected):
                adapter.disconnect()
            raise
        self._executor = ThreadPoolExecutor(
            max_workers=3, thread_name_prefix="so101_capture"
        )

    def _invalid_result(
        self,
        samples: dict[str, CapturedSample[np.ndarray]],
        errors: dict[str, str],
        *,
        error: str,
    ) -> SynchronizationResult:
        timestamps = {
            key: sample.timestamp_ns for key, sample in samples.items()
        }
        span_ns = (
            max(timestamps.values()) - min(timestamps.values())
            if len(timestamps) == 4
            else None
        )
        return SynchronizationResult(
            None,
            SyncDiagnostics(timestamps, span_ns, errors),
            error=error,
        )

    def capture(self) -> SynchronizationResult:
        if self._executor is None:
            raise RuntimeError("Synchronizer must be connected before capture")

        overview_future: Future[Any] = self._executor.submit(
            self.overview.read, self.timeout_ms
        )
        wrist_future: Future[Any] = self._executor.submit(
            self.wrist.read, self.timeout_ms
        )
        state_future: Future[Any] = self._executor.submit(
            self.state.read, self.timeout_ms
        )
        try:
            rgbd = overview_future.result(timeout=self.timeout_ms / 1000 + 0.1)
            wrist = wrist_future.result(timeout=self.timeout_ms / 1000 + 0.1)
            state = state_future.result(timeout=self.timeout_ms / 1000 + 0.1)
        except Exception as exc:  # noqa: BLE001 - worker failures reject the sample
            self._worker_failed = True
            return SynchronizationResult(
                None,
                SyncDiagnostics({}, None, {"synchronizer": f"{type(exc).__name__}: {exc}"}),
                error="capture worker failed",
            )

        samples = {
            SENSOR_TIMESTAMP_KEYS[0]: rgbd.overview,
            SENSOR_TIMESTAMP_KEYS[1]: wrist,
            SENSOR_TIMESTAMP_KEYS[2]: rgbd.depth,
            SENSOR_TIMESTAMP_KEYS[3]: state,
        }
        errors = {
            key: sample.error or "invalid sample"
            for key, sample in samples.items()
            if not sample.valid
        }
        if errors:
            return self._invalid_result(
                samples, errors, error="one or more sensors returned an invalid sample"
            )

        duplicate_errors: dict[str, str] = {}
        for key, sample in samples.items():
            if self._last_sequences.get(key) == sample.sequence_id:
                duplicate_errors[key] = f"reused sequence_id={sample.sequence_id}"
        if duplicate_errors:
            return self._invalid_result(
                samples, duplicate_errors, error="one or more sensor frames were reused"
            )

        timestamps = {key: sample.timestamp_ns for key, sample in samples.items()}
        span_ns = max(timestamps.values()) - min(timestamps.values())
        if span_ns > self.tolerance_ns:
            return SynchronizationResult(
                None,
                SyncDiagnostics(timestamps, span_ns, {}),
                error=(
                    f"sensor timestamp span {span_ns / 1_000_000:.3f} ms exceeds "
                    f"{self.tolerance_ns / 1_000_000:.3f} ms"
                ),
            )

        self._last_sequences.update(
            {key: sample.sequence_id for key, sample in samples.items()}
        )
        observation = SynchronizedObservation(
            overview=rgbd.overview.value,
            wrist=wrist.value,
            overview_depth=rgbd.depth.value,
            state=state.value,
            overview_timestamp_ns=rgbd.overview.timestamp_ns,
            wrist_timestamp_ns=wrist.timestamp_ns,
            overview_depth_timestamp_ns=rgbd.depth.timestamp_ns,
            state_timestamp_ns=state.timestamp_ns,
        )
        return SynchronizationResult(
            observation,
            SyncDiagnostics(timestamps, span_ns, {}),
        )

    def disconnect(self) -> None:
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        for adapter in (self.state, self.wrist, self.overview):
            adapter.disconnect()

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.disconnect()
