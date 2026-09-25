"""Concurrent timestamp synchronization for schema-v2 observations."""

from __future__ import annotations

from collections import deque
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
from .time_matching import closest_observation_triplet


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
        self._overview_buffer: deque[Any] = deque(maxlen=4)
        self._wrist_buffer: deque[CapturedSample[np.ndarray]] = deque(maxlen=4)
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
            max_workers=2, thread_name_prefix="so101_capture"
        )
        self._last_sequences.clear()
        self._overview_buffer.clear()
        self._wrist_buffer.clear()

    def _read_cameras(self) -> tuple[Any, CapturedSample[np.ndarray]]:
        assert self._executor is not None
        overview_future: Future[Any] = self._executor.submit(
            self.overview.read, self.timeout_ms
        )
        wrist_future: Future[Any] = self._executor.submit(
            self.wrist.read, self.timeout_ms
        )
        timeout_s = self.timeout_ms / 1000 + 0.1
        return (
            overview_future.result(timeout=timeout_s),
            wrist_future.result(timeout=timeout_s),
        )

    def _append_camera_samples(
        self,
        rgbd: Any,
        wrist: CapturedSample[np.ndarray],
    ) -> None:
        if (
            rgbd.overview.valid
            and rgbd.depth.valid
            and not any(
                old.overview.sequence_id == rgbd.overview.sequence_id
                for old in self._overview_buffer
            )
        ):
            self._overview_buffer.append(rgbd)
        if wrist.valid and not any(
            old.sequence_id == wrist.sequence_id for old in self._wrist_buffer
        ):
            self._wrist_buffer.append(wrist)

    def _unused_camera_samples(self) -> tuple[list[Any], list[CapturedSample[np.ndarray]]]:
        overview_last = self._last_sequences.get(SENSOR_TIMESTAMP_KEYS[0], -1)
        wrist_last = self._last_sequences.get(SENSOR_TIMESTAMP_KEYS[1], -1)
        return (
            [
                sample
                for sample in self._overview_buffer
                if sample.overview.sequence_id > overview_last
            ],
            [
                sample
                for sample in self._wrist_buffer
                if sample.sequence_id > wrist_last
            ],
        )

    def _read_older_camera(self, rgbd: Any, wrist: CapturedSample[np.ndarray]) -> None:
        if rgbd.overview.timestamp_ns <= wrist.timestamp_ns:
            next_rgbd = self.overview.read(self.timeout_ms)
            self._append_camera_samples(next_rgbd, wrist)
        else:
            next_wrist = self.wrist.read(self.timeout_ms)
            self._append_camera_samples(rgbd, next_wrist)

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

        try:
            rgbd, wrist = self._read_cameras()
            self._append_camera_samples(rgbd, wrist)
            # Read state after camera delivery so its host timestamp describes the
            # robot state paired with the images instead of thread scheduling time.
            state = self.state.read(self.timeout_ms)
        except Exception as exc:  # noqa: BLE001 - worker failures reject the sample
            self._worker_failed = True
            return SynchronizationResult(
                None,
                SyncDiagnostics({}, None, {"synchronizer": f"{type(exc).__name__}: {exc}"}),
                error="capture worker failed",
            )

        initial_samples = {
            SENSOR_TIMESTAMP_KEYS[0]: rgbd.overview,
            SENSOR_TIMESTAMP_KEYS[1]: wrist,
            SENSOR_TIMESTAMP_KEYS[2]: rgbd.depth,
            SENSOR_TIMESTAMP_KEYS[3]: state,
        }
        errors = {
            key: sample.error or "invalid sample"
            for key, sample in initial_samples.items()
            if not sample.valid
        }
        if errors:
            return self._invalid_result(
                initial_samples, errors, error="one or more sensors returned an invalid sample"
            )

        selected = None
        for _ in range(3):
            overview_candidates, wrist_candidates = self._unused_camera_samples()
            selected = closest_observation_triplet(
                overview_candidates, wrist_candidates, state
            )
            if selected is None or selected[2] <= self.tolerance_ns:
                break
            selected_rgbd, selected_wrist, _ = selected
            try:
                self._read_older_camera(selected_rgbd, selected_wrist)
            except Exception as exc:  # noqa: BLE001 - reject driver retry failures
                self._worker_failed = True
                return SynchronizationResult(
                    None,
                    SyncDiagnostics(
                        {}, None, {"synchronizer": f"{type(exc).__name__}: {exc}"}
                    ),
                    error="capture worker failed while matching nearest frames",
                )

        if selected is None:
            duplicate_errors = {
                key: f"no new sample after sequence_id={sequence_id}"
                for key, sequence_id in self._last_sequences.items()
                if key in {SENSOR_TIMESTAMP_KEYS[0], SENSOR_TIMESTAMP_KEYS[1]}
            }
            return self._invalid_result(
                initial_samples,
                duplicate_errors,
                error="one or more sensor frames were reused",
            )

        rgbd, wrist, span_ns = selected
        samples = {
            SENSOR_TIMESTAMP_KEYS[0]: rgbd.overview,
            SENSOR_TIMESTAMP_KEYS[1]: wrist,
            SENSOR_TIMESTAMP_KEYS[2]: rgbd.depth,
            SENSOR_TIMESTAMP_KEYS[3]: state,
        }
        timestamps = {key: sample.timestamp_ns for key, sample in samples.items()}
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
        self._overview_buffer = deque(
            (
                sample
                for sample in self._overview_buffer
                if sample.overview.sequence_id > rgbd.overview.sequence_id
            ),
            maxlen=4,
        )
        self._wrist_buffer = deque(
            (
                sample
                for sample in self._wrist_buffer
                if sample.sequence_id > wrist.sequence_id
            ),
            maxlen=4,
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
        self._overview_buffer.clear()
        self._wrist_buffer.clear()

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.disconnect()
