"""Dependency-light nearest-timestamp selection for captured sensor samples."""

from __future__ import annotations

from collections.abc import Sequence

from .capture_types import CapturedSample, RGBDFrame


def closest_camera_pair[T](
    overview_frames: Sequence[RGBDFrame],
    wrist_frames: Sequence[CapturedSample[T]],
) -> tuple[RGBDFrame, CapturedSample[T], int] | None:
    """Return the unique camera pair with the smallest host-time difference."""

    candidates = (
        (
            abs(rgbd.overview.timestamp_ns - wrist.timestamp_ns),
            rgbd.overview.timestamp_ns,
            wrist.timestamp_ns,
            rgbd,
            wrist,
        )
        for rgbd in overview_frames
        for wrist in wrist_frames
        if rgbd.overview.valid and rgbd.depth.valid and wrist.valid
    )
    try:
        span_ns, _, _, rgbd, wrist = min(candidates, key=lambda item: item[:3])
    except ValueError:
        return None
    return rgbd, wrist, span_ns


def closest_observation_triplet[W, S](
    overview_frames: Sequence[RGBDFrame],
    wrist_frames: Sequence[CapturedSample[W]],
    state: CapturedSample[S],
) -> tuple[RGBDFrame, CapturedSample[W], int] | None:
    """Select camera frames that minimize total span around one state read."""

    if not state.valid:
        return None
    candidates = []
    for rgbd in overview_frames:
        if not rgbd.overview.valid or not rgbd.depth.valid:
            continue
        for wrist in wrist_frames:
            if not wrist.valid:
                continue
            timestamps = (
                rgbd.overview.timestamp_ns,
                wrist.timestamp_ns,
                state.timestamp_ns,
            )
            candidates.append(
                (max(timestamps) - min(timestamps), *timestamps, rgbd, wrist)
            )
    if not candidates:
        return None
    span_ns, _, _, _, rgbd, wrist = min(candidates, key=lambda item: item[:4])
    return rgbd, wrist, span_ns
