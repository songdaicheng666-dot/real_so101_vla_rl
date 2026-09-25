"""Fail-closed SO-101 connection helpers for real-hardware recording."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class SO101CalibrationMismatchError(RuntimeError):
    """Raised when the attached arm does not match its configured calibration."""


@dataclass(frozen=True, slots=True)
class FollowerConnectionSnapshot:
    """Follower state used to prime a motionless position hold."""

    present_position_raw: dict[str, int | float]
    goal_position_raw: dict[str, int | float]


def _disconnect_with_torque_disabled(bus: Any) -> None:
    if not bool(getattr(bus, "is_connected", False)):
        return
    try:
        bus.disable_torque()
    finally:
        bus.disconnect(disable_torque=False)


def connect_calibrated_so101_leader(leader: Any) -> None:
    """Connect a leader while refusing any implicit calibration rewrite."""

    if bool(getattr(leader, "is_connected", False)):
        raise RuntimeError("SO-101 leader is already connected")
    bus = leader.bus
    bus.connect()
    try:
        if not bool(leader.is_calibrated):
            raise SO101CalibrationMismatchError(
                "SO-101 leader calibration does not match its configured file"
            )
        leader.configure()
    except Exception:
        _disconnect_with_torque_disabled(bus)
        raise


def connect_calibrated_so101_follower(
    follower: Any, *, num_retry: int = 2
) -> FollowerConnectionSnapshot:
    """Connect a follower without pursuing a stale position target.

    LeRobot's SO-101 follower configuration temporarily disables torque and then
    enables it. Before that enable transition, copy the current raw joint positions
    into ``Goal_Position`` so connection starts as a motionless position hold.
    """

    if bool(getattr(follower, "is_connected", False)):
        raise RuntimeError("SO-101 follower is already connected")
    bus = follower.bus
    bus.connect()
    try:
        if not bool(follower.is_calibrated):
            raise SO101CalibrationMismatchError(
                "SO-101 follower calibration does not match its configured file"
            )
        bus.disable_torque(num_retry=num_retry)
        present_raw = bus.sync_read(
            "Present_Position", normalize=False, num_retry=num_retry
        )
        bus.sync_write(
            "Goal_Position", present_raw, normalize=False, num_retry=num_retry
        )
        follower.configure()
        goal_raw = bus.sync_read(
            "Goal_Position", normalize=False, num_retry=num_retry
        )
        if goal_raw != present_raw:
            raise RuntimeError(
                "SO-101 follower position-hold priming could not be verified"
            )
    except Exception:
        _disconnect_with_torque_disabled(bus)
        raise
    return FollowerConnectionSnapshot(
        present_position_raw=dict(present_raw),
        goal_position_raw=dict(goal_raw),
    )
