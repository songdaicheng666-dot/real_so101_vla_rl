from __future__ import annotations

import json
from dataclasses import replace

import pytest

from real_so101_vla_rl.data import (
    build_robot_profile,
    load_robot_profile,
    write_robot_profile,
)

CALIBRATION = {
    "shoulder_pan": {
        "id": 1,
        "drive_mode": 0,
        "homing_offset": -1860,
        "range_min": 683,
        "range_max": 3416,
    },
    "shoulder_lift": {
        "id": 2,
        "drive_mode": 0,
        "homing_offset": -1937,
        "range_min": 831,
        "range_max": 3223,
    },
    "elbow_flex": {
        "id": 3,
        "drive_mode": 0,
        "homing_offset": -1983,
        "range_min": 925,
        "range_max": 3142,
    },
    "wrist_flex": {
        "id": 4,
        "drive_mode": 0,
        "homing_offset": -1897,
        "range_min": 853,
        "range_max": 3214,
    },
    "wrist_roll": {
        "id": 5,
        "drive_mode": 0,
        "homing_offset": -120,
        "range_min": 0,
        "range_max": 4095,
    },
    "gripper": {
        "id": 6,
        "drive_mode": 0,
        "homing_offset": -1468,
        "range_min": 1477,
        "range_max": 2953,
    },
}


def test_profile_uses_actual_lerobot_calibration_units(tmp_path) -> None:
    calibration_path = tmp_path / "my_follower_arm.json"
    calibration_path.write_text(json.dumps(CALIBRATION), encoding="utf-8")

    profile, calibration_bytes = build_robot_profile(
        calibration_path,
        robot_id="my_follower_arm",
        camera_setup_id="dual_rgbd_camera_v2",
        lerobot_commit="abc123",
    )

    assert profile.nominal_ranges["shoulder_pan.pos"].minimum == pytest.approx(-120.131868)
    assert profile.nominal_ranges["shoulder_lift.pos"].maximum == pytest.approx(105.142857)
    assert profile.nominal_ranges["elbow_flex.pos"].maximum == pytest.approx(97.450549)
    assert profile.nominal_ranges["wrist_flex.pos"].maximum == pytest.approx(103.78022)
    assert profile.nominal_ranges["wrist_roll.pos"].minimum == pytest.approx(-180.0)
    assert profile.nominal_ranges["gripper.pos"].minimum == 0
    assert profile.nominal_ranges["gripper.pos"].maximum == 100
    assert profile.nominal_ranges["gripper.pos"].unit == "lerobot_range_0_100"

    metadata_dir = tmp_path / "dataset" / "project_meta"
    write_robot_profile(metadata_dir, profile, calibration_bytes)

    assert (metadata_dir / "calibration.json").read_bytes() == calibration_bytes
    assert load_robot_profile(metadata_dir / "robot_profile.json") == profile

    with pytest.raises(ValueError, match="schema_version=1"):
        replace(profile, schema_version=1)
