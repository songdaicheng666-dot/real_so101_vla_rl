"""Immutable camera-rig provenance stored beside real LeRobot datasets."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from real_so101_vla_rl.hardware.cameras.discovery import CameraProbeReport
from real_so101_vla_rl.hardware.cameras.profile import LoadedCameraRigProfile

CAMERA_PROFILE_FILENAME = "camera_profile.yaml"
CAMERA_CALIBRATION_FILENAME = "orbbec_factory_calibration.json"
CAMERA_SETUP_FILENAME = "camera_setup.json"
CAMERA_PROBES_FILENAME = "camera_probe_reports.jsonl"


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _setup_payload(loaded: LoadedCameraRigProfile) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "rig_id": loaded.profile.rig_id,
        "camera_profile_file": CAMERA_PROFILE_FILENAME,
        "camera_profile_sha256": loaded.sha256,
        "factory_calibration_file": CAMERA_CALIBRATION_FILENAME,
        "factory_calibration_sha256": loaded.calibration_sha256,
        "installation_calibration_status": (
            loaded.profile.installation_calibration.status
        ),
    }


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def write_camera_setup_metadata(
    dataset_root: str | Path,
    loaded: LoadedCameraRigProfile,
    report: CameraProbeReport,
) -> None:
    """Create or validate immutable camera snapshots and append this probe report."""

    if not report.valid:
        raise ValueError("Cannot persist an invalid camera probe report")
    if report.rig_id != loaded.profile.rig_id:
        raise ValueError("Camera probe rig_id differs from loaded profile")
    if report.profile_sha256 != loaded.sha256:
        raise ValueError("Camera probe profile SHA-256 differs from loaded profile")
    if report.calibration_sha256 != loaded.calibration_sha256:
        raise ValueError("Camera probe calibration SHA-256 differs from loaded profile")

    metadata_root = Path(dataset_root) / "project_meta"
    profile_path = metadata_root / CAMERA_PROFILE_FILENAME
    calibration_path = metadata_root / CAMERA_CALIBRATION_FILENAME
    setup_path = metadata_root / CAMERA_SETUP_FILENAME
    probe_path = metadata_root / CAMERA_PROBES_FILENAME
    expected_setup = _canonical_json(_setup_payload(loaded))

    immutable_paths = (profile_path, calibration_path, setup_path)
    existing_count = sum(path.exists() for path in immutable_paths)
    if existing_count not in (0, len(immutable_paths)):
        raise RuntimeError("Dataset contains a partial camera metadata snapshot")
    if existing_count:
        if profile_path.read_bytes() != loaded.raw_bytes:
            raise RuntimeError("Dataset camera profile differs; refusing mixed-rig recording")
        if calibration_path.read_bytes() != loaded.calibration_bytes:
            raise RuntimeError(
                "Dataset camera calibration differs; refusing mixed-calibration recording"
            )
        if setup_path.read_bytes() != expected_setup:
            raise RuntimeError("Dataset camera setup manifest differs from active profile")
    else:
        _atomic_write(profile_path, loaded.raw_bytes)
        _atomic_write(calibration_path, loaded.calibration_bytes)
        _atomic_write(setup_path, expected_setup)

    metadata_root.mkdir(parents=True, exist_ok=True)
    with probe_path.open("a", encoding="utf-8") as output:
        json.dump(report.to_dict(), output, ensure_ascii=False, separators=(",", ":"))
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
