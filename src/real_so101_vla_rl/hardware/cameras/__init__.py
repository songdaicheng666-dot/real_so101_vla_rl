"""RGB and RGB-D camera adapters used by the schema-v2 synchronizer."""

from .base import RGBAdapter, RGBDAdapter
from .discovery import (
    CameraProbeReport,
    probe_camera_rig,
    require_valid_camera_probe,
    write_camera_probe_report,
)
from .opencv_rgb import OpenCVRGBAdapter
from .orbbec_rgbd import OrbbecRGBDAdapter
from .profile import (
    CameraRigProfile,
    LoadedCameraRigProfile,
    load_camera_rig_profile,
)

__all__ = [
    "CameraProbeReport",
    "CameraRigProfile",
    "LoadedCameraRigProfile",
    "OpenCVRGBAdapter",
    "OrbbecRGBDAdapter",
    "RGBAdapter",
    "RGBDAdapter",
    "load_camera_rig_profile",
    "probe_camera_rig",
    "require_valid_camera_probe",
    "write_camera_probe_report",
]
