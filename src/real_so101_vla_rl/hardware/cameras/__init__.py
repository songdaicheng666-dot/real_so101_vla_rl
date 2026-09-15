"""RGB and RGB-D camera adapters used by the schema-v2 synchronizer."""

from .base import RGBAdapter, RGBDAdapter
from .opencv_rgb import OpenCVRGBAdapter
from .orbbec_rgbd import OrbbecRGBDAdapter

__all__ = ["OpenCVRGBAdapter", "OrbbecRGBDAdapter", "RGBAdapter", "RGBDAdapter"]
