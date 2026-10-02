"""Versioned real-to-MuJoCo alignment and overview sensor modelling."""

from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ALIGNMENT_PATH = (
    Path(__file__).resolve().parent
    / "assets"
    / "mujoco"
    / "competition_2026"
    / "alignment.yaml"
)
JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=4)
def load_alignment(path: str | Path = ALIGNMENT_PATH) -> dict[str, Any]:
    """Load and minimally validate the immutable alignment contract."""

    alignment_path = Path(path).resolve()
    with alignment_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config.get("schema_version") != 1:
        raise ValueError("unsupported alignment schema")
    if not str(config.get("alignment_id", "")).strip():
        raise ValueError("alignment_id must be non-empty")
    joints = config["robot"]["joints"]
    if tuple(joints) != JOINT_NAMES:
        raise ValueError(f"alignment joint order differs: {tuple(joints)}")
    camera = config["overview_camera"]
    if camera["resolution"] != [640, 480]:
        raise ValueError("overview alignment must use 640x480")
    if len(camera["distortion_opencv"]) != 8:
        raise ValueError("overview distortion must be OpenCV rational k1..k6/p1/p2")
    return config


def alignment_sha256(path: str | Path = ALIGNMENT_PATH) -> str:
    """Return the exact alignment file digest used by a run."""

    return _sha256(Path(path))


def real_state_to_mujoco_qpos(
    state: np.ndarray | list[float] | tuple[float, ...],
    *,
    alignment: dict[str, Any] | None = None,
) -> np.ndarray:
    """Map LeRobot degrees/percent to MuJoCo radians without silent clipping."""

    config = alignment or load_alignment()
    values = np.asarray(state, dtype=np.float64)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("real SO-101 state must contain six finite values")
    robot = config["robot"]
    offsets = np.asarray(robot["zero_offsets_deg"][:5], dtype=np.float64)
    qpos = np.empty(6, dtype=np.float64)
    qpos[:5] = np.deg2rad(values[:5] - offsets)
    gripper = robot["gripper_mapping"]
    physical_min, physical_max = gripper["physical_percent"]
    sim_min, sim_max = gripper["mujoco_deg"]
    qpos[5] = math.radians(
        sim_min
        + (values[5] - physical_min)
        * (sim_max - sim_min)
        / (physical_max - physical_min)
    )
    limits = np.deg2rad(np.asarray(robot["mujoco_ranges_deg"], dtype=np.float64))
    outside = np.flatnonzero((qpos < limits[:, 0] - 1e-10) | (qpos > limits[:, 1] + 1e-10))
    if outside.size:
        details = ", ".join(
            f"{JOINT_NAMES[index]}={values[index]:.6g}" for index in outside
        )
        raise ValueError(f"real state maps outside calibrated MuJoCo limits: {details}")
    return qpos


def mujoco_qpos_to_real_state(
    qpos: np.ndarray | list[float] | tuple[float, ...],
    *,
    alignment: dict[str, Any] | None = None,
) -> np.ndarray:
    """Invert :func:`real_state_to_mujoco_qpos`."""

    config = alignment or load_alignment()
    values = np.asarray(qpos, dtype=np.float64)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("MuJoCo qpos must contain six finite values")
    robot = config["robot"]
    result = np.empty(6, dtype=np.float64)
    result[:5] = np.rad2deg(values[:5]) + np.asarray(
        robot["zero_offsets_deg"][:5], dtype=np.float64
    )
    gripper = robot["gripper_mapping"]
    physical_min, physical_max = gripper["physical_percent"]
    sim_min, sim_max = gripper["mujoco_deg"]
    result[5] = physical_min + (
        np.rad2deg(values[5]) - sim_min
    ) * (physical_max - physical_min) / (sim_max - sim_min)
    return result


def home_qpos(*, alignment: dict[str, Any] | None = None) -> np.ndarray:
    config = alignment or load_alignment()
    return np.deg2rad(np.asarray(config["robot"]["home_mujoco_deg"], dtype=np.float64))


def reset_to_home_keyframe(model: Any, data: Any, *, key_name: str = "home") -> None:
    """Reset an MjData to the named full-scene home keyframe."""

    import mujoco

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
    if key_id < 0:
        raise RuntimeError(f"MuJoCo scene has no {key_name!r} keyframe")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)


class OverviewSensorModel:
    """Apply the calibrated Orbbec optical and photometric response."""

    def __init__(self, alignment: dict[str, Any] | None = None) -> None:
        self.alignment = alignment or load_alignment()
        camera = self.alignment["overview_camera"]
        self.width, self.height = (int(value) for value in camera["resolution"])
        self.fx, self.fy = (float(value) for value in camera["focalpixel"])
        self.cx, self.cy = (float(value) for value in camera["principalpixel"])
        self.distortion = np.asarray(camera["distortion_opencv"], dtype=np.float64)
        self.style = self.alignment["overview_sensor_model"]
        self._map_x, self._map_y = self._build_distortion_map()

    def _distort_normalized(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        k1, k2, p1, p2, k3, k4, k5, k6 = self.distortion
        radius2 = x * x + y * y
        numerator = 1 + k1 * radius2 + k2 * radius2**2 + k3 * radius2**3
        denominator = 1 + k4 * radius2 + k5 * radius2**2 + k6 * radius2**3
        radial = numerator / denominator
        xy2 = 2 * x * y
        return (
            x * radial + p1 * xy2 + p2 * (radius2 + 2 * x * x),
            y * radial + p1 * (radius2 + 2 * y * y) + p2 * xy2,
        )

    def _build_distortion_map(self) -> tuple[np.ndarray, np.ndarray]:
        output_x, output_y = np.meshgrid(
            np.arange(self.width, dtype=np.float64),
            np.arange(self.height, dtype=np.float64),
        )
        desired_x = (output_x - self.cx) / self.fx
        desired_y = (output_y - self.cy) / self.fy
        source_x = desired_x.copy()
        source_y = desired_y.copy()
        for _ in range(8):
            actual_x, actual_y = self._distort_normalized(source_x, source_y)
            source_x += desired_x - actual_x
            source_y += desired_y - actual_y
        return (
            np.asarray(source_x * self.fx + self.cx, dtype=np.float32),
            np.asarray(source_y * self.fy + self.cy, dtype=np.float32),
        )

    def distort_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply the calibrated optical warp to a boolean segmentation mask."""

        if mask.shape != (self.height, self.width):
            raise ValueError(
                f"overview mask must be [{self.height},{self.width}], got {mask.shape}"
            )
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is required for the overview sensor model; install .[sim]"
            ) from exc
        warped = cv2.remap(
            mask.astype(np.uint8),
            self._map_x,
            self._map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return np.ascontiguousarray(warped.astype(bool))

    def apply(self, image: np.ndarray, *, seed: int = 0, noise: bool = True) -> np.ndarray:
        """Return a reproducible ``uint8[480,640,3]`` styled frame."""

        if image.shape != (self.height, self.width, 3) or image.dtype != np.uint8:
            raise ValueError(
                f"overview input must be uint8[{self.height},{self.width},3], "
                f"got {image.dtype}{image.shape}"
            )
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is required for the overview sensor model; install .[sim]"
            ) from exc

        styled = cv2.remap(
            image,
            self._map_x,
            self._map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        ).astype(np.float32)
        styled = styled * np.asarray(self.style["color_gain_rgb"], dtype=np.float32)
        styled += np.asarray(self.style["color_bias_rgb"], dtype=np.float32)
        styled = np.clip(styled / 255.0, 0.0, 1.0)
        styled = np.power(styled, 1.0 / float(self.style["gamma"]))

        yy, xx = np.ogrid[-1.0:1.0:complex(self.height), -1.0:1.0:complex(self.width)]
        radius2 = np.clip(xx * xx + yy * yy, 0.0, 1.0)
        vignette = 1.0 - float(self.style["vignette_strength"]) * radius2
        styled *= vignette[..., None]
        sigma = float(self.style["blur_sigma_px"])
        if sigma > 0:
            styled = cv2.GaussianBlur(styled, (0, 0), sigmaX=sigma, sigmaY=sigma)
        if noise and float(self.style["noise_std_255"]) > 0:
            generator = np.random.default_rng(seed)
            styled += generator.normal(
                0.0,
                float(self.style["noise_std_255"]) / 255.0,
                styled.shape,
            ).astype(np.float32)
        return np.ascontiguousarray(
            np.clip(np.rint(styled * 255.0), 0, 255).astype(np.uint8)
        )
