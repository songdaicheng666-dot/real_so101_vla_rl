"""SO-101 LeRobot joint readings and MuJoCo joint coordinates.

The endpoint pairs were verified with the live follower mirror.  This module
only changes coordinates; it never clips model actions or steps a simulation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from real_so101_vla_rl.alignment import JOINT_NAMES, load_alignment


class JointAngleMapping:
    """Vectorized six-joint conversion using the versioned alignment contract."""

    def __init__(self, alignment: dict[str, Any] | None = None) -> None:
        self.alignment = alignment if alignment is not None else load_alignment()
        if self.alignment.get("schema_version") != 3:
            raise ValueError("unsupported joint alignment schema")
        robot = self.alignment["robot"]
        if tuple(robot["joints"]) != JOINT_NAMES:
            raise ValueError("joint order differs from SO-101")
        self.real_ranges = np.asarray(
            robot["physical_ranges_deg_or_percent"], dtype=np.float64
        )
        self.sim_ranges_deg = np.asarray(robot["mujoco_ranges_deg"], dtype=np.float64)
        signs = np.asarray(robot["joint_direction_signs"], dtype=np.int8)
        if (
            self.real_ranges.shape != (6, 2)
            or not np.all(np.isfinite(self.real_ranges))
            or not np.all(self.real_ranges[:, 0] < self.real_ranges[:, 1])
            or self.sim_ranges_deg.shape != (6, 2)
            or not np.all(np.isfinite(self.sim_ranges_deg))
            or not np.all(self.sim_ranges_deg[:, 0] < self.sim_ranges_deg[:, 1])
            or signs.shape != (5,)
            or not np.all(signs == 1)
        ):
            raise ValueError("invalid SO-101 joint endpoint alignment")
        if not np.array_equal(self.real_ranges[4], self.sim_ranges_deg[4]):
            raise ValueError("wrist_roll direct mapping requires matching endpoints")

    @staticmethod
    def _values(values: ArrayLike) -> NDArray[np.float64]:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim < 1 or array.shape[-1] != 6 or not np.all(np.isfinite(array)):
            raise ValueError("joint values must be finite with final dimension 6")
        return array

    def real_to_mujoco_qpos(self, values: ArrayLike) -> NDArray[np.float64]:
        """Map LeRobot degrees/percent to MuJoCo radians without clipping."""

        real = self._values(values)
        result = np.empty_like(real)
        for index in (0, 1, 2, 3, 5):
            real_min, real_max = self.real_ranges[index]
            sim_min, sim_max = self.sim_ranges_deg[index]
            fraction = (real[..., index] - real_min) / (real_max - real_min)
            result[..., index] = sim_min + fraction * (sim_max - sim_min)
        result[..., 4] = real[..., 4]
        return np.deg2rad(result)

    def mujoco_qpos_to_real_state(self, qpos: ArrayLike) -> NDArray[np.float64]:
        """Map MuJoCo radians back to LeRobot degrees/percent."""

        sim = np.rad2deg(self._values(qpos))
        result = np.empty_like(sim)
        for index in (0, 1, 2, 3, 5):
            real_min, real_max = self.real_ranges[index]
            sim_min, sim_max = self.sim_ranges_deg[index]
            fraction = (sim[..., index] - sim_min) / (sim_max - sim_min)
            result[..., index] = real_min + fraction * (real_max - real_min)
        result[..., 4] = sim[..., 4]
        return result

    def mujoco_ranges_deg(self) -> NDArray[np.float64]:
        """Return the six formal joint/control intervals in qpos degrees."""

        return self.sim_ranges_deg.copy()

    def validate_mujoco_ranges(self, model: Any) -> None:
        """Ensure the compiled robot accepts the complete mapped intervals."""

        import mujoco

        expected = np.deg2rad(self.mujoco_ranges_deg())
        for index, name in enumerate(JOINT_NAMES):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if joint_id < 0 or actuator_id < 0:
                raise RuntimeError(f"MuJoCo model has no {name!r} joint/actuator")
            if (
                not model.jnt_limited[joint_id]
                or not model.actuator_ctrllimited[actuator_id]
                or not np.allclose(
                    model.jnt_range[joint_id], expected[index], rtol=0, atol=1e-7
                )
                or not np.allclose(
                    model.actuator_ctrlrange[actuator_id],
                    expected[index],
                    rtol=0,
                    atol=1e-7,
                )
            ):
                raise RuntimeError(f"MuJoCo limits differ from alignment for {name}")
