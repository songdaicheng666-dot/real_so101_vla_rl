"""MuJoCo task-state extraction and privileged observations."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from numpy.typing import NDArray

BATTERY_COLORS = ("red", "yellow", "blue", "green")
BATTERY_BODY_NAMES = tuple(f"battery_{color}" for color in BATTERY_COLORS)


@dataclass(frozen=True, slots=True)
class BatteryState:
    color: str
    position: NDArray[np.float64]
    quaternion: NDArray[np.float64]
    linear_velocity: NDArray[np.float64]
    angular_velocity: NDArray[np.float64]
    axis: NDArray[np.float64]
    fixed_jaw_contact: bool
    moving_jaw_contact: bool
    board_contact: bool


@dataclass(frozen=True, slots=True)
class TaskState:
    robot_qpos: NDArray[np.float64]
    robot_qvel: NDArray[np.float64]
    tcp_position: NDArray[np.float64]
    tcp_orientation_6d: NDArray[np.float64]
    batteries: tuple[BatteryState, ...]
    target_color: str
    target_center_xy: NDArray[np.float64]
    target_inside: bool

    @property
    def target(self) -> BatteryState:
        return next(battery for battery in self.batteries if battery.color == self.target_color)


def cylinder_projection_extent(
    axis: NDArray[np.floating],
    *,
    radius: float,
    half_length: float,
) -> NDArray[np.float64]:
    """Return the XY support extent of an arbitrarily oriented cylinder."""

    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    radial = np.sqrt(np.maximum(0.0, 1.0 - axis[:2] ** 2))
    return half_length * np.abs(axis[:2]) + radius * radial


class PrivilegedStateBuilder:
    """Extract exact simulator state for reward/debugging and the MLP baseline."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        target_center_xy: tuple[float, float],
        target_size_xy: tuple[float, float],
        battery_radius: float,
        battery_half_length: float,
    ) -> None:
        self.model = model
        self.target_center_xy = np.asarray(target_center_xy, dtype=np.float64)
        self.target_half_size_xy = 0.5 * np.asarray(target_size_xy, dtype=np.float64)
        self.battery_radius = battery_radius
        self.battery_half_length = battery_half_length

        self.robot_joint_ids = np.asarray(
            [self._id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in (
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "gripper",
            )],
            dtype=np.int32,
        )
        self.robot_qpos_addresses = model.jnt_qposadr[self.robot_joint_ids].copy()
        self.robot_dof_addresses = model.jnt_dofadr[self.robot_joint_ids].copy()
        self.tcp_site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.board_geom_id = self._id(mujoco.mjtObj.mjOBJ_GEOM, "board_collision")
        self.fixed_jaw_body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "gripper")
        self.moving_jaw_body_id = self._id(
            mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1"
        )
        self.battery_body_ids = {
            color: self._id(mujoco.mjtObj.mjOBJ_BODY, f"battery_{color}")
            for color in BATTERY_COLORS
        }
        self.battery_joint_ids = {
            color: self._id(mujoco.mjtObj.mjOBJ_JOINT, f"battery_{color}_joint")
            for color in BATTERY_COLORS
        }

        # robot qpos/qvel + TCP pos/rotation6d + 4 battery states + target one-hot
        # + target-to-T0 delta/inside + 3 contact flags for every battery.
        self.observation_size = 6 + 6 + 3 + 6 + 4 * 13 + 4 + 3 + 4 * 3

    def _id(self, object_type: int, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise RuntimeError(f"required MuJoCo object is missing: {name}")
        return object_id

    def extract(
        self,
        data: mujoco.MjData,
        *,
        target_color: str,
    ) -> TaskState:
        contact_flags = {
            color: {"fixed": False, "moving": False, "board": False}
            for color in BATTERY_COLORS
        }
        body_to_color = {body_id: color for color, body_id in self.battery_body_ids.items()}
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            geom_1 = int(contact.geom1)
            geom_2 = int(contact.geom2)
            body_1 = int(self.model.geom_bodyid[geom_1])
            body_2 = int(self.model.geom_bodyid[geom_2])
            for battery_body, other_body, other_geom in (
                (body_1, body_2, geom_2),
                (body_2, body_1, geom_1),
            ):
                color = body_to_color.get(battery_body)
                if color is None:
                    continue
                if other_body == self.fixed_jaw_body_id:
                    contact_flags[color]["fixed"] = True
                if other_body == self.moving_jaw_body_id:
                    contact_flags[color]["moving"] = True
                if other_geom == self.board_geom_id:
                    contact_flags[color]["board"] = True

        batteries = []
        for color in BATTERY_COLORS:
            body_id = self.battery_body_ids[color]
            joint_id = self.battery_joint_ids[color]
            dof_address = int(self.model.jnt_dofadr[joint_id])
            rotation = data.xmat[body_id].reshape(3, 3)
            flags = contact_flags[color]
            batteries.append(
                BatteryState(
                    color=color,
                    position=data.xpos[body_id].astype(np.float64, copy=True),
                    quaternion=data.xquat[body_id].astype(np.float64, copy=True),
                    linear_velocity=data.qvel[dof_address : dof_address + 3].astype(
                        np.float64, copy=True
                    ),
                    angular_velocity=data.qvel[dof_address + 3 : dof_address + 6].astype(
                        np.float64, copy=True
                    ),
                    axis=rotation[:, 2].astype(np.float64, copy=True),
                    fixed_jaw_contact=flags["fixed"],
                    moving_jaw_contact=flags["moving"],
                    board_contact=flags["board"],
                )
            )

        target = next(battery for battery in batteries if battery.color == target_color)
        extent = cylinder_projection_extent(
            target.axis,
            radius=self.battery_radius,
            half_length=self.battery_half_length,
        )
        inside = bool(
            np.all(
                np.abs(target.position[:2] - self.target_center_xy) + extent
                <= self.target_half_size_xy
            )
        )
        tcp_rotation = data.site_xmat[self.tcp_site_id].reshape(3, 3)
        return TaskState(
            robot_qpos=data.qpos[self.robot_qpos_addresses].astype(np.float64, copy=True),
            robot_qvel=data.qvel[self.robot_dof_addresses].astype(np.float64, copy=True),
            tcp_position=data.site_xpos[self.tcp_site_id].astype(np.float64, copy=True),
            tcp_orientation_6d=tcp_rotation[:, :2].reshape(-1).astype(
                np.float64, copy=True
            ),
            batteries=tuple(batteries),
            target_color=target_color,
            target_center_xy=self.target_center_xy.copy(),
            target_inside=inside,
        )

    def observation(self, state: TaskState) -> NDArray[np.float32]:
        values: list[float] = []
        values.extend(state.robot_qpos)
        values.extend(state.robot_qvel)
        values.extend(state.tcp_position)
        values.extend(state.tcp_orientation_6d)
        for battery in state.batteries:
            values.extend(battery.position)
            values.extend(battery.quaternion)
            values.extend(battery.linear_velocity)
            values.extend(battery.angular_velocity)
        values.extend(float(color == state.target_color) for color in BATTERY_COLORS)
        values.extend(state.target.position[:2] - state.target_center_xy)
        values.append(float(state.target_inside))
        for battery in state.batteries:
            values.extend(
                (
                    float(battery.fixed_jaw_contact),
                    float(battery.moving_jaw_contact),
                    float(battery.board_contact),
                )
            )
        observation = np.asarray(values, dtype=np.float32)
        if observation.shape != (self.observation_size,):
            raise RuntimeError(
                f"privileged observation has unexpected shape {observation.shape}"
            )
        return observation
