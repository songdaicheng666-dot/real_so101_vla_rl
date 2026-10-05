#!/usr/bin/env python3
"""Read-only SO-101/MuJoCo alignment capture and reporting utilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from real_so101_vla_rl.alignment import (
    ALIGNMENT_PATH,
    OverviewSensorModel,
    load_alignment,
    real_state_to_mujoco_qpos,
    reset_to_home_keyframe,
    validate_mujoco_joint_refs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = (
    PROJECT_ROOT
    / "src"
    / "real_so101_vla_rl"
    / "assets"
    / "mujoco"
    / "competition_2026"
    / "scene_basic_t0.xml"
)
DEFAULT_RECORDING_CONFIG = (
    PROJECT_ROOT / "configs" / "recording" / "so101_t0_100_lowlight_v1.yaml"
)
DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "so101_t0_100_lowlight_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _capture(args: argparse.Namespace) -> None:
    """Capture without configuring servos, changing torque, or writing targets."""

    try:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ImportError as exc:
        raise SystemExit("capture requires the project's hardware environment") from exc

    from real_so101_vla_rl.data.schema import ordered_joint_vector
    from real_so101_vla_rl.hardware.cameras import OrbbecRGBDAdapter
    from real_so101_vla_rl.recording import load_recording_config

    config = load_recording_config(args.config, require_hardware_ready=True)
    follower = SO101Follower(
        SO101FollowerConfig(
            port=config.follower.port,
            id=config.follower.robot_id,
            calibration_dir=(
                Path(config.follower.calibration_dir)
                if config.follower.calibration_dir
                else None
            ),
            cameras={},
            use_degrees=True,
            max_relative_target=config.max_relative_target,
        )
    )
    overview = OrbbecRGBDAdapter.from_profile(config.cameras.rig)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    bus_connected = False
    try:
        # Deliberately bypass follower.connect(): that path configures the arm.
        follower.bus.connect()
        bus_connected = True
        if not bool(follower.is_calibrated):
            raise RuntimeError("attached follower does not match its calibration")
        overview.connect()
        frame = overview.read(args.timeout_ms)
        if not frame.overview.valid or not frame.depth.valid:
            raise RuntimeError(frame.overview.error or frame.depth.error or "RGB-D failed")
        observation = follower.get_observation()
        state = np.asarray(ordered_joint_vector(observation), dtype=np.float64)
        present_raw = follower.bus.sync_read(
            "Present_Position", normalize=False, num_retry=2
        )
    finally:
        overview.disconnect()
        if bus_connected:
            # Never disable/enable torque: leave the pre-existing hardware state intact.
            follower.bus.disconnect(disable_torque=False)

    rgb = np.asarray(frame.overview.value, dtype=np.uint8)
    depth = np.asarray(frame.depth.value, dtype=np.uint16)
    Image.fromarray(rgb).save(output / "overview_rgb.png")
    Image.fromarray(depth[..., 0]).save(output / "overview_depth_mm.png")
    np.save(output / "overview_depth_mm.npy", depth)
    metadata = {
        "capture_mode": "read_only",
        "writes_goal_position": False,
        "changes_torque": False,
        "alignment_id": load_alignment()["alignment_id"],
        "timestamp_utc_ns": time.time_ns(),
        "overview_timestamp_ns": frame.overview.timestamp_ns,
        "depth_timestamp_ns": frame.depth.timestamp_ns,
        "state_deg_or_percent": state.tolist(),
        "present_position_raw": dict(present_raw),
        "camera_profile_sha256": config.cameras.rig.sha256,
        "factory_calibration_sha256": config.cameras.rig.calibration_sha256,
        "rgb_sha256": _sha256(output / "overview_rgb.png"),
        "depth_npy_sha256": _sha256(output / "overview_depth_mm.npy"),
    }
    _write_json(output / "capture.json", metadata)
    print(output / "capture.json")


def _farthest_pose_indices(states: np.ndarray, count: int, home: np.ndarray) -> list[int]:
    ranges = np.ptp(states, axis=0)
    ranges[ranges < 1e-9] = 1.0
    normalized = states / ranges
    seed = int(np.argmin(np.linalg.norm((states - home) / ranges, axis=1)))
    selected = [seed]
    minimum_distance = np.linalg.norm(normalized - normalized[seed], axis=1)
    while len(selected) < min(count, len(states)):
        index = int(np.argmax(minimum_distance))
        selected.append(index)
        distance = np.linalg.norm(normalized - normalized[index], axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[selected] = -1.0
    return selected


def _select_poses(args: argparse.Namespace) -> None:
    try:
        from pyarrow import parquet
    except ImportError as exc:
        raise SystemExit("select-poses requires pyarrow/LeRobot") from exc

    dataset = args.dataset.resolve()
    rows: list[dict[str, Any]] = []
    for path in sorted((dataset / "data").rglob("*.parquet")):
        table = parquet.read_table(
            path,
            columns=["observation.state", "episode_index", "frame_index", "index"],
        )
        rows.extend(table.to_pylist())
    if not rows:
        raise RuntimeError(f"no dataset rows found under {dataset}")
    states = np.asarray([row["observation.state"] for row in rows], dtype=np.float64)
    home = np.asarray(
        load_alignment()["robot"]["current_real_home_deg_or_percent"],
        dtype=np.float64,
    )
    chosen = _farthest_pose_indices(states, args.count, home)
    manifest = {
        "alignment_id": load_alignment()["alignment_id"],
        "dataset": str(dataset),
        "selection": "normalized_joint_state_farthest_point",
        "count": len(chosen),
        "frames": [
            {
                "episode_index": int(rows[index]["episode_index"]),
                "frame_index": int(rows[index]["frame_index"]),
                "dataset_index": int(rows[index]["index"]),
                "state_deg_or_percent": states[index].tolist(),
            }
            for index in chosen
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, manifest)
    print(args.output)


def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    quaternion = np.empty(4, dtype=np.float64)
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion[:] = (
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        following = (index + 1) % 3
        remaining = (index + 2) % 3
        scale = np.sqrt(
            1.0
            + matrix[index, index]
            - matrix[following, following]
            - matrix[remaining, remaining]
        ) * 2.0
        quaternion[index + 1] = 0.25 * scale
        quaternion[0] = (
            matrix[remaining, following] - matrix[following, remaining]
        ) / scale
        quaternion[following + 1] = (
            matrix[following, index] + matrix[index, following]
        ) / scale
        quaternion[remaining + 1] = (
            matrix[remaining, index] + matrix[index, remaining]
        ) / scale
    return quaternion if quaternion[0] >= 0 else -quaternion


def _solve_pnp(args: argparse.Namespace) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("solve-pnp requires OpenCV") from exc

    points = json.loads(args.points.read_text(encoding="utf-8"))
    world = np.asarray(points["world_points_m"], dtype=np.float64)
    pixels = np.asarray(points["image_points_px"], dtype=np.float64)
    if world.shape[0] < 8 or world.shape != (pixels.shape[0], 3):
        raise ValueError("PnP requires at least eight paired 3D/2D boundary points")
    camera = load_alignment(args.alignment)["overview_camera"]
    fx, fy = camera["focalpixel"]
    cx, cy = camera["principalpixel"]
    intrinsic = np.asarray(((fx, 0, cx), (0, fy, cy), (0, 0, 1)), dtype=np.float64)
    distortion = np.asarray(camera["distortion_opencv"], dtype=np.float64)
    ok, rotation_vector, translation = cv2.solvePnP(
        world,
        pixels,
        intrinsic,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("solvePnP failed")
    rotation_cv, _ = cv2.Rodrigues(rotation_vector)
    position_world = (-rotation_cv.T @ translation).reshape(3)
    # OpenCV x-right/y-down/z-forward -> MuJoCo x-right/y-up/z-back.
    rotation_world_mujoco = rotation_cv.T @ np.diag((1.0, -1.0, -1.0))
    quaternion = _matrix_to_quaternion(rotation_world_mujoco)
    projected, _ = cv2.projectPoints(
        world, rotation_vector, translation, intrinsic, distortion
    )
    errors = np.linalg.norm(projected[:, 0, :] - pixels, axis=1)
    result = {
        "method": "opencv_solvepnp_iterative_planar",
        "point_count": len(world),
        "world_position_m": position_world.tolist(),
        "world_quaternion_wxyz": quaternion.tolist(),
        "mean_reprojection_error_px": float(errors.mean()),
        "max_reprojection_error_px": float(errors.max()),
        "per_point_error_px": errors.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, result)
    print(args.output)


def _static_paper_mask(real: np.ndarray, simulated: np.ndarray) -> np.ndarray:
    import cv2

    yy, xx = np.indices(real.shape[:2])
    sim_spread = simulated.max(axis=2).astype(np.int16) - simulated.min(axis=2)
    real_spread = real.max(axis=2).astype(np.int16) - real.min(axis=2)
    paper = (simulated.min(axis=2) > 80) & (sim_spread < 15)
    paper &= real_spread < 35
    paper &= yy >= int(real.shape[0] * 0.15)
    paper &= ~(
        (yy >= int(real.shape[0] * 0.70))
        & (xx >= int(real.shape[1] * 0.25))
        & (xx <= int(real.shape[1] * 0.86))
    )

    def gradient_magnitude(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        return cv2.magnitude(dx, dy)

    paper &= gradient_magnitude(simulated) < 20
    paper &= gradient_magnitude(real) < 30
    return cv2.erode(paper.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def _comparison_metrics(
    real: np.ndarray, simulated: np.ndarray
) -> dict[str, float | int]:
    import cv2

    difference = np.abs(real.astype(np.float32) - simulated.astype(np.float32))
    static = _static_paper_mask(real, simulated)
    if int(static.sum()) < 1000:
        raise RuntimeError("fewer than 1000 static paper pixels remain for comparison")
    real_lab = cv2.cvtColor(real.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    sim_lab = cv2.cvtColor(simulated.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    delta_lab = np.linalg.norm(real_lab - sim_lab, axis=2)
    return {
        "mean_absolute_rgb_255": float(difference.mean()),
        "static_paper_pixel_count": int(static.sum()),
        "static_brightness_mean_error_255": float(
            abs(real[static].mean() - simulated[static].mean())
        ),
        "static_lab_median_delta": float(np.median(delta_lab[static])),
    }


def _robot_silhouette_metrics(
    real_depth_mm: np.ndarray,
    real: np.ndarray,
    simulated: np.ndarray,
    simulated_robot: np.ndarray,
) -> dict[str, float | int]:
    import cv2

    depth = np.asarray(real_depth_mm)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[..., 0]
    if depth.shape != real.shape[:2]:
        raise ValueError(f"real depth shape {depth.shape} differs from RGB {real.shape[:2]}")
    yy, xx = np.indices(depth.shape)
    valid = depth > 0
    fit_mask = _static_paper_mask(real, simulated) & valid
    if int(fit_mask.sum()) < 1000:
        raise RuntimeError("fewer than 1000 paper-depth pixels remain for plane fit")
    design = np.column_stack((xx[fit_mask], yy[fit_mask], np.ones(fit_mask.sum())))
    coefficients = np.linalg.lstsq(design, depth[fit_mask], rcond=None)[0]
    board_depth = coefficients[0] * xx + coefficients[1] * yy + coefficients[2]
    roi = (
        (yy >= int(depth.shape[0] * 0.70))
        & (xx >= int(depth.shape[1] * 0.25))
        & (xx <= int(depth.shape[1] * 0.86))
    )
    real_robot = valid & ((board_depth - depth) > 6.0) & roi
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    real_robot = cv2.morphologyEx(
        real_robot.astype(np.uint8), cv2.MORPH_OPEN, kernel
    )
    real_robot = cv2.morphologyEx(real_robot, cv2.MORPH_CLOSE, kernel).astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        real_robot.astype(np.uint8), 8
    )
    keep = [
        index
        for index in range(1, count)
        if stats[index, cv2.CC_STAT_AREA] >= 64
    ]
    real_robot = np.isin(labels, keep)
    simulated_robot = np.asarray(simulated_robot, dtype=bool) & roi
    intersection = int(np.count_nonzero(real_robot & simulated_robot))
    union = int(np.count_nonzero(real_robot | simulated_robot))
    return {
        "robot_silhouette_iou": float(intersection / union) if union else 0.0,
        "real_robot_mask_pixels": int(real_robot.sum()),
        "sim_robot_mask_pixels": int(simulated_robot.sum()),
        "robot_depth_above_board_threshold_mm": 6.0,
    }


def _report(args: argparse.Namespace) -> None:
    import mujoco

    alignment = load_alignment(args.alignment)
    state_record = json.loads(args.state.read_text(encoding="utf-8"))
    if isinstance(state_record, dict) and "frames" in state_record:
        state_record = state_record["frames"][0]
    state = np.asarray(
        (
            state_record.get("state_deg_or_percent", state_record)
            if isinstance(state_record, dict)
            else state_record
        ),
        dtype=np.float64,
    )
    qpos = real_state_to_mujoco_qpos(state, alignment=alignment)
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    validate_mujoco_joint_refs(model, alignment=alignment)
    data = mujoco.MjData(model)
    reset_to_home_keyframe(model, data)
    data.qpos[:6] = qpos
    data.ctrl[:] = qpos
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=480, width=640)
    try:
        renderer.update_scene(data, camera="overview")
        raw = renderer.render().copy()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera="overview")
        segmentation = renderer.render().copy()
    finally:
        renderer.close()
    sensor_model = OverviewSensorModel(alignment)
    styled = sensor_model.apply(raw, seed=args.seed, noise=False)
    real = np.asarray(Image.open(args.real_rgb).convert("RGB"), dtype=np.uint8)
    if real.shape != styled.shape:
        raise ValueError(f"real RGB shape {real.shape} differs from {styled.shape}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.real_rgb, output / "real.png")
    Image.fromarray(raw).save(output / "sim_raw.png")
    Image.fromarray(styled).save(output / "sim_styled.png")
    overlay = np.rint(0.5 * real + 0.5 * styled).astype(np.uint8)
    difference = np.abs(real.astype(np.int16) - styled.astype(np.int16)).astype(np.uint8)
    Image.fromarray(overlay).save(output / "overlay.png")
    Image.fromarray(difference).save(output / "difference.png")
    comparison = _comparison_metrics(real, styled)
    depth_path = args.real_depth
    if depth_path is None:
        inferred_depth = args.real_rgb.with_name("overview_depth_mm.npy")
        depth_path = inferred_depth if inferred_depth.is_file() else None
    if depth_path is not None:
        base_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        robot_bodies = {base_body}
        for body_id in range(1, model.nbody):
            if int(model.body_parentid[body_id]) in robot_bodies:
                robot_bodies.add(body_id)
        robot_geoms = np.asarray(
            [
                geom_id
                for geom_id in range(model.ngeom)
                if int(model.geom_bodyid[geom_id]) in robot_bodies
            ],
            dtype=np.int32,
        )
        raw_robot = (
            segmentation[..., 1] == mujoco.mjtObj.mjOBJ_GEOM
        ) & np.isin(segmentation[..., 0], robot_geoms)
        simulated_robot = sensor_model.distort_mask(raw_robot)
        comparison.update(
            _robot_silhouette_metrics(
                np.load(depth_path), real, styled, simulated_robot
            )
        )
        comparison["silhouette_depth_source"] = str(depth_path)
    metrics = {
        "alignment_id": alignment["alignment_id"],
        "alignment_sha256": _sha256(args.alignment),
        "state_deg_or_percent": state.tolist(),
        "mujoco_qpos_rad": qpos.tolist(),
        "sensor_noise": False,
        **comparison,
    }
    _write_json(output / "metrics.json", metrics)
    print(output / "metrics.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    capture = commands.add_parser("capture", help="read-only RGB-D and follower capture")
    capture.add_argument("--config", type=Path, default=DEFAULT_RECORDING_CONFIG)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--timeout-ms", type=int, default=2000)
    capture.set_defaults(run=_capture)

    select = commands.add_parser("select-poses", help="joint-space farthest-point sampling")
    select.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    select.add_argument("--count", type=int, default=16)
    select.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "calibration" / "pose_manifest.json",
    )
    select.set_defaults(run=_select_poses)

    pnp = commands.add_parser("solve-pnp", help="solve overview extrinsics from boundaries")
    pnp.add_argument("--points", type=Path, required=True)
    pnp.add_argument("--alignment", type=Path, default=ALIGNMENT_PATH)
    pnp.add_argument("--output", type=Path, required=True)
    pnp.set_defaults(run=_solve_pnp)

    report = commands.add_parser("report", help="write real/sim/overlay/difference report")
    report.add_argument("--real-rgb", type=Path, required=True)
    report.add_argument(
        "--real-depth",
        type=Path,
        help="optional aligned depth .npy; inferred beside overview_rgb.png",
    )
    report.add_argument("--state", type=Path, required=True)
    report.add_argument("--alignment", type=Path, default=ALIGNMENT_PATH)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--seed", type=int, default=0)
    report.set_defaults(run=_report)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
