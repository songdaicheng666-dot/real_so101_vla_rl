"""Record synchronized SO-101 demonstrations directly as schema-v2 LeRobotDataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/recording/so101_schema_v2.yaml",
        help="project recording YAML",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="validate structure without requiring unresolved hardware identifiers",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="append to an interrupted dataset after validating its plan prefix",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Optional LeRobot imports can emit warnings and install a root handler.
    # Configure logging first, and force our INFO level so operator prompts are
    # never hidden by an import-time warning handler.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    from real_so101_vla_rl.recording import (
        load_recording_config,
        validate_recording_root_mode,
    )

    config = load_recording_config(
        args.config, require_hardware_ready=not args.check_config
    )
    if args.check_config:
        print(
            f"Schema-v2 recording config is structurally valid: {args.config}; "
            f"camera_rig={config.cameras.rig.profile.rig_id}"
        )
        return
    validate_recording_root_mode(config, resume=args.resume)

    try:
        from lerobot.configs.video import RGBEncoderConfig
        from lerobot.datasets import LeRobotDataset, VideoEncodingManager
        from lerobot.processor import make_default_processors
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
        from lerobot.utils.keyboard_input import create_key_listener
    except ImportError as exc:
        raise SystemExit(
            "Real recording requires LeRobot dataset and hardware support. "
            "The project hardware extra will be frozen after device identification."
        ) from exc

    from real_so101_vla_rl.data import build_so101_lerobot_features
    from real_so101_vla_rl.hardware import (
        SO101ObservationSynchronizer,
        SO101StateAdapter,
        connect_calibrated_so101_follower,
        connect_calibrated_so101_leader,
    )
    from real_so101_vla_rl.hardware.cameras import (
        OpenCVRGBAdapter,
        OrbbecRGBDAdapter,
        probe_camera_rig,
        require_valid_camera_probe,
    )
    from real_so101_vla_rl.recording.camera_metadata import (
        write_camera_setup_metadata,
    )
    from real_so101_vla_rl.recording.dataset_lifecycle import (
        validate_resumed_recording_dataset,
        write_recording_robot_profile,
    )
    from real_so101_vla_rl.recording.lerobot_v2 import (
        RecordingControls,
        run_recording_session,
    )

    camera_probe = probe_camera_rig(
        config.cameras.rig,
        check_streams=True,
        duration_s=10.0,
        timeout_ms=max(2000, config.sync.timeout_ms),
        timestamp_tolerance_ms=config.sync.tolerance_ms,
    )
    if not camera_probe.valid:
        logger.error("Camera preflight report:\n%s", camera_probe.to_json())
    require_valid_camera_probe(camera_probe)
    logger.info("Camera preflight passed:\n%s", camera_probe.to_json())
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
    leader = SO101Leader(
        SO101LeaderConfig(
            port=config.leader.port,
            id=config.leader.robot_id,
            calibration_dir=(
                Path(config.leader.calibration_dir)
                if config.leader.calibration_dir
                else None
            ),
            use_degrees=True,
        )
    )
    state_adapter = SO101StateAdapter(
        follower,
        max_consecutive_failures=config.sync.max_consecutive_failures,
    )
    overview_adapter = OrbbecRGBDAdapter.from_profile(
        config.cameras.rig,
        max_consecutive_failures=config.sync.max_consecutive_failures,
    )
    wrist_adapter = OpenCVRGBAdapter.from_profile(
        config.cameras.rig,
        max_consecutive_failures=config.sync.max_consecutive_failures,
    )
    synchronizer = SO101ObservationSynchronizer(
        overview_adapter,
        wrist_adapter,
        state_adapter,
        tolerance_ms=config.sync.tolerance_ms,
        timeout_ms=config.sync.timeout_ms,
    )
    controls = RecordingControls()
    listener = create_key_listener(
        controls.dispatch,
        controls_help=(
            "Enter/s=ready or save, Right/n=end, Left/r=discard, Esc/q=stop"
        ),
    )
    if listener is None:
        raise SystemExit("An interactive keyboard is required to confirm recorded episodes")

    dataset = None
    try:
        connect_calibrated_so101_leader(leader)
        follower_connection = connect_calibrated_so101_follower(follower)
        logger.info(
            "Follower connected with current-position hold: %s",
            follower_connection.present_position_raw,
        )
        synchronizer.connect()
        features = build_so101_lerobot_features(
            rgb_use_videos=config.dataset.rgb_use_videos
        )
        common_writer_kwargs = {
            "repo_id": config.dataset.repo_id,
            "root": Path(config.dataset.root),
            "image_writer_threads": 12,
            # DatasetWriter validates its encoder configs even for image-backed data.
            "rgb_encoder": RGBEncoderConfig(vcodec="h264"),
        }
        if args.resume:
            dataset = LeRobotDataset.resume(**common_writer_kwargs)
            try:
                validate_resumed_recording_dataset(config, dataset)
            except BaseException:
                dataset.finalize()
                raise
            logger.info(
                "Resuming accepted episode %d of %d",
                dataset.num_episodes + 1,
                config.dataset.num_episodes,
            )
        else:
            dataset = LeRobotDataset.create(
                **common_writer_kwargs,
                fps=config.dataset.fps,
                robot_type="so101_follower",
                features=features,
                use_videos=config.dataset.rgb_use_videos,
            )
        write_camera_setup_metadata(
            config.dataset.root,
            config.cameras.rig,
            camera_probe,
        )
        write_recording_robot_profile(config, follower)
        teleop_processor, robot_processor, _ = make_default_processors()
        with VideoEncodingManager(dataset):
            summaries = run_recording_session(
                config,
                robot=follower,
                teleop=leader,
                synchronizer=synchronizer,
                state_adapter=state_adapter,
                dataset=dataset,
                controls=controls,
                teleop_action_processor=teleop_processor,
                robot_action_processor=robot_processor,
            )
        accepted = sum(summary.accepted for summary in summaries)
        aborted = sum(summary.aborted for summary in summaries)
        print(f"Recording complete: accepted={accepted}, capture_aborted={aborted}")
    finally:
        listener.stop()
        synchronizer.disconnect()
        if follower.is_connected:
            follower.disconnect()
        if leader.is_connected:
            leader.disconnect()


if __name__ == "__main__":
    main()
