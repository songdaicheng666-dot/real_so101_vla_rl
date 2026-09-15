"""Record synchronized SO-101 demonstrations directly as schema-v2 LeRobotDataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from real_so101_vla_rl.recording import load_recording_config

    config = load_recording_config(
        args.config, require_hardware_ready=not args.check_config
    )
    if args.check_config:
        print(f"Schema-v2 recording config is structurally valid: {args.config}")
        return

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
    )
    from real_so101_vla_rl.hardware.cameras import (
        OpenCVRGBAdapter,
        OrbbecRGBDAdapter,
    )
    from real_so101_vla_rl.recording.lerobot_v2 import (
        RecordingControls,
        run_recording_session,
    )

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
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
    overview_adapter = OrbbecRGBDAdapter.from_sdk(
        serial_number=config.cameras.orbbec_serial,
        max_consecutive_failures=config.sync.max_consecutive_failures,
    )
    wrist_adapter = OpenCVRGBAdapter.from_device(
        config.cameras.wrist_device,
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
            "Right/n=end, Enter/s=save, Left/r=discard, Esc/q=stop"
        ),
    )
    if listener is None:
        raise SystemExit("An interactive keyboard is required to confirm recorded episodes")

    dataset = None
    try:
        leader.connect()
        follower.connect()
        synchronizer.connect()
        features = build_so101_lerobot_features(
            rgb_use_videos=config.dataset.rgb_use_videos
        )
        create_kwargs = {
            "repo_id": config.dataset.repo_id,
            "fps": config.dataset.fps,
            "root": Path(config.dataset.root),
            "robot_type": "so101_follower",
            "features": features,
            "use_videos": config.dataset.rgb_use_videos,
            "image_writer_threads": 12,
            # DatasetWriter validates its encoder configs even for image-backed data.
            "rgb_encoder": RGBEncoderConfig(vcodec="h264"),
        }
        dataset = LeRobotDataset.create(**create_kwargs)
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
