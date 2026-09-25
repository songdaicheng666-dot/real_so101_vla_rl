"""Validate and finalize a recorded SO-101 LeRobotDataset for SFT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recording-config",
        default="configs/recording/so101_real_pilot_lowlight_v2.yaml",
    )
    parser.add_argument(
        "--sft-config",
        default="configs/sft/openvla_oft_real_pilot_lowlight.yaml",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate an incomplete or complete recording without writing SFT metadata",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from real_so101_vla_rl.models import load_sft_config
    from real_so101_vla_rl.recording import load_recording_config
    from real_so101_vla_rl.recording.finalization import (
        file_sha256,
        finalize_recorded_so101_dataset,
    )

    recording_path = Path(args.recording_config)
    sft_path = Path(args.sft_config)
    recording_config = load_recording_config(
        recording_path,
        require_hardware_ready=False,
    )
    sft_config = load_sft_config(sft_path)
    summary = finalize_recorded_so101_dataset(
        recording_config,
        sft_config,
        validate_only=args.validate_only,
        recording_config_sha256=file_sha256(recording_path),
        sft_config_sha256=file_sha256(sft_path),
    )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
