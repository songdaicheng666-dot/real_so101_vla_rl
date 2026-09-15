"""Run the project LeRobotDataset-to-OpenVLA-OFT SFT pipeline."""

from __future__ import annotations

import argparse
import os
from dataclasses import replace

from real_so101_vla_rl.models import load_sft_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/sft/openvla_oft_synthetic_overfit.yaml",
    )
    parser.add_argument("--dataset-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--checkpoint",
        help="Reload a saved checkpoint for a preflight forward pass",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.checkpoint and not args.preflight_only:
        raise ValueError("--checkpoint currently requires --preflight-only")
    os.environ.setdefault("OPENVLA_ROBOT_PLATFORM", "SO101")
    if args.local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    config = load_sft_config(args.config)
    if args.dataset_root:
        config = replace(
            config,
            dataset=replace(config.dataset, root=args.dataset_root),
        )
    if args.output_dir:
        config = replace(
            config,
            training=replace(
                config.training, output_dir=args.output_dir
            ),
        )
    config.validate_training_ready()

    from real_so101_vla_rl.models.openvla_oft_training import (
        run_training,
    )

    run_training(
        config,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        config_path=args.config,
        preflight_only=args.preflight_only,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
