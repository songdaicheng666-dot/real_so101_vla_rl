"""Create the deterministic LeRobotDataset used by the cloud SFT test."""

from __future__ import annotations

import argparse
import json

from real_so101_vla_rl.data.synthetic_dataset import (
    DEFAULT_SYNTHETIC_REPO_ID,
    create_synthetic_so101_dataset,
    summary_to_dict,
    validate_synthetic_lerobot_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--repo-id", default=DEFAULT_SYNTHETIC_REPO_ID
    )
    parser.add_argument(
        "--rgb-image-backed",
        action="store_true",
        help=(
            "Store the two RGB streams as PNG instead of MP4; aligned depth "
            "always remains lossless TIFF."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = create_synthetic_so101_dataset(
        args.root,
        repo_id=args.repo_id,
        rgb_use_videos=not args.rgb_image_backed,
    )
    contract = validate_synthetic_lerobot_dataset(
        args.root, repo_id=args.repo_id
    )
    print(
        json.dumps(
            {
                "dataset": summary_to_dict(summary),
                "contract": contract,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
