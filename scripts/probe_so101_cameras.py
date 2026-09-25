"""Probe the frozen SO-101 dual-camera rig and emit a JSON report."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        default="configs/hardware/cameras/so101_competition_2026.yaml",
        help="camera rig profile YAML",
    )
    parser.add_argument(
        "--check-streams",
        action="store_true",
        help="open both cameras and acquire frames concurrently",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=1.0,
        help="stream-check duration; use 600 for the hardware soak test",
    )
    parser.add_argument("--timeout-ms", type=int, default=2000)
    parser.add_argument("--timestamp-tolerance-ms", type=float, default=25.0)
    parser.add_argument(
        "--report",
        type=Path,
        help="optional JSON output path; the report is always printed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from real_so101_vla_rl.hardware.cameras import (
        load_camera_rig_profile,
        probe_camera_rig,
        write_camera_probe_report,
    )

    loaded = load_camera_rig_profile(args.profile)
    report = probe_camera_rig(
        loaded,
        check_streams=args.check_streams,
        duration_s=args.duration_s,
        timeout_ms=args.timeout_ms,
        timestamp_tolerance_ms=args.timestamp_tolerance_ms,
    )
    print(report.to_json())
    if args.report is not None:
        write_camera_probe_report(args.report, report)
    if not report.valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
