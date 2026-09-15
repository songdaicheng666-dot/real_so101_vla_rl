"""Create patched working copies of the pinned OpenVLA-OFT dependencies."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

OPENVLA_OFT_COMMIT = "e4287e94541f459edc4feabc4e181f537cd569a8"
TRANSFORMERS_COMMIT = "bc339d9ad707454c0c115970db43c260067c61ab"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openvla-source", type=Path, required=True)
    parser.add_argument("--transformers-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def git_head(source: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def require_commit(source: Path, expected: str) -> None:
    actual = git_head(source)
    if actual != expected:
        raise RuntimeError(
            f"{source} must be checked out at {expected}, got {actual}"
        )


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected one patch target in {path}, found {count}: {old!r}"
        )
    path.write_text(text.replace(old, new, 1))


def copy_source(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(
            f"Compatibility destination already exists: {destination}"
        )
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
    )


def patch_openvla(source: Path) -> None:
    init_path = source / "prismatic" / "vla" / "__init__.py"
    replace_once(
        init_path,
        "from .materialize import get_vla_dataset_and_collator\n",
        '"""VLA model components.\n\n'
        "Dataset materialization is intentionally imported only by callers that use "
        "the\nupstream RLDS input pipeline. The SO-101 project supplies a PyTorch "
        "Dataset and\ndoes not install TensorFlow or dlimp.\n"
        '"""\n',
    )

    constants_path = source / "prismatic" / "vla" / "constants.py"
    replace_once(constants_path, "import sys\n", "import os\nimport sys\n")
    replace_once(
        constants_path,
        "\n\n\n# Function to detect robot platform from command line arguments\n",
        "\n\nSO101_CONSTANTS = {\n"
        '    "NUM_ACTIONS_CHUNK": 8,\n'
        '    "ACTION_DIM": 6,\n'
        '    "PROPRIO_DIM": 6,\n'
        '    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,\n'
        "}\n\n\n"
        "# Function to detect robot platform from command line arguments\n",
    )
    replace_once(
        constants_path,
        "def detect_robot_platform():\n    cmd_args = \" \".join(sys.argv).lower()\n",
        "def detect_robot_platform():\n"
        '    requested_platform = os.environ.get("OPENVLA_ROBOT_PLATFORM")\n'
        "    if requested_platform is not None:\n"
        "        requested_platform = requested_platform.strip().upper()\n"
        '        supported_platforms = {"LIBERO", "ALOHA", "BRIDGE", "SO101"}\n'
        "        if requested_platform not in supported_platforms:\n"
        "            raise ValueError(\n"
        '                "OPENVLA_ROBOT_PLATFORM must be one of "\n'
        '                f"{sorted(supported_platforms)}, got {requested_platform!r}"\n'
        "            )\n"
        "        return requested_platform\n\n"
        '    cmd_args = " ".join(sys.argv).lower()\n',
    )
    replace_once(
        constants_path,
        '    if "libero" in cmd_args:\n        return "LIBERO"\n',
        '    if "so101" in cmd_args:\n'
        '        return "SO101"\n'
        '    elif "libero" in cmd_args:\n'
        '        return "LIBERO"\n',
    )
    replace_once(
        constants_path,
        'elif ROBOT_PLATFORM == "BRIDGE":\n    constants = BRIDGE_CONSTANTS\n',
        'elif ROBOT_PLATFORM == "BRIDGE":\n'
        "    constants = BRIDGE_CONSTANTS\n"
        'elif ROBOT_PLATFORM == "SO101":\n'
        "    constants = SO101_CONSTANTS\n",
    )


def patch_transformers(source: Path) -> None:
    old = '"huggingface-hub>=0.19.3,<1.0"'
    new = '"huggingface-hub>=0.19.3,<2.0"'
    replace_once(source / "setup.py", old, new)
    table_path = source / "src" / "transformers" / "dependency_versions_table.py"
    replace_once(table_path, old, new)


def main() -> None:
    args = parse_args()
    require_commit(args.openvla_source, OPENVLA_OFT_COMMIT)
    require_commit(args.transformers_source, TRANSFORMERS_COMMIT)
    args.output_root.mkdir(parents=True, exist_ok=True)

    openvla_destination = args.output_root / "openvla-oft"
    transformers_destination = args.output_root / "transformers-openvla-oft"
    copy_source(args.openvla_source, openvla_destination)
    copy_source(args.transformers_source, transformers_destination)
    patch_openvla(openvla_destination)
    patch_transformers(transformers_destination)

    print(
        {
            "openvla_oft": str(openvla_destination),
            "transformers": str(transformers_destination),
            "robot_platform": "SO101",
            "action_chunk_size": 8,
            "action_dim": 6,
            "proprio_dim": 6,
        }
    )


if __name__ == "__main__":
    main()
