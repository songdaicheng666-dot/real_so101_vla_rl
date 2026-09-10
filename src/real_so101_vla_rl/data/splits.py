"""Deterministic, episode-level dataset splits."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from .episode_manifest import EpisodeRecord, validate_episode_manifest
from .robot_profile import _atomic_write
from .schema import SCHEMA_VERSION, BatteryColor, TargetSlot


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    schema_version: int
    seed: int
    group_by: tuple[str, ...]
    train: tuple[int, ...]
    val: tuple[int, ...]
    test: tuple[int, ...]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> DatasetSplits:
        raw = dict(raw)
        for key in ("group_by", "train", "val", "test"):
            raw[key] = tuple(raw[key])
        return cls(**raw)


def _split_group_counts(group_count: int, val_ratio: float, test_ratio: float) -> tuple[int, int]:
    if group_count < 3:
        raise ValueError("At least three distinct layouts are required for train/val/test splits")
    val_count = max(1, round(group_count * val_ratio))
    test_count = max(1, round(group_count * test_ratio))
    if val_count + test_count >= group_count:
        raise ValueError("Split ratios leave no layout group for training")
    return val_count, test_count


def _training_has_full_coverage(train_records: Iterable[EpisodeRecord]) -> bool:
    train_records = tuple(train_records)
    return (
        {record.target_color for record in train_records} == set(BatteryColor)
        and {record.target_slot for record in train_records} == set(TargetSlot)
    )


def generate_dataset_splits(
    records: Iterable[EpisodeRecord],
    *,
    seed: int = 42,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> DatasetSplits:
    """Split successful episodes while keeping each layout in one partition."""

    records = validate_episode_manifest(records)
    successful = tuple(record for record in records if record.success)
    if not successful:
        raise ValueError("No successful episodes are available for SFT")
    if not 0 < val_ratio < 1 or not 0 < test_ratio < 1 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be positive and sum to less than one")

    layouts: dict[str, list[EpisodeRecord]] = defaultdict(list)
    for record in successful:
        layouts[record.layout_id].append(record)
    val_count, test_count = _split_group_counts(len(layouts), val_ratio, test_ratio)

    layout_ids = sorted(layouts)
    chosen = None
    # Try deterministic shuffles until training retains all required colors and slots.
    for offset in range(10_000):
        shuffled = layout_ids.copy()
        random.Random(seed + offset).shuffle(shuffled)
        test_layouts = set(shuffled[:test_count])
        val_layouts = set(shuffled[test_count : test_count + val_count])
        train_layouts = set(shuffled[test_count + val_count :])
        train_records = [record for layout in train_layouts for record in layouts[layout]]
        if _training_has_full_coverage(train_records):
            chosen = train_layouts, val_layouts, test_layouts
            break
    if chosen is None:
        raise ValueError(
            "Unable to create grouped splits whose training partition covers all colors and target slots"
        )

    train_layouts, val_layouts, test_layouts = chosen

    def indices(layout_set: set[str]) -> tuple[int, ...]:
        return tuple(sorted(record.episode_index for layout in layout_set for record in layouts[layout]))

    splits = DatasetSplits(
        schema_version=SCHEMA_VERSION,
        seed=seed,
        group_by=("layout_id",),
        train=indices(train_layouts),
        val=indices(val_layouts),
        test=indices(test_layouts),
    )
    validate_dataset_splits(splits, records)
    return splits


def validate_dataset_splits(
    splits: DatasetSplits, records: Iterable[EpisodeRecord]
) -> DatasetSplits:
    records = validate_episode_manifest(records)
    if splits.schema_version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported split schema_version={splits.schema_version}")
    if splits.group_by != ("layout_id",):
        raise ValueError("The v1 split schema groups episodes by layout_id")

    split_sets = {
        "train": set(splits.train),
        "val": set(splits.val),
        "test": set(splits.test),
    }
    if split_sets["train"] & split_sets["val"] or split_sets["train"] & split_sets["test"]:
        raise ValueError("Dataset splits overlap")
    if split_sets["val"] & split_sets["test"]:
        raise ValueError("Dataset splits overlap")

    successful = {record.episode_index: record for record in records if record.success}
    assigned = set().union(*split_sets.values())
    if assigned != set(successful):
        raise ValueError("Splits must contain every successful episode exactly once and no failed episodes")

    layout_partition: dict[str, str] = {}
    for partition, episode_indices in split_sets.items():
        for episode_index in episode_indices:
            layout_id = successful[episode_index].layout_id
            previous = layout_partition.setdefault(layout_id, partition)
            if previous != partition:
                raise ValueError(f"layout_id={layout_id!r} occurs in both {previous} and {partition}")

    if not _training_has_full_coverage(successful[index] for index in splits.train):
        raise ValueError("Training split must cover all four colors and T0/P1/P2/P3")
    return splits


def write_dataset_splits(path: str | Path, splits: DatasetSplits) -> None:
    payload = json.dumps(splits.to_dict(), ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    _atomic_write(Path(path), payload)


def load_dataset_splits(path: str | Path) -> DatasetSplits:
    return DatasetSplits.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
