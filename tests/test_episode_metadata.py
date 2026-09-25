from __future__ import annotations

from dataclasses import replace

import pytest

from real_so101_vla_rl.data import (
    AtomicTask,
    CubeColor,
    EpisodeRecord,
    FailureType,
    SplitPolicy,
    TargetSlot,
    TaskType,
    append_episode_record,
    generate_dataset_splits,
    load_dataset_splits,
    load_episode_manifest,
    validate_episode_manifest,
    write_dataset_splits,
)


def make_record(
    episode_index: int,
    *,
    layout: str,
    trial: str,
    color: CubeColor,
    slot: TargetSlot,
    success: bool = True,
) -> EpisodeRecord:
    task_type = TaskType.SINGLE_T0 if slot is TargetSlot.T0 else TaskType.SEQUENCE_STEP
    step = {TargetSlot.T0: 0, TargetSlot.P1: 1, TargetSlot.P2: 2, TargetSlot.P3: 3}[slot]
    return EpisodeRecord.from_task(
        episode_index=episode_index,
        trial_id=trial,
        atomic_task=AtomicTask(task_type, color, slot, step),
        layout_id=layout,
        success=success,
        failure_type=None if success else FailureType.GRASP_FAILED,
        num_frames=180,
        duration_s=6.0,
    )


def test_episode_manifest_round_trip_and_atomic_append(tmp_path) -> None:
    path = tmp_path / "project_meta" / "episodes.jsonl"
    first = make_record(
        0,
        layout="layout_0001",
        trial="single_0001",
        color=CubeColor.RED,
        slot=TargetSlot.T0,
    )
    append_episode_record(path, first)

    assert load_episode_manifest(path) == (first,)

    with pytest.raises(ValueError, match="episode_index values must be unique"):
        append_episode_record(path, first)


def test_episode_record_checks_outcome_and_task_consistency() -> None:
    record = make_record(
        0,
        layout="layout_0001",
        trial="single_0001",
        color=CubeColor.RED,
        slot=TargetSlot.T0,
    )
    with pytest.raises(ValueError, match="must not have a failure_type"):
        replace(record, failure_type=FailureType.TIMEOUT)
    with pytest.raises(ValueError, match="task does not match"):
        replace(record, task="Pick up the blue cube and place it in T0.")


def test_manifest_keeps_one_layout_and_unique_steps_per_trial() -> None:
    p1 = make_record(
        0,
        layout="layout_a",
        trial="sequence_a",
        color=CubeColor.BLUE,
        slot=TargetSlot.P1,
    )
    with pytest.raises(ValueError, match="spans multiple layouts"):
        validate_episode_manifest((p1, replace(p1, episode_index=1, layout_id="layout_b", sequence_step=2,
                                               target_slot=TargetSlot.P2,
                                               task="Pick up the blue cube and place it in P2.")))


def test_grouped_split_excludes_failures_and_round_trips(tmp_path) -> None:
    records = []
    episode_index = 0
    for layout_index in range(10):
        layout = f"layout_{layout_index:04d}"
        records.append(
            make_record(
                episode_index,
                layout=layout,
                trial=f"single_{layout_index:04d}",
                color=CubeColor.RED,
                slot=TargetSlot.T0,
            )
        )
        episode_index += 1
        for color, slot in zip(
            (CubeColor.BLUE, CubeColor.YELLOW, CubeColor.GREEN),
            (TargetSlot.P1, TargetSlot.P2, TargetSlot.P3),
            strict=True,
        ):
            records.append(
                make_record(
                    episode_index,
                    layout=layout,
                    trial=f"sequence_{layout_index:04d}",
                    color=color,
                    slot=slot,
                )
            )
            episode_index += 1

    failed = make_record(
        episode_index,
        layout="layout_failed",
        trial="failed_0001",
        color=CubeColor.RED,
        slot=TargetSlot.T0,
        success=False,
    )
    records.append(failed)

    splits = generate_dataset_splits(records)
    assigned = set(splits.train) | set(splits.val) | set(splits.test)

    assert failed.episode_index not in assigned
    assert len(splits.val) == 4
    assert len(splits.test) == 4
    assert len(splits.train) == 32

    path = tmp_path / "splits.json"
    write_dataset_splits(path, splits)
    assert load_dataset_splits(path) == splits


def test_single_task_pilot_split_is_grouped_and_deterministic() -> None:
    records = tuple(
        make_record(
            index,
            layout=f"pilot-lowlight-blue-t0-pose-{index + 1:03d}",
            trial=f"pilot-{index:06d}",
            color=CubeColor.BLUE,
            slot=TargetSlot.T0,
        )
        for index in range(20)
    )

    splits = generate_dataset_splits(
        records,
        policy=SplitPolicy.PILOT_SINGLE_TASK_GROUPED_V1,
    )

    assert splits.policy is SplitPolicy.PILOT_SINGLE_TASK_GROUPED_V1
    assert splits.test == (5, 19)
    assert splits.val == (4, 14)
    assert len(splits.train) == 16
    assert set(splits.train) | set(splits.val) | set(splits.test) == set(
        range(20)
    )

    with pytest.raises(ValueError, match="covers all colors"):
        generate_dataset_splits(records)


def test_single_task_pilot_split_rejects_mixed_tasks() -> None:
    records = (
        make_record(
            0,
            layout="layout_a",
            trial="trial_a",
            color=CubeColor.BLUE,
            slot=TargetSlot.T0,
        ),
        make_record(
            1,
            layout="layout_b",
            trial="trial_b",
            color=CubeColor.RED,
            slot=TargetSlot.T0,
        ),
        make_record(
            2,
            layout="layout_c",
            trial="trial_c",
            color=CubeColor.BLUE,
            slot=TargetSlot.T0,
        ),
    )

    with pytest.raises(ValueError, match="exactly one canonical task"):
        generate_dataset_splits(
            records,
            policy=SplitPolicy.PILOT_SINGLE_TASK_GROUPED_V1,
        )
