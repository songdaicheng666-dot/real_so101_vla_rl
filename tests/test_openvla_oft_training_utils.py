from __future__ import annotations

import csv
import json

import pytest
import torch
from PIL import Image

from real_so101_vla_rl.models.openvla_oft_training import (
    MetricsWriter,
    build_cosine_scheduler,
    moving_average,
    save_loss_curve,
)


def test_moving_average_and_loss_artifacts(tmp_path) -> None:
    losses = [1.0, 0.8, 0.5, 0.4]
    smoothed = moving_average(losses, window=2)
    rows = []
    with MetricsWriter(tmp_path) as writer:
        for step, (loss, smooth) in enumerate(
            zip(losses, smoothed, strict=True), start=1
        ):
            row = {
                "step": step,
                "epoch": 1,
                "train_loss": loss,
                "smoothed_train_loss": smooth,
                "val_loss": 0.45 if step == 4 else None,
                "learning_rate": 0.001,
                "grad_norm": 1.0,
            }
            rows.append(row)
            writer.write(row)

    assert smoothed == pytest.approx([1.0, 0.9, 0.65, 0.45])
    json_rows = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    assert len(json_rows) == 4
    with (tmp_path / "metrics.csv").open(newline="") as metrics_file:
        assert len(list(csv.DictReader(metrics_file))) == 4

    curve_path = tmp_path / "loss_curve.png"
    save_loss_curve(curve_path, rows)
    with Image.open(curve_path) as curve:
        assert curve.size == (1200, 720)
        assert curve.format == "PNG"


def test_cosine_scheduler_warms_up_then_decays() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = build_cosine_scheduler(
        optimizer, max_steps=10, warmup_steps=2
    )

    initial = optimizer.param_groups[0]["lr"]
    optimizer.step()
    scheduler.step()
    warmed = optimizer.param_groups[0]["lr"]
    for _ in range(8):
        optimizer.step()
        scheduler.step()
    final = optimizer.param_groups[0]["lr"]

    assert initial == pytest.approx(0.5)
    assert warmed == pytest.approx(1.0)
    assert final < warmed
