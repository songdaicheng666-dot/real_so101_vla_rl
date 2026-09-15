# SO-101 schema-v2 real recording

`so101_schema_v2.yaml` configures the project-owned recorder. It writes accepted
demonstrations directly through `LeRobotDataset.add_frame()`; there is no raw-to-
LeRobot conversion pass.

`dataset.rgb_use_videos` controls only `overview` and `wrist`: `true` encodes
those RGB streams as MP4 and `false` retains per-frame PNG. Aligned overview
depth is always encoded as raw/lossless uint16 TIFF in millimetres; LeRobot v3
embeds that TIFF payload in Parquet after removing its temporary frame file.
The reader exposes depth as CHW float32 millimetres, with pixel values unchanged.
The removed `dataset.use_videos` key is intentionally rejected.

The dataset/video side requires `av>=15,<16`. The currently installed
`pyorbbecsdk2==2.1.2` package metadata pins `av==12.3.0`; its USB capture path
imports successfully with PyAV 15 and this project does not use its PyAV-based
network-camera example, but `pip check` still reports the declared conflict.
Resolve that packaging boundary when the hardware extra is frozen; do not treat
the successful import as final hardware compatibility certification.

The checked-in file deliberately leaves the follower port, leader port, Orbbec
serial and wrist camera identity unresolved. Validate the non-hardware structure with:

```bash
conda run --no-capture-output -n lerobot \
  python scripts/record_so101_lerobot.py --check-config
```

After hardware discovery and calibration are frozen, fill those values and run
without `--check-config`. `/dev/video4` must not be used as an implicit default.

Controls are phase-aware:

- During recording, `Right`/`n` ends the episode early, `Left`/`r` discards it,
  and `Esc`/`q` stops the session.
- During confirmation, `Enter`/`s` saves and `Left`/`r` discards.
- During manual reset, `Right`/`n` starts the next attempt early.

Any invalid or out-of-sync sensor sample clears the whole pending episode.
Diagnostics go to `project_meta/capture_failures.jsonl`; rejected attempts are
never assigned a persisted LeRobot episode index.
