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
The project therefore does not publish an unsatisfiable `hardware` extra in this
revision. Runtime camera probing plus RGB-video/depth-image finalization is the
compatibility gate; do not treat a successful import by itself as certification.

The checked-in files use stable leader/follower `/dev/serial/by-id` identities.
Camera identity and stream settings live in the reusable
`../hardware/cameras/so101_competition_2026.yaml` profile. Validate a complete
configuration without touching hardware with:

```bash
conda run --no-capture-output -n lerobot \
  python scripts/record_so101_lerobot.py --check-config
```

Before connecting either arm, a live recording run probes the exact Orbbec
serial/firmware and wrist by-id device, opens both fixed stream profiles, sets
and reads back wrist manual exposure `300` with gain `0`, and requires at least
99% valid samples plus a p95 camera timestamp span no greater than 25 ms. Raw
`/dev/videoN` selectors are rejected. The recorder runs this gate for ten
seconds on every startup.

The synchronizer performs nearest-unused-frame matching in a short timestamp
window before state attachment and before any video/image encoding. This fixes
the old false skew caused by hard-pairing two independent cameras' next frames;
it is not an MP4/TIFF encoder workaround. The 2026-09-19 formal ten-second probe
passed with 300/300 valid pairs, 29.92 FPS, and 21.40 ms p95 camera skew. A
ten-minute soak remains required before long sessions.

Real demonstrations use red, yellow, blue, and green cubes because the intended
battery props are unavailable. The canonical instruction is therefore
`Pick up the <color> cube and place it in <slot>.` Existing MuJoCo AAA assets are
kept as a documented sim-to-real shape mismatch until the simulation is revised.

Every real dataset receives immutable `project_meta/camera_profile.yaml`, the
Orbbec factory calibration snapshot, `camera_setup.json`, and an append-only
`camera_probe_reports.jsonl`. A profile or calibration hash mismatch prevents
episodes from different rigs being mixed.

Controls are phase-aware:

- Before each pilot attempt, place the blue cube at the displayed layout and
  press `Enter`/`s`; `Esc`/`q` safely stops without consuming that layout ID.
- During recording, `Right`/`n` ends the episode early, `Left`/`r` discards it,
  and `Esc`/`q` stops the session.
- During confirmation, `Enter`/`s` saves and `Left`/`r` discards.
- During manual reset, `Right`/`n` starts the next attempt early.

Any invalid or out-of-sync sensor sample clears the whole pending episode.
Diagnostics go to `project_meta/capture_failures.jsonl`; rejected attempts are
never assigned a persisted LeRobot episode index.

The calibration IDs are frozen as `my_follower_arm` and `my_leader_arm`, and
the recording entry refuses to continue if device EEPROM calibration differs
from either file. Before enabling follower torque it copies the current raw
position into `Goal_Position` and verifies the copy, preventing connection from
pursuing a stale target left by an earlier session.

`so101_static_validation_v2.yaml` is diagnostic-only: it records ten automatic
eight-second episodes without a reset delay and stores them under
`artifacts/datasets/so101_static_validation_v2`. Saved episode records use
`success=false` and `failure_type=static_validation`, so normal SFT split
generation rejects the dataset even though its LeRobot and OpenVLA adapter
contracts can be inspected. It must not be reused as a successful task dataset.

`so101_real_pilot_lowlight_v2.yaml` is the isolated 20-episode low-light pilot.
It fixes the task to `Pick up the blue cube and place it in T0.` and assigns the
unique layouts `pilot-lowlight-blue-t0-pose-001` through `020`. A discarded or
failed attempt does not advance the layout. It uses the versioned exposure-150
wrist profile with gain 0; the historical exposure-300 profile remains
unchanged. Start a fresh dataset with:

```bash
conda run --no-capture-output -n lerobot \
  python scripts/record_so101_lerobot.py \
  --config configs/recording/so101_real_pilot_lowlight_v2.yaml
```

After saving two episodes, stop at the next layout prompt and validate the
incomplete dataset without writing split or normalization metadata:

```bash
conda run --no-capture-output -n lerobot \
  python scripts/finalize_so101_dataset.py --validate-only
```

The validator decodes each episode boundary. Also inspect both recorded streams
manually under `videos/observation.images.overview/` and
`videos/observation.images.wrist/`, then continue at pose 003 with `--resume`:

```bash
conda run --no-capture-output -n lerobot \
  python scripts/record_so101_lerobot.py \
  --config configs/recording/so101_real_pilot_lowlight_v2.yaml \
  --resume
```

Once all 20 successful episodes exist, run the finalizer without
`--validate-only`. It atomically creates `splits.json`, `norm_stats.json`, and
`finalization.json`. The low-light directory is frozen at that point: it cannot
be resumed, and data captured after the fill light arrives must use a new
dataset root.

`max_relative_target` remains `null` for the initial low-speed pilot by explicit
project decision; enable per-step limiting before policy-controlled real-robot
rollouts.
