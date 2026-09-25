# SO-101 competition camera rig

`so101_competition_2026.yaml` is the historical exposure-300 hardware profile.
`so101_competition_2026_exposure150.yaml` is its versioned successor for the
low-light pilot. Both select the overview Orbbec camera by SDK serial number and
the wrist camera through its persistent Linux
`/dev/v4l/by-id/...-video-index0` link. Raw `/dev/videoN` selectors are rejected.

The historical profile revision is `so101_competition_2026_dual_camera_mounted_v2`;
the pilot revision is
`so101_competition_2026_dual_camera_mounted_exposure150_v3`.
The Orbbec factory calibration belongs to the exact device, firmware, and stream
profiles recorded in the YAML. Its SHA-256 is checked whenever the profile is
loaded. The pilot wrist camera is reopened with manual exposure `150` and gain
`0`; the adapter must successfully set and read back those controls before
recording. The v2 exposure-300 file is retained so existing dataset provenance
continues to identify its original settings.
Wrist intrinsics and both installation transforms remain `uncalibrated`, so image
recording is allowed but world-frame geometry must not use this profile yet.

Inspect identities without opening streams:

```bash
conda run --no-capture-output -n lerobot \
  python scripts/probe_so101_cameras.py
```

Open both cameras and produce a JSON report:

```bash
conda run --no-capture-output -n lerobot \
  python scripts/probe_so101_cameras.py \
  --check-streams \
  --duration-s 1 \
  --report Log/hardware/camera_probe.json
```

Use `--duration-s 600` for the ten-minute acceptance run. Both devices currently
enumerate at 480 Mbps on separate physical ports (`3-4` for Orbbec and `3-2` for
wrist). Independent 30 Hz cameras are not phase locked, so the preflight and the
recorder retain a short window and select the unused pair with the nearest host
monotonic timestamps instead of pairing the two arbitrary "next" frames.

The 2026-09-19 formal ten-second run passed: 300/300 pairs were valid, paired
throughput was 29.92 FPS, and p95 camera timestamp span was 21.40 ms against the
25 ms limit. This matching happens before MP4/TIFF encoding. The ten-minute soak
test is still required before a long recording session; do not weaken the data
contract or silently reduce stream profiles if it fails.

Firmware updates, camera replacement, stream changes, or later installation
calibration require a new profile revision. Do not overwrite a profile already
referenced by recorded data.
