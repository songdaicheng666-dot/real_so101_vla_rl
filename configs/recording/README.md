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
`Pick up the <color> cube and place it in <slot>.` MuJoCo uses the same proxy
shape and physical specification: 20 mm edge length and 0.096 kg net mass.

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
`datasets/so101_static_validation_v2`. Saved episode records use
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

## T0 四色 100 条正式采集

`so101_t0_100_lowlight_v1.yaml` 是 pilot 之后的正式 T0 数据集配置。它固定
30 FPS、60 秒录制上限、30 秒人工复位、双 RGB 视频、无损对齐深度和腕部
曝光 150。任务计划共 100 条，红、黄、蓝、绿各 25 条；每个 20 条绝对分段
中每色恰好 5 条，顺序由 seed 42 确定性生成。首次四条是每色一条的 canary。

这里的 `scene_id` 只是每条成功示范的唯一身份，不表示需要复现坐标或固定
布局。每次开始前，操作员应在合法初始区域内重新随机摆放四个方块，保证
互不重叠、完整可见且机械臂可抓取；不采集精确坐标，也不恢复历史摆法。
丢弃、同步失败或人工中止不会推进颜色计划，下一次仍重试同一颜色。
`project_meta/attempts.jsonl` 记录每次接受、丢弃或失败尝试，但失败帧不会
进入 SFT 数据集。

正式开录前，用同一曝光 150 profile 做十分钟稳定性测试：

```bash
conda run --no-capture-output -n lerobot \
  python scripts/probe_so101_cameras.py \
  --profile configs/hardware/cameras/so101_competition_2026_exposure150.yaml \
  --check-streams \
  --duration-s 600 \
  --report Log/hardware/camera_probe_t0_100_soak.json
```

首次录制仍只使用录制入口，不联动定稿或训练：

```bash
conda run --no-capture-output -n lerobot \
  python scripts/record_so101_lerobot.py \
  --config configs/recording/so101_t0_100_lowlight_v1.yaml
```

启动会要求至少 15 GiB 可用空间并保存计划哈希。完成前四条 canary 后，在
下一条场景准备提示按 `Esc`/`q` 退出，独立运行只读校验并人工检查两路 RGB、
深度和任务标签：

```bash
conda run --no-capture-output -n lerobot \
  python scripts/finalize_so101_dataset.py \
  --recording-config configs/recording/so101_t0_100_lowlight_v1.yaml \
  --sft-config configs/sft/openvla_oft_t0_100_lowlight.yaml \
  --validate-only
```

校验通过后显式续录：

```bash
conda run --no-capture-output -n lerobot \
  python scripts/record_so101_lerobot.py \
  --config configs/recording/so101_t0_100_lowlight_v1.yaml \
  --resume
```

录制器最多运行到下一个 20 条绝对边界。每段结束后重复 `--validate-only`，
再用 `--resume` 开始下一段。续录会核对计划哈希、已录颜色和场景 ID、
LeRobot task table、机械臂身份及相机 profile，任何不一致都会拒绝继续。
校验命令只投影读取全量数值列，并按 Parquet 文件解码每条的首帧、中间帧
和末帧；少量 Hugging Face 元数据缓存使用一次性目录并在退出时删除。因此
不会再物化或在全局缓存中累计 4/20/40 条等各阶段的完整 Arrow 副本。

100 条全部完成后，独立执行正式定稿：

```bash
conda run --no-capture-output -n lerobot \
  python scripts/finalize_so101_dataset.py \
  --recording-config configs/recording/so101_t0_100_lowlight_v1.yaml \
  --sft-config configs/sft/openvla_oft_t0_100_lowlight.yaml
```

定稿会检查全部表格帧的有效位、有限数值和 25 ms 同步上限，解码每条 episode
的首帧、中间帧和末帧，并按每色 20/2/3 生成 80/8/12 的 train/val/test。
完成后数据集被冻结。SFT 仍由 `train_openvla_oft_sft.py` 独立启动；正式训练
前先在 SFT 配置中填写训练步数、batch、梯度累积、worker、评估/保存间隔和
输出目录。
