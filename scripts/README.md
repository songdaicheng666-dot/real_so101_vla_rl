# Scripts

## SO-101 实机—MuJoCo 对齐标定

版本化参数保存在
`src/real_so101_vla_rl/assets/mujoco/competition_2026/alignment.yaml`。
只读采集命令只读取 follower 的 `Present_Position` 和 Orbbec RGB-D；它不会调用
`configure()`、不会写 `Goal_Position`、不会开关扭矩：

```bash
conda run --no-capture-output -n lerobot \
  python scripts/calibrate_so101_alignment.py capture \
  --output calibration/capture-001
```

用任务纸源区和 T0 区至少 8 个已知边界点求 overview 外参：

```bash
python scripts/calibrate_so101_alignment.py solve-pnp \
  --points calibration/pnp_points.json \
  --output calibration/pnp_result.json
```

从 100 条演示按归一化关节距离做最远点抽样，并为单帧生成
`real / sim_raw / sim_styled / overlay / difference` 与 JSON 指标：

```bash
python scripts/calibrate_so101_alignment.py select-poses --count 16
python scripts/calibrate_so101_alignment.py report \
  --real-rgb calibration/capture-001/overview_rgb.png \
  --real-depth calibration/capture-001/overview_depth_mm.npy \
  --state calibration/capture-001/capture.json \
  --output calibration/report-home
```

省略 `--real-depth` 时，报告会自动查找 RGB 同目录下的
`overview_depth_mm.npy`。JSON 使用标准浮点 CIELAB 计算静态纸面 ΔE，并通过
拟合板面深度、排除动态方块和细线缆来报告机械臂轮廓 IoU。

`calibration/` 作为可复核的标定证据纳入仓库。移动 overview 相机、任务纸或桌面后
必须重新采集并生成新的 `alignment_id`；不要改写历史 100 条数据及其相机 profile。

## SO-101 schema-v2 real recording

`record_so101_lerobot.py` is the project-owned synchronized recorder. It uses
LeRobot's SO-101 hardware, processors and Dataset writer, but constructs the
project's dual-RGB/aligned-depth frame, rejects invalid or over-25-ms-skewed
observations, and records the action returned by `robot.send_action()`.
The two RGB streams are optionally encoded as MP4, while aligned millimetre
depth always remains lossless per-frame uint16 TIFF.

The checked-in recording config references a strict reusable camera profile and
stable leader/follower `/dev/serial/by-id` identities. Its static structure can
be checked with:

```bash
conda run --no-capture-output -n lerobot \
  python scripts/record_so101_lerobot.py \
  --config configs/recording/so101_schema_v2.yaml \
  --check-config
```

Probe the frozen dual-camera identity and streams independently with:

```bash
conda run --no-capture-output -n lerobot \
  python scripts/probe_so101_cameras.py \
  --check-streams \
  --duration-s 1 \
  --report Log/hardware/camera_probe.json
```

After passing camera preflight, omit `--check-config` to record. Demonstration
episodes are saved only after explicit `Enter`/`s` confirmation. A separate
`configs/recording/so101_static_validation_v2.yaml` diagnostic config records
ten automatic eight-second static episodes, marks them `static_validation`, and
therefore prevents them from entering successful SFT splits. See
`configs/recording/README.md` for controls and rejection behavior.

The low-light pilot uses
`configs/recording/so101_real_pilot_lowlight_v2.yaml`. Each attempt has a
layout-setup prompt, and only a confirmed save advances its unique layout ID.
If a partial dataset already exists, pass `--resume`; without that explicit flag
the recorder refuses to touch a nonempty root. Resume verifies the LeRobot
episode count, project manifest, task, and planned layout prefix, and is refused
after split/normalization metadata has been generated.

`finalize_so101_dataset.py` validates the real dataset against both its recording
and SFT configs. `--validate-only` supports the two-episode inspection gate and
writes nothing. A full run requires all 20 successful episodes, checks every
tabular frame plus each episode's first/middle/last media, computes q01/q99 only from
the train split, and atomically writes deterministic final metadata:

Validation projects only the numeric columns needed for whole-dataset checks and
reads first/middle/last media one Parquet file at a time. Each invocation also
uses a disposable Hugging Face Datasets metadata cache and removes it on exit,
so repeated checkpoints no longer materialize or accumulate full Arrow copies
in the user's global cache.

```bash
conda run --no-capture-output -n lerobot \
  python scripts/finalize_so101_dataset.py --validate-only

conda run --no-capture-output -n lerobot \
  python scripts/finalize_so101_dataset.py
```

The real recording task uses colored cubes and the canonical English instruction
`Pick up the <color> cube and place it in <slot>.` The MuJoCo grasp demo now
uses matching 20 mm, 0.096 kg colored battery-proxy cubes.

The formal low-light T0 collection uses
`configs/recording/so101_t0_100_lowlight_v1.yaml`. Its deterministic plan has
100 accepted demonstrations, 25 per color and five per color in every 20-episode
session. Every attempt still asks the operator to randomize all four cubes inside
the legal initial area; the displayed scene ID is unique metadata, not a pose to
reproduce. Rejections and synchronization failures are appended to
`project_meta/attempts.jsonl` and retry the same planned color.

The recorder stops at the next absolute 20-episode boundary. Use the recording
command with `--resume` only after the separate finalizer command with explicit
formal recording and SFT configs has passed `--validate-only`. After episode 100,
run that finalizer without `--validate-only`; start SFT later with the separate
`train_openvla_oft_sft.py` entry. The exact soak, canary, resume and finalization
commands are kept in `configs/recording/README.md`.

## MuJoCo Basic T0 PPO/GRPO training

`train_mujoco_rl.py` runs the complete state-based RL engineering loop: it
creates the Basic T0 environments, samples continuous `[8, 6]` action chunks,
steps MuJoCo, records reward events, computes PPO or group-relative advantages,
updates the MLP policy, evaluates fixed resets, writes JSONL metrics, and saves
update-boundary checkpoints.

```bash
conda run --no-capture-output -n lerobot \
  python scripts/train_mujoco_rl.py \
  --config configs/rl/ppo_t0_mlp.yaml \
  --smoke

conda run --no-capture-output -n lerobot \
  python scripts/train_mujoco_rl.py \
  --config configs/rl/grpo_t0_mlp.yaml \
  --smoke
```

Remove `--smoke` for the configured 1000-update run. Resume only from a saved
update boundary with `--checkpoint`. Evaluate a checkpoint independently with:

```bash
conda run --no-capture-output -n lerobot \
  python scripts/eval_mujoco_rl.py \
  --checkpoint runs/rl/hybrid/<run_id>/checkpoint-000050.pt
```

Add `--video evaluation.gif` to record the first fixed evaluation reset with
the `overview` camera. Headless video rendering uses MuJoCo EGL. See
`configs/rl/README.md` for the observation, action, PPO collection-window, and
GRPO grouping semantics.

## MuJoCo T0 interactive grasp demo

`demo_mujoco_t0_grasp.py` loads the Competition 2026 `basic_t0` scene, runs
the dynamics in real time, and lets you test grasping the four 20 mm,
0.096 kg colored cubes with either keyboard commands or the MuJoCo Viewer's
right-side actuator sliders:

```bash
conda run --no-capture-output -n lerobot \
  python scripts/demo_mujoco_t0_grasp.py
```

Use the top-row numbers to select the matching physical servo ID: `1`
shoulder pan, `2` shoulder lift, `3` elbow flex, `4` wrist flex, `5` wrist
roll, and `6` gripper. Press `A` to decrease the selected target (or close the
gripper) and `D` to increase it (or open the gripper). `Space` pauses/resumes,
and `Backspace` resets the robot and cubes. The demo automatically undoes
the native Viewer visibility toggles attached to `1`-`5`, `A`, and `D`, so
using them does not hide or alter the rendered model.

The defaults are 2-degree arm increments and 1-degree gripper increments;
override them with `--joint-step-deg` and `--gripper-step-deg`.

The terminal reports fixed-jaw, moving-jaw, and two-sided cube contacts. A
cube is reported as lifted after its center rises 10 mm above its initial
height, and as successfully grasped when it is both lifted and in two-sided
contact. These messages are diagnostics rather than task rewards.

The archived manual validation grasped and lifted the previous AAA model. It is
historical control-chain evidence, not validation of the current cube mass and
contact model. Re-run the same manual check for the cubes; this demo remains a
regression tool separate from the Gymnasium environment and task rewards.

For live pose comparison in the formal T0 scene, support the real follower arm
and run:

```bash
conda run --no-capture-output -n lerobot \
  python scripts/demo_so101_live_mirror.py --scene basic_t0
```

After calibration is checked, press Enter to disable follower torque and move
one joint at a time by hand. The mirror prints each real reading, requested and
applied MuJoCo angle, and any out-of-range or model-clipping flag. Use the
Viewer's left Camera panel to switch between Free and overview views. The
formal scene's cubes stay at home while the six robot joints mirror the real
arm. Closing the Viewer disconnects the bus without re-enabling torque. The
default command without `--scene` still opens the standalone robot mirror.

For a display-free dependency and dynamics check, run:

```bash
conda run -n lerobot python scripts/demo_mujoco_t0_grasp.py --headless-check
```

`smoke_openvla_processor.py` loads the real OpenVLA processor without loading
model weights, then converts one synthetic SO-101 sample through the project
adapter. Set `OPENVLA_ROBOT_PLATFORM=SO101` before importing OpenVLA-OFT.

`smoke_openvla_forward.py` loads the real OpenVLA checkpoint on CUDA and runs
one synthetic sample through the vision-language model, proprio projector, and
continuous action head. It checks the 48 action-token hidden states, `[1, 8, 6]`
action prediction, and padding-aware L1 loss.

`smoke_openvla_train_step.py` adds LoRA, enables gradient checkpointing, and
optimizes one synthetic batch. It verifies a finite padding-aware action loss,
a nonzero finite LoRA gradient, and an actual LoRA parameter update.

`generate_synthetic_lerobot_dataset.py` creates six deterministic SO-101
episodes with the real LeRobot writer. Its default path stores distinct overview
and wrist RGB streams plus aligned millimetre depth, timestamps, and validity
fields, then writes the project's split, normalization, robot, calibration, and
generation metadata:

```bash
python scripts/generate_synthetic_lerobot_dataset.py \\
  --root /root/autodl-tmp/datasets/so101_synthetic_overfit_v2_lossless_depth
```

Use `--rgb-image-backed` to keep the two RGB streams as per-frame PNG for local
debugging. This option never changes the lossless TIFF depth representation.

`train_openvla_oft_sft.py` reads that finalized dataset through the project
adapter and runs single-GPU OpenVLA-OFT LoRA SFT. The preflight mode performs
one real batch forward before the full run:

```bash
python scripts/train_openvla_oft_sft.py \\
  --cache-dir /root/autodl-tmp/huggingface \\
  --local-files-only \\
  --preflight-only

python scripts/train_openvla_oft_sft.py \\
  --cache-dir /root/autodl-tmp/huggingface \\
  --local-files-only
```

After training, reload a saved LoRA adapter and the two continuous-action
modules, then run a real forward pass without starting a new run:

```bash
python scripts/train_openvla_oft_sft.py \\
  --cache-dir /root/autodl-tmp/huggingface \\
  --local-files-only \\
  --preflight-only \\
  --checkpoint /root/autodl-tmp/runs/so101_synthetic_overfit_v2_lossless_depth/checkpoint-000200
```

The full command records every optimizer step in CSV and JSONL, saves the LoRA
adapter and continuous-action components, and renders `loss_curve.png`.

For a completed formal run, enhanced offline evaluation selects the best finite
validation checkpoint, verifies its training statistics, evaluates the full
test split, and writes both evaluation and acceptance reports:

```bash
python scripts/evaluate_openvla_oft_sft.py \
  --run-dir /root/autodl-tmp/runs/so101_t0_100_lowlight_v1_s42_5k_aug \
  --cache-dir /root/autodl-tmp/huggingface \
  --local-files-only
```
