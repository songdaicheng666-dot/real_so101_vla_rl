# Scripts

## SO-101 schema-v2 real recording

`record_so101_lerobot.py` is the project-owned synchronized recorder. It uses
LeRobot's SO-101 hardware, processors and Dataset writer, but constructs the
project's dual-RGB/aligned-depth frame, rejects invalid or over-25-ms-skewed
observations, and records the action returned by `robot.send_action()`.
The two RGB streams are optionally encoded as MP4, while aligned millimetre
depth always remains lossless per-frame uint16 TIFF.

The checked-in configuration intentionally has unresolved hardware identifiers.
Its non-hardware structure can already be checked with:

```bash
conda run --no-capture-output -n lerobot \
  python scripts/record_so101_lerobot.py \
  --config configs/recording/so101_schema_v2.yaml \
  --check-config
```

After the hardware identity and calibration step fills the follower/leader
ports, Orbbec serial and wrist selector, omit `--check-config` to record. Only
episodes explicitly accepted with `Enter`/`s` are saved. See
`configs/recording/README.md` for controls and rejection behavior.

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
the dynamics in real time, and lets you test grasping the four AAA batteries
with either keyboard commands or the MuJoCo Viewer's right-side actuator
sliders:

```bash
conda run --no-capture-output -n lerobot \
  python scripts/demo_mujoco_t0_grasp.py
```

Use the top-row numbers to select the matching physical servo ID: `1`
shoulder pan, `2` shoulder lift, `3` elbow flex, `4` wrist flex, `5` wrist
roll, and `6` gripper. Press `A` to decrease the selected target (or close the
gripper) and `D` to increase it (or open the gripper). `Space` pauses/resumes,
and `Backspace` resets the robot and batteries. The demo automatically undoes
the native Viewer visibility toggles attached to `1`-`5`, `A`, and `D`, so
using them does not hide or alter the rendered model.

The defaults are 2-degree arm increments and 1-degree gripper increments;
override them with `--joint-step-deg` and `--gripper-step-deg`.

The terminal reports fixed-jaw, moving-jaw, and two-sided battery contacts. A
battery is reported as lifted after its center rises 10 mm above its initial
height, and as successfully grasped when it is both lifted and in two-sided
contact. These messages are diagnostics rather than task rewards.

Manual validation has successfully controlled the robot, closed the gripper,
grasped an AAA battery, and lifted it in the `basic_t0` scene. The demo is kept
as a repeatable regression tool for later contact, actuator, and model changes;
it is not yet a Gymnasium environment or an RL task implementation.

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
