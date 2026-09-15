# RL configurations

这里保存 SO101 强化学习的可复现配置。当前提供两个 `basic_t0` 低维特权状态
baseline：

- `ppo_t0_mlp.yaml`：带 critic 的连续动作 PPO。
- `grpo_t0_mlp.yaml`：无 critic、同一 reset 完整采样四条轨迹的 GRPO。

两个配置都让策略一次输出 `[8, 6]` 连续绝对关节目标。动作先限制在
`[-1, 1]`，再映射到 MuJoCo 六个 actuator 的实际 `ctrlrange`。这里没有将
动作离散为 token；MLP actor 使用 tanh-squashed Gaussian，能够提供 PPO/GRPO
所需的随机采样和带 Jacobian 修正的 `log_prob`。

安装及启动：

```bash
pip install -e '.[rl]'

python scripts/train_mujoco_rl.py --config configs/rl/ppo_t0_mlp.yaml
python scripts/train_mujoco_rl.py --config configs/rl/grpo_t0_mlp.yaml
```

先用两个短更新检查依赖、MuJoCo 采样、反向传播、评估和 checkpoint：

```bash
python scripts/train_mujoco_rl.py \
  --config configs/rl/ppo_t0_mlp.yaml \
  --smoke
```

PPO 配置中的 `steps_per_env_per_update: 32` 是采样窗口，不是 episode 长度：
每个环境每次参数更新前采集 32 个 action chunk，一个 chunk 包含 8 个 30 Hz
控制点。episode 上限由 `max_episode_chunks: 128` 单独控制。两者都可以在 YAML
中调整。

GRPO 不读取这个固定窗口。它将 `num_envs: 16` 分为四组，每组
`group_size: 4`；组内环境加载完全相同的 `ResetSnapshot`，然后真实执行四条完整
MuJoCo 轨迹。完成后按组内总回报计算相对优势。

训练结果写入 `runs/rl/hybrid/<run_id>/`：

```text
resolved_config.yaml
run_metadata.json
metrics.jsonl
checkpoint-000050.pt
```

从更新边界恢复或独立评估：

```bash
python scripts/train_mujoco_rl.py \
  --config configs/rl/ppo_t0_mlp.yaml \
  --checkpoint runs/rl/hybrid/<run_id>/checkpoint-000050.pt

python scripts/eval_mujoco_rl.py \
  --checkpoint runs/rl/hybrid/<run_id>/checkpoint-000050.pt

python scripts/eval_mujoco_rl.py \
  --checkpoint runs/rl/hybrid/<run_id>/checkpoint-000050.pt \
  --video evaluation.gif
```

当前 MLP 的输入不是纯 proprioception，而是 MuJoCo 提供的低维特权状态：关节
状态、TCP、所有电池的真实位姿/速度、目标颜色、T0 相对几何和接触标志。它用于
先验证环境、奖励和 RL 算法。未来 OpenVLA 策略的输入边界是 RGB、
proprioception 和文本；环境仍可在内部使用仿真真值计算奖励。
