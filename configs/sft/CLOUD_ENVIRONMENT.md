# OpenVLA-OFT 云端环境

当前已验证的环境使用 Python 3.12、PyTorch 2.7.1 CUDA 12.8、
FlashAttention 2.7.4.post1 和 A100 40 GB。模型、conda 环境与缓存应放在
云服务器数据盘上，不要放在 30 GB 系统盘。

## 固定上游版本

- LeRobot: `4aaff99be4a1d81568c08c8f0296b41b40c99ec4` (`0.6.2`)
- OpenVLA-OFT: `e4287e94541f459edc4feabc4e181f537cd569a8`
- transformers-openvla-oft: `bc339d9ad707454c0c115970db43c260067c61ab`

先将三个上游仓库 checkout 到上述提交。LeRobot 可直接 editable
安装。OpenVLA-OFT 和其 Transformers fork 先生成兼容工作副本：

```bash
python scripts/prepare_openvla_oft_compat.py \
  --openvla-source /root/autodl-tmp/external/openvla-oft \
  --transformers-source /root/autodl-tmp/external/transformers-openvla-oft \
  --output-root /root/autodl-tmp/compat
```

该脚本会验证 Git commit，再完成三项最小兼容改动：

- 取消 `prismatic.vla` 对 RLDS/TensorFlow/dlimp 的导入副作用。
- 增加显式的 `OPENVLA_ROBOT_PLATFORM=SO101`，对应 8 步动作块、
  6 维动作和 6 维状态。
- 放宽 Transformers fork 的 `huggingface-hub` 声明上限。

上游 checkout 始终保持不变，editable 安装指向 `/root/autodl-tmp/compat` 下的
工作副本。

## Python 依赖

`cloud_requirements.txt` 记录了已验证的关键版本。FlashAttention 使用
官方 GitHub release 的预编译 wheel，文件 SHA-256 已写入 URL。云端直连
GitHub 过慢时，可在本机下载、校验后传入数据盘，再使用
`pip install /path/to/wheel` 安装。

```bash
python -m pip install -r configs/sft/cloud_requirements.txt
python -m pip install -e '/root/autodl-tmp/external/lerobot[dataset]'
python -m pip install -e /root/autodl-tmp/compat/transformers-openvla-oft --no-deps
python -m pip install -e /root/autodl-tmp/compat/openvla-oft --no-deps
python -m pip install -e '/root/autodl-tmp/real_so101_vla_rl[dev]'
```

`tokenizers==0.19.1` 要求 `huggingface-hub<1.0`，而 LeRobot 0.6.2 的包声明
要求更新的 Hub 版本。项目在 LeRobot 加载层中为普通数据集提供了
`sync_bucket` 导入兼容处理；该环境不支持 Hugging Face bucket 类型数据集。
OpenVLA-OFT 的包元数据仍声明了旧 PyTorch 和整套 TensorFlow/RLDS 依赖，
因此对兼容工作副本使用 `--no-deps`。

## 运行时约束

启动任何 OpenVLA-OFT 进程前设置：

```bash
export OPENVLA_ROBOT_PLATFORM=SO101
export HF_HOME=/root/autodl-tmp/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

脚本先注册本地 OpenVLA-OFT 类，再使用 `trust_remote_code=False` 读取
checkpoint。这会使用 OpenVLA-OFT 的多图像和 proprio 前向逻辑，同时保持
checkpoint 缓存不变。

## 验收顺序

```bash
python scripts/smoke_openvla_processor.py \
  --cache-dir /root/autodl-tmp/huggingface --local-files-only

python scripts/smoke_openvla_forward.py \
  --cache-dir /root/autodl-tmp/huggingface --local-files-only

python scripts/smoke_openvla_train_step.py \
  --cache-dir /root/autodl-tmp/huggingface --local-files-only
```

以下 A100 40 GB 数值来自旧 schema v1 单 RGB 验证，只作为历史基线；
schema v2 双 RGB 的显存和吞吐必须重新测量。旧结果中完整前向峰值显存为 14.76 GiB；LoRA rank 8
单步反向峰值显存为 16.08 GiB。LoRA rank 32、真实 batch size 和数据
worker 数需在真实数据集到位后再做显存与吞吐测量。

## 模拟数据全链路 SFT

模型缓存和环境已经存在时，使用离线模式生成 LeRobot v3 视频数据并训练：

```bash
export OPENVLA_ROBOT_PLATFORM=SO101
export HF_HOME=/root/autodl-tmp/huggingface
export HF_DATASETS_CACHE=/root/autodl-tmp/huggingface/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/generate_synthetic_lerobot_dataset.py \\
  --root /root/autodl-tmp/datasets/so101_synthetic_overfit_v2_lossless_depth

python scripts/train_openvla_oft_sft.py \\
  --cache-dir /root/autodl-tmp/huggingface \\
  --local-files-only \\
  --preflight-only

python scripts/train_openvla_oft_sft.py \\
  --cache-dir /root/autodl-tmp/huggingface \\
  --local-files-only
```

训练产物位于
`/root/autodl-tmp/runs/so101_synthetic_overfit_v2_lossless_depth`。其中
`metrics.jsonl`、`metrics.csv` 和 `loss_curve.png` 用于人工判断 loss 是否
先下降再进入平台；checkpoint 只保存 LoRA adapter、processor、action head、
proprio projector 和归一化统计，不复制或合并完整 7B 权重。
