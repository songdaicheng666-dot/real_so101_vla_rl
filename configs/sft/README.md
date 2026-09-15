# SFT 配置

`openvla_oft_synthetic_overfit.yaml` 是模拟数据全链路测试的可运行配置。
它读取云端数据盘中的 6 个短 episode，使用 batch size 2 运行 200 个
optimizer step。该配置关闭图像增强，让模拟数据的 loss 趋势更容易复现；
生产配置仍保持图像增强开启。

云端 A100 环境、固定上游提交、兼容工作副本和冒烟验收顺序记录在
`CLOUD_ENVIRONMENT.md`。

`openvla_oft_lora.yaml` 定义 SO-101 schema v2 的 OpenVLA-OFT LoRA 微调契约。当前文件可以通过结构校验，但有意把依赖数据规模或云端 GPU 的参数保留为 `null`。

固定的数据和模型约束如下：

- 模型 RGB 输入顺序固定为 `observation.images.overview`、`observation.images.wrist`；两幅图分别预处理后沿通道维组合。
- `observation.images.overview_depth` 是对齐到全局 RGB 的毫米制深度，始终以无损 uint16 TIFF 保存，只录制，不进入当前 OpenVLA 图像骨干。
- 三路视觉和 `observation.state` 都带有 host monotonic 纳秒时间戳及有效位；任一有效位为假时不得写入正式数据集。
- follower 状态和绝对目标动作均为六维，顺序为肩部旋转、肩部抬升、肘部、手腕俯仰、手腕旋转、夹爪；动作必须取自 `robot.send_action()` 的返回值。
- 动作窗口长度为 8，episode 尾部补齐位置通过 `action_is_pad` 从损失中排除。
- 使用 OpenVLA-OFT 连续动作头、L1 损失和 `BOUNDS_Q99` 归一化。
- LoRA 使用 rank 32、alpha 16、dropout 0 和 `all-linear` 目标模块。

项目自有的 `record_so101_lerobot.py` 已保存 `robot.send_action()`
返回的最终动作，不使用裁剪前的 leader 目标。录制时的两路
RGB 可编码为 MP4，深度则固定保留为逐帧 TIFF。

数据接口直接读取 LeRobotDataset，不需要转换为 RLDS。`repo_id` 是
LeRobotDataset 必需的数据集标识，本地数据也需要填写；`root` 可以指向本地
数据集目录，留空时使用 Hub 及其默认缓存。`revision` 留空时使用 LeRobot
v3.0 数据版本。

录制端写入的 RGB/深度分别是 HWC `uint8`/`uint16`。RGB 可存为
MP4 或 PNG；深度磁盘数据固定为无损 uint16 TIFF 编码，LeRobot v3
会将 TIFF 字节嵌入 Parquet。读取时返回 CHW `float32`毫米值，数值与
原始整数逐像素一致；`[1]`
标量特征会被压成零维 Tensor。项目适配器按该读取语义处理。
LeRobot 自动生成的顶层 `timestamp` 是 30 Hz 名义时间，不能替代各传感器
的 `observation.timestamps.*_ns`。

数据适配依赖通过独立 extra 安装，基础包不会安装 Torch 或 LeRobot：

```bash
pip install -e '.[sft-data,dev]'
```

该命令只安装代码依赖，不下载 OpenVLA 权重。正式启动训练前必须填写
`repo_id`、训练步数、batch size、梯度累积、worker 数、评估与保存间隔以及
输出目录。配置加载入口为：

```python
from real_so101_vla_rl.models import load_sft_config

config = load_sft_config(
    "configs/sft/openvla_oft_lora.yaml",
    require_training_ready=False,
)
```

把 `require_training_ready` 改为 `True` 后，任何尚未填写的运行参数都会触发明确错误。
