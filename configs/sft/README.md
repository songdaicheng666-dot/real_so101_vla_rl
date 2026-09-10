# SFT 配置

`openvla_oft_lora.yaml` 定义 SO-101 的首版 OpenVLA-OFT LoRA 微调契约。当前文件可以通过结构校验，但有意把依赖数据规模或云端 GPU 的参数保留为 `null`。

固定的数据和模型约束如下：

- 单个前置相机字段：`observation.images.front`。
- follower 状态和绝对目标动作均为六维，顺序为肩部旋转、肩部抬升、肘部、手腕俯仰、手腕旋转、夹爪；动作必须取自 `robot.send_action()` 的返回值。
- 动作窗口长度为 8，episode 尾部补齐位置通过 `action_is_pad` 从损失中排除。
- 使用 OpenVLA-OFT 连续动作头、L1 损失和 `BOUNDS_Q99` 归一化。
- LoRA 使用 rank 32、alpha 16、dropout 0 和 `all-linear` 目标模块。

正式启动训练前必须填写数据集位置、训练步数、batch size、梯度累积、worker 数、评估与保存间隔以及输出目录。配置加载入口为：

```python
from real_so101_vla_rl.models import load_sft_config

config = load_sft_config(
    "configs/sft/openvla_oft_lora.yaml",
    require_training_ready=False,
)
```

把 `require_training_ready` 改为 `True` 后，任何尚未填写的运行参数都会触发明确错误。
