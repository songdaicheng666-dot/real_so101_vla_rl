# SO-101 T0 100 Lowlight v1

这是本项目第一版冻结的正式真机示范数据集，用于训练 SO-101 机械臂根据颜色指令
抓取单个方块并放入 `T0` 区域。数据遵循 LeRobotDataset v3.0 目录格式。

## 数据概览

| 项目 | 数值 |
| --- | --- |
| Episode | 100 条成功示范 |
| 帧数 | 46,909 |
| 采样率 | 30 FPS |
| 任务 | 红、黄、蓝、绿方块各 25 条，均放入 T0 |
| 划分 | train 80 / validation 8 / test 12 |
| 机器人 | SO-101 follower，6 维绝对关节目标 |
| 视觉 | overview RGB、wrist RGB、overview 对齐毫米深度 |
| 冻结载荷 | 79 个文件，4,080,796,024 字节 |
| 许可证 | CC BY 4.0 |

`data/` 保存 Parquet 数据分片，`videos/` 保存两路 H.264 RGB 视频，`meta/`
保存 LeRobot v3 元数据，`project_meta/` 保存采集计划、标定、归一化统计、固定
split 和质量验收信息。深度图以无损 `uint16` 毫米 TIFF 载荷保存在 Parquet 中。

冻结载荷的组合 SHA-256 为
`57a347891fea3dd3c45409b62b1a79c1bef55ab4467712ddcc98994c60ea99f5`，计算规则为
`sorted_sha256sum_lines_v1`。该指纹覆盖发布 README 和 LICENSE 加入前的 79 个
训练数据文件，便于与训练运行记录逐字节对应。

## 获取和使用

大体积 `*.parquet` 和 `*.mp4` 文件由 Git LFS 管理。完整克隆前先安装 Git LFS：

```bash
git lfs install
git clone git@github.com:songdaicheng666-dot/real_so101_vla_rl.git
cd real_so101_vla_rl
git lfs pull --include="datasets/so101_t0_100_lowlight_v1/**"
```

项目录制与训练配置使用相对路径
`datasets/so101_t0_100_lowlight_v1`。数据的 schema、成功 episode 筛选、固定划分
和归一化统计由仓库中的数据加载代码直接读取。

## 适用范围与限制

- 数据仅覆盖低照度条件下的单方块 T0 放置，不包含 P1～P3 顺序任务、失败恢复
  或闭环策略 rollout。
- 100 条正式 episode 均为人工确认的成功示范；另外 8 次人工丢弃只记录在
  `project_meta/attempts.jsonl`，没有进入训练 episode。
- OpenVLA-OFT 训练使用两路 RGB 和 proprioception；深度数据保留在数据合同中，
  但未用于该轮模型输入。
- RGB 视频包含真实采集环境，项目元数据包含原始硬件序列号、USB 拓扑和设备路径。
  使用者应自行评估其环境、隐私和部署风险。
- 离线动作误差不能替代真实机械臂上的安全评估或抓取成功率。

## 许可与署名

数据集使用 [CC BY 4.0](LICENSE)。复用、修改或再分发时请注明数据集名称、项目
来源、许可证，并说明是否做过修改。
