# 2026 挑战赛 MuJoCo 双底图场景

这个目录保存项目自有的 2026 挑战赛场景资产，并引用项目中原样 vendored 的
`../SO101_menagerie/` 机械臂网格。当前包含完整 SO101 动力学模型、静态底板、
四节动态 AAA 电池、干净纹理、灯光、腕部相机和俯视相机。任务奖励和
Gymnasium/RL 代码位于项目的 `rewards/`、`envs/basic_t0.py` 和 `rl/`，不写入
MJCF 资产文件。

## 当前完成状态

两套任务底图、SO101 完整动力学模型、四节动态 AAA 电池及交互抓取 Demo 已完成集成。
在 `basic_t0` 场景中已经过实际人工操作验证：六个舵机关节可正常控制，夹爪能够与
电池发生接触、闭合夹持并将电池抓起。这说明当前场景中的机械臂执行器、夹爪碰撞、
电池自由刚体和接触动力学能够共同完成基本抓取，任务仿真环境的基础场景搭建已大致完成。

本次人工抓取验收针对 `basic_t0` 场景；`sequence_p1_p2_p3` 复用同一套机械臂和电池
动力学模型，但仍应在后续任务流程接入时单独验收其完整操作路径。当前已经为
`basic_t0` 实现目标区域判定、轻量随机复位、特权状态观测、连续 action chunk、
分阶段奖励、终止条件以及 PPO/GRPO baseline；序列场景的对应任务环境尚未实现。

## 文件

- `source/challenge_task_board_2026.pdf`：原始双页底图留档。
- `layouts.yaml`：物理尺寸、坐标、颜色和纹理生成参数的唯一数据源。
- `textures/basic_t0.png`：电池摆放区 + T0，`2800×2200` RGB。
- `textures/sequence_p1_p2_p3.png`：电池摆放区 + P1/P2/P3，`2800×2200` RGB。
- `so101_competition.xml`：基于 Menagerie 模型的比赛坐标适配层。
- `batteries_aaa.xml`：两个场景共享的四节动态 AAA 电池定义。
- `scene_basic_t0.xml`、`scene_sequence_p1_p2_p3.xml`：可独立加载的 MJCF。

两套第三方原始资产分别位于相邻的 `../SO101/` 和 `../SO101_menagerie/`；
来源、许可证与逐文件校验值记录在 `../THIRD_PARTY.md`、
`../SO101_NEXUS_LICENSE.md` 和 `../SO101_NEXUS_SHA256SUMS`。

## 模型引用关系

两个 `scene_*.xml` 是最终场景入口，同时共享机械臂适配模型和 AAA 电池集合。
机械臂适配模型再读取原样 vendored 的 Menagerie STL 网格：

```text
scene_basic_t0.xml ───────────────┬─ include → so101_competition.xml
                                  │                         │
                                  │                         └─ meshdir → ../SO101_menagerie/assets/*.stl
                                  └─ include → batteries_aaa.xml → 四节电池

scene_sequence_p1_p2_p3.xml ──────┬─ include → so101_competition.xml
                                   │                         │
                                   │                         └─ meshdir → ../SO101_menagerie/assets/*.stl
                                   └─ include → batteries_aaa.xml → 四节电池
```

运行 MuJoCo 时只需加载对应的 `scene_*.xml`。`so101_competition.xml` 和
`batteries_aaa.xml` 都不是额外任务场景，分别是共享的机械臂定义和电池定义。

## 尺寸与坐标

原 PDF 页面标注尺寸为 `700×540 mm`，而派生仿真底板按项目决定使用
`700×550 mm`。这个 10 mm 差异是有意保留的，使数字尺寸优先的
`250 mm 电池区 + 100 mm 间隔 + 200 mm 底座圆`恰好容纳于底板高度。

世界原点在机械臂底座圆心：`x` 指向底图右侧，`y` 指向远离机械臂的方向，
`z` 向上。底座圆直径为 `0.20 m`，与底图下边缘相切。底板世界边界为：

```text
x: [-0.392853, 0.307147] m
y: [-0.100000, 0.450000] m
```

MJCF 中的 `board_collision` 是顶面位于 `z=0` 的有限静态 box；
`board_visual` 是稍高于顶面的无碰撞纹理 plane，因此图像不会改变未来物体的接触行为。

## SO101 模型

比赛场景使用 `SO101_menagerie/so101.xml` 的完整动力学模型，包括 6 个关节、
6 个位置执行器、操作碰撞体、`gripperframe` TCP site 和 `wrist_cam`：

```text
shoulder_pan
shoulder_lift
elbow_flex
wrist_flex
wrist_roll
gripper
```

vendored 原件保持不变。项目自有的 `so101_competition.xml` 只调整网格路径，并将
机械臂根基座绕世界 `z` 轴旋转 `+90°`，使模型自身的 `+x` 前向与任务底图的
世界 `+y` 对齐。基座中心仍为 `(0,0,0)`，默认关节位置和控制量均为 Menagerie 零位。

## AAA 电池模型

`batteries_aaa.xml` 集中定义红、黄、蓝、绿四节 AAA（7号）电池，分别对应底图的
A、B、C、D 位置。模型采用 Energizer E92 数据表中的最大尺寸和典型质量：直径
`10.5 mm`、总长 `44.5 mm`、质量 `11.5 g`。每节电池由一个承担全部质量和接触的
圆柱碰撞体，以及无碰撞的彩色桶身和银色正负极组成。

每个电池 body 都有独立 `freejoint`，因此能够被夹爪抓取、推动、滚动和掉落：

```text
battery_red       / battery_red_joint
battery_yellow    / battery_yellow_joint
battery_blue      / battery_blue_joint
battery_green     / battery_green_joint
```

共享文件中的规范布局使用 `basic_t0` 的 A/B/C/D 中心，圆柱长轴沿世界 `+y`。
`scene_basic_t0.xml` 以零偏移引用它；序列场景通过外层 `<frame>` 将整组电池沿
世界 `-x` 平移 `0.037866 m`，从而与该页底图的四个占位中心对齐。

增加两个场景共同使用的电池时，只需在 `batteries_aaa.xml` 中增加 body，并同步更新
`layouts.yaml`。若以后两个场景需要不同数量或不能用同一个刚性偏移表达的布局，应由
场景生成脚本或 RL reset 逻辑设置，而不复制电池的几何定义。

Energizer E92 官方数据表：<https://data.energizer.com/pdfs/e92-1119.pdf>

## 底图源文件校验

```text
SHA-256  f537b63235542f3a509b6f695651a486232e65c02d21fc457a876f893eb277c6
```

生成脚本会先校验此哈希，再使用 `Noto Sans CJK SC` 重新绘制干净纹理。
它不拉伸或截图 PDF，也不保留尺寸标注与测量虚线。运行 MuJoCo 时只读取已提交的 PNG，
不需要字体、Pillow 或原 PDF。

## 重新生成纹理

```bash
conda run -n lerobot python scripts/generate_competition_2026_boards.py
```

若字体不在默认 Linux 路径，传入 `--font /path/to/NotoSansCJK-Regular.ttc`。
对 TrueType Collection 还可用 `--font-index` 选择 SC 字体面；系统默认 TTC 的 SC 索引为 2。

## 人工验收

验收结论：`basic_t0` 交互 Demo 已实际完成机械臂控制、夹爪闭合和电池抓取/抬升，
基本抓取链路可用。以下命令保留用于重复验证和后续模型参数调整后的回归检查。

![T0 场景中 SO101 双侧夹持并抬升蓝色 AAA 电池的人工验收截图](evidence/t0_manual_grasp_validation.png)

截图左侧显示蓝色电池被夹爪夹持并离开底板；右侧终端记录了固定夹爪与活动夹爪
同时接触、电池高于初始位置 `10 mm` 和抓取成功诊断。该证据验证的是手动控制下的
基本抓取链路，不代表自动放置、序列任务、奖励或 RL 策略已经完成。

要在 T0 场景中通过键盘或 Viewer 右侧 actuator 滑块控制机械臂、手动测试电池抓取，运行：

```bash
conda run --no-capture-output -n lerobot \
  python scripts/demo_mujoco_t0_grasp.py
```

先用顶排数字 `1～6` 选择对应真实舵机 ID：底座旋转、肩部抬升、肘部、腕部俯仰、
腕部旋转和夹爪；再用 `A/D` 减小/增大所选目标，选择夹爪时分别表示闭合/打开。
`Space` 暂停/继续，`Backspace` 完整复位。脚本会撤销这些控制键同时触发的 Viewer
显示开关，因此操作时不会隐藏机械臂、电池或改变可视化效果。终端会报告当前选择、
目标角度、单侧/双侧夹爪接触、电池抬升 10 mm，以及“双侧接触并抬升”的成功诊断。

若只想确认场景能加载并稳定运行而不打开 GUI，可执行：

```bash
conda run -n lerobot python scripts/demo_mujoco_t0_grasp.py --headless-check
```

也可以继续直接打开两个原始 Viewer 场景：

```bash
/usr/bin/python3 -m mujoco.viewer \
  --mjcf=src/real_so101_vla_rl/assets/mujoco/competition_2026/scene_basic_t0.xml

/usr/bin/python3 -m mujoco.viewer \
  --mjcf=src/real_so101_vla_rl/assets/mujoco/competition_2026/scene_sequence_p1_p2_p3.xml
```

Viewer 默认使用斜俯视的 `Free` 相机，可用鼠标拖拽旋转、平移和缩放。
选择固定相机 `overview` 可以查看完整任务底图；`wrist_cam` 是机械臂自带的腕部相机。
