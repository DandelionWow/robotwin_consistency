# 本机两套 RoboTwin 数据说明

本文记录以下两个本机目录在 **2026-10-07** 的实际内容、格式和用途：

- `/data1/liuwenhao/Datasets/RoboTwin2.0_640_480_lerobot`
- `/data1/liuwenhao/Datasets/data`

结论：两者覆盖相同的 50 个 RoboTwin 任务，但不是相同 episode 的两份副本。
前者是 Clean-50 专家示范的 LeRobot 训练数据；后者是 Pi0.5 策略在同一批任务上的
评测 rollout。训练世界模型时使用前者，做 Pi0.5 分布偏移或策略 rollout 评测时使用
后者。

## 总览

| 项目 | `RoboTwin2.0_640_480_lerobot` | `data` |
|---|---:|---:|
| 本机占用（`du -sh`） | 14 GB | 35 GB |
| 顶层任务数 | 50 | 50 |
| 每任务 episode 数 | 50 | 10 |
| episode 总数 | 2,500 | 500 |
| 总帧数 | 552,287 | 310,500 |
| 单 episode 帧数 | 75–895，平均 220.9 | 401–1,701，平均 621.0 |
| 主要格式 | Parquet + AV1 MP4 + JSON/JSONL | HDF5 + Pickle + H.264 MP4 + TXT |
| 图像视角 | 6 个 | 5 个 |
| 主要用途 | 专家数据训练、GT 验证 | Pi0.5 策略 rollout 评测 |
| 是否包含语言指令 | 是，每个 episode 一条 | 未保存实际选中的指令文本 |
| 软链接 | 0 | 0 |

两个目录去掉 `data` 中任务名末尾的 `_pi05` 后，50 个任务名完全一致。例如：

```text
RoboTwin2.0_640_480_lerobot/adjust_bottle
data/adjust_bottle_pi05
```

这只表示任务集合相同，不表示其中的 episode 一一对应。

## `RoboTwin2.0_640_480_lerobot`

### 定位

这是按任务拆分的 LeRobot v2.1 数据集。每个任务都包含一个
`aloha-agilex_clean_50` 子集，共 50 条专家示范。磁盘上共有：

- 2,500 个 Parquet episode；
- 15,000 个 MP4，即每个 episode 有 6 个相机视频；
- 150 个 JSONL 和 101 个 JSON；
- 17,751 个文件，文件内容合计约 14.20 GB。

目录结构如下：

```text
RoboTwin2.0_640_480_lerobot/
├── stats.json                         # 根目录 joint_abs/eef_abs 统计
├── adjust_bottle/
│   └── aloha-agilex_clean_50/
│       ├── data/chunk-000/
│       │   ├── episode_000000.parquet
│       │   └── ... episode_000049.parquet
│       ├── videos/chunk-000/
│       │   ├── observation.images.front/
│       │   ├── observation.images.head/
│       │   ├── observation.images.left/
│       │   ├── observation.images.right/
│       │   ├── observation.images.table_left/
│       │   └── observation.images.table_right/
│       └── meta/
│           ├── info.json
│           ├── stats.json
│           ├── episodes.jsonl
│           ├── episodes_stats.jsonl
│           └── tasks.jsonl
└── ... 其余 49 个任务
```

### 图像与时序

`meta/info.json` 声明并由样本文件验证：

- 机器人类型为 `aloha-agilex`；
- 采样率为 30 FPS；
- 图像为 640×480、三通道、AV1、`yuv420p`；
- 六个视角为 `front`、`head`、`left`、`right`、`table_left` 和
  `table_right`；
- 每个相机同时保存 `cam2world_gl` 4×4、`extrinsic_cv` 3×4 和
  `intrinsic_cv` 3×3 标定矩阵。

### Parquet 字段

每行对应一帧。核心字段是：

| 字段 | 形状 | 含义 |
|---|---:|---|
| `joint_abs` | 14 | 双臂六关节加两个夹爪的记录值；名称本身不证明它是实际 qpos |
| `eef_abs` | 16 | 左右臂各自的 XYZ + WXYZ 四元数 + 夹爪 |
| `observation.cameras.*` | 4×4 / 3×4 / 3×3 | 六个相机的标定矩阵 |
| `timestamp` | 1 | 帧时间戳 |
| `frame_index` | 1 | episode 内连续帧号 |
| `episode_index` | 1 | episode 编号 |
| `index` | 1 | 当前任务子数据集内的全局索引 |
| `task_index` | 1 | 任务索引 |

图像本身不放在 Parquet 中，而是放在对应相机目录的 MP4 中。
`episodes.jsonl` 保存每个 episode 的长度和自然语言指令；`stats.json` 与
`episodes_stats.jsonl` 分别保存任务级和 episode 级统计。

注意：每个任务自身的 LeRobot 元数据把 `0:50` 全部标为 `train`。当前
Ctrl-World 实验在配置层另外划分为：

- episode 0–39：训练；
- episode 40–49：验证。

这不是磁盘数据的第二份物理切分。

### 在当前项目中的用途

`world_simulator_baseline/runners/ctrl_world/configs/train.yaml` 将该目录配置为
`source_root`，期望 50 个任务、2,500 个 episode 和 552,287 个原始帧。
转换器读取 `head` 相机视频和 Parquet 中的 16D `eef_abs`，将 WXYZ 四元数转换成
RPY 后得到 14D EEF 条件：

```text
[left xyz, left rpy, left gripper,
 right xyz, right rpy, right gripper]
```

因此，这套数据是当前 Ctrl-World 训练和 GT-action 离线验证的主要源数据。

## `data`

### 定位

尽管目录名只有 `data`，其内容不是上面 LeRobot 专家数据的原始版，而是
RoboTwin Pi0.5 策略评测 rollout。每个任务名以 `_pi05` 结尾，并包含
`demo_clean/run_0000`。磁盘上共有：

- 500 个 HDF5 episode；
- 500 个 Pickle 动作轨迹文件；
- 500 个 MP4；
- 50 个 `seed.txt` 和 50 个 `_result.txt`；
- 1,600 个文件，文件内容合计约 37.17 GB。

每个任务恰好 10 个 episode，编号为 0–9：

```text
data/
├── adjust_bottle_pi05/
│   └── demo_clean/run_0000/
│       ├── data/
│       │   ├── episode0.hdf5
│       │   └── ... episode9.hdf5
│       ├── _traj_data/
│       │   ├── episode0.pkl
│       │   └── ... episode9.pkl
│       ├── video/
│       │   ├── episode0_fail.mp4
│       │   └── ... episode9_fail.mp4
│       ├── seed.txt
│       └── _result.txt
└── ... 其余 49 个任务
```

### HDF5 字段

HDF5 保存逐帧仿真观测，主要结构为：

```text
endpose/
  left_endpose       [T, 7]
  left_gripper       [T]
  right_endpose      [T, 7]
  right_gripper      [T]
joint_action/
  left_arm           [T, 6]
  left_gripper       [T]
  right_arm          [T, 6]
  right_gripper      [T]
  vector             [T, 14]
observation/
  front_camera/
  head_camera/
  left_camera/
  right_camera/
  tabletop_camera/
pointcloud           [T, 0]
```

每个相机组含 RGB 字节、`cam2world_gl`、`extrinsic_cv` 和 `intrinsic_cv`。
这里有一个 `tabletop_camera`，而 LeRobot 专家数据有 `table_left` 和
`table_right` 两个桌面视角，所以两套数据的相机集合并不相同。

抽检 `adjust_bottle` 的 episode 0：HDF5 有 401 帧；对应 Pickle 的
`actions` 是 8 个 `[50, 14]` 动作块；对应 MP4 是 640×480、30 FPS、H.264、
401 帧。不同 episode 长度不固定，全体 500 条合计 310,500 帧。

### 成功标签和指令限制

当前目录中的 500 个 MP4 文件名都以 `_fail.mp4` 结尾。每个 `_result.txt`
记录 `Instruction Type: unseen` 和任务级汇总成功率，但它不是逐 episode 标签；
当前 50 个汇总值中有 36 个为 0，其余为不同的非零值。因此：

- 对这 500 条已保存轨迹，可记录文件名给出的 `fail` 标记；
- 不应把 `_result.txt` 中的任务级比例反推到某个具体 episode；
- 实际抽取到的 unseen 指令文本没有随 rollout 保存。

当前 `policy_rollout_adapter.py` 因此从 LeRobot 数据每个任务的验证 episode
40–49 中确定性选择一条中性指令作为 fallback，并在输出 manifest 中明确记录
“原始 rollout prompt 不可用”。使用这套数据做语言条件实验时必须保留这一限制。

### 在当前项目中的用途

`world_simulator_baseline/runners/ctrl_world/policy_rollout_adapter.py` 将该目录作为
默认 `source_root`，并校验：

- 50 个任务；
- 每任务 episode 0–9；
- 共 500 个 episode；
- 310,500 个原始帧。

适配器从 HDF5 的左右 `endpose` 和 gripper 字段重建 16D `eef_abs`，再转换成与
训练数据相同顺序的 14D XYZ/RPY/gripper 条件。它不直接使用 `_traj_data` 中的
策略动作块作为世界模型条件。

## 两者应如何选择

| 场景 | 应使用的数据 |
|---|---|
| 训练当前 Ctrl-World RoboTwin 模型 | `RoboTwin2.0_640_480_lerobot` 的 episode 0–39 |
| 使用专家 GT 条件做离线验证 | `RoboTwin2.0_640_480_lerobot` 的 episode 40–49 |
| 检查模型面对 Pi0.5 rollout 的分布偏移 | `data`，经 `policy_rollout_adapter.py` 转换 |
| 训练 Pi0.5 策略 | 不能仅凭目录名认定使用其中任意一个；需按策略训练配置确认 |
| 比较专家轨迹与策略轨迹 | 可以按任务比较，但不能按 episode 编号配对 |

简言之：

```text
LeRobot Clean-50 专家示范
    ├── 40 episodes/task -> 当前世界模型训练
    └── 10 episodes/task -> GT 验证与 fallback 指令来源

Pi0.5 rollout (`data`)
    └── 10 episodes/task -> 策略分布下的世界模型评测
```

## 可复现性注意事项

1. 两套目录都在 Git 仓库之外，Git 提交不会包含数据本体。
2. 配置中使用了绝对路径，迁移服务器时需要同步修改路径或保持相同目录结构。
3. `data` 根目录没有独立的数据集说明或完整 manifest；本文对其来源和用途的判断
   同时依据磁盘结构、RoboTwin 字段以及当前消费它的
   `policy_rollout_adapter.py`。
4. 不要用 `data` 这个泛化名称判断其为训练集；它在当前实验中是 Pi0.5 rollout
   评测源。
5. 不要把 HDF5 的 `joint_action/vector`、Pickle 的策略输出块和转换后的 14D EEF
   条件视为同一语义。当前世界模型适配器明确使用执行后的 end-effector 记录。
