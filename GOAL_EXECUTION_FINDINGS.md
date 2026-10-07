# `goal.md` 执行阶段核查结果

记录日期：2026-10-07

执行目标：`/data1/liuwenhao/goal.md`

本文只记录截至当前已经得到的事实、正式实验的准入判断以及仍待解决的阻塞项。
尚未通过验证的事项不会标记为完成。

## 结论

目前不能直接用 `/data1/liuwenhao/Datasets/data` 中的 500 条旧 Pi0.5 episode
开展正式 matched-pair 实验。有限范围的数据考古没有找到可以直接证明
`episode -> env_seed`、逐 episode 成败标签、Pi0.5 checkpoint 身份和精确 task config 的
原始记录，所以旧数据必须从正式统计中排除。

按照 `goal.md` 的规则，正式实验需要改为重新采集 Expert/Pi0.5 matched pairs。
但新采集目前还有一个硬阻塞：本机尚未找到并验证一套可由当前 RoboTwin Pi0.5
loader 直接运行、身份和预处理信息完整、适用于三个正式任务的 Pi0.5 checkpoint/config。

## 1. 代码版本和工作区状态

已记录四个代码仓库的 commit、dirty 状态和 patch SHA256：

| 仓库 | commit | 当前状态 |
|---|---|---|
| `robotwin_consistency` | `325a5e5e30551a1b2693c3a8c5dc91d524e1c0a6` | 有本次新增文件及既有子模块状态变化 |
| `third_party/robotwin` | `2d3a8eb9804d2e5e239435d32f45da6277503d73` | detached HEAD，存在既有未提交修改 |
| `third_party/boundless-world-model` | `8a85a222fd65a48c778027d4aa44ffc8fa04206e` | clean，`dev/liuwenhao` |
| `third_party/WorldArena` | `b30f352bc727e84c5f6bcd484a3f348f983c708f` | clean，`custom/eight-metrics` |

RoboTwin 的既有 patch SHA256 为：

```text
4c232904d6e1b6b668f240480b418575e14e8e385850d7daf7b5b07bed1ececb
```

该 patch 涉及 `README.md` 和 `envs/adjust_bottle.py`，不是本次核查创建的修改，
因此没有重置、覆盖或随本文一起提交。未跟踪目录
`third_party/WorldArena-2.0/` 也没有加入正式实现或本次提交。

机器可读的 provenance 当前生成在：

```text
experiments/policy_shift/provenance/code_provenance.json
experiments/policy_shift/provenance/root.patch
experiments/policy_shift/provenance/robotwin.patch
experiments/policy_shift/provenance/bwm.patch
experiments/policy_shift/provenance/worldarena.patch
```

这些文件应在正式采集前再次生成，确保记录的是实际运行代码，而不是当前中间状态。

## 2. 旧 Pi0.5 数据完整审计

审计对象：

```text
/data1/liuwenhao/Datasets/data
```

已检查的内容包括：

- 全部 50 个 task 目录；
- 全部 50 个 `seed.txt`；
- 全部 500 个 HDF5 episode 及其 attributes；
- 全部 500 个 trajectory Pickle；
- 全部 500 个 MP4；
- 所有 `_result.txt`；
- 隐藏文件和其他 sidecar；
- 当前 RoboTwin 脚本及本地可达 Git 历史；
- 可用 shell history。

数据层面确认得到：

| 项目 | 结果 |
|---|---|
| task 数 | 50 |
| episode 数 | 500 |
| 每个 task 的 seed token 数 | 10 |
| 每个 task 的唯一 seed 数 | 10 |
| HDF5 head-RGB 长度 | 401–1701 帧 |
| 视频尺寸 | 640×480 |
| 视频 FPS | 30 |
| Pickle schema | 500 条均为 `{"actions": ...}` |
| 额外/隐藏 provenance sidecar | 0 |

但是没有找到以下直接证据：

- 证明 `seed.txt[i] == episode{i}` 的 writer、manifest 或逐 episode 日志；
- 生成 `_fail.mp4` 文件名的可达代码；
- 独立于文件名的逐 episode failure metadata；
- 原始 Pi0.5 checkpoint/config 身份和哈希；
- 数据生成时的精确 task config bytes；
- 初始 scene state 或 scene fingerprint。

当前和历史 eval 代码会把 seed 加入内存中的 `suc_test_seed_list`，但没有发现持久化
该列表的实现。`collect_data.py` 虽然会写 expert collection seed，但其 trajectory schema
和旧 Pi0.5 数据的 `{"actions": ...}` schema 不一致，不能作为旧数据的直接来源证明。

因此 500 条 episode 的正式状态全部为：

| 字段 | 状态 | 原因 |
|---|---|---|
| `seed_mapping_status` | `AMBIGUOUS` | 只有位置对应候选，没有直接证据 |
| `failure_label_status` | `AMBIGUOUS` | `_fail` 仅是文件名标签 |
| `policy_checkpoint_status` | `UNKNOWN` | 没有归档 checkpoint/config 身份 |
| `task_config_status` | `AMBIGUOUS` | `demo_clean` 目录名不等于配置原文 |
| 正式 matched-pair eligibility | `NO` | 不满足 `goal.md` 的 provenance gate |

正式决定为：

```text
usable for matched experiment: NO
```

文件顺序、文件时间戳以及 seed 数量一致都只能算辅助线索，不能把状态提升为
`CONFIRMED`。

## 3. 为什么必须重新采集 matched pairs

相同 seed 本身不足以证明 Expert 和 Pi0.5 来自同一个 initial scene。正式配对还需要固定
或记录：

- RoboTwin commit 和 dirty patch hash；
- task config；
- embodiment 和 camera config；
- domain randomization 配置；
- RNG 初始化与消耗顺序；
- instruction 和 scene setup 顺序；
- action 执行前的机器人状态、关键 actor pose 和相机状态 fingerprint。

新采集应以同一进程按 pair 执行，并分别保存 canonical state fingerprint、render
fingerprint 和完整 provenance。只有通过 `STATE_MATCH` gate 的 pair 才能进入 BWM 评测。

## 4. Pi0.5 checkpoint 的当前阻塞

当前 `third_party/robotwin/policy/pi05/checkpoints` 下没有可直接使用的 checkpoint。
RoboTwin 的 Pi0.5 loader 期望 checkpoint 目录同时具有模型和对应 assets/norm stats，
并且需要明确：

- `train_config_name`；
- `model_name`；
- `checkpoint_id`；
- `asset_id`；
- checkpoint commit/来源和 SHA256；
- observation preprocessing；
- action chunk 与机器人动作约定。

在本机目前只发现两份可能相关的较大权重：

```text
checkpoints/WorldArena2.0/pi05_adjust_bottle/model.safetensors
checkpoints/WorldArena2.0/pi05_click_bell/model_state_dict/full_weights.pt
```

它们看起来是 task-specific RLInf/OpenPi 产物，但尚未证明可被当前 RoboTwin Pi0.5
loader 正确加载，也没有找到覆盖三个正式任务的第三份对应 checkpoint。不能仅凭文件名
假设兼容，更不能把一个 task-specific checkpoint 静默当成通用三任务策略。

在以下条件满足前，正式新采集必须保持阻塞：

1. 明确选择正式使用的三个 task；
2. 找到每个 task 对应的可运行 Pi0.5 checkpoint，或明确指定一个有证据支持的通用
   checkpoint；
3. 用当前 RoboTwin loader 完成最小加载和单 episode smoke test；
4. 固化 checkpoint、assets、norm stat、config 和代码 commit 的哈希。

## 5. BWM runtime 当前状态

本机 BWM 代码、基础模型和 finetuned checkpoint 文件存在。已开始创建独立环境，目标版本
按照 BWM README 固定为：

```text
Python 3.10.20
torch 2.8.0
torchvision 0.23.0
torchaudio 2.8.0
CUDA 12.8 wheels
diffsynth 2.0.11
```

截至本文记录时，环境安装仍在进行，尚未完成 import/CUDA 验证，也尚未完成官方
one-window inference。因此 BWM runtime gate 当前应写为：

```text
NOT YET VERIFIED
```

GPU 检查只使用了物理卡 0–3；后续所有 GPU 命令也必须显式限制为
`CUDA_VISIBLE_DEVICES=0,1,2,3` 或其中的子集。

## 6. 对正式实验的直接影响

当前可以继续进行的工作：

- CPU-only 协议、schema、fingerprint、cadence 和转换单元测试；
- BWM 独立 runtime 环境和 one-window smoke test；
- WorldArena adapter 的正式路径和输出 schema 修正；
- 新 matched collector 的预检、manifest 和 fail-fast 实现。

当前不能宣称完成的工作：

- 3 tasks × 2 seeds × Expert/Pi0.5 的 6 个正式 matched pairs；
- `MATCHED_INITIAL_STATE_AUDIT.md` 中的正式 `STATE_MATCH` 结论；
- 12 个 BWM middle-window generation；
- 基于上述 generation 的正式 smoke comparison；
- 任何 expanded experiment 或统计结论。

如果 Pi0.5 checkpoint/config 无法补齐，正确处理方式是报告硬阻塞，不是回退到 provenance
不完整的旧数据，也不是猜测 episode/seed 对应关系。

## 7. 当前证据文件

本次核查已经在本地生成：

```text
PI05_OLD_DATA_PROVENANCE.md
experiments/policy_shift/provenance/pi05_old_episode_audit.jsonl
experiments/policy_shift/provenance/code_provenance.json
```

其中 JSONL 包含全部 500 条 episode 的逐条审计结果；这些执行产物将在 schema 和代码稳定后
统一纳入正式提交。本文是当前阶段的简明、可审阅结论。
