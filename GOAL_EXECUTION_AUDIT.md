# `goal.md` 可执行性与冲突审计

审计日期：2026-10-07

审计对象：`/data1/liuwenhao/goal.md`

## 结论

实验的核心设计是合理的：使用相同 task、environment seed 和初始场景构造
Expert/Pi0.5 matched pair；用 GT history 和 realized EEF 运行 BWM；最后以 pair 为统计单位。

但当前状态还不能直接开展正式 matched experiment。主要阻塞不是 BWM checkpoint 本身，
而是旧 Pi0.5 数据的 provenance、episode 与 seed 的对应关系、逐 episode failure 标签，以及
现有评测代码尚未实现 `goal.md` 规定的 pair-level 协议。

如果不能把旧数据的 seed mapping 和 failure label 提升到 `CONFIRMED`，不应猜测或继续把
这 500 条旧 episode 用作正式核心实验，而应重新采集 matched 数据。

## 一、阻断性问题

### 1. 旧 Pi0.5 的 `episode -> env_seed` 尚不能标记为 `CONFIRMED`

当前 RoboTwin `script/eval_policy.py` 可以确认以下运行逻辑：

```text
candidate seed
-> expert planner 检查
-> 仅保留 expert-success seed
-> 使用相同 seed 重置环境
-> 执行 policy
```

但是代码只把 seed 加入内存中的 `suc_test_seed_list`，没有发现把该列表写成
`seed.txt` 的逻辑。

旧数据虽然包含 50 个 `seed.txt`，每个文件含 10 个 seed token，并同时存在
`episode0` 到 `episode9`，但没有找到以下交叉证据：

- 原始生成脚本；
- 逐 episode 日志；
- `scene_info.json`；
- 初始场景 metadata；
- 数据生成时的代码 commit；
- 明确声明 `seed.txt[i] == episode{i}` 的 manifest。

此外，旧数据与当前代码还存在 provenance 差异：

- 旧 `_traj_data/episode0.pkl` 是 `{"actions": ...}`，其中包含 8 个 `[50, 14]`
  Pi0.5 action chunk；
- 当前 RoboTwin `save_traj_data()` 保存的是 `left_joint_path` 和
  `right_joint_path`；
- 旧 MP4 是 640x480、30 FPS；
- 当前 `eval_policy.py` 的录像命令使用 10 FPS，并保存为 `episodeN.mp4`，没有发现
  `_fail.mp4` 的命名逻辑。

因此，现阶段最严谨的状态是：

```text
mapping_status = AMBIGUOUS
```

按照 `goal.md` 的准入条件，这 500 条旧数据目前不能直接进入正式 matched experiment。

### 2. 相同 seed 不足以证明相同 initial scene

相同 seed 只有在以下条件同时一致时才有意义：

- RoboTwin commit；
- task config；
- embodiment；
- camera config；
- domain randomization 配置；
- RNG 初始化和消耗顺序；
- scene setup 与 instruction generation 顺序；
- renderer/denoiser 相关配置。

当前仓库中的 `third_party/robotwin` 存在可复现性风险：

- 父仓库记录的 gitlink 与实际 checkout commit 不一致；
- 子模块处于 detached HEAD；
- 子模块还有未提交修改。

因此，在固定或完整记录 RoboTwin 源码状态以前，即使 seed 数字相同，也不能证明新生成
expert 和旧 Pi0.5 rollout 来自相同初始场景。

当前 `pair_id = task + env_seed` 也不够表达完整实验身份。建议 manifest 另外保存：

```text
task_config_hash
robotwin_commit
robotwin_patch_hash
camera_config_hash
scene_fingerprint
```

其中 `scene_fingerprint` 应由 action 执行前的相机参数、机器人状态和关键 actor pose
确定性生成。

### 3. BWM 的精确训练 distribution 和 normalization stat 未确认

可以确认 public BWM checkpoint 的以下信息：

- checkpoint 文件和 SHA256；
- 14D `state_pose/eef_abs` 条件；
- 81 total frames；
- 9 history frames；
- 72 future frames。

但目前不能确认：

- checkpoint 使用的精确训练数据版本；
- 本机 Clean-50 是否就是该 checkpoint 的训练集或严格 ID distribution；
- checkpoint 训练时使用的确切 normalization stat。

`BWM_NORMALIZATION_AUDIT.md` 的结论仍是：

```text
BWM training stat provenance = UNCONFIRMED
```

因此 Group A 更严谨的名称应是：

```text
expert reference distribution
```

而不是未经证明的 `BWM training/ID distribution`。

使用候选 `--reference-stat` 计算出的 clip rate，只能解释为：

```text
out-of-reference-range rate
```

不能在 stat provenance 未确认时直接称为相对于 BWM 真实训练分布的 OOD rate。

### 4. 旧数据的 failure-only 标签证据有限

当前 500 个已保存 MP4 文件名全部以 `_fail.mp4` 结尾。但是每个 `_result.txt` 只包含
任务级汇总成功率，不是逐 episode 标签；50 个任务中还有部分任务的汇总成功率非零。

所以，在没有恢复原始保存逻辑以前，正式文档最好使用：

```text
filename-labelled Pi0.5 failure rollouts
```

不应仅根据文件名把每条 episode 的失败状态当作已经由独立 metadata 验证的事实。

## 二、科研解释边界

### 1. 这是 privileged GT-conditioned offline evaluation

当前 public BWM 的 `eef_abs` 实际读取的是执行后记录的 realized EEF/gripper state，而不是
Pi0.5 原始 action command。对于每个 future window，BWM 会得到未来真实 realized EEF
trajectory。

因此本实验实际测量的是：

> 给定 GT history 和真实未来 EEF 状态轨迹时，BWM 对未来视觉结果的条件生成能力，是否在
> Pi0.5 failure-induced trajectory 上弱于 expert trajectory。

它不能直接证明：

- BWM 能否从 policy command 正确预测动力学；
- BWM 的 online planning 能力下降；
- autoregressive rollout 稳定性下降；
- 所有 policy rollout 都会使世界模型退化。

论文标题、摘要、图表和结论都应持续保留
`GT realized-EEF-conditioned offline prediction` 这一限定。

### 2. 当前样本存在条件选择

按照可见的 RoboTwin eval 逻辑，seed 首先经过 expert-success 筛选；旧目录又只保存了
带 `_fail` 名称的 Pi0.5 视频。因此核心样本更接近：

```text
expert-feasible initial scenes
+ filename-labelled Pi0.5 failure trajectories
```

这个实验可以回答 failure-conditioned 问题，但不能推广到 generic Pi0.5 distribution。
`goal.md` 对 claim 的限制方向是正确的，建议进一步把上述双重选择条件写入最终结论。

## 三、现有 `experiments/policy_shift` 与目标协议的冲突

当前代码已经实现 GT-history 独立窗口、14D stat 校验、checkpoint/stat hash 和严格的
PSNR/SSIM shape 检查，但距离新版协议还有以下差距。

### 1. WorldArena 路径冲突

当前 `experiments/policy_shift/README.md` 示例仍然使用：

```text
third_party/WorldArena-2.0/video_quality_ood
```

这与 `goal.md` 指定的唯一正式实现：

```text
third_party/WorldArena
commit b30f352bc727e84c5f6bcd484a3f348f983c708f
```

直接冲突。

当前 adapter 还假定传入目录下直接存在 `WorldArena/basic_metrics.py`；指定子模块的真实路径
是 `third_party/WorldArena/video_quality/WorldArena/basic_metrics.py`，CLI 根目录语义也需要
统一。

### 2. 只实现了 PSNR 和 SSIM

`worldarena_adapter.py` 当前只调用 WorldArena 的 PSNR 和 SSIM。以下指标尚未接入现有
policy-shift evaluator：

- Aesthetic Quality；
- Image Quality；
- JEPA Similarity；
- Subject Consistency；
- Trajectory Accuracy；
- Depth Accuracy。

### 3. 输出 artifact 不符合目标格式

当前 evaluator 输出 PNG 帧目录：

```text
windows/.../gt_future/frame_*.png
windows/.../pred_future/frame_*.png
```

目标要求的是严格对齐的：

```text
gt_future.mp4
pred_future.mp4
```

并要求全局唯一 sample ID、相同 frame count、resolution 和 fps。

### 4. env seed 与 generation seed 没有分离

当前输出只有一个含义模糊的 `seed` 字段，并使用：

```python
generation_seed = base_seed + global_index
```

这会依赖 metadata 顺序和 global sample index，也不能保证 matched expert/policy sides 得到
相同 generation seed。

需要改为独立字段：

```text
env_seed
generation_seed
```

并从稳定编码后的 `pair_id + window_slot + repeat_id` 计算 generation seed。不能直接使用
Python 内置 `hash()`，因为它可能受 hash randomization 影响。

### 5. 窗口协议不一致

当前代码按固定 stride 构造所有窗口，再截取前 N 个。它没有实现严格定义的：

```text
early / middle / late
```

也没有记录 `window_slot` 和 `repeat_id`。

### 6. 当前统计不是 pair-level

当前聚合器：

- 直接在 window 值上计算 bootstrap；
- 分别计算 expert/policy distribution mean；
- 再用两个总体均值相减；
- 没有验证 expert 与 policy 是否属于同一个 pair。

这不满足：

```text
windows -> episode-side score
expert score - policy score -> per-pair gap
paired bootstrap over pairs
```

### 7. 缺少 conditioning OOD diagnostics

尚未实现 normalization 前的：

- below-p01 rate；
- above-p99 rate；
- total clip rate；
- per-dimension clip rate。

### 8. 输入格式尚未统一

当前 evaluator 期望 BWM 转换格式，包括视频 metadata 和宽度为 26 的
`observation.state` Parquet。

但是现有三组源数据分别是：

- Clean-50：LeRobot Parquet + 16D quaternion `eef_abs`；
- Pi0.5：原始 HDF5 + policy action Pickle；
- 待生成 matched expert：保存格式尚未最终确定。

因此方案中还缺少一个正式的 BWM conversion 阶段。A/B/C 三组必须使用同一套：

- head-camera 选择；
- WXYZ quaternion 到 RPY 的转换；
- 14D 顺序；
- frame indexing；
- fps/cadence；
- metadata schema；
- stat application convention。

Ctrl-World 转换器只能作为历史参考，不能把其 checkpoint、配置或 stat 混入本实验。

### 9. 测试覆盖不足

当前已有 3 个单元测试并全部通过：

- window 连续性；
- 14D reference stat；
- gap 符号。

但 `goal.md` 要求的 13 类约束中，seed mapping、matched identity、pair bootstrap、JEPA 排除、
sample ID、WorldArena schema 等仍没有测试。

## 四、WorldArena 指标风险

### 1. Trajectory Accuracy 尚不能直接作为成熟 primary metric

Trajectory Accuracy 需要先对 GT 和 prediction 执行一致的 gripper detection/tracking，并对齐：

- tracker identity；
- 像素坐标定义；
- window 边界；
- 双臂轨迹顺序；
- detection 缺失值；
- tracking failure 的统计规则。

如果只剔除 tracker 失败样本，会产生与生成质量相关的选择偏差。必须预先规定失败样本如何
计分或报告，并单独输出 tracker success rate。

在完成并验证该 adapter 前，建议 PSNR/SSIM 保持 primary，Trajectory Accuracy 标为
`NEEDS_ADAPTATION`。

### 2. Depth Accuracy 更适合作为辅助几何指标

当前 WorldArena Depth Accuracy 来自单目深度估计，并进行尺度对齐。它不是 RoboTwin
simulator ground-truth depth error。因此它可以诊断几何一致性，但不应被解释为真实 metric
depth 或完整动作后果正确性。

### 3. Image/Aesthetic/Subject 指标不是 GT fidelity

这些指标可以帮助解释生成失败类型，但高分不能说明 future 与 GT 相同。例如，一个清晰、
美观、主体稳定但动作结果错误的视频仍可能得到较高分。

### 4. JEPA 必须保持 set-level

当前 WorldArena JEPA 输出一个 dataset-level score，但聚合器会为了 CSV schema 把同一个
JEPA 值复制到每个 sample 行。后续 policy-shift 聚合必须读取原始 set-level result，不能
把这些复制值当作独立窗口或 episode 进行 bootstrap。

比较 A/B/C 的 JEPA 时还必须平衡：

- task 组成；
- episode 数；
- 每 episode 窗口数；
- window slot；
- generation repeat 数。

否则 set-level 差异可能来自集合组成而不是 distribution shift。

### 5. 应同时保存 raw 与 normalized metric

当前 WorldArena fork 会使用固定 leaderboard bounds 归一化 Trajectory/Depth，并截断到
`[0, 1]`。在新的 72-frame RoboTwin window 分布上，这种截断可能饱和并掩盖 pair gap。

建议至少同时保存：

```text
raw_metric
worldarena_normalized_metric
normalization_bounds
```

正式分析需要预先指定使用哪一个值，不能看到结果后再选择。

## 五、协议中尚未定义清楚的细节

### 1. early/middle/late 的公式

必须预先固定三个窗口的 start index 公式，以及窗口重叠和短 episode 的处理方式。不能只写
语义上的 early/middle/late。

### 2. 少于 81 帧的 episode

Clean-50 中最短 episode 只有 75 帧，无法形成 9+72 的完整窗口。需要预先选择：

- 排除；
- 使用一个有科学依据的 padding/masking 协议；
- 或缩短所有组的统一窗口长度。

如果只排除短 expert episode，必须报告排除数量和任务分布，避免产生 selection bias。

### 3. temporal cadence / FPS

目前存在多个口径：

- Clean-50 和抽检 Pi0.5 视频：30 FPS；
- 当前 RoboTwin eval 录像代码：10 FPS；
- BWM CLI 默认输出参数：24 FPS。

必须区分：

- 输入帧的真实时间间隔；
- 是否进行了 temporal subsampling；
- 输出 MP4 metadata fps。

仅仅让 GT 和 prediction 的 MP4 metadata 相同，并不能证明两组窗口覆盖相同物理时长。

### 4. generation repeats 的统计层级

正式实验若使用 3 个 generation seeds，应先在每个 pair/side/window 内聚合 repeat，再聚合
window，最后形成 per-pair gap。三个 diffusion repeats 不能被当作三个独立 pair，否则会造成
pseudoreplication。

### 5. “禁止 resize”的边界

建议明确为：

> 禁止在保存 GT/pred paired artifacts 或对齐两侧时改变几何尺寸；允许 WorldArena 各
> backbone 按其官方实现执行必要的内部模型预处理，但必须记录。

完全禁止所有 metric-internal resize 会使 DINO、MUSIQ、Depth 等模型无法按其规定输入运行。

## 六、仓库与运行环境现状

### 已存在的资源

- BWM `step-12000.safetensors` 存在；
- Wan2.2-TI2V-5B 底模文件存在；
- WorldArena 和 WorldArena_JEPA Conda 环境存在；
- 八指标需要的大部分权重存在于另一份 `/data1/liuwenhao/Projects/WorldArena`。

### 当前阻塞

- 当前检查到的 Conda 环境均没有 `diffsynth`，现有 BWM evaluator 无法加载 pipeline；
- 当前会话中 `nvidia-smi` 无法连接 NVIDIA driver，尚不能验证 GPU 推理；
- 指定的 `third_party/WorldArena` 子模块内部没有指标权重；
- WorldArena config 的 checkpoint 路径需要配置到一个正式、可复现的外部 checkpoint 目录，
  不能继续依赖第二份 WorldArena 的代码目录；
- `third_party/robotwin` 需要先固定 commit/patch 状态。

这些问题是可修复的环境和工程问题，不代表实验设计不可实现。

## 七、计算规模风险

如果正式实验包含：

```text
500 pairs
x 2 sides
x 3 windows
x 3 generation seeds
```

则最多需要约 9,000 次 BWM 81-frame generation，尚未计算 Group A 和八指标开销。
BWM 是 5B video model，并默认使用 50 inference steps，因此必须先在 6-pair smoke test 中记录：

- 每 window 推理时间；
- 峰值 GPU 显存；
- artifact 磁盘占用；
- WorldArena 各指标耗时；
- tracker/depth/JEPA 失败率。

之后再根据预算决定完整规模、generation repeat 数和 Group A 的确定性采样量。

## 八、建议的最小修订

### 1. 把科研问题写得更精确

建议改为：

> 在经过验证的相同 RoboTwin 初始场景、统一候选 reference stat 和相同 BWM generation
> randomness 下，public BWM 的 GT-realized-EEF-conditioned future video fidelity，是否在
> filename-labelled Pi0.5 failure trajectories 上低于 matched expert trajectories？

如果以后确认了原始 failure label、training dataset 和 training stat，再去掉相应限定词。

### 2. 在正式 smoke test 前增加四个 hard gate

```text
Gate 0: old-data provenance
  seed mapping + failure label + code/config provenance

Gate 1: matched initial state
  camera/robot/object/scene fingerprint 全部通过

Gate 2: common BWM conversion
  A/B/C 使用相同 schema、14D convention、cadence 和 geometry

Gate 3: metric adapter validation
  paired MP4、trajectory failure handling、raw/normalized metric、JEPA isolation
```

任一 hard gate 失败，就不进入正式统计。

### 3. 如果旧数据 provenance 无法恢复，优先重新采集

推荐先重新采集：

```text
3 tasks x 2 seeds x {expert, Pi0.5}
```

collector 应一次性保存双方轨迹及完整 manifest。新数据验证通过后，再决定扩展到更多
task/seed，而不是花费大量时间从不完整文件反推旧映射。

### 4. 指标分层建议

在 adapter 完成前：

```text
Primary ready:
  PSNR, SSIM

Primary after validation:
  Trajectory Accuracy

Secondary:
  Depth Accuracy, Subject Consistency

Diagnostic only:
  Image Quality, Aesthetic Quality

Set-level auxiliary only:
  JEPA Similarity
```

### 5. 统计层级建议

```text
generation repeats
-> window-side score
-> episode/pair-side score
-> per-pair expert-policy gap
-> paired bootstrap over pair_id
-> optional task-cluster bootstrap
```

Group A 只作为 expert reference sanity check，不参与虚假的 matched pairing。

## 最终判断

| 项目 | 当前判断 |
|---|---|
| 科研问题本身 | 可以研究，方向合理 |
| 旧 500 条 Pi0.5 直接进入正式 matched experiment | 当前不可以 |
| seed mapping | `AMBIGUOUS`，待原始 collector/日志证明 |
| same-seed initial scene | 尚未验证 |
| BWM checkpoint/base model | 已存在 |
| BWM training stat | `UNCONFIRMED` |
| 现有 GT-history PSNR/SSIM 原型 | 基础逻辑可用 |
| 八指标完整评测 | 尚未接入 |
| pair-level 统计 | 尚未实现 |
| 当前环境直接启动 BWM GPU 推理 | 尚不可行 |
| 推荐下一步 | 先解决 provenance；失败则重采 6-pair matched smoke 数据 |
