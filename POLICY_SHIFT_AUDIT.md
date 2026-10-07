# Policy-induced BWM blind-spot experiments: local audit

审计时间：2026-10-07。范围仅为 `/data1/liuwenhao/goal.md` 的第一阶段：理解三个 kill experiments、审查当前本地代码与数据、列出缺口和最小改造。**未实现 Residual/Gated Residual，也未宣称三个实验已完成。** 以下判断基于本地文件、Git 状态和只读检查；未执行 GPU 推理/训练。

## A. Local repository state

| repo | branch | local HEAD | dirty? |
|---|---|---|---|
| `robotwin_consistency` | `dev/liuwenhao` | `ff8af3d65200b117f2b06a087f963f236ec5fa56` | 是：`third_party/robotwin` 修改、`third_party/WorldArena-2.0/` 未跟踪；审计前状态 |
| `third_party/robotwin` | detached HEAD（不在 `dev/liuwenhao`） | `2d3a8eb9804d2e5e239435d32f45da6277503d73` | 是：`README.md`、`envs/adjust_bottle.py` 未提交 |
| `third_party/boundless-world-model` | `dev/liuwenhao` | `8a85a222fd65a48c778027d4aa44ffc8fa04206e` | 否 |

主仓库记录的 RoboTwin gitlink 是 `b90a1097dae9e9e0208c5c06bffbbae7d9ba0907`，**不是**当前实际检出的 `2d3a8eb`；本地扰动数据还依赖未提交的 `adjust_bottle.py` hook。仅保存主仓库 SHA 无法重现实验。BWM gitlink 与当前工作目录一致。`debug/sunyang` 已合入 BWM `dev/liuwenhao`，但当前分支并非直接名为 `debug/sunyang`。

按要求执行的三个 `git log -5 --oneline`：主仓库 `ff8af3d, 5fbe33e, 8f9ee94, b876c12, 76908fc`；RoboTwin `2d3a8eb, be8eae9, c3ddfa8, 2a817a8, 0aeea2d`；BWM `8a85a22, 7b6889d, b47d2ac, 382eb40, cc0bc74`。

## B. Existing capability map

状态仅用 `READY`、`PARTIAL`、`MISSING`、`BROKEN/UNCERTAIN`；`READY` 表示该局部能力可用，不代表端到端实验就绪。

| Capability | Existing? | File | Function/Class | Notes |
|---|---|---|---|---|
| Expert trajectory collection | READY | `third_party/robotwin/script/collect_data.py`, `envs/_base_task.py` | `main`, `_take_picture`, `merge_pkl_to_hdf5_video` | HDF5、head 视频和多相机 RGB/EEF/qpos；相机及 `save_freq` 由配置指定。 |
| Controlled paired perturbation | PARTIAL | `position_perturbation/collect_adjust_bottle_pairs.py` | `Collector.find_boundary`, `record_episode`, `validate_dataset` | 100 对、100 seed、200 段数据已在 `outputs/adjust_bottle_paired_100/`；固定专家轨迹和 seed，成功/失败同帧数。依赖本地未提交的 RoboTwin hook；不是 policy rollout。 |
| Policy rollout execution | PARTIAL | `third_party/robotwin/script/eval_policy.py`; `policy/*/deploy_policy.py` | `eval_policy`, `eval`, `TASK_ENV.take_action` | 有同 seed 的 expert 可解性预筛、reset 和 policy 执行；但同 seed 的 expert HDF5 不随 eval 保存。 |
| Policy rollout recording | MISSING | `script/eval_policy.py`, `envs/_base_task.py` | `eval_policy`, `get_obs` | eval 主要保存成功率及可选 head MP4；不记录 policy 命令、每帧 EEF/qpos、wrist RGB 或 BWM HDF5。`save_data`/`merge_pkl_to_hdf5_video` 可复用，但 `take_action` 不调用 `_take_picture`。 |
| RoboTwin→BWM conversion | READY | `robotwin_to_bwm.py` | `read_robotwin_episode`, `convert`, `convert_paired` | HDF5/MP4→Parquet/metadata/stat；paired 转换保留 pair/seed/label；普通 expert 转换的 metadata 缺 seed/success。 |
| Fixed action normalization | MISSING | `robotwin_to_bwm.py`, BWM `data/operators.py` | `build_stat`, `LoadCobotAction` | 每次转换重新算 `stat.json`；`infer()` 默认传该输出 stat。BWM 本身可接受固定 `--action_stat_path`，但 root 尚无引用/校验机制。 |
| BWM autoregressive inference | READY | BWM `scripts/infer.py` | `_run_autoregressive` | 从首帧开始回灌预测帧，适合长时 rollouts；当前 `outputs/adjust_bottle_paired_100_bwm/bwm_outputs/` 无预测 MP4。 |
| BWM GT-history inference | MISSING | BWM `scripts/infer.py`, `pipelines/wan_video_action.py` | `_run_autoregressive`, `WanVideoActionPipeline.__call__` | CLI 没有 teacher-forced sliding-window 模式；底层 pipeline 支持多帧 `input_video`，可由 root wrapper 实现，无需先改 BWM 内部。 |
| Video metrics | PARTIAL | BWM `metrics/basic_metrics.py`; untracked WorldArena `video_quality_ood/WorldArena/` | `compute_basic_video_metrics`, `compute_basic_metrics` | PSNR/SSIM 可用但无按 window/seed/success 分层；未找到直接可复用的 LPIPS。WorldArena 有 DINO/RAFT/轨迹指标代码，但数据格式不同，且本地 `action_following.py` 源文件缺失而 `__init__.py` 引用它，不可直接当作现成 evaluator。 |
| Train/test split | MISSING | `robotwin_to_bwm.py`, BWM `data/wan_dataset.py` | `RoboTwinUnifiedDataset` | BWM 可读取任意 JSONL metadata，但 root 没有按 episode/seed/pair 固定分割及泄漏检查。 |
| Pretrained BWM training init | MISSING | BWM `scripts/train.py`, `pipelines/wan_video_action.py` | `WanTrainingModule`, `from_pretrained`, `load_checkpoint_weights` | `train.py` 未传 `ckpt_path`；会加载 Wan base 后新建 action encoder。`resume_from` 只恢复 Accelerate 全状态，不能用单独的公开 `.safetensors` 替代。 |
| Full FT | PARTIAL | BWM `scripts/train.py`, `runner.py`, train YAML | `WanTrainingModule`, `launch_training_task` | `trainable=[dit,action_encoder]`、loss/AdamW/存盘链存在；但未从 WM₀ 初始化，故目前不能当有效 adaptation baseline。VAE 不在 trainable 列表。 |
| LoRA | BROKEN/UNCERTAIN | BWM `scripts/train.py`, `parsers.py`, train YAML | `switch_pipe_to_training_mode` hook | 参数存在，但训练 YAML 的 `lora_base_model=null`、targets 空；当前无 WM₀ 初始化，action encoder 的冻结/更新与导出后的 inference 需实际核对。 |
| Checkpoint export/reload | PARTIAL | BWM `wan_video_action/logger.py`, `pipelines/wan_video_action.py` | `ModelLogger.save_model`, `load_checkpoint_weights` | full-FT 导出 trainable 权重，推理 loader 接受 DiT 和 `pipe.action_encoder.*`；需检查 missing/unexpected keys 和输出一致性。LoRA 仅导出 adapter 时，当前推理 loader 无明确 adapter 装载/融合链。 |
| Policy A/B tagging | MISSING | `script/eval_policy.py`; root converter | `eval_policy`, `convert` | 路径里有 policy 名，但 BWM metadata 未有稳定的 policy/checkpoint ID、rollout provenance 字段。 |
| Blind-spot matching | MISSING | `position_perturbation/collect_adjust_bottle_pairs.py` | `find_boundary` | 已有受控区域/方向/幅度 metadata，仅覆盖 expert replay；没有 A/B state-action feature 和 WM₀ error matching。 |

### 代码级关键核对

- `robotwin_to_bwm.read_robotwin_episode` 从 HDF5 `/joint_action/vector`、`/endpose/*` 组装 26D `observation.state`，另以 `state_to_action` 生成 26D delta `action`。BWM `LoadCobotAction` 将 `eef_abs` 映射到 `state_pose`，读取 **`observation.state` 的 14 个 EEF/gripper 维度**；此模式下 Parquet 的 delta `action` 不作为条件。`/joint_action/vector` 在 RoboTwin `get_obs` 中来自 `get_left/right_arm_jointState()`，是实际关节观测，不是原始 policy command；`take_action` 的 policy command 是另一个变量。必须分开保存。
- BWM `LoadCobotAction` 用对应 `stat.json` 的 p01/p99（或 min/max）线性归一化并裁剪到 `[-1,1]`。本地 `demo/stat.json` 与 paired converter 的 `stat.json` SHA 不同，数值也不同。`demo/stat.json` 可作候选 reference，但 README 只说明它是 demo 统计量，**未证明它就是 `step-12000.safetensors` 的训练统计量**。在查明 checkpoint 训练所用 stat 前，不能宣称绝对归一化匹配 WM₀。
- 本地有 BWM `ckpt/BLM/step-12000.safetensors`（约 9.4G）、Wan2.2 base weights 和 `demo/stat.json`。只读 safetensors header 检查显示 8 个 action-encoder tensor key；推理 `build_pipeline` 会传 `ckpt_path`，训练 `WanTrainingModule` 不传。`load_checkpoint_weights` 使用 `strict=False` 并只打印 missing/unexpected；正式实验应将关键缺失视为错误。
- `RoboTwinUnifiedDataset._build_temporal_sample_info` 选择允许范围内的第一个 future start（非随机采样）；`runner.py` 的 DataLoader `shuffle=False`。若只有每 episode 一条 metadata，训练会反复看到同一前缀 window。需显式构建覆盖全轨迹的 train-window manifest，并冻结顺序用于公平比较。
- `eval_policy` 的可选 MP4 是每个 `take_action` 级别的 head 画面；expert `save_freq=15` 是控制循环采样。两者时间尺度不能直接一帧对一帧比较。expert HDF5 可含 head/wrist RGB 和 EEF/qpos（取决于 `data_type`）；默认 BWM 视频是 head MP4。相同 seed 仍须核对相机型号、分辨率、帧率、时序和 `eval_mode`/随机化。
- 本机有 `checkpoints/WorldArena2.0/pi05_adjust_bottle/` 的 7G 权重、14D stats，以及另一个 click_bell 权重；未找到 RoboTwin `policy/pi0`/`pi05`/`ACT`/`DP` 的预期 checkpoint 目录。前者是 WorldArena/RLinf 格式，不等于可由 `robotwin/policy/pi05/deploy_policy.py` 直接加载的 checkpoint。**同一 adjust_bottle task 上当前没有已验证可直接运行的 Policy A/B 两个权重**。`pi0`、`pi05`、OpenVLA-OFT、ACT、DP 代码有 qpos/ALOHA 路径，但权重、依赖、相机输入和具体 14D 输出均须逐一验证。
- 现有 Python 环境检查未发现 `diffsynth`；GPU 在沙箱外可见，但训练/推理环境尚未验证。WorldArena 副本为未跟踪文件，不能当作可复现依赖。

| Policy candidate | 同任务本地权重 | 当前可直接用 RoboTwin eval? | Action/camera 判断 |
|---|---|---|---|
| RoboTwin pi0 | 未发现 | 否 | deploy 读取 head+双 wrist 与 14D joint state，调用默认 qpos `take_action`；仍需同任务权重 |
| RoboTwin pi05 | 未发现其原生 checkpoint 树 | 否 | 同 pi0；本地 WorldArena pi05 adjust_bottle 权重是另一套加载格式，需 bridge 和实测 |
| OpenVLA-OFT | 未发现 | 否 | deploy 可调用 qpos `take_action`，但需要 checkpoint/unnorm key 和动作维度验证 |
| ACT | 未发现 | 否 | deploy 有 qpos `take_action`，需训练权重和 dataset stats |
| DP | 未发现 | 否 | deploy 动作维由 embodiment 计算，但缺对应 `.ckpt` |

paired perturbation 可作为 Experiment 1 的**受控 off-expert proxy**、Experiment 3 的**同环境区域对照基础**、policy recorder 的**HDF5/视频/metadata 格式参考**；三者都不使它自动成为真实 policy-induced shift 数据。

## C. Experiment readiness and scientific decision

### Experiment 1 — `BLOCKED`

**为什么做：** 先证实固定 WM₀ 在真实 policy 访问区域是否比 matched expert 访问区域更不可靠。若 `D_expert ≈ D_policy`，后续 adaptation/residual 没有充分动机。固定 policy，不研究 policy evolution。

现有链：expert HDF5/MP4 → root converter → BWM AR inference → PSNR/SSIM；policy eval → 成败/可选 MP4。现有链尚不能形成同 seed、同时间基准、同 normalization 的 `D_expert_test`/`D_policyA_test`。缺：已验证 policy 权重、policy 完整 recorder、固定 reference stat、GT-history sliding windows、split/分层聚合。最小补丁：root 的 recorder + manifest/split + teacher-forced evaluator；仅复用 BWM pipeline。**Primary** 用每个真实 window 的 GT history、真实 `eef_abs` chunk 预测未来，下一 window 再从 GT 取 history；报告 PSNR/SSIM 和若可用的 feature/motion 指标的 `Error(policy)-Error(expert)`、按 task/seed/成败/step/window 分层及 bootstrap CI。**Secondary** 才是现有 AR rollout 的画质、无效生成率和长时动力学指标；不得把 AR compounding error 解释为 conditional gap。

### Experiment 2 — `BLOCKED`

**为什么做：** 在 Experiment 1 确有 gap 后，实测 Full FT/LoRA 的 plasticity、retention、效率；若普通 LoRA 已同时改善 policy 且保留 expert，复杂 Residual 没必要。

现有链：BWM JSONL/video/action dataset → Wan2.2 pipeline → flow-matching loss → AdamW → safetensors/full-state checkpoint。缺：可信 `WM₀` 训练初始化、固定 `D_policyA_train/test` 与 `D_expert_test`、LoRA 参数/导出语义验证、固定评估协议和 trainable 参数计数。最小补丁：BWM 训练入口 `--init_ckpt_path`，在 `from_pretrained` 后、`switch_pipe_to_training_mode` 前加载 DiT + action encoder；独立于 `resume_from`。训练前后断言关键权重匹配、无关键 missing/unexpected。Base/FT/LoRA 均从同一 WM₀、同数据顺序/seed/steps/stat 出发；VAE 冻结。LoRA 要明确 target/rank/action encoder 冻结与否，并使导出可与 WM₀ 融合或显式加载。用相同 evaluator 比较 `D_policyA_test` 的改善和 `D_expert_test` 的损伤，另报参数数、GPU 时与显存。

### Experiment 3 — `BLOCKED`

**为什么做：** 检验 correction 是对 world-model 错误区域的修复，还是 Policy A 记忆；若 A 的改善不能迁移至 B 的同类区域，policy-agnostic 命题不成立。

现有链：paired perturbation 给出了「同 seed、同专家轨迹、同方向、临界幅度」的受控 state-region proxy；没有两个同任务可运行 policy rollout，更没有跨 policy 区域匹配。最小补丁：先用 root metadata 标记 `policy_id/checkpoint_id/seed/task/success/region`；选定 A/B 后，预注册 region 定义。三种定义中，**受控 object XY perturbation + direction/magnitude bin** 最适合三个月 pilot，但只能说明同环境区域；更严格的定义需 EEF/object pose/gripper/task phase 特征近邻，同时要求 WM₀ GT-history error 超过仅由 train/validation 确定的阈值，避免用 test error 挑样本。用 A train blind-spot data 做 LoRA/FT，不用新 Residual；测试 A same-region、B same-region、B unrelated-region、expert 四格。如果只有同架构不同 checkpoint，明确标注是 weaker cross-policy pilot。

## D. Missing components by priority

**P0（不解决则结论不可信）**

1. 固定 WM₀ checkpoint、模型代码、`eef_abs` 14D schema 与**同一 reference stat** 的来源/哈希；不能每分布重算 stat，也不能仅假定 demo/stat 是预训练 stat。
2. 同 task/seed/scene 的 expert 与 policy 完整轨迹；统一 head 相机、画幅、渲染/denoiser、采样时钟和 horizon。保存 policy 命令及实际 EEF/qpos、每帧 head/wrist RGB、成功/失败和失败时间。最好 root wrapper 对 `get_obs`/`take_action` 做捕获并复用 HDF5 schema；不修改 RoboTwin 源码除非证明确实无法 wrapper。
3. GT-history sliding-window inference、严格未来帧 mask/action 对齐、window 级指标与异常计数；AR 作为独立二级结果。
4. 按 seed/episode/pair 分 train/test；同一轨迹重叠 window 不能跨分割；split 文件冻结并记录哈希。
5. Full FT/LoRA 必须从同一 WM₀ 初始化；验证 DiT + action encoder 权重；验证导出的 checkpoint 在推理中重现并避免 LoRA only 文件被误当完整 BWM。
6. 两个同任务、同 embodiment、动作约定可比较的真实 policy 权重；否则 Experiment 3 不能开展。当前 RoboTwin 子模块 SHA/本地 hook 不可复现，也需固定 provenance。

**P1（可先 smoke，正式实验必须有）**：逐层任务/成败/rollout step/window/horizon 统计、按 seed 配对 bootstrap CI 和预注册 kill 判据；对象 pose/EEF 运动或 optical-flow 指标；action encoder 冻结策略及 trainable-count/时长/显存记录；跨 policy 相同区域特征匹配与 balance check；记录相机/渲染/任务配置及模型权重哈希。

**P2（增强）**：LPIPS/DINO 等视觉特征距离、可靠可运行的 WorldArena 物理/action evaluator 适配、更多任务/策略架构、expert replay 消融和更严格 error-based blind spot 定义。不可用这些增强替代 P0。

## E. Minimal implementation plan (not executed)

优先在 root 新建 `experiments/policy_shift/`：

| root file | 最小职责 |
|---|---|
| `collect_policy_rollouts.py` | 复用 `eval_policy` 的 expert-valid seed/reset 和 policy `get_model/eval`；通过 root wrapper 捕获每个 `get_obs`、`take_action`、实际 state/EEF、head/wrist RGB、指令与结果；输出与 expert 对齐的 HDF5/视频和 policy command sidecar。必须验证 chunk policy 多次 `take_action` 的每一步均被记录。 |
| `build_dataset.py` | 复用 `robotwin_to_bwm.read_robotwin_episode`/`write_parquet`；增加 `--reference_stat_path`，所有分布复制/引用同一文件且记录 SHA；保留 seed/task/policy/checkpoint/success/camera/fps/provenance。 |
| `build_splits.py` | 以 seed/episode/pair 为分割单位，输出固定 JSONL 和 seed 清单；验证无 pair/window/seed 泄漏。 |
| `eval_gap.py` | 从完整 GT 视频与 action 读多 window，复用 `WanVideoActionPipeline.__call__`；每次提供 GT history + true action，只评未来帧；同时调用现有 AR 路径，输出逐窗口与逐 episode JSONL 和聚合表。 |
| `run_adaptation.py`, `eval_adaptation.py` | 固定 WM₀/stat/split/steps/seed；Base/FT/LoRA 运行记录、关键权重校验、参数数/显存/GPU 时、同协议 retention/plasticity。 |
| `compare_policies.py`, `configs/`, `README.md` | region 定义/阈值/匹配、A/B/专家四格评估；预注册决策规则及可复现实验清单。 |

现有 `robotwin_to_bwm.py` 可薄改固定 stat 与 metadata，不应复制整套转换。RoboTwin submodule 原则上不改：`get_obs` 和 `take_action` 可由 wrapper 捕获；若需以 expert 相同控制步长记录内循环，则先验证 wrapper 是否足够，不够时才提最小 hook。**BWM submodule 只需训练初始化及可能的 LoRA export/reload 修复**；GT-history 可先由 root 调已有 pipeline。任何 submodule 修改都单独 commit、更新主仓库 gitlink，并保留用户现有未提交改动。

## F. Minimal smoke-test plan (not executed)

先做 Stage-0 结构性 smoke，再做真实 policy smoke；下面命令的第二段需完成 P0 补丁后才存在，不能当作当前已可运行命令。

```bash
cd /data1/liuwenhao/Projects/robotwin_consistency
# 现有 100 对中取 2 对/4 段，检查格式；此步生成的 stat 只用于格式 smoke，不能用于 gap 结论。
python robotwin_to_bwm.py --stage convert \
  --robotwin_dir outputs/adjust_bottle_paired_100 \
  --output_dir outputs/policy_shift_stage0_format --num_episodes 4
```

预期：4 个 metadata 行、2 对同 seed/trajectory hash 的成功/失败片段，视频与 Parquet 帧数一致、`eef_abs` 为 14D。然后由待实现的 `build_dataset.py` 绑定固定 reference stat，`build_splits.py` 按 pair 分割，`eval_gap.py` 在每对各取 2–3 个同索引 GT-history window；验证相机、帧数、normalization SHA、未来帧 mask、无 window 泄漏、指标对相同 GT/预测有合理自检。再用 **1 task、2–4 expert episodes、2–4 policy A episodes、1 GPU** 做 matched-seed pilot；要求每条 policy 命令与实际 EEF/qpos、head 视频同时间戳，teacher-forced 和 AR 输出独立，且失败样本不被静默跳过。初始化补丁完成后再做 1–2 optimizer-step FT/LoRA smoke：初始权重与 WM₀ 一致、应更新参数变化、VAE 不变、导出重载预测可复现。此 smoke 不用于论文效应结论。

拟议的后续调用契约（**脚本/参数尚未实现，此时不能执行**）：

```bash
python experiments/policy_shift/build_splits.py --metadata outputs/policy_shift_stage0_format/metadata.jsonl --group-by pair_id --seed 42 --output-dir outputs/policy_shift_stage0_format/splits
python experiments/policy_shift/eval_gap.py --mode gt_history --metadata outputs/policy_shift_stage0_format/splits/test.jsonl --reference-stat third_party/boundless-world-model/demo/stat.json --checkpoint third_party/boundless-world-model/ckpt/BLM/step-12000.safetensors --windows-per-episode 3 --output-dir outputs/policy_shift_stage0_eval
```

上述 `demo/stat.json` 只是**占位候选**；在确认 WM₀ 训练 stat 来源前不得作为可发表 gap 结果。policy smoke 的输入应是 recorder 产出的同 seed expert/policy JSONL/HDF5，输出应分别为 GT-history window JSONL、AR MP4 和按 episode/seed 的指标表。

## G. Full three-experiment execution plan

1. **冻结协议与资源**：锁定三仓库 SHA/工作树或补丁、任务/ALOHA embodiment、head 摄像头和分辨率/帧率/denoiser、WM₀ checkpoint 与训练 stat、policy 权重及依赖；记录 SHA。现有两张 WorldArena pi05 权重分别是 adjust_bottle/click_bell，不能当作同任务 Policy A/B；另需取得/训练一个同任务 policy 或两个同架构不同 checkpoint。先验证离线推理、动作 14D/qpos 约定和任务成功率，再收集。
2. **数据与分割**：每 task 建议先 50–100 个 matched valid seeds 的 expert/policy A 测试对，并独立 50–100 个 policy A 训练 episodes；Policy B 在相同区域另取至少 50 个独立测试 episodes。若跨 2–3 task，按 task 分层。数量是 pilot 预算，不是统计功效保证；依 episode-level bootstrap CI/预先定义的最小有意义 gap 追加样本。训练与测试按 seed 分离；paired perturbation 按 pair 划分；固定相机、时钟、horizon 与成功分层。
3. **Experiment 1**：只用未参与 WM₀ 训练/调参的 expert/policy test；GT-history 为主指标，AR 为次指标。每 task/成功标签/时间位置报告 error gap 和置信区间；若无稳健且有意义 gap，停止方向。若 WM₀ 的训练集 seed 不可追溯，先标记可能预训练泄漏并使用新随机测试 seeds。
4. **Experiment 2**：同一 WM₀、同一 `D_policyA_train`、同一 window 顺序/steps/seed/stat，运行 Base/Full FT/LoRA；统一 D_policyA_test 和 D_expert_test；报告改善、retention、效率及多 seed 变异。若 LoRA 已达到目标且无可观 retention loss，停止开发复杂 Residual。
5. **Experiment 3**：事前锁定 perturbation bin 或 state-action feature 阈值，且不使用 Policy B test 标签选阈值；只用 A train region 适配。比较 A same-region、B same-region、B unrelated-region、expert；同时看相似度、基线误差及负控制。若 A 提升但 B 同区域不提升，不支持 policy-agnostic correction。Stage-0 的固定专家轨迹位置扰动只能证明 controlled off-expert/state-shift proxy，**不能代替真实 policy-induced shift**。

## Confound checklist / 判定门槛

| Confound | 当前证据 | 必须采取的控制 |
|---|---|---|
| Normalization leakage | converter 每次 `build_stat`；demo 与 paired stat 不同 | 追溯 WM₀ 训练 stat，全部分布固定一个 SHA；记录裁剪比例 |
| Seed/scene mismatch | paired 已配对；eval 预筛后重置；普通 expert/policy 未成对保存 | 同 task/seed/scene 参数，保留失败及拒绝 seed 清单 |
| Different horizon/timebase | expert `save_freq=15`，eval 视频每 `take_action`，长度可异 | 同一记录频率，或物理时间戳重采样；同 horizon/step-bin 报告 |
| Autoregressive confound | 现有 infer 回灌预测帧 | GT-history 主结果，AR 只做二级长时结果 |
| Success/failure confound | paired 有标签；普通 expert metadata 无标签 | 所有 rollout 保存 outcome、失败时刻，分层/匹配而不只报总体均值 |
| Camera/render mismatch | expert MP4 是 head；policy 可看三相机；本地 denoiser 有 OIDN/OptiX 差异 | BWM 输入固定 head、相同型号/画幅/渲染；policy 可看 wrist 但不改变 BWM 评价视角 |
| Action convention | `take_action` 默认 qpos；HDF5 是实际 joint state；BWM `eef_abs` 是实际 EEF pose | command 与 realized state 分字段，统一 EEF 坐标/旋转/夹爪/schema，逐帧校准 |
| BWM checkpoint init | 推理加载 `.safetensors`，训练未传 | Full FT/LoRA 从同一 WM₀；断言 DiT/action encoder 载入与导出重载 |
| Test leakage | 目前无 split；train 默认同前缀 window | seed/episode/pair 整组分割，window 派生后校验交集为空 |
| Metric sensitivity | BWM 仅 PSNR/SSIM；WorldArena metrics 需适配/有缺文件 | 画质 + EEF/object motion/action-following + invalid rate；相机/对齐一致后解释 |
| Code/data provenance | RoboTwin 实际 HEAD 与 gitlink 不同，hook 未提交 | 固定完整 worktree patch/commit、数据生成配置、权重/stat/manifest 哈希 |

结论：**目前适合先用现有 paired 数据做 Stage-0 管线验证，不适合宣称 Experiment 1–3 的科学结论，更不能直接开始 Residual 方法。**
