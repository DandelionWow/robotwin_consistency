# goal.md 执行描述反馈

本文只针对 `/data1/liuwenhao/Projects/robotwin_consistency/goal.md` 和当前本地代码状态做核对，重点说明冲突、有误、容易误解或会影响实验正确性的地方。

## 总体判断

`goal.md` 的实验目标是合理的：在 RoboTwin 中生成 RPY 方向验证数据，包含视频、endpose、qpos，再用于后续世界模型对比。

但它不是对当前根目录 `robotwin_to_bwm.py` 的直接修改说明，而是一个新的 RoboTwin 数据采集任务说明。当前 `robotwin_to_bwm.py` 只负责把已经采集好的 RoboTwin 数据转换成 BWM 输入、执行 BWM 推理、生成对比视频；它不会创建 RoboTwin 任务，也不会驱动仿真采集。

因此需要把任务分成两层：

1. RoboTwin 侧：生成 `data/rpy_direction_validation/rpy_direction_clean/{data,video,scene_info.json,...}`。
2. BWM 侧：用现有 `robotwin_to_bwm.py` 转换、推理、比较。

## 1. 和当前代码直接冲突或有误的点

### 1.1 如果使用官方 `collect_data.sh`，任务文件不能只放在 `third_party` 同级目录

`goal.md` 后面问到是否可以在 `third_party` 同级新建文件夹实现。结论是：

- 如果仍然使用命令 `bash collect_data.sh rpy_direction_validation rpy_direction_clean 0`，不可以只放在 `third_party` 同级目录。
- 原因是 `third_party/robotwin/script/collect_data.py` 固定从 `envs.<task_name>` 导入任务，并且固定读取 `./task_config/<task_config>.yml`。
- 也就是说官方入口会找：
  - `third_party/robotwin/envs/rpy_direction_validation.py`
  - `third_party/robotwin/task_config/rpy_direction_clean.yml`

当前代码依据：

- `third_party/robotwin/script/collect_data.py` 中 `class_decorator()` 使用 `importlib.import_module(f"envs.{task_name}")`。
- 同文件 `main()` 使用 `config_path = f"./task_config/{task_config}.yml"`。

可行替代方案：

1. 官方 RoboTwin 方式：在 `third_party/robotwin/envs/` 和 `third_party/robotwin/task_config/` 下新增文件。优点是完全兼容 `collect_data.sh`；缺点是会修改 third_party。
2. 外部紧凑方式：在 `third_party` 同级新建一个文件夹，例如 `rpy_direction_validation/`，写一个独立 runner，自己导入 RoboTwin 的 `Base_Task` 和采集函数。优点是基本不动 third_party；缺点是运行命令不能再是原始 `collect_data.sh rpy_direction_validation rpy_direction_clean 0`。
3. 折中方式：外部文件夹放主要实现，`third_party/robotwin/envs/rpy_direction_validation.py` 只放一个很薄的 import shim，配置仍放 `task_config`。这能使用官方命令，但仍会在 third_party 下新增少量文件。

### 1.2 任务类名不能用示例里的 `RpyDirectionValidation`

`goal.md` 伪代码写的是：

```python
class RpyDirectionValidation(Base_Task):
```

这和当前 RoboTwin 加载机制不匹配。当前 `collect_data.py` 会执行：

```python
env_class = getattr(envs_module, task_name)
```

对于任务名 `rpy_direction_validation`，类名必须也是：

```python
class rpy_direction_validation(Base_Task):
```

如果按伪代码使用大驼峰类名，官方 `collect_data.sh` 会报 `No such task`。

### 1.3 配置中的 `D435` 不是 480p

`goal.md` 的目标写了“视频分辨率：480p”，但配置示例仍然使用：

```yaml
camera:
  head_camera_type: D435
  wrist_camera_type: D435
```

当前本地相机配置在：

```text
third_party/robotwin/task_config/_camera_config.yml
```

其中：

```yaml
D435:
  w: 320
  h: 240

Large_D435:
  w: 640
  h: 480
```

所以如果目标是 480p，配置应该优先使用：

```yaml
camera:
  head_camera_type: Large_D435
  wrist_camera_type: Large_D435
```

如果继续使用 `D435`，输出会是 320x240，不满足 480p。

### 1.4 `goal.md` 里说“如果没有分辨率字段就 README 说明”，但当前其实有明确的相机类型配置

当前项目不是通过 `width / height / image_size / resolution` 放在任务配置里设置分辨率，而是通过 `_camera_config.yml` 里的相机型号间接设置。

因此不能简单写“未发现官方配置分辨率字段，暂用默认输出”。更准确的描述应该是：

```text
当前 task_config 没有直接的 width/height 字段，但 RoboTwin 使用 task_config/_camera_config.yml 中的相机型号控制分辨率。D435=320x240，Large_D435=640x480。若需要 480p，应将 head/wrist camera type 设为 Large_D435。
```

### 1.5 `goal.md` 缺少和现有 `robotwin_to_bwm.py` 对接所需的 `experiment_start_frame`

当前 `robotwin_to_bwm.py` 已支持 `--crop_to_experiment`，但它要求 `scene_info.json` 中存在：

```json
{
  "episode_0": {
    "info": {
      "experiment_start_frame": 123
    }
  }
}
```

代码依据：

```python
scene_info[f"episode_{episode_id}"]["info"]["experiment_start_frame"]
```

`goal.md` 的实验方式 a 是“先从默认初始位姿移动到目标实验位置，再执行 RPY 旋转”。如果后续 BWM 只想验证 RPY 方向，而不是验证“移动到目标点 + RPY 旋转”的完整轨迹，就必须记录“开始执行 RPY 旋转之前”的帧号，也就是 `experiment_start_frame`。

否则现有转换脚本默认会从 episode 第 0 帧开始，BWM 输入首帧就是默认初始状态，action 也包含移动到目标点的阶段。这和“验证 TCP local RPY 方向”这个目标不完全一致。

建议补充：

```json
"experiment_start_frame": <before_rpy_pose 被记录时的 FRAME_IDX>
```

并在转换时使用：

```bash
python robotwin_to_bwm.py \
  --stage convert \
  --robotwin_dir third_party/robotwin/data/rpy_direction_validation/rpy_direction_clean \
  --output_dir outputs/robotwin_bwm/rpy_direction_validation_clean \
  --crop_to_experiment \
  --overwrite
```

### 1.6 metadata 不能只写普通 `meta/*.json`，还应写入或同步到 `scene_info.json`

`goal.md` 要求每条 episode 保存：

```text
data/rpy_direction_validation/rpy_direction_clean/meta/episode_000000.json
```

这本身没有问题。但当前 RoboTwin 的官方采集流程已经会把 `play_once()` 返回值写入：

```text
data/rpy_direction_validation/rpy_direction_clean/scene_info.json
```

代码流程是：

1. 数据采集阶段调用 `TASK_ENV.play_once()`。
2. 把返回的 `info` 写入 `scene_info.json` 的 `episode_<idx>`。

如果只写 `meta/*.json`，不把关键字段同步到 `self.info["info"]` 并由 `play_once()` 返回，则现有 `robotwin_to_bwm.py --crop_to_experiment` 无法读取这些字段。

建议：

- `scene_info.json` 作为和现有 pipeline 对接的主 metadata。
- `meta/episode_000000.json` 可以作为人类可读副本。
- 至少以下字段必须在 `scene_info.json` 中出现：
  - `pos_id`
  - `axis`
  - `angle_deg`
  - `rotation_frame`
  - `experiment_start_frame`
  - `target_position_xyz`
  - `before_rpy_pose`
  - `target_pose`
  - `final_pose`
  - `success`

### 1.7 metadata 写入时机需要避开 seed 搜索阶段

当前 RoboTwin 采集分两阶段：

1. seed / pre-motion 阶段：`need_plan=True`，只找成功 seed 并保存 `_traj_data`。
2. data collection 阶段：`need_plan=False`，回放 `_traj_data`，保存 HDF5 和视频。

如果在 seed 阶段就直接写 `meta/episode_*.json`，可能产生两类问题：

- 某个 seed 后续失败，但已经留下 metadata。
- data collection 阶段 replay 时同一个 episode 会再次写 metadata，容易不一致。

建议：

- `check_success()` 使用内存中的 `self.before_rpy_pose / self.final_pose / self.condition`。
- `meta/*.json` 只在 `self.save_data == True` 的 data collection 阶段写。
- `play_once()` 返回的 `self.info` 也应以 replay 阶段的 `FRAME_IDX` 为准，因为只有此时帧号才对应最终 HDF5/video。

### 1.8 `save_freq: 1` 可用，但会显著增加帧数和数据量

`goal.md` 示例配置把：

```yaml
save_freq: 1
```

当前 `demo_clean.yml` 是：

```yaml
save_freq: 15
```

`save_freq: 1` 的好处是轨迹采样更密，对 RPY 验证和定位 `experiment_start_frame` 更精确。缺点是：

- HDF5 更大。
- 视频更长。
- 162 条完整实验的数据量明显增加。
- 后续 BWM 推理耗时更长。

这不是错误，但需要明确它是“高密度采样设置”，不是默认配置。建议 smoke test 用 `save_freq: 1`，完整 162 条如果数据量过大，可以考虑 `save_freq: 3` 或 `save_freq: 5`，但要保证动作方向仍可判定。

### 1.9 `eval_video_log: true` 不等于最终视频保存开关

当前 HDF5 和 episode 视频由 `Base_Task.merge_pkl_to_hdf5_video()` 生成，路径是：

```text
<save_path>/data/episodeN.hdf5
<save_path>/video/episodeN.mp4
```

只要 `collect_data: true` 且 `data_type.rgb: true`，正式采集阶段就会通过缓存 pkl 生成视频。`eval_video_log` 不是这个标准 episode 视频的核心开关。

因此 `goal.md` 中验证输出时应重点检查：

```text
data/rpy_direction_validation/rpy_direction_clean/data/episode0.hdf5
data/rpy_direction_validation/rpy_direction_clean/video/episode0.mp4
```

### 1.10 输出视频默认是 30fps，不是 BWM 的 24fps

当前 RoboTwin 视频生成工具 `images_to_video()` 默认 `fps=30.0`。BWM 推理和当前对比视频使用的是 24fps。

这不是阻塞问题，因为当前 `robotwin_to_bwm.py` 的 compare 阶段已经改成按帧号对齐，再输出 24fps。但是在文档中需要说清楚：

- RoboTwin 原始视频：30fps。
- BWM 预测视频：24fps。
- 对比视频：24fps，使用 frame-aligned 拼接。

不要把 RoboTwin 采集 fps 和 BWM 推理 fps 混为一谈。

### 1.11 `check_success` 不能只看 Euler 分量，尤其是 180°

`goal.md` 已经提到 180° Euler 表示可能不唯一，这个提醒是正确的。但实现要求里仍然强调 `relative_quat_to_local_rpy` 和 Euler 表格，容易让实现者过度依赖 Euler。

建议成功判定分两层：

1. 30° / 90°：可以用 local relative Euler 做直观检查。
2. 180°：应该使用四元数或旋转矩阵的角距离作为主判据，Euler 只用于打印。

否则 180° 情况可能因为 Euler 分解方式不同，被误判失败。

### 1.12 `right_arm: fixed_or_origin` 描述太模糊，实际应固定为一种策略

`goal.md` 中多处写“右臂保持不动或回到 origin”。为了实验可重复，metadata 不能记录模糊值。

建议固定为：

```text
right_arm = "origin"
```

并在每个 episode 开始时执行：

```python
self.move(self.back_to_origin(ArmTag("right")))
```

如果后续发现右臂回 origin 增加规划失败，再改为：

```text
right_arm = "fixed"
```

但不要在同一批数据里混用两种语义。

## 2. 描述正确但需要按本地 API 改写的点

### 2.1 本地确实使用 `envs/`，不是 `env/`

本地已有任务在：

```text
third_party/robotwin/envs/beat_block_hammer.py
third_party/robotwin/envs/place_a2b_left.py
third_party/robotwin/envs/stamp_seal.py
```

所以如果采用官方集成方式，新增任务路径应该是：

```text
third_party/robotwin/envs/rpy_direction_validation.py
```

不是：

```text
third_party/robotwin/env/rpy_direction_validation.py
```

### 2.2 空桌面任务需要实现 `load_actors(self): pass`

`Base_Task._init_task_env_()` 会调用 `self.load_actors()`。已有任务都实现了 `load_actors()`。RPY 验证虽然不需要物体，也必须实现空函数：

```python
def load_actors(self):
    pass
```

否则初始化阶段会找不到该方法。

### 2.3 `setup_demo()` 应按已有任务写法调用 `_init_task_env_`

已有任务写法是：

```python
def setup_demo(self, **kwags):
    super()._init_task_env_(**kwags)
```

`goal.md` 伪代码写：

```python
super().setup_demo(**kwargs)
```

这和本地已有任务不一致。应使用本地风格：

```python
def setup_demo(self, **kwags):
    super()._init_task_env_(**kwags)
```

### 2.4 `move_to_pose / back_to_origin / get_arm_pose` API 是存在的

`goal.md` 使用这些 API 的方向是对的。本地 `Base_Task` 已有：

- `move_to_pose(arm_tag, target_pose)`
- `back_to_origin(arm_tag)`
- `get_arm_pose(arm_tag)`

并且 `move_to_pose()` 在 seed 阶段会规划路径、保存 joint path；在 data collection 阶段会按 `_traj_data` 回放。因此新增 RPY 任务可以走标准 RoboTwin 两阶段采集流程。

### 2.5 四元数格式 `[x, y, z, qw, qx, qy, qz]` 的说明是正确的

当前 `get_arm_pose()` 返回的是：

```text
[x, y, z, qw, qx, qy, qz]
```

这和当前 `robotwin_to_bwm.py` 读取 `/endpose/left_endpose` 并用 `Rotation.from_quat(..., scalar_first=True)` 的方式一致。

因此 `make_local_rpy_target_quat()` 中使用 wxyz/xyzw 转换是必要的。

### 2.6 `q_target = q_current * q_delta` 的方向是正确的

如果使用 `scipy.spatial.transform.Rotation`，要表达 TCP local frame 下的增量旋转，推荐：

```python
target = current * delta
```

这对应旋转矩阵：

```text
R_target = R_current @ R_delta_local
```

不要写成：

```python
target = delta * current
```

后者通常对应世界坐标系下的增量旋转。

### 2.7 3x3 位置不能完全按文档硬跑，需要先 smoke test IK

`goal.md` 推荐 `dx=0.06, dy=0.06` 是合理初值，但不是必然安全。当前 `left_plan_path()` 规划失败会让 `plan_success=False`，该 seed 失败并继续找下一个 seed。

为了避免完整 162 条里大量失败，建议：

1. 先中心点 6 条。
2. 再只跑 9 个位置、单一小角度，比如 roll +30°，验证 3x3 可达。
3. 最后再跑完整 162 条。

如果外圈点失败，再把网格缩到 `0.04m`。

## 3. 和当前 `robotwin_to_bwm.py` 的对接要求

当前转换脚本要求 RoboTwin 输出目录结构为：

```text
<robotwin_dir>/
  data/episode0.hdf5
  video/episode0.mp4
  scene_info.json  # 可选，--crop_to_experiment 时必需
```

HDF5 必须包含：

```text
/joint_action/vector
/endpose/left_endpose
/endpose/right_endpose
/endpose/left_gripper
/endpose/right_gripper
```

当前 RoboTwin 标准采集流程会生成这些字段，所以只要 RPY 任务走 `Base_Task` 的正常采集路径，就能被 `robotwin_to_bwm.py` 读取。

转换后 action 维度是 14：

```text
left_xyz + left_rpy + left_gripper + right_xyz + right_rpy + right_gripper
```

这和 BWM 当前 `--action_type eef_abs`、`--action_dim 14` 的设置一致。

需要特别注意：

- 如果不使用 `--crop_to_experiment`，BWM 输入包含“移动到目标位置 + RPY 旋转”。
- 如果使用 `--crop_to_experiment`，必须在 `scene_info.json` 写入 `experiment_start_frame`。
- 对比视频现在已经按 24fps frame-aligned 拼接，不再依赖 RoboTwin 原始 30fps 时间戳。

## 4. 是否可以在 `third_party` 同级目录实现

可以，但要区分“实现位置”和“运行入口”。

### 方案 A：完全官方集成

文件：

```text
third_party/robotwin/envs/rpy_direction_validation.py
third_party/robotwin/task_config/rpy_direction_clean.yml
third_party/robotwin/script/build_rpy_index.py
third_party/robotwin/script/check_rpy_outputs.py
```

运行：

```bash
cd third_party/robotwin
bash collect_data.sh rpy_direction_validation rpy_direction_clean 0
```

优点：

- 和 RoboTwin 官方流程完全一致。
- 最少踩导入路径问题。

缺点：

- 会修改 third_party。
- 需要至少 4 个新增文件，不满足“尽可能一个 py 文件”的偏好。

### 方案 B：third_party 同级紧凑实现

文件：

```text
rpy_direction_validation/
  rpy_direction_experiment.py
```

这个单文件可以包含：

- `rpy_direction_validation(Base_Task)` 任务类。
- quaternion 工具函数。
- condition 构建。
- smoke/full 配置。
- 采集 runner。
- index 生成。
- 输出检查。

运行命令类似：

```bash
cd /data1/liuwenhao/Projects/robotwin_consistency
python rpy_direction_validation/rpy_direction_experiment.py collect --gpu 0 --smoke
python rpy_direction_validation/rpy_direction_experiment.py index --root third_party/robotwin/data/rpy_direction_validation/rpy_direction_clean
python rpy_direction_validation/rpy_direction_experiment.py check --root third_party/robotwin/data/rpy_direction_validation/rpy_direction_clean
```

优点：

- 不需要改 boundless-world-model。
- 可以不改或少改 third_party。
- 代码集中，便于后续删掉或迁移。

缺点：

- 不能直接使用 `bash collect_data.sh rpy_direction_validation rpy_direction_clean 0`。
- 需要在 runner 里处理 RoboTwin 的 cwd、sys.path、denoiser 初始化、embodiment config、seed/data 两阶段流程。
- 如果完全复用 `collect_data.run()`，它最后会尝试生成 language instruction；外部任务可能没有 `description/task_instruction/rpy_direction_validation.json`，这一步可能报错或产生无关输出。

### 推荐选择

如果当前目标是“先最小可运行，且尽量不要冗余代码”，推荐方案 B：

```text
/data1/liuwenhao/Projects/robotwin_consistency/rpy_direction_validation/rpy_direction_experiment.py
```

同时让输出仍然写到 RoboTwin 标准数据目录：

```text
third_party/robotwin/data/rpy_direction_validation/rpy_direction_clean/
```

这样后续可以直接接当前转换脚本：

```bash
python robotwin_to_bwm.py \
  --stage convert \
  --robotwin_dir third_party/robotwin/data/rpy_direction_validation/rpy_direction_clean \
  --output_dir outputs/robotwin_bwm/rpy_direction_validation_clean \
  --crop_to_experiment \
  --overwrite
```

如果后续确认要长期保留为 RoboTwin 标准任务，再把单文件里的任务类拆到 `third_party/robotwin/envs/`，配置拆到 `task_config/`。

## 5. 建议修正后的最小实现要求

为了和当前代码完全对齐，最小 smoke test 版本建议只做这些：

1. 单文件或任务文件里定义 `class rpy_direction_validation(Base_Task)`。
2. `setup_demo()` 调用 `super()._init_task_env_(**kwags)`。
3. `load_actors()` 为空。
4. condition 用 `self.ep_num` 映射：
   - episode 0: roll +30
   - episode 1: roll -30
   - episode 2: pitch +30
   - episode 3: pitch -30
   - episode 4: yaw +30
   - episode 5: yaw -30
5. 每条 episode：
   - 右臂固定为 `origin` 或固定不动，必须二选一。
   - 左臂先移动到中心点。
   - 记录 `experiment_start_frame`。
   - 再执行 TCP local RPY。
6. `play_once()` 返回的 `self.info["info"]` 中写入所有 condition 和 pose 信息。
7. 若额外写 `meta/*.json`，只在 `self.save_data == True` 时写。
8. `check_success()` 用四元数相对旋转做主判定，Euler 只用于打印和 CSV。
9. 相机用 `Large_D435` 才能满足 640x480。
10. 输出目录保持 RoboTwin 标准结构，方便当前 `robotwin_to_bwm.py` 直接转换。

## 6. 需要从 goal.md 中删除或改写的表述

建议改写以下内容：

1. “任务文件应对应 `env/<task_name>.py`”
   - 本地项目应明确为 `envs/<task_name>.py`。

2. “class RpyDirectionValidation(Base_Task)”
   - 官方入口下应改为 `class rpy_direction_validation(Base_Task)`。

3. “D435 + 480p”
   - 应改为 `Large_D435` 才是 640x480。

4. “如果没有分辨率字段则 README 说明”
   - 当前不是没有，而是通过 `_camera_config.yml` 的 camera type 控制。

5. “metadata 只放 meta/”
   - 应补充同步到 `scene_info.json`，尤其是 `experiment_start_frame`。

6. “right_arm: fixed_or_origin”
   - 应固定为 `origin` 或 `fixed`，不要在同一批数据里语义混合。

7. “先直接跑 collect_data.sh”
   - 如果采用 third_party 同级单文件实现，不能使用这个命令；需要改成外部 runner 命令。

8. “完整实验时再改任务脚本条件开关”
   - 更建议通过 CLI 参数或环境变量控制 smoke/full，避免手改代码造成记录不可复现。

## 7. 结论

`goal.md` 的实验设计方向正确，但按当前代码直接执行会遇到以下主要问题：

- 官方 `collect_data.sh` 不支持任务只放在 `third_party` 同级目录。
- 任务类名示例和 RoboTwin 的动态加载规则冲突。
- 相机配置示例不是 480p。
- 缺少当前转换脚本需要的 `experiment_start_frame`。
- metadata 只写单独 JSON 不够，必须和 `scene_info.json` 对齐。
- 右臂策略和 180° 成功判定需要更严格定义。

在保证实验正确且尽量减少冗余代码的前提下，推荐先做 `third_party` 同级的单文件 runner，实现 smoke test 的 6 条数据；输出仍写入 RoboTwin 标准 `data/rpy_direction_validation/rpy_direction_clean/`。这样可以最小化 third_party 修改，同时保持和现有 `robotwin_to_bwm.py` 的后续转换、推理、对比流程兼容。
