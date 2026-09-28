# RoboTwin / robotwin_consistency 环境配置与问题排查记录

> 记录人：liuwenhao  
> 服务器：`server-pro6000-1`  
> 主要路径：`/data1/liuwenhao/Projects/robotwin_consistency`  
> 主要环境：`conda activate RoboTwin`  
> 记录内容：从新服务器用户环境、项目结构、RoboTwin 安装，到 Blackwell GPU / Curobo / SAPIEN OIDN 问题的排查与解决过程。

---

## 1. 新服务器用户与基础环境

### 1.1 新用户登录提示

新建用户 `liuwenhao` 后登录服务器时出现：

```bash
New release '24.04.4 LTS' available.
*** System restart required ***
/usr/bin/xauth:  file /home/liuwenhao/.Xauthority does not exist
```

判断：

- Ubuntu 版本升级提示不是错误。
- `System restart required` 表示系统更新后建议重启。
- `.Xauthority does not exist` 多数是 SSH X11 forwarding 相关提示，不影响命令行使用。

处理建议：

```bash
sudo -u liuwenhao touch /home/liuwenhao/.Xauthority
sudo chown liuwenhao:liuwenhao /home/liuwenhao/.Xauthority
chmod 600 /home/liuwenhao/.Xauthority
```

---

## 2. Conda 和 cache 迁移到 `/data1`

管理员要求：

> conda 等环境以及各类 cache 全部转移到 `/data1`

最终采用目录结构：

```text
/data1/liuwenhao/
├── Projects/
├── conda/
│   ├── envs/
│   └── pkgs/
├── cache/
└── tmp/
```

创建目录：

```bash
mkdir -p /data1/liuwenhao/{Projects,conda/envs,conda/pkgs,cache,tmp}
```

配置 conda 环境目录和包缓存目录：

```bash
conda config --add envs_dirs /data1/liuwenhao/conda/envs
conda config --add pkgs_dirs /data1/liuwenhao/conda/pkgs
```

配置 pip cache：

```bash
mkdir -p ~/.config/pip
cat > ~/.config/pip/pip.conf <<'EOF_PIP'
[global]
cache-dir = /data1/liuwenhao/cache/pip
EOF_PIP
```

在 `~/.bashrc` 中加入常用 cache 路径：

```bash
export TMPDIR=/data1/liuwenhao/tmp
export XDG_CACHE_HOME=/data1/liuwenhao/cache
export PIP_CACHE_DIR=/data1/liuwenhao/cache/pip
export HF_HOME=/data1/liuwenhao/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/data1/liuwenhao/cache/huggingface/hub
export TRANSFORMERS_CACHE=/data1/liuwenhao/cache/huggingface/transformers
export HF_DATASETS_CACHE=/data1/liuwenhao/cache/huggingface/datasets
export TORCH_HOME=/data1/liuwenhao/cache/torch
export CUDA_CACHE_PATH=/data1/liuwenhao/cache/cuda
export MPLCONFIGDIR=/data1/liuwenhao/cache/matplotlib
```

生效：

```bash
source ~/.bashrc
```

---

## 3. 项目仓库结构理解

项目由 3 个仓库组成：

```text
DandelionWow/robotwin_consistency        # 主仓库
DandelionWow/RoboTwin                    # 三方仓库，作为 submodule
DandelionWow/boundless-world-model       # 三方仓库，作为 submodule
```

主仓库结构：

```text
robotwin_consistency/
├── third_party/
│   ├── robotwin/
│   └── boundless-world-model/
└── 其他主仓库代码
```

开发原则：

- 优先在 `robotwin_consistency` 主仓库中写 pipeline、数据转换、实验入口、输出管理等封装逻辑。
- 如果必须修改 RoboTwin 或 BWM 内部源码，进入对应 submodule 修改，并提交到自己的 `dev/liuwenhao` 分支。
- 修改 submodule 后，需要回到主仓库提交 submodule 指针。

clone 方式：

```bash
cd /data1/liuwenhao/Projects
git clone --recurse-submodules https://github.com/DandelionWow/robotwin_consistency.git
cd robotwin_consistency
```

若 submodule 没拉下来：

```bash
git submodule update --init --recursive
```

切换分支：

```bash
cd /data1/liuwenhao/Projects/robotwin_consistency
git fetch origin
git switch dev/liuwenhao || git switch -c dev/liuwenhao --track origin/dev/liuwenhao

cd third_party/robotwin
git fetch origin
git switch dev/liuwenhao || git switch -c dev/liuwenhao --track origin/dev/liuwenhao

cd ../boundless-world-model
git fetch origin
git switch dev/liuwenhao || git switch -c dev/liuwenhao --track origin/dev/liuwenhao
```

---

## 4. RoboTwin 环境安装

RoboTwin 位于：

```bash
/data1/liuwenhao/Projects/robotwin_consistency/third_party/robotwin
```

创建并激活环境：

```bash
cd /data1/liuwenhao/Projects/robotwin_consistency/third_party/robotwin
conda create -n RoboTwin python=3.10 -y
conda activate RoboTwin
```

执行 RoboTwin 安装脚本：

```bash
bash script/_install.sh
```

下载 RoboTwin assets：

```bash
bash script/_download_assets.sh
```

检查 Python、ffmpeg、Vulkan：

```bash
which python
python --version
ffmpeg -version
vulkaninfo | head
```

实际输出中确认：

```text
/data1/liuwenhao/conda/envs/RoboTwin/bin/python
Python 3.10.20
ffmpeg version 8.0.1
Vulkan Instance Version: 1.3.204
```

说明：

- Python 版本正常。
- ffmpeg 正常。
- `DISPLAY environment variable not set` 在无图形界面服务器上常见，不是致命问题。
- Vulkan 能输出版本，说明基本可用。

---

## 5. 第一次运行 RoboTwin 采集遇到 Curobo / CUDA 问题

测试命令：

```bash
bash collect_data.sh beat_block_hammer demo_clean 1
```

遇到错误：

```text
[planner.py]: Something wrong happened when importing CuroboPlanner!
RuntimeError: CUDA error: no kernel image is available for execution on the device
ImportError: cannot import name 'CuroboPlanner' from 'envs.robot.planner'
```

### 5.1 定位 GPU 和 PyTorch 版本

执行：

```bash
nvidia-smi
```

服务器 GPU：

```text
NVIDIA RTX PRO 6000 Blackwell Workstation Edition
Driver Version: 580.105.08
CUDA Version: 13.0
```

执行：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
PY
```

实际输出：

```text
torch: 2.4.1+cu121
torch cuda: 12.1
cuda available: True
device count: 2
NVIDIA RTX PRO 6000 Blackwell Workstation Edition (12, 0)
```

并且 PyTorch 警告：

```text
NVIDIA RTX PRO 6000 Blackwell Workstation Edition with CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
```

结论：

- 当前 GPU 是 Blackwell，算力 `sm_120`。
- 旧 PyTorch `2.4.1+cu121` 只支持到 `sm_90`。
- 所以 Curobo / torch CUDA kernel 在 Blackwell 上无法运行。

---

## 6. 解决 Blackwell GPU 与 PyTorch 不兼容问题

卸载旧 torch：

```bash
pip uninstall -y torch torchvision torchaudio
```

安装支持 CUDA 12.8 的 PyTorch：

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

清理旧的 torch extensions：

```bash
rm -rf ~/.cache/torch_extensions
rm -rf /data1/liuwenhao/cache/torch_extensions
mkdir -p /data1/liuwenhao/cache/torch_extensions
```

设置 Blackwell 编译架构：

```bash
export TORCH_CUDA_ARCH_LIST="12.0"
export TORCH_EXTENSIONS_DIR=/data1/liuwenhao/cache/torch_extensions
```

建议写入 `~/.bashrc`：

```bash
cat >> ~/.bashrc <<'EOF_BASH'

# PyTorch CUDA extensions for Blackwell GPU
export TORCH_CUDA_ARCH_LIST="12.0"
export TORCH_EXTENSIONS_DIR=/data1/liuwenhao/cache/torch_extensions
EOF_BASH

source ~/.bashrc
```

---

## 7. Curobo 旧编译产物和 ninja 问题

升级 PyTorch 后再次运行，遇到新错误：

```text
ImportError: ... kinematics_fused_cu.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZN3c106detail23torchInternalAssertFailEPKcS2_jS2_RKSs
RuntimeError: Ninja is required to load C++ extensions (pip install ninja to get it)
```

判断：

- 旧的 Curobo `.so` 编译产物仍然存在，和新 PyTorch ABI 不兼容。
- Curobo 需要重新 JIT 编译 CUDA/C++ extension，但环境缺 `ninja`。

安装 ninja：

```bash
pip install ninja
ninja --version
```

删除旧 Curobo 编译产物：

```bash
cd /data1/liuwenhao/Projects/robotwin_consistency/third_party/robotwin
find envs/curobo/src/curobo -name "*.so" -print
find envs/curobo/src/curobo -name "*.so" -delete
find envs/curobo/src/curobo -name "*.so" -print
```

再次清理 torch extensions：

```bash
rm -rf ~/.cache/torch_extensions
rm -rf /data1/liuwenhao/cache/torch_extensions
mkdir -p /data1/liuwenhao/cache/torch_extensions
```

重新运行：

```bash
bash collect_data.sh beat_block_hammer demo_clean 1
```

结果：Curobo 开始重新 JIT 编译：

```text
kinematics_fused_cu not found, JIT compiling...
geom_cu binary not found, jit compiling...
tensor_step_cu not found, jit compiling...
lbfgs_step_cu not found, JIT compiling...
line_search_cu not found, JIT compiling...
```

---

## 8. RoboTwin 成功生成视频，但出现 SAPIEN OIDN 报错

运行时出现大量日志：

```text
[svulkan2] [error] OIDN Error: unsupported device type: CUDA
[svulkan2] [error] OIDN Error: invalid handle
```

但同时也出现：

```text
🎬 Video is saved to `./data/beat_block_hammer/demo_clean/video/episode11.mp4`, containing 126 frames at 320×240 resolution and 30.0 FPS.
```

判断：

- RoboTwin 数据采集主流程已经能跑通。
- 视频确实被保存。
- OIDN 报错主要来自 SAPIEN / svulkan2 的 ray tracing denoiser，和 Blackwell GPU 兼容性有关。
- 该问题不是立即阻塞数据生成，但会刷大量错误日志，并可能影响渲染质量或稳定性。

---

## 9. Codex 实现 denoiser 可切换机制

目标：实现两类修复方案，并可通过参数切换。

### 9.1 方案设计

支持三种 denoiser backend：

```text
oidn    # 默认，保持原始行为
optix   # 使用 NVIDIA OptiX 替代 OIDN
none    # 关闭 denoiser，兜底
```

命令行用法：

```bash
bash collect_data.sh beat_block_hammer demo_clean 1 --denoiser oidn
bash collect_data.sh beat_block_hammer demo_clean 1 --denoiser optix
bash collect_data.sh beat_block_hammer demo_clean 1 --denoiser none
```

环境变量用法：

```bash
ROBOTWIN_DENOISER=optix bash collect_data.sh beat_block_hammer demo_clean 1
```

优先级：

```text
命令行 --denoiser > 环境变量 ROBOTWIN_DENOISER > 默认 oidn
```

### 9.2 Codex 完成的主要改动

新增文件：

```text
third_party/robotwin/envs/render_denoiser_config.py
third_party/robotwin/script/manage_sapien_oidn.py
```

修改文件：

```text
third_party/robotwin/script/collect_data.py
third_party/robotwin/collect_data.sh
third_party/robotwin/envs/_base_task.py
third_party/robotwin/script/test_render.py
third_party/robotwin/script/create_messy_data.py
third_party/robotwin/README.md
```

功能：

- `render_denoiser_config.py`：统一设置 `sapien.render.set_ray_tracing_denoiser(...)`。
- `collect_data.py`：提前解析 `--denoiser` 和 `--oidn-library-dir`，确保在 import SAPIEN 前完成 OIDN 库切换。
- `collect_data.sh`：保持旧命令兼容，并透传新增参数。
- `manage_sapien_oidn.py`：支持查看、替换、dry-run、备份、恢复 SAPIEN OIDN 库。
- README：新增 Blackwell / OIDN compatibility 说明。

实际验证时命令能输出：

```text
[RoboTwin] Using SAPIEN ray tracing denoiser: oidn
```

说明参数切换机制已经生效。

---

## 10. 关于 `--denoiser oidn` 仍然有原问题的原因

运行：

```bash
bash collect_data.sh beat_block_hammer demo_clean 1 --denoiser oidn
```

仍然出现 OIDN 原报错。

原因：

- `--denoiser oidn` 只是选择继续使用 OIDN。
- 如果没有替换新的 OIDN 动态库，它仍然使用 SAPIEN 自带旧版本 OIDN。
- 所以原问题仍然存在，这是预期行为。

真正绕开 OIDN 的命令是：

```bash
bash collect_data.sh beat_block_hammer demo_clean 1 --denoiser optix
```

兜底命令是：

```bash
bash collect_data.sh beat_block_hammer demo_clean 1 --denoiser none
```

---

## 11. 查看当前 SAPIEN OIDN 版本

执行：

```bash
python script/manage_sapien_oidn.py status
```

最初检测到的版本为：

```text
libOpenImageDenoise.so.2.0.1
libOpenImageDenoise_core.so.2.0.1
libOpenImageDenoise_device_cuda.so.2.0.1
```

即：

```text
OIDN 2.0.1
```

判断：

- SAPIEN 自带 OIDN 2.0.1。
- Blackwell GPU 对 OIDN 版本有兼容性要求。
- 需要尝试 OIDN 2.3.3 或更新版本。

---

## 12. 下载并替换 OIDN 2.3.3

下载 OIDN 2.3.3：

```bash
cd /data1/liuwenhao
mkdir -p libs
cd libs
wget https://github.com/RenderKit/oidn/releases/download/v2.3.3/oidn-2.3.3.x86_64.linux.tar.gz
tar -xzf oidn-2.3.3.x86_64.linux.tar.gz
```

检查库文件：

```bash
find /data1/liuwenhao/libs/oidn-2.3.3.x86_64.linux -maxdepth 3 -type f | grep "libOpenImageDenoise"
```

用脚本 dry-run：

```bash
cd /data1/liuwenhao/Projects/robotwin_consistency/third_party/robotwin
conda activate RoboTwin

python script/manage_sapien_oidn.py use-custom \
  --source /data1/liuwenhao/libs/oidn-2.3.3.x86_64.linux/lib \
  --dry-run
```

正式替换：

```bash
python script/manage_sapien_oidn.py use-custom \
  --source /data1/liuwenhao/libs/oidn-2.3.3.x86_64.linux/lib
```

替换后查看状态：

```bash
python script/manage_sapien_oidn.py status
```

实际输出：

```text
[RoboTwin] OIDN libraries:
  libOpenImageDenoise.so
  libOpenImageDenoise.so.2
  libOpenImageDenoise.so.2.3.3
  libOpenImageDenoise_core.so.2.3.3
  libOpenImageDenoise_device_cpu.so.2.3.3
  libOpenImageDenoise_device_cuda.so.2.3.3
  libOpenImageDenoise_device_hip.so.2.3.3
  libOpenImageDenoise_device_sycl.so.2.3.3
[RoboTwin] Detected OIDN version(s): 2.3.3
```

说明：

- SAPIEN OIDN 已经替换为 2.3.3。

---

## 13. 替换 OIDN 2.3.3 后的新问题：动态库加载失败

运行：

```bash
bash collect_data.sh beat_block_hammer demo_clean 1 --denoiser oidn
```

出现错误：

```text
ImportError: libOpenImageDenoise.so.2: cannot open shared object file: No such file or directory
```

判断：

- OIDN 2.3.3 文件已经在 SAPIEN 的 `oidn_library` 目录中。
- 但系统动态链接器在 import `sapien/pysapien` 时没有找到 `libOpenImageDenoise.so.2`。
- 这是动态库搜索路径问题，不是 OIDN 文件不存在。

检查 OIDN 目录：

```bash
OIDN_DIR=/data1/liuwenhao/conda/envs/RoboTwin/lib/python3.10/site-packages/sapien/oidn_library
ls -lah "$OIDN_DIR"
readlink -f "$OIDN_DIR/libOpenImageDenoise.so.2"
```

补齐软链接：

```bash
cd "$OIDN_DIR"

for lib in \
  libOpenImageDenoise \
  libOpenImageDenoise_core \
  libOpenImageDenoise_device_cpu \
  libOpenImageDenoise_device_cuda \
  libOpenImageDenoise_device_hip \
  libOpenImageDenoise_device_sycl
do
  if [ -f "${lib}.so.2.3.3" ]; then
    ln -sf "${lib}.so.2.3.3" "${lib}.so.2"
    ln -sf "${lib}.so.2.3.3" "${lib}.so"
  fi
done
```

临时设置 `LD_LIBRARY_PATH`：

```bash
export LD_LIBRARY_PATH="$OIDN_DIR:$LD_LIBRARY_PATH"
```

测试 SAPIEN import：

```bash
python - <<'PY'
import sapien
print("sapien import OK:", sapien.__file__)
PY
```

如果 import 成功，再运行：

```bash
bash collect_data.sh beat_block_hammer demo_clean 1 --denoiser oidn
```

若成功，则可将路径写入 `~/.bashrc`：

```bash
cat >> ~/.bashrc <<'EOF_BASH'

# SAPIEN OIDN library path
export LD_LIBRARY_PATH=/data1/liuwenhao/conda/envs/RoboTwin/lib/python3.10/site-packages/sapien/oidn_library:$LD_LIBRARY_PATH
EOF_BASH

source ~/.bashrc
```

---

## 14. 旧数据导致采集从已有进度开始

运行时出现：

```text
Exist seed file, Start from: 64 / 50
Complete simulation, failed 0 times / 64 tries
```

判断：

- 之前已经生成过 `data/beat_block_hammer/demo_clean` 下的数据。
- 当前任务检测到已有 seed / episode 数据。
- 因为已有数量超过目标数量，所以不会从 0 开始重新生成。

检查已有数据：

```bash
find data/beat_block_hammer/demo_clean -maxdepth 3 -type f | head -50
find data/beat_block_hammer/demo_clean/video -name "*.mp4" | wc -l
ls -lht data/beat_block_hammer/demo_clean/video | head
```

如果需要干净重跑，先备份旧目录：

```bash
mv data/beat_block_hammer/demo_clean \
   data/beat_block_hammer/demo_clean.bak_$(date +%Y%m%d_%H%M%S)
```

再重新运行：

```bash
bash collect_data.sh beat_block_hammer demo_clean 1 --denoiser optix
```

或：

```bash
bash collect_data.sh beat_block_hammer demo_clean 1 --denoiser oidn
```

---

## 15. 当前状态总结

已经完成：

- 新服务器用户可正常使用。
- conda 环境和 cache 已迁移到 `/data1`。
- `robotwin_consistency` 主仓库和 submodule 结构已理解并配置。
- RoboTwin 环境安装完成。
- Python、ffmpeg、Vulkan 基本检查通过。
- Blackwell GPU 与旧 PyTorch 不兼容问题已定位。
- PyTorch 已升级到支持 Blackwell 的 CUDA 12.8 版本。
- ninja 缺失问题已解决。
- 旧 Curobo `.so` 编译产物已清理，Curobo 可以重新 JIT 编译。
- RoboTwin 已能生成视频文件。
- Codex 已实现 `--denoiser oidn|optix|none` 参数切换机制。
- SAPIEN OIDN 已从 2.0.1 替换到 2.3.3。

当前仍需验证或处理：

- 替换 OIDN 2.3.3 后，需要解决 `LD_LIBRARY_PATH` / 动态库加载问题。
- `--denoiser oidn` 是否在 OIDN 2.3.3 + 正确动态库路径后不再报错，需要继续实测。
- `--denoiser optix` 是优先推荐的绕开 OIDN 方案，也需要真实采集测试。
- `--denoiser none` 可作为最稳兜底方案。
- 旧数据目录会影响是否重新生成，需要测试前备份或删除旧 `demo_clean` 数据目录。

---

## 16. 推荐后续执行顺序

### 16.1 先解决 OIDN 2.3.3 动态库路径

```bash
cd /data1/liuwenhao/Projects/robotwin_consistency/third_party/robotwin
conda activate RoboTwin

OIDN_DIR=/data1/liuwenhao/conda/envs/RoboTwin/lib/python3.10/site-packages/sapien/oidn_library
export LD_LIBRARY_PATH="$OIDN_DIR:$LD_LIBRARY_PATH"

python - <<'PY'
import sapien
print("sapien import OK:", sapien.__file__)
PY
```

### 16.2 干净重跑 OIDN 方案

```bash
mv data/beat_block_hammer/demo_clean \
   data/beat_block_hammer/demo_clean.bak_$(date +%Y%m%d_%H%M%S)

bash collect_data.sh beat_block_hammer demo_clean 1 --denoiser oidn
```

### 16.3 测试 OptiX 方案

```bash
bash collect_data.sh beat_block_hammer demo_clean 1 --denoiser optix
```

### 16.4 测试 none 兜底方案

```bash
bash collect_data.sh beat_block_hammer demo_clean 1 --denoiser none
```

### 16.5 修改完成后提交 RoboTwin submodule

```bash
cd /data1/liuwenhao/Projects/robotwin_consistency/third_party/robotwin

git status
git add envs/render_denoiser_config.py \
        script/manage_sapien_oidn.py \
        script/collect_data.py \
        collect_data.sh \
        envs/_base_task.py \
        script/test_render.py \
        script/create_messy_data.py \
        README.md

git commit -m "fix(render): add configurable sapien denoiser"
git push origin dev/liuwenhao
```

回主仓库提交 submodule 指针：

```bash
cd ../..

git status
git add third_party/robotwin
git commit -m "chore/deps: update robotwin denoiser config"
git push origin dev/liuwenhao
```

注意：不要直接 `git add .`，避免误提交 `goal.md` 或 `third_party/boundless-world-model` 的异常状态。

---

## 17. 常用命令备忘

查看 GPU：

```bash
nvidia-smi
```

查看 PyTorch 和 GPU 架构：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("arch list:", torch.cuda.get_arch_list())
print("cuda available:", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
PY
```

查看当前 OIDN 状态：

```bash
python script/manage_sapien_oidn.py status
```

恢复最近一次 OIDN 备份：

```bash
python script/manage_sapien_oidn.py restore --latest
```

测试不同 denoiser：

```bash
bash collect_data.sh beat_block_hammer demo_clean 1 --denoiser oidn
bash collect_data.sh beat_block_hammer demo_clean 1 --denoiser optix
bash collect_data.sh beat_block_hammer demo_clean 1 --denoiser none
```

查看视频输出：

```bash
find data/beat_block_hammer/demo_clean/video -name "*.mp4" | wc -l
ls -lht data/beat_block_hammer/demo_clean/video | head
```

检查视频信息：

```bash
ffprobe -v error \
  -select_streams v:0 \
  -show_entries stream=width,height,r_frame_rate,nb_frames,duration \
  -of default=noprint_wrappers=1 \
  data/beat_block_hammer/demo_clean/video/episode11.mp4
```
