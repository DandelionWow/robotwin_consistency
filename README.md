# robotwin_consistency

对比 RoboTwin 合成视频数据和世界模型对合成视频首帧 + action 序列的预测视频数据.

## 初始化项目

### 1. 克隆仓库

```bash
git clone --recurse-submodules https://github.com/DandelionWow/robotwin_consistency.git
cd robotwin_consistency
```

如果已经克隆但没有拉取 submodule:

```bash
git submodule update --init --recursive
```

### 2. 切换开发分支

每个人使用自己的 `dev/<name>` 分支开发:

```bash
proxy_up git fetch origin
git switch -c dev/<name> --track origin/dev/<name>
```

### 3. 准备 RoboTwin 环境

RoboTwin 位于:

```text
third_party/robotwin
```

按照 RoboTwin 官方文档安装环境和资源:

```text
https://robotwin-platform.github.io/doc/usage/robotwin-install.html
```

数据采集入口示例:

```bash
cd third_party/robotwin
bash collect_data.sh <task_name> <task_config> <gpu_id>
```

### 4. 准备 Boundless World Model 环境

Boundless World Model 位于:

```text
third_party/boundless-world-model
```

环境初始化参考:

```bash
cd third_party/boundless-world-model
conda create -n BWM python=3.10.20
conda activate BWM
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install diffsynth==2.0.11
pip install -r requirements.txt
```

模型权重需要本地准备:

```bash
modelscope download --model Wan-AI/Wan2.2-TI2V-5B --local_dir models/Wan2.2-TI2V-5B
hf download BLM-Lab/Boundless-World-Model step-12000.safetensors --local-dir ckpt/BLM
```

推理前复制并修改本地配置:

```bash
cp scripts/local.example.sh scripts/local.sh
```

然后更新 `scripts/local.sh` 中的 `MODEL_PATHS` 和 `CKPT_PATH`.
