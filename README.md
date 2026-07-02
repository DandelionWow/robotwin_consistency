# robotwin_consistency

对比 RoboTwin 合成视频数据和世界模型对合成视频首帧 + action 序列的预测视频数据.

## 开发方式

本仓库支持两种开发方式.

第一种是基于三方源码库在主仓库根目录做封装. 这种方式不直接修改 `third_party` 源码, 而是在当前仓库中编写 pipeline, 数据转换, 轨迹编辑, 推理封装和输出管理代码. 这是项目最终希望收敛到的方式.

第二种是直接修改 fork 后的三方库源码, 用于快速打通逻辑流程或验证功能. 这种方式可以修改 `third_party/robotwin` 或 `third_party/boundless-world-model`, 但改动需要提交到对应 fork 仓库, 再回主仓库提交 submodule 指针.

现阶段可以两种方式并行使用. 如果只是新增项目编排和封装逻辑, 优先改主仓库; 如果必须快速调整 RoboTwin 或 BWM 内部行为, 再进入对应 submodule 修改.

## 仓库结构

当前项目由 3 个 GitHub 仓库协作组成:

```text
DandelionWow/robotwin_consistency
DandelionWow/RoboTwin
DandelionWow/boundless-world-model
```

主仓库负责项目封装层, pipeline, 数据转换, 实验入口和输出组织. 两个三方库以 submodule 形式放在:

```text
third_party/robotwin
third_party/boundless-world-model
```

现阶段为了快速实现功能, 可以临时修改两个 fork 后的三方库源码. 后续当根目录封装层稳定后, 目标是尽量不再修改三方库源码, 让 submodule 切回稳定的 `main` commit 后由根目录代码直接调用.

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

如果是已有仓库, 更新 submodule URL 和内容:

```bash
proxy_up git pull
git submodule sync --recursive
git submodule update --init --recursive
```

### 2. 切换开发分支

每个人在 3 个仓库中都使用自己的 `dev/<name>` 分支开发.

主仓库:

```bash
proxy_up git fetch origin
git switch -c dev/<name> --track origin/dev/<name>
```

RoboTwin submodule:

```bash
cd third_party/robotwin
proxy_up git fetch origin
git switch -c dev/<name> --track origin/dev/<name>
```

Boundless World Model submodule:

```bash
cd ../boundless-world-model
proxy_up git fetch origin
git switch -c dev/<name> --track origin/dev/<name>
```

已有开发分支:

```text
dev/sunyang
dev/tanwentao
dev/wangbowen
dev/lizhe
dev/wangzequn
dev/liuwenhao
dev/fangxuebin
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

## 提交和推送流程

开发者需要根据实际改动位置分别提交. 三个仓库是独立 Git 仓库, 主仓库只记录 submodule 的具体 commit 指针.

只修改主仓库时:

```bash
git switch dev/<name>
git add <files>
git commit -m "<message>"
proxy_up git push
```

修改 RoboTwin 时:

```bash
cd third_party/robotwin
git switch dev/<name>
git add <files>
git commit -m "<message>"
proxy_up git push

cd ../..
git add third_party/robotwin
git commit -m "<message>"
proxy_up git push
```

修改 Boundless World Model 时:

```bash
cd third_party/boundless-world-model
git switch dev/<name>
git add <files>
git commit -m "<message>"
proxy_up git push

cd ../..
git add third_party/boundless-world-model
git commit -m "<message>"
proxy_up git push
```

同时修改主仓库和两个 submodule 时, 先分别提交并 push submodule, 最后回主仓库一起提交根目录改动和 submodule 指针:

```bash
cd third_party/robotwin
git switch dev/<name>
git add <files>
git commit -m "<message>"
proxy_up git push

cd ../boundless-world-model
git switch dev/<name>
git add <files>
git commit -m "<message>"
proxy_up git push

cd ../..
git add <main-repo-files> third_party/robotwin third_party/boundless-world-model
git commit -m "<message>"
proxy_up git push
```

提交信息请按项目规范整理, 例如:

```text
chore(repo): update submodule workflow
docs(docs): document setup flow
chore(deps): update robotwin submodule
chore(deps): update bwm submodule
```

## 后续收敛方式

现阶段:

```text
robotwin_consistency                  -> dev/<name>
third_party/robotwin                  -> dev/<name>
third_party/boundless-world-model     -> dev/<name>
```

封装层稳定后:

```text
robotwin_consistency                  -> dev/<name> 或 main
third_party/robotwin                  -> main 的稳定 commit
third_party/boundless-world-model     -> main 的稳定 commit
```

也就是说, 后续尽量把三方库源码改动收敛回 fork 的 `main` 或官方 `main`, 主仓库只维护根目录封装逻辑和 submodule 指针.
