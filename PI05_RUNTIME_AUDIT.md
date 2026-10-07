# Pi0.5 Runtime and Checkpoint Audit

Audit date: 2026-10-07 UTC

## Decision

**READY for the six-pair fresh matched collection.**

All three task-specific Pi0.5 inference checkpoints have immutable source
provenance, complete canonical inference manifests, and a successful real model
load plus first inference on physical GPU 0. The RoboTwin motion-planning stack
also completed a real CuRobo CUDA kernel call on the same Blackwell GPU.

This decision applies to the compatibility environment described below. It is
not a claim that the original `uv.lock` environment runs unmodified on this
machine.

## Source lock versus runnable compatibility environment

The unmodified Pi0.5 source lock is preserved and was installed separately at:

```text
/data1/liuwenhao/.conda/envs/pi05-matched
```

Source hashes:

```text
uv.lock        330f8db019cf3aee909e92177c5d138f123317240937baedff9c328742787ddf
pyproject.toml  891795ba02ac6b1d88c02003b3be6a8d072f1764538b62e9002a8bb17b967a68
```

That exact environment passes its dependency check, but its JAX 0.5.0/CUDA
12.6 stack cannot execute the Pi0.5 model on this compute-capability 12.0 GPU:
the bundled PTX toolchain predates SM120, and substituting CUDA 12.8 still ends
in an unsupported BF16-to-F16 lowering during the full model inference.

The runnable environment is an explicit compatibility overlay cloned from the
locked environment:

```text
/data1/liuwenhao/.conda/envs/pi05-blackwell
Python 3.11.17
PYTHONNOUSERSITE=1
```

The material runtime versions are:

```text
jax / jaxlib / jax-cuda12-*  0.6.2
torch                         2.10.0+cu128
torchvision                   0.25.0
nvidia-cudnn-cu12             9.10.2.21
nvidia-cublas-cu12            12.8.4.1
nvidia-curobo                 0.7.8 (editable local checkout)
warp-lang                     1.12.0
chardet                       5.2.0
pip                           24.2
numpy                         1.26.4
```

CUDA compiler/toolkit provenance:

```text
CUDA_ROOT=/data1/liuwenhao/.conda/envs/cuda128-robotwin
ptxas build: cuda_12.8.r12.8/compiler.35583870_0
```

NVIDIA documents CUDA 12.8 as the first CUDA 12.x toolkit with Blackwell
compute-capability 12.0 support in its
[CUDA toolkit/driver/architecture matrix](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html).
The [JAX installation documentation](https://github.com/jax-ml/jax/blob/main/docs/installation.md)
also documents the local-toolkit and `CUDA_ROOT` behavior used here.

The final dependency check has exactly two intentional incompatibilities:

1. `openpi` declares `jax[cuda12]==0.5.0`; this is overridden by JAX 0.6.2 for
   executable SM120 support.
2. `lerobot` declares `torch<2.7`; this is overridden by Torch 2.10.0+cu128 so
   RoboTwin/CuRobo can execute on SM120.

There are no duplicate package distributions. These two overrides are not
silently represented as an exact-lock installation.

## GPU and CuRobo verification

Every GPU command exposed only physical GPU 0; no card outside 0--3 was used.

Torch reported:

```text
device count: 1
device: NVIDIA RTX 6000D
compute capability: 12.0
compiled architectures: sm_70, sm_75, sm_80, sm_86, sm_90, sm_100, sm_120
matrix-multiply smoke: PASS
```

JAX 0.6.2 reported exactly one `gpu` device and completed a JIT matrix
multiplication: **PASS**.

CuRobo was built from the clean local checkout:

```text
path:   /data1/liuwenhao/.cache/curobo-v0.7.8
commit: d64c4b005459db10c5dd867d8b30a87d5bda9bdb
dirty:  false
install: editable nvidia-curobo==0.7.8
```

All five CPython 3.11 extensions contain an `sm_120` target:

```text
geom_cu
kinematics_fused_cu
lbfgs_step_cu
line_search_cu
tensor_step_cu
```

They were imported together on GPU 0. The fused-kinematics extension converted
an identity rotation matrix to the expected CuRobo `xyzw` quaternion
`[0, 0, 0, 1]`: **CUROBO_SM120_KERNEL_SMOKE=PASS**.

## Pi0.5 checkpoint provenance

Upstream repository:
[JackieMM/RoboTwin-pi05-30000-checkpoints](https://huggingface.co/JackieMM/RoboTwin-pi05-30000-checkpoints)

Pinned revision:

```text
4c5a1c2be00d4019b649d72dc81fe1bbf9d999a4
```

The canonical inference identity covers exactly:

```text
_CHECKPOINT_METADATA
params/**
assets/**
```

`train_state/**` is deliberately excluded because RoboTwin inference does not
load it. Therefore these are complete **inference** checkpoints, not complete
training-resume snapshots. Every included local file was compared with the
pinned Hugging Face tree by relative path, byte size, and LFS SHA-256.

| Task | Files | Bytes | Canonical inference-manifest SHA-256 | Remote match |
|---|---:|---:|---|---|
| `blocks_ranking_size` | 14 | 6,335,780,960 | `17a388a5cabd65fe3d7d6564dd6f828ebc017890e44d7ed61d003fa13d479015` | PASS |
| `hanging_mug` | 14 | 6,335,777,835 | `357547e06169c782f2e974aba4d48d2f38af91526fa1b8423017ce483b090951` | PASS |
| `stamp_seal` | 15 | 6,335,782,944 | `1aafcbd5d54fc6c3e83ae21b6a0b4a62b295d675eb89d60696260c95b2cb9349` | PASS |

Machine-readable evidence is stored in
`experiments/policy_shift/provenance/pi05_checkpoint_provenance.json`.

## Real model-load and first-inference smoke

Each task used its own task-specific checkpoint and norm-stat asset. Each run
loaded the full checkpoint, set a language instruction, consumed one real
RoboTwin observation (three RGB views plus a 14D state), and materialized the
returned action array on GPU 0.

| Task | Action shape | Finite | Load seconds | First inference seconds | Result file |
|---|---:|---|---:|---:|---|
| `blocks_ranking_size` | `[50, 14]` | yes | 13.320 | 18.825 | `pi05_runtime_blocks_ranking_size.json` |
| `hanging_mug` | `[50, 14]` | yes | 16.010 | 16.600 | `pi05_runtime_hanging_mug.json` |
| `stamp_seal` | `[50, 14]` | yes | 14.011 | 17.593 | `pi05_runtime_stamp_seal.json` |

The three result files are under `experiments/policy_shift/provenance/` and
contain the complete per-file checkpoint manifests.

## Collector preflight

The final collector preflight for `blocks_ranking_size`, seed `200001`, passed
with:

```text
checkpoint SHA-256: 17a388a5cabd65fe3d7d6564dd6f828ebc017890e44d7ed61d003fa13d479015
RoboTwin commit:     2d3a8eb9804d2e5e239435d32f45da6277503d73
RoboTwin patch SHA:  d88f67a8a3ba7ce7030e774765000991f30d8be33bf24af4630174e3f39fc23e
CuRobo commit:       d64c4b005459db10c5dd867d8b30a87d5bda9bdb
CUDA toolkit:        12.8
```

The collector additionally hard-fails unless the installed `nvidia-curobo`
editable source matches the supplied clean checkout, and each completed pair
manifest records Torch/CUDA/device/architecture, JAX devices, and CuRobo source
identity.
