# BWM runtime audit

Audit date: 2026-10-07

## Verdict

```text
BWM runtime gate: PASS
```

The dedicated BWM environment successfully:

1. imported PyTorch, DiffSynth, BWM data operators, and the action-conditioned
   Wan pipeline;
2. executed a real CUDA kernel;
3. loaded the local Wan2.2 TI2V 5B base model;
4. loaded `step-12000.safetensors`;
5. loaded 81 aligned RGB/state rows from the checked-in demo;
6. completed one non-autoregressive window with 9 GT history frames and 72
   future frames;
7. returned a `[1, 3, 81, 480, 640]` tensor and encoded a 72-frame future MP4.

This passes only the BWM runtime gate. It does not clear the separate Pi0.5
checkpoint, matched-collection, initial-state, or cadence gates.

## 1. Formal interpreter and GPU scope

```text
BWM_PYTHON=/data1/liuwenhao/.conda/envs/bwm/bin/python
Python=3.10.20
CUDA_VISIBLE_DEVICES=0
physical GPU used=0
GPU=NVIDIA RTX 6000D
driver=580.95.05
```

The initial visibility check used only physical cards 0–3. The actual model-load
and one-window inference exposed only physical GPU 0. No GPU outside 0–3 was
used.

## 2. Environment

Core versions:

| package | version |
|---|---|
| PyTorch | `2.8.0+cu128` |
| TorchVision | `0.23.0+cu128` |
| TorchAudio | `2.8.0+cu128` |
| DiffSynth | `2.0.11` |
| NumPy | `1.26.4` |
| OmegaConf | `2.3.0` |
| ImageIO | `2.37.3` |
| ImageIO-FFmpeg | `0.6.0` |
| PyArrow | `23.0.1` |
| Pillow | `12.1.1` |
| Einops | `0.8.2` |
| tqdm | `4.67.3` |
| safetensors | `0.7.0` |
| transformers | `4.57.6` |
| datasets | `4.8.5` |

The first DiffSynth install selected `transformers 5.19.0` and `datasets 5.1.0`.
Those versions conflicted with BWM's pinned `safetensors 0.7.0` and
`pyarrow 23.0.1`. They were constrained to compatible `<5` releases shown
above. Final validation:

```text
python -m pip check
No broken requirements found.
```

CUDA validation with only cards 0–3 visible reported four RTX 6000D devices and
successfully evaluated a tensor on CUDA. The formal inference was then narrowed
to one visible device, GPU 0.

## 3. Code identity

```text
BWM repo: third_party/boundless-world-model
commit: 8a85a222fd65a48c778027d4aa44ffc8fa04206e
branch: dev/liuwenhao
status: clean
```

The smoke entrypoint is:

```text
experiments/policy_shift/bwm_runtime_smoke.py
```

It invokes the checked-in `WanVideoActionPipeline`, video operator, and
`LoadCobotAction`; it does not use Ctrl-World code or checkpoints.

## 4. Checkpoint identity

```text
path:
third_party/boundless-world-model/ckpt/BLM/step-12000.safetensors

bytes:
10051484872

SHA256:
75f863b9474d6e74934db45bb85728fef0adece3d123c667b78349bdade9c7f3
```

The loader reported:

```text
Loaded dit keys: 821 (missing=4, unexpected=0)
Loaded action_encoder keys: 8 (missing=4, unexpected=0)
```

Loading uses `strict=False`. The four action-encoder missing parameters are the
legacy `action_embedding` layer; the active forward path uses `action_mlp1` and
`action_mlp2`, whose eight parameter tensors are present. The audit preserves
the loader counts instead of silently describing the load as strict. The full
one-window inference nevertheless completed deterministically with this exact
code/checkpoint combination.

## 5. Wan2.2 base model identity

Model identity from the checked-in config:

```text
Wan-AI/Wan2.2-TI2V-5B
```

Files used by the `dit + vae` inference profile:

| file | bytes | SHA256 |
|---|---:|---|
| `diffusion_pytorch_model-00001-of-00003.safetensors` | 9825014472 | `720b06c4ade5e87c1246bba8ac95b664c638749cd9b102cf84d823bb44c026a1` |
| `diffusion_pytorch_model-00002-of-00003.safetensors` | 9995661736 | `09ec5ef720d8396f6cfa51fbdcbdb2327e37722afd6e89fd38f1e7e5e782c283` |
| `diffusion_pytorch_model-00003-of-00003.safetensors` | 178558176 | `6306f7894c345de9093ad588771c2abfaeb668a81f7a6d9a918bd26ba3568e49` |
| `Wan2.2_VAE.pth` | 2818839170 | `20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36` |
| `diffusion_pytorch_model.safetensors.index.json` | 72865 | `bfa2337f1163e195d24151a72298daf34a620543898109be47e414c8daa5b3fe` |
| `configuration.json` | 309 | `b79968dbf190e7f4802d48dbb7e8c3a4d1064f5faf90a44435d86239c3e6198a` |
| `config.json` | 251 | `d1fea36899d00c2501b836c13ad65af56e2f9529ba622e50886d3f5c3e6c02bc` |

The same identities are stored in machine-readable form at:

```text
experiments/policy_shift/provenance/bwm_model_provenance.json
```

Text and image encoders are disabled by this BWM profile and were not loaded.

## 6. Inference contract

```text
action mode: adaln
action type: eef_abs -> state_pose
action dimension: 14
conditioning semantics: GT realized dual-arm EEF/gripper state
num_frames: 81
num_history_frames: 9
num_future_frames: 72
num_inference_steps: 50
cfg_scale: 1.0
mixed_precision: bf16
text mode: none
image mode: none
VAE mode: raw
diffusion seed: 0
```

The test made exactly one pipeline call. It did not feed predicted frames into a
second window.

## 7. Runtime smoke result

Input:

```text
video: demo/adjust_bottle/videos/chunk-000/episode_000040.mp4
state: demo/adjust_bottle/data/chunk-000/episode_000040.parquet
selected source frames: 0..80 inclusive
video tensor: 1×3×81×480×640
conditioning tensor: 1×81×14
GT history: frames 0..8
requested future: 72 frames
```

Measured result:

| field | value |
|---|---:|
| base/checkpoint load time | 26.408 s |
| one-window generation time | 44.517 s |
| PyTorch peak allocated GPU memory | 15,347,507,712 bytes |
| prediction shape | `1×3×81×480×640` |
| encoded future frames | 72 |
| encoded geometry | 640×480 |
| future artifact bytes | 830,717 |

Runtime artifacts:

```text
outputs/policy_shift/runtime_smoke/runtime_smoke.json
outputs/policy_shift/runtime_smoke/pred_future.mp4
```

The MP4 was reopened successfully and reports 72 frames at 640×480.

## 8. Reference stat boundary

The runtime smoke used:

```text
path: third_party/boundless-world-model/demo/stat.json
SHA256: 77bd14fad80f855a28bb6767e379aac44660a1a588568ed0a2f8f2806a68c06c
entry: state_pose
bounds: p01/p99
dimensions: 14
```

This proves loader compatibility only. It does not prove that the demo stat is
the public checkpoint's true training stat. Its formal name remains
`reference_stat`, and BWM training-stat provenance remains `UNCONFIRMED`.

## 9. Reproduction command

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/data1/liuwenhao/.conda/envs/bwm/bin/python \
  experiments/policy_shift/bwm_runtime_smoke.py \
  --bwm-root third_party/boundless-world-model \
  --model-root third_party/boundless-world-model/models/Wan2.2-TI2V-5B \
  --checkpoint third_party/boundless-world-model/ckpt/BLM/step-12000.safetensors \
  --dataset-base third_party/boundless-world-model/demo \
  --metadata third_party/boundless-world-model/demo/demo.jsonl \
  --reference-stat third_party/boundless-world-model/demo/stat.json \
  --output-dir outputs/policy_shift/runtime_smoke \
  --num-inference-steps 50 \
  --cfg-scale 1.0 \
  --seed 0
```
