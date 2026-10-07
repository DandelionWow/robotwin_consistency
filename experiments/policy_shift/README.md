# Matched GT-history policy-shift experiment

This directory implements Experiment 1 only: matched Expert/Pi0.5 trajectories,
verified initial state, one common realized-EEF conversion, fixed temporal
cadence, and independent BWM GT-history windows.

The scientific interpretation is deliberately narrow: this measures offline
future-video fidelity under GT realized-EEF conditioning. It does not measure
policy-command simulation, online planning, or autoregressive stability.

## Hard gates

Before GPU generation, all of the following must pass:

1. old Pi0.5 data provenance or fresh matched collection;
2. exact Expert/Pi0.5 initial-state fingerprint match;
3. common 16D realized-EEF quaternion input to 14D `state_pose` conversion;
4. equal effective temporal cadence;
5. both sides have at least 81 frames;
6. BWM one-window runtime smoke test;
7. paired MP4 frame count, geometry, and FPS validation.

Old episodes whose seed mapping is `AMBIGUOUS` or `UNKNOWN` cannot enter formal
matched-pair statistics. A `_fail` filename is not a verified episode label.

Pi0.5 identity is the SHA256 of a canonical inference-file manifest. For an
Orbax checkpoint this covers all `params/`, `assets/`, and available
`_CHECKPOINT_METADATA` files, rather than selecting one arbitrary shard.

## Metadata schema

The evaluator consumes two rows per `pair_id`, one `expert` and one `pi05`:

```json
{
  "pair_id": "adjust_bottle__seed100123",
  "task": "adjust_bottle",
  "env_seed": 100123,
  "side": "expert",
  "policy_id": "expert",
  "success": true,
  "video": "videos/adjust_bottle__seed100123__expert.mp4",
  "action": "data/adjust_bottle__seed100123__expert.parquet",
  "start_frame": 0,
  "length": 181,
  "raw_fps": 30.0,
  "bwm_sampling_stride": 1
}
```

`env_seed` and diffusion `generation_seed` are separate fields. Generation
seeds are deterministic SHA256-derived values of `pair_id`, window slot, and
repeat ID, so the two sides receive identical diffusion randomness.

## Fixed windows

The only primary window rule is:

```text
history = 9
future = 72
max_start = L - 81
early  = 0
middle = floor(max_start / 2)
late   = max_start
```

Duplicate starts are removed in early/middle/late order. There is no
stride-and-truncate mode and no padding. If either side is shorter than 81
frames, the entire pair is excluded.

## WorldArena code and assets

The only executable evaluation code is the pinned submodule:

```text
third_party/WorldArena
```

Model assets remain external. Point `WORLD_ARENA_ASSETS_ROOT` to the asset-rich
checkout and generate an absolute runtime config plus hash manifest:

```bash
export WORLD_ARENA_ASSETS_ROOT=/data1/liuwenhao/Projects/WorldArena
python experiments/policy_shift/resolve_worldarena_assets.py \
  --worldarena-root third_party/WorldArena \
  --output-config outputs/policy_shift/worldarena.runtime.json \
  --output-manifest outputs/policy_shift/worldarena.assets.json
```

The resolver reads and hashes model assets only. It never executes Python from
the external checkout.

## CPU protocol validation

```bash
/data1/liuwenhao/.conda/envs/robotwin-from-worldarena/bin/python \
  experiments/policy_shift/eval_gt_history.py \
  --bwm-root third_party/boundless-world-model \
  --worldarena-root third_party/WorldArena \
  --dataset-base outputs/policy_shift/matched_bwm \
  --metadata outputs/policy_shift/matched_bwm/metadata.jsonl \
  --reference-stat outputs/policy_shift/reference_stat.json \
  --checkpoint third_party/boundless-world-model/ckpt/BLM/step-12000.safetensors \
  --output-dir outputs/policy_shift/plan \
  --history-frames 9 \
  --future-frames 72 \
  --slots early middle late \
  --repeat-id 0 \
  --plan-only
```

`--reference-stat` is mandatory because the training-stat provenance of the
public checkpoint remains unconfirmed. Range diagnostics must be called
out-of-reference-range rates, not BWM training-distribution OOD rates.

Use `--overwrite` only when replacing an existing plan or output is intended.

## Middle-window GPU smoke

After Smoke A and the BWM runtime gate pass, use the same command in the BWM
environment, remove `--plan-only`, and add:

```bash
--model-root third_party/boundless-world-model/models/Wan2.2-TI2V-5B \
--slots middle
```

For six pairs this produces at most 12 generations. Each sample writes:

```text
windows/<pair_id>/<side>/<slot>/r<repeat_id>/
  gt_future.mp4
  pred_future.mp4
```

Both MP4 files contain exactly 72 future frames and use the effective source
cadence as MP4 metadata. Optional PNG frames require `--debug-png`.

The evaluator also writes:

- `run_manifest.json`: checkpoint/stat/protocol identity and exclusions;
- `window_plan.jsonl`: validated fixed windows;
- `windows.jsonl`: per-window scores and artifacts;
- `worldarena_summary.json`: official `sample_id`, `gt_path`,
  `generated_video` schema;
- `summary.json`: repeat -> window -> pair-side -> paired-gap PSNR/SSIM summary.

PSNR and SSIM call the pinned WorldArena functions through a thin adapter.
The remaining metrics must be run through the official
`third_party/WorldArena/video_quality` CLI; their formulas are not copied here.
JEPA remains a balanced set-level auxiliary score and never enters pair
bootstrap statistics.

## Tests

```bash
python -m unittest discover -s experiments/policy_shift/tests -v
```
