# BWM normalization provenance audit

## Final status: UNCONFIRMED

The local checkpoint is byte-identical to the official released checkpoint,
but no checked-in file, training log, resolved launch configuration, checkpoint
metadata, Git history entry, or official model-repository artifact binds it to
a specific `state_pose` statistics file. In particular,
`demo/stat.json` is compatible with inference but is **not proven** to be the
training stat.

```text
demo/stat.json provenance: UNKNOWN
WM0 training-stat status: UNCONFIRMED
```

## Checkpoint identity

```text
path: third_party/boundless-world-model/ckpt/BLM/step-12000.safetensors
size: 10,051,484,872 bytes
SHA256: 75f863b9474d6e74934db45bb85728fef0adece3d123c667b78349bdade9c7f3
```

The SHA256 and size match the official
[Hugging Face file](https://huggingface.co/BLM-Lab/Boundless-World-Model/blob/main/step-12000.safetensors).

## Candidate statistics files

| path | SHA256 | evidence / disposition |
|---|---|---|
| `third_party/boundless-world-model/demo/stat.json` | `77bd14fad80f855a28bb6767e379aac44660a1a588568ed0a2f8f2806a68c06c` | `scripts/local.example.sh` pairs it with `step-12000` for demo inference. Added with demo assets in commit `f26b103`; no training provenance. Candidate, not confirmed. |
| `outputs/robotwin_bwm/rpy_grid_rpy_rotation_position_only_clean/stat.json` | `77bd14fad80f855a28bb6767e379aac44660a1a588568ed0a2f8f2806a68c06c` | Byte-identical to demo stat, but locally generated/copied output; identity does not prove training use. |
| `outputs/robotwin_bwm/rpy_grid_rpy_rotation_position_only_clean_uncropped/stat.json` | `77bd14fad80f855a28bb6767e379aac44660a1a588568ed0a2f8f2806a68c06c` | Same bytes; no checkpoint linkage. |
| `outputs/adjust_bottle_paired_100_bwm/stat.json` | `bb99e00e973853a92a27643cecb974bd3e3c9facd23ae4152738ab208341eb21` | Recomputed from the local paired 100 dataset. Suitable only when explicitly evaluating that reference convention; not the public checkpoint's proven training stat. |
| `outputs/robotwin_bwm/beat_block_hammer_demo_clean/stat.json` | `8cbf7cc44199d1a16d3a91c8387bb4df209f67dccb87c8f465503fc373c82e26` | Recomputed conversion output; no checkpoint linkage. |
| `outputs/robotwin_bwm/rpy_direction_validation_clean/stat.json` | `4e607f24bdcf0e31fd625dd0b7f0d391d4766f7e94873f3e931b2567a03edf41` | Recomputed validation output; no checkpoint linkage. |
| `outputs/robotwin_bwm/rpy_grid_rpy_rotation_clean/stat.json` | `63405dc9fa55dd95ec3bc4b1716cc4365a446addf78fc096d38ac0f5d574b115` | Recomputed conversion output; no checkpoint linkage. |

Files found but excluded as BWM candidates:

| path | SHA256 | reason excluded |
|---|---|---|
| `checkpoints/WorldArena2.0/pi05_adjust_bottle/rlinf/robotwin_headcam_adjust_bottle/norm_stats.json` | `b02a165bb92257bce72007800dc452b71d7efcc2f96588dd8c108b263de1ceec` | pi0.5 policy normalization, not BWM `state_pose`. |
| `checkpoints/WorldArena2.0/pi05_click_bell/rlinf/robotwin_headcam_click_bell/norm_stats.json` | `41e220f52db56827605aa5d757d99c4a51b0d43ed8453149c90987b9ff32b288` | pi0.5 policy normalization. |
| `third_party/robotwin/policy/RDT/configs/dataset_stat.json` | `6ca2603e9e7105d578490e6d7a425ee024882d213a666ff3eeb54e8a4baf42ad` | RDT policy statistics. |

## Evidence checked

- `LoadCobotAction` prefers `p01/p99`, falls back to `min/max`, applies
  `2*(x-low)/(high-low)-1`, and clips to `[-1,1]`
  (`wan_video_action/data/operators.py:572-586,617-684`).
- `robotwin_to_bwm.py:274-294` recomputes all four stat groups from every
  conversion. A nearby `stat.json` therefore identifies a conversion dataset,
  not automatically the public checkpoint's training set.
- `scripts/local.example.sh` selects `demo/stat.json`; this proves the intended
  demo invocation only.
- `scripts/infer_example.sh` instead names
  `data/RoboTwin2.0_lerobot/metadata/stat.json`, which is absent locally.
- Training examples pass `${DATASET_DIR}/stat.json`; no resolved path or hash
  for the public run is committed. `scripts/train_local.sh` is gitignored and
  absent.
- Repository history shows `demo/stat.json` entered in `f26b103` as a demo
  asset. No commit message or content says it was copied from the training run.
- The official model's
  [`config.json`](https://huggingface.co/BLM-Lab/Boundless-World-Model/blob/main/config.json)
  confirms `action_type: state_pose`, 81 frames, 9 history frames, and 12,000
  steps, but contains no dataset revision, stat path, stat values, or stat hash.
  The model repository contains no stat file.
- Official GitHub
  [issue #4](https://github.com/boundless-large-model/boundless-world-model/issues/4)
  asks whether `demo/stat.json` was computed from training data. It remains an
  unanswered question, not provenance evidence.
- No repository training logs or checkpoint-side metadata were found that name
  a stat artifact.

Numerical similarity or plausibility was deliberately not used as evidence.

## What to request from the BWM trainers

Ask for all of the following, preferably as one immutable artifact bundle:

1. the exact `stat.json` bytes supplied to the 12,000-step run and its SHA256;
2. the resolved training launch/config log showing `action_type=state_pose` and
   `action_stat_path`;
3. the training metadata/manifest and dataset revision used to compute it;
4. confirmation that `p01/p99` (rather than min/max) were selected;
5. the converter commit and exact 14D order/frame sampling/camera convention;
6. an explicit statement whether `demo/stat.json` is that exact file.

Until those are supplied, experiments must record an explicitly chosen
`--reference-stat` and its SHA256 and must not label it “WM0 training stat.”
