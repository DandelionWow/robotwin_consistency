# GT-history policy-shift evaluator

This directory implements the protocol in `GT_HISTORY_METRIC_AUDIT.md`. It is
deliberately separate from BWM's autoregressive `scripts/infer.py`:

- every window loads its history from the GT video;
- the matching `observation.state` rows are normalized by BWM's own
  `LoadCobotAction(action_type="eef_abs")`;
- no predicted frame becomes history for another window;
- video frame count, Parquet row count, 26D state width, tensor length, and
  GT/prediction shape must agree exactly;
- PSNR and SSIM call the checked-in WorldArena implementation through a thin
  adapter. JEPA and trajectory accuracy stay `null` until their official
  set-level/trajectory preprocessing is adapted to windows.

The evaluator always requires `--reference-stat`. This is intentional because
the training-stat provenance of `step-12000.safetensors` is unconfirmed.

## CPU protocol validation

This validates paths, normalization schema, exact frame/row alignment, and the
window plan without loading BWM:

```bash
/data1/liuwenhao/.conda/envs/robotwin-from-worldarena/bin/python \
  experiments/policy_shift/eval_gt_history.py \
  --bwm-root third_party/boundless-world-model \
  --worldarena-root third_party/WorldArena-2.0/video_quality_ood \
  --dataset-base outputs/adjust_bottle_paired_100_bwm \
  --metadata outputs/adjust_bottle_paired_100_bwm/metadata.jsonl \
  --reference-stat outputs/adjust_bottle_paired_100_bwm/stat.json \
  --checkpoint third_party/boundless-world-model/ckpt/BLM/step-12000.safetensors \
  --output-dir outputs/policy_shift/plan \
  --history-frames 9 \
  --future-frames 72 \
  --stride 36 \
  --plan-only
```

Use `--overwrite` only when replacing an existing plan/output is intended.

## GPU generation and evaluation

Run the same command in the BWM environment, remove `--plan-only`, and add the
Wan model root:

```bash
python experiments/policy_shift/eval_gt_history.py \
  ... \
  --model-root third_party/boundless-world-model/models/Wan2.2-TI2V-5B
```

Outputs are:

- `run_manifest.json`: checkpoint/stat identity and protocol parameters;
- `window_plan.jsonl`: validated GT windows;
- `windows.jsonl`: per-window metric records;
- `summary.json`: overall, per-episode, per-task, distribution, policy, and
  policy-success/failure summaries with bootstrap mean CIs;
- `windows/.../{gt_future,pred_future}/`: exactly aligned frame artifacts.

For all currently emitted metrics, higher is better. The summary preserves the
requested raw gap `policy_mean - expert_mean` and also reports
`gap_error = expert_mean - policy_mean`, so a positive `gap_error` always means
the policy distribution is harder for the world model.

## Tests

```bash
python -m unittest discover -s experiments/policy_shift/tests -v
```
