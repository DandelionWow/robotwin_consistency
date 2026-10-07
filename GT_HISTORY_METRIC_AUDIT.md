# WorldArena metrics for GT-history policy-shift evaluation

## Scope and source state

The target quantity is conditional prediction fidelity for independently reset
windows:

```text
GT history + matching true BWM conditioning -> predicted future vs GT future
```

It is not the full-video WorldArena EWMScore. Sources inspected were:

- local WorldArena 2.0 repository at commit
  `e295378c702b2e87617ebddcad193be3608e00c3`;
- local `video_quality_ood/WorldArena/*` implementations, including the recent
  continuous Dynamic Degree, Flow Score, photometric, and VFI smoothness code;
- official WorldArena
  [video-quality instructions](https://github.com/tsinghua-fib-lab/WorldArena/blob/main/video_quality/README.md)
  and [Track-1 submission protocol](https://github.com/tsinghua-fib-lab/WorldArena/blob/main/assets/README_submission.md);
- official WorldArena
  [`action_following.py`](https://github.com/tsinghua-fib-lab/WorldArena/blob/main/video_quality/WorldArena/action_following.py).

The local WorldArena 2.0 checkout imports `action_following.py` but does not
contain its source file, so that one metric was verified against the official
WorldArena repository rather than a `.pyc` file.

## Metric classification

| metric | GT-reference based? | short-window valid? | measures conditional fidelity? | recommended role |
|---|---|---|---|---|
| PSNR | Yes, frame-aligned RGB | Yes, with exact length/resolution alignment | Yes, pixel error; sensitive to harmless appearance shifts | PRIMARY |
| SSIM | Yes, frame-aligned RGB | Yes | Yes, local structural similarity | PRIMARY |
| JEPA Similarity | Yes, paired real/generated video sets | Conditionally; official code uniformly samples/pads 16 frames | Indirectly, in learned video-feature distribution; official result is set-level, not per-window | NEEDS_ADAPTATION |
| Trajectory Accuracy | Yes, predicted versus GT tracked 2D trajectories | Potentially, after slicing/re-extracting both trajectories to the same window | Yes for motion consequence, but not until tracker identity/coordinates and window boundaries are aligned | NEEDS_ADAPTATION |
| Depth Accuracy | Yes, estimated depth from generated and GT frames | Yes, but the official code samples up to 40 frames and median-scale aligns depth | Partly; tests geometry rather than exact RGB | SECONDARY |
| Photometric Consistency | No; forward/backward flow cycle consistency inside generated video | Technically >=2 frames, but unstable for short/fps-changed clips | No; a self-consistent wrong outcome can score well | SECONDARY |
| Flow Score | No; mean generated-video optical-flow magnitude | Numerically possible, scientifically weak for short windows and fps changes | No; measures amount of motion | AR_ONLY |
| Dynamic Degree | No; thresholded/soft RAFT motion in generated video | No robustly; sampling interval is derived from fps and thresholding depends on resolution/frame count | No | AR_ONLY |
| Motion Smoothness | No; VFI prediction of odd frames from even frames | >=3 frames, but strongly cadence/frame-count dependent | No; measures temporal interpolatability | AR_ONLY |
| Interaction Quality | No GT; VLM judges visible contact/physics | Sometimes, only if the short window contains the interaction | Not direct conditional fidelity | SECONDARY |
| Subject Consistency | No; DINO similarity to previous/first generated frames plus Dynamic Degree | Computable but unstable and motion-coupled on short clips | No | AR_ONLY |
| Background Consistency | No; CLIP similarity to previous/first generated frames plus Dynamic Degree | Computable but unstable and motion-coupled | No | AR_ONLY |
| Image Quality | No; MUSIQ technical quality per generated frame | Yes | No | NOT_USE |
| Aesthetic Quality | No; CLIP aesthetic predictor | Yes | No | NOT_USE |
| Instruction Following | No GT video; VLM judges generated frames against task text | Often incomplete for a short sub-action window | No direct state/action-consequence comparison | AR_ONLY |
| Semantic Alignment | GT-derived and generated video captions compared in CLIP text space | Not without window-specific recaptioning | Only coarse semantic fidelity, not frame/action alignment | NEEDS_ADAPTATION |
| Action Following | No GT consequence; cross-action generated-video diversity | Requires multiple counterfactual prompts/videos, not one GT window | No | NOT_USE |

`PRIMARY` here means primary for the current implementation and scientific
question. `NEEDS_ADAPTATION` means the metric could become useful but the
official entrypoint cannot be relabeled as a per-window metric without changing
its protocol.

## Special finding: Action Following

The official implementation does the following for each episode:

1. encode every frame of each generated action/prompt variant with CLIP;
2. average frames into one feature per variant;
3. compute pairwise cosine distance among variants `gid=1,2,3`;
4. average those cross-variant distances.

The official README instructs users to create variants 2 and 3 by modifying the
prompt to request different actions. No predicted trajectory is compared to the
GT action consequence. A model can receive a high diversity score while all
three outcomes are wrong. Therefore Action Following cannot be a primary
Expert-to-Policy GT-history gap metric.

## Special finding: motion metrics

The local implementations make cadence dependence explicit:

- Dynamic Degree samples MP4 frames at `round(fps/8)` and uses thresholds tied
  to resolution and sampled count.
- Flow Score averages flow magnitude over adjacent decoded frames, so changing
  fps changes displacement per pair.
- Motion Smoothness splits even/odd frames and predicts each odd midpoint from
  its even neighbors; it needs at least three frames and its meaning changes
  with temporal spacing.
- Photometric Consistency is the reciprocal of flow-cycle EPE, additionally
  multiplied by a Dynamic Degree term for low-motion videos.

These can diagnose long autoregressive rollouts under one fixed fps/frame-count
protocol. They must not be primary short-window policy-gap results.

## Minimal evaluator decision

Implemented under `experiments/policy_shift/`:

- contiguous windows with Wan-valid lengths (`history=9`, `future=72` by
  default);
- an explicit, mandatory reference stat with path and SHA256 in the manifest;
- pre-generation validation of video count, Parquet count, metadata range, and
  26D `observation.state` width;
- each call loads its own GT history and matching conditioning rows;
- BWM generation uses `WanVideoActionPipeline.__call__()` without modifying
  BWM internals;
- a thin adapter calls WorldArena's `peak_signal_noise_ratio` and `cal_ssim`;
- strict shape equality: unlike the original full-benchmark wrapper, the
  evaluator neither silently truncates unequal frame lists nor resizes the
  prediction to GT during scoring;
- per-window JSONL and aggregate mean/median/sample-std/bootstrap mean CI for
  per-episode, per-task, distribution, policy, and policy success/failure;
- raw `Gap = policy_mean - expert_mean`, plus `Gap_error` whose positive sign
  always means policy windows are worse.

JEPA and Trajectory Accuracy fields are present as `null`. The official JEPA
implementation computes one set-level JEDi distance after uniform 16-frame
sampling, while Trajectory Accuracy assumes separately extracted `traj.npy`
files and chooses a GT trajectory by spatial extent. Both require a documented
window adapter before their numbers can be scientifically comparable. They are
not silently replaced with lookalike metrics.

GT-history and AR evaluation remain independent entrypoints. BWM's existing
`scripts/infer.py` is the AR path; `eval_gt_history.py` never consumes its prior
predictions.
