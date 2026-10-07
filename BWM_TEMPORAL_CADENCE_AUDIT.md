# BWM temporal cadence audit

Audit date: 2026-10-07

Code under audit:

```text
third_party/boundless-world-model
commit 8a85a222fd65a48c778027d4aa44ffc8fa04206e
```

## Verdict

BWM does not infer physical temporal cadence from `num_frames=81`,
`time_division_factor=4`, or output `fps=24`.

For the formal matched evaluator, input cadence is determined entirely by the
source trajectory sampling plus the explicitly selected frame indices:

```text
effective_bwm_fps = raw_fps / bwm_sampling_stride
effective_frame_dt = bwm_sampling_stride / raw_fps
```

Expert and Pi0.5 must have identical `effective_bwm_fps`. A mismatch is a hard
failure. The evaluator now checks this before producing any generation plan.

The fresh dataset has not yet been collected because Pi0.5 checkpoint identity
is unresolved, so dataset-level cadence status remains:

```text
PENDING FRESH MATCHED DATA
```

## 1. Loader behavior

`wan_video_action/data/operators.py` constructs `LoadVideoChunk` with defaults:

```text
frame_rate=24
fix_frame_rate=False
```

There are two distinct code paths.

### Explicit `frame_indices`

When metadata supplies `frame_indices`, `LoadVideoChunk.__call__()` reads every
listed source frame directly with `reader.get_data(frame_id)`. It does not use
MP4 FPS, `frame_rate=24`, or `map_single_frame_id()` to resample those indices.

The policy-shift evaluator always supplies the exact 81 indices for each fixed
window. Therefore contiguous indices mean source-frame stride 1.

### Start/end range without explicit indices

When `frame_indices` is absent, the loader reads source FPS from video metadata.
However, with `fix_frame_rate=False`, `available_frames` is the raw clip length,
and no fixed-24-FPS conversion is requested. The default 24 value is not proof
that BWM training or inference uses 24 physical frames per second.

Formal evaluation avoids this ambiguity by always supplying explicit indices.

## 2. `time_division_factor=4` is not a physical sampling stride

Wan uses:

```text
time_division_factor=4
time_division_remainder=1
```

to enforce a VAE-compatible temporal length such as 81 frames and to map pixel
frames into latent groups `[0]`, `[1:5]`, `[5:9]`, and so on. It does not mean
“take every fourth source frame.”

The primary window stays 81 consecutive selected source frames:

```text
9 GT history + 72 GT future
```

## 3. Output MP4 FPS is metadata, not conditioning cadence

`configs/infer/infer.yaml` contains:

```text
infer.fps: 24
```

`scripts/infer.py` passes that value only to `save_video()` after generation.
It controls playback metadata for the encoded output and does not resample the
input trajectory or action/state tensor.

The formal evaluator instead writes both `gt_future.mp4` and
`pred_future.mp4` with the verified effective source FPS and then reopens both
files to check:

- exactly 72 frames;
- identical width and height;
- identical FPS metadata;
- no one-sided resize.

## 4. Public demo observations

The three checked-in `demo/adjust_bottle` videos report:

| video | frames | geometry | MP4 FPS |
|---|---:|---:|---:|
| `episode_000040.mp4` | 143 | 640×480 | 30 |
| `episode_000041.mp4` | 150 | 640×480 | 30 |
| `episode_000042.mp4` | 143 | 640×480 | 30 |

Their metadata JSONL describes contiguous start/end ranges. This confirms that
the public demo assets happen to be encoded at 30 FPS; it does not independently
prove the complete checkpoint training-time temporal distribution.

## 5. Fresh matched collection cadence

The collector protocol records trajectories from the same physics-step sampler
on both sides. With RoboTwin's default physics timestep:

```text
physics_timestep = 1/250 s
physics_steps_per_sample = 25
raw_fps = 10
raw_frame_dt = 0.1 s
bwm_sampling_stride = 1
effective_bwm_fps = 10
effective_frame_dt = 0.1 s
```

These are proposed defaults, not reconstructed claims about the old data or BWM
training. The actual values are written per pair and checked at runtime.

Sampling at 10 Hz is preferable to mixing the current eval recorder's 10-FPS
video with differently sampled HDF5 state. The fresh collector must drive RGB,
realized EEF, robot state, and timestamp capture from one common physics-step
clock.

## 6. Required per-side fields

Every converted trajectory must include:

```json
{
  "raw_fps": 10.0,
  "raw_frame_dt": 0.1,
  "bwm_sampling_stride": 1,
  "effective_bwm_fps": 10.0,
  "effective_frame_dt": 0.1
}
```

The evaluator also checks declared `raw_fps` against the source MP4 metadata.
Both pair sides must be present before cadence comparison.

## 7. Hard-fail rules

A pair is ineligible if any of the following occurs:

- a cadence field is missing;
- FPS or stride is non-positive/non-finite;
- declared raw FPS disagrees with decoded MP4 FPS;
- Expert and Pi0.5 effective FPS differ;
- RGB and realized-EEF rows do not share the same frame timestamps;
- one side is resampled without applying the identical operation to the other;
- either side is shorter than 81 frames after the common sampling rule.

No padding, one-sided horizon change, or MP4-metadata-only “fix” is allowed.
