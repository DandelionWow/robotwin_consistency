# RPY Direction Validation

This experiment validates TCP-local roll, pitch, and yaw directions in RoboTwin.
The core task implementation lives outside `third_party`; RoboTwin only gets a
thin `envs/rpy_direction_validation.py` shim and a task config.

## Files

- `rpy_direction_validation/rpy_direction_core.py`: task implementation,
  quaternion helpers, condition generation, metadata construction.
- `third_party/robotwin/envs/rpy_direction_validation.py`: thin import shim for
  RoboTwin's official loader.
- `third_party/robotwin/task_config/rpy_direction_clean.yml`: smoke-test config.
- `third_party/robotwin/description/task_instruction/rpy_direction_validation.json`:
  minimal metadata-only instruction file.  `language_num` is set to 0, but
  RoboTwin's official collection script still loads this file at the end.

## Smoke Test

```bash
cd /data1/liuwenhao/Projects/robotwin_consistency/third_party/robotwin
bash collect_data.sh rpy_direction_validation rpy_direction_clean 0
```

The smoke test contains 6 episodes:

- center position, roll +30 / -30
- center position, pitch +30 / -30
- center position, yaw +30 / -30

The 3x3 grid is built around the current left TCP pose with a default +8cm z
offset.  This keeps the validation pose close to home while giving the planner
enough clearance for pitch rotations.

Output should be written to:

```text
third_party/robotwin/data/rpy_direction_validation/rpy_direction_clean/
  data/episode0.hdf5
  video/episode0.mp4
  scene_info.json
```

`scene_info.json` is the authoritative metadata source.  The task also writes a
human-readable copy under `meta/` during the official data collection stage.

## Full Experiment

The core supports the full condition set with:

```bash
RPY_FULL=1 bash collect_data.sh rpy_direction_validation rpy_direction_clean 0
```

Before running full collection, set `episode_num: 162` in the task config or use
a copied full config.  The condition set is:

```text
9 positions x 3 axes x 6 signed angles = 162 episodes
```

## BWM Conversion

After collection, the output can be converted with the existing compact pipeline:

```bash
cd /data1/liuwenhao/Projects/robotwin_consistency
python robotwin_to_bwm.py \
  --stage convert \
  --robotwin_dir third_party/robotwin/data/rpy_direction_validation/rpy_direction_clean \
  --output_dir outputs/robotwin_bwm/rpy_direction_validation_clean \
  --crop_to_experiment \
  --overwrite
```

`--crop_to_experiment` uses `scene_info.json -> episode_N -> info ->
experiment_start_frame`, which is recorded immediately before the RPY rotation.

## Notes

- Rotation is TCP-local: `q_target = q_current * q_delta`.
- Pose format is `[x, y, z, qw, qx, qy, qz]`.
- Camera type is `Large_D435`, which is 640x480 in this RoboTwin checkout.
- RoboTwin source videos are generated at 30fps.  The existing comparison stage
  in `robotwin_to_bwm.py` aligns by frame index when producing 24fps comparison
  videos.
- For 180 degree rotations, quaternion angle error is the primary success
  metric; Euler angles are only used for logging.
