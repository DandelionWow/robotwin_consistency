# BWM `eef_abs` action semantics audit

## Verdict

`eef_abs` is an alias for BWM's `state_pose`. It reads 14 values from the
Parquet `observation.state` column:

```text
[left EEF xyz, left EEF Euler xyz, left gripper,
 right EEF xyz, right EEF Euler xyz, right gripper]
```

For RoboTwin data produced by this repository, the 12 pose values are sampled
from the simulated robot's current end-effector pose after physics/controller
execution. They are not the policy command passed into `take_action()`. The two
gripper values are the environment's recorded normalized gripper controller
state. `/joint_action/vector` is also not realized qpos in this RoboTwin
version: its arm joints come from `joint.get_drive_target()`. Actual joint qpos
has separate `get_*_arm_real_jointState()` functions and is not used by
`get_obs()`.

Consequently, “action” in the public BWM wording is misleading for this mode:
the checkpoint is conditioned on an absolute recorded EEF state trajectory.

## Exact 26D and 14D layouts

`robotwin_to_bwm.py:19-27,233-271` constructs `observation.state` as follows:

| 26D indices | meaning | RoboTwin source |
|---|---|---|
| `0:6` | left arm joint drive targets | `/joint_action/vector[0:6]` |
| `6` | left normalized gripper state | `/endpose/left_gripper` (overwrites joint vector value) |
| `7:10` | left realized EEF xyz | `/endpose/left_endpose[0:3]` |
| `10:13` | left realized EEF Euler xyz, radians | `/endpose/left_endpose[3:7]`, quaternion wxyz converted by SciPy |
| `13:19` | right arm joint drive targets | `/joint_action/vector[7:13]` |
| `19` | right normalized gripper state | `/endpose/right_gripper` |
| `20:23` | right realized EEF xyz | `/endpose/right_endpose[0:3]` |
| `23:26` | right realized EEF Euler xyz, radians | `/endpose/right_endpose[3:7]`, quaternion wxyz converted by SciPy |

The 14D `eef_abs` projection is exactly
`[7,8,9,10,11,12,6,20,21,22,23,24,25,19]`.

## End-to-end trace

1. **Policy command and controller.**
   `third_party/robotwin/policy/pi05/deploy_policy.py:40-48` obtains a 14D
   policy output, calls `TASK_ENV.take_action(action)`, then calls `get_obs()`.
   `third_party/robotwin/envs/_base_task.py:1480-1656` interprets the default
   command as two 6-joint targets plus grippers, plans/interpolates paths, writes
   drive targets, and advances physics. `action_type="ee"` is an alternative
   planner input, but it is still an input command rather than the recorded EEF.

2. **Observation sampling.**
   `third_party/robotwin/envs/_base_task.py:438-501` calls `get_arm_pose()` for
   `/endpose/*` and `get_*_arm_jointState()` for `/joint_action/vector`.
   `get_arm_pose()` delegates at lines `1398-1404` to
   `robot.get_*_ee_pose()`.

3. **Realized EEF versus joint target.**
   `third_party/robotwin/envs/robot/robot.py:557-603` reads
   `left_ee/right_ee.global_pose` and applies the RoboTwin tool-frame transforms.
   This is the current simulated EEF pose. Conversely, lines `493-506` use
   `joint.get_drive_target()` for the so-called joint state. Lines `508-524`
   show the unused functions that actually read entity qpos. Lines `609-617`
   are where controller targets are written.

4. **PKL to HDF5.**
   `_take_picture()` uses `get_obs()`; then
   `third_party/robotwin/envs/utils/pkl2hdf5.py:78-100` sorts the sampled PKLs
   and writes their nested arrays to HDF5. It does not reconstruct the original
   policy command.

5. **HDF5 to BWM Parquet.**
   `robotwin_to_bwm.py:233-271` reads the HDF5 fields above, assembles the 26D
   state, and derives a separate 26D delta `action`. Lines `297-314` write both
   `observation.state` and `action`. The delta column is irrelevant to
   `eef_abs`; it is used by `eef_delta`/`action_pose`.

6. **BWM action loader and dataset.**
   `third_party/boundless-world-model/wan_video_action/data/operators.py:497-511`
   declares the exact 14D EEF order. `LoadCobotAction` maps `eef_abs` to
   `state_pose` at lines `537-545`, marks it as state-backed at `553-557`, reads
   `observation.state` at `661-684`, projects the 14 indices, and normalizes to
   `[-1,1]` using p01/p99 (min/max fallback). Both train and inference dataset
   builders instantiate that loader in
   `wan_video_action/data/wan_dataset.py:179-256`.

7. **Pipeline to DiT.**
   `WanVideoActionPipeline.from_pretrained()` creates a 14D action encoder at
   `wan_video_action/pipelines/wan_video_action.py:15-64`. The pipeline passes
   the tensor through `WanVideoUnit_ActionEmbedder` at lines `337-363`.
   `wan_video_action/models/wan_video_action_encoder.py:5-35` produces
   per-frame context tokens and 4-frame grouped modulation embeddings. In
   `model_fn_wan_video_action` (`wan_video_action.py:200-219`), action tokens
   become the DiT cross-attention context (text is disabled), while the grouped
   embedding is added to the timestep embedding before the DiT blocks.

## Where future conditioning comes from

### Public BWM demo/offline inference

`third_party/boundless-world-model/scripts/infer.py:45-59` takes both history
and future conditioning rows from the already loaded full episode tensor.
`_run_autoregressive()` then generates video chunks at lines `62-141`, but the
future `eef_abs` rows still come from the GT Parquet, not from the generated
frames or a controller. Therefore the complete future realized EEF trajectory
is known before demo generation.

This is valid for an explicitly named **GT-conditioned/offline conditional
prediction** experiment. It is privileged post-execution information and is
not, by itself, a deployable policy-action interface or a planning rollout.

### WorldArena 2.0 RL environment

The local WorldArena 2.0 integration is a different checkpoint/data path and
does not establish semantic compatibility with public BWM `step-12000`:

- `RL_env_benchmark/examples/embodiment/config/env/wan_robotwin_adjust_bottle.yaml:39-46`
  configures 8-step chunks, 14 dimensions, and `action_key: abs_action`.
- `RL_env_benchmark/rlinf/data/datasets/world_model.py:258-312` reads raw
  `abs_action`; there is no BWM `stat.json` normalization there.
- `RL_env_benchmark/rlinf/envs/world_model/world_model_wan_env.py:616-676`
  concatenates history action when configured and gives the tensor directly to
  the Wan pipeline. Lines `788-799` show that this tensor is the current
  `policy_output_action` chunk.

Thus the current 8-step policy chunk is known before that chunk is generated,
but the entire future episode is not known at reset. More importantly, the code
does not convert policy joint commands into the public checkpoint's realized
EEF-state convention. Treating these as identical without checkpoint-specific
evidence would be a semantic error.

## Data-flow diagram

```text
policy output (normally 14D dual-arm joint/gripper target)
    |
    v
take_action -> path planner/interpolator -> drive targets -> physics steps
    |                                              |
    |                                              +-> /joint_action/vector
    |                                                  (drive targets, not qpos)
    v
get_obs -> left_ee/right_ee.global_pose + tool transform
    |
    v
/endpose/{left,right}_endpose + gripper values
    |
    v
HDF5 -> robotwin_to_bwm.py -> 26D Parquet observation.state
    |
    v  select [7:13,6,20:26,19]
LoadCobotAction(eef_abs -> state_pose) -> explicit-stat normalization
    |
    v
14D realized EEF/gripper trajectory -> action encoder -> DiT
```

The real chain therefore does **not** contain a direct arrow from the original
policy command to public-BWM `eef_abs`. A live planner would need an explicit,
validated command-to-future-EEF bridge or a model retrained on commands.
