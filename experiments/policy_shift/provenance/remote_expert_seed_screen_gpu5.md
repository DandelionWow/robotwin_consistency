# Remote Expert-only seed screen (GPU 5)

Status date: 2026-10-07

## Decision

The first ascending replacement candidate for each task passed the unchanged
Expert gate in two independent Python processes. The recommended formal seeds
are therefore:

| Task | Retained seed | Replacement seed | Final formal seeds |
|---|---:|---:|---|
| `blocks_ranking_size` | 200002 | 200003 | **200002, 200003** |
| `hanging_mug` | 200001 | 200003 | **200001, 200003** |
| `stamp_seal` | 200004 | 200005 | **200004, 200005** |

This was an Expert-only screen. Pi0.5 was not loaded or run, formal matched
collection was not started, and no formal matched pair was generated.

## Fixed code and runtime identities

- Root commit used for the replacement screen:
  `81754f12719b2bc5ca05fbabcb7877f52d53ced9`
- RoboTwin commit: `ce63ccb13e7b3e891ed6209b9b374c07e73c9311`
- Task config: `third_party/robotwin/task_config/demo_clean.yml`
- Task-config SHA256:
  `a3261c7f1e9f18e939722a13d2558a125735d9c49cd4e64727d6d4d67f64387c`
- CuRobo tag: `v0.7.8`
- CuRobo commit: `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`
- CuRobo source state: clean editable checkout
- Physical GPU: 5, NVIDIA RTX 6000D, 85,651 MiB
- NVIDIA driver: 580.105.08
- CUDA toolkit: 12.8 (`ptxas` build 12.8.93)
- Python: 3.11.14
- Torch: 2.7.1+cu128
- JAX: 0.5.3
- SAPIEN: 3.0.1
- CuRobo Python package: 0.7.8
- SAPIEN denoiser: OptiX

Physical GPU 5 was explicitly authorized by the user and is formally accepted
by root commit `81754f1`. `CUDA_VISIBLE_DEVICES=5` exposed only that physical
GPU to every screening process. The root repository, RoboTwin submodule, and
CuRobo source were clean before screening.

## Screening method

Each attempt constructed the unchanged RoboTwin task with the exact environment
seed and `demo_clean.yml`, then ran `play_once()` with normal CuRobo planning.
Trajectory saving was disabled, but the task, planner, task predicate, physics
timestep, and scene setup were not modified. The diagnostic wrapper only
recorded high-level call boundaries, planner status, gripper status,
`plan_success`, and `check_success()`.

For each task, seeds were examined in ascending order from the requested start:

- `blocks_ranking_size`: 200003 onward
- `hanging_mug`: 200003 onward
- `stamp_seal`: 200005 onward

The first seed with both booleans true was rerun in a separate Python process.
It was accepted only when the independent rerun also returned both booleans
true. Because the first candidate passed twice for every task, no higher seed
was attempted.

## Complete replacement-screen results

| Task | Seed | Run | `plan_success` | `check_success` | Stage and reason |
|---|---:|---:|---:|---:|---|
| `blocks_ranking_size` | 200003 | 1 | true | true | PASS: full Expert rollout completed; all recorded left/right planner statuses were `Success`; both grippers were open at the final predicate. |
| `blocks_ranking_size` | 200003 | 2 | true | true | PASS: independent process reproduced the complete rollout; all recorded planner statuses were `Success`; both grippers were open. |
| `hanging_mug` | 200003 | 1 | true | true | PASS: full Expert rollout completed; all recorded planner statuses were `Success`; final hanging-mug predicate passed. |
| `hanging_mug` | 200003 | 2 | true | true | PASS: independent process reproduced the full rollout and final task predicate. |
| `stamp_seal` | 200005 | 1 | true | true | PASS: full right-arm grasp, displacement, placement, and retreat completed; all recorded planner statuses were `Success`; final seal predicate passed. |
| `stamp_seal` | 200005 | 2 | true | true | PASS: independent process reproduced the full rollout and final task predicate. |

Task-specific deterministic scene details also matched across the two runs:

- `blocks_ranking_size/200003`: arm assignment was large=left,
  medium=left, small=right; each run produced 25 high-level trace records.
- `hanging_mug/200003`: mug asset `039_mug/base2`, rack asset
  `040_rack/base0`; each run produced 17 high-level trace records.
- `stamp_seal/200005`: seal asset `100_seal/base6`, purple target,
  right arm; each run produced 6 high-level trace records.

## Original fixed-seed screen retained for the final plan

The original six-seed Expert screen ran with root commit `aba3668` plus the
provenance-captured temporary GPU-5 validation extension. Root commit `81754f1`
formally incorporates that GPU-5 authorization. RoboTwin, CuRobo, the task
config, and the Expert/task implementation were unchanged between the original
screen and the replacement screen.

| Task | Seed | `plan_success` | `check_success` | Result, failure stage, and reason |
|---|---:|---:|---:|---|
| `blocks_ranking_size` | 200001 | false | false | FAIL: the first two blocks completed; during placement of the third/large block, the right-arm CuRobo path returned `Fail`. The right gripper remained closed. This failure was reproduced by the formal attempt and a separate Expert-only diagnostic. |
| `blocks_ranking_size` | 200002 | true | true | PASS: full rollout completed, all recorded planner statuses were `Success`, and the final ranking predicate passed. |
| `hanging_mug` | 200001 | true | true | PASS: full rollout completed and the final hanging-mug predicate passed. |
| `hanging_mug` | 200002 | true | false | FAIL: all planned paths returned `Success`, both grippers were open, but the final physical hanging-mug predicate was false. |
| `stamp_seal` | 200003 | false | false | FAIL: the left-arm CuRobo placement path returned `Fail`; the left gripper remained closed. |
| `stamp_seal` | 200004 | true | true | PASS: full rollout completed and the final seal predicate passed. |

## Preserved local evidence

The ignored remote-server logs are retained under `outputs/policy_shift/logs/`:

- `expert_seed_screen_gpu5_environment.txt`
- `expert_seed_screen_gpu5_blocks_ranking_size_200003_run1.log`
- `expert_seed_screen_gpu5_blocks_ranking_size_200003_run2.log`
- `expert_seed_screen_gpu5_hanging_mug_200003_run1.log`
- `expert_seed_screen_gpu5_hanging_mug_200003_run2.log`
- `expert_seed_screen_gpu5_stamp_seal_200005_run1.log`
- `expert_seed_screen_gpu5_stamp_seal_200005_run2.log`
- `expert_seed200001_diagnostic.log`
- `remaining_expert_seed_diagnostics.log`

The earlier failed formal staging directory remains preserved at
`outputs/policy_shift/matched_raw/.staging/blocks_ranking_size__seed200001/`.
No staging directory or prior log was deleted or overwritten.

No HDF5, MP4, checkpoint, console log, staging content, or formal matched data
is included in this provenance document or intended for Git.
