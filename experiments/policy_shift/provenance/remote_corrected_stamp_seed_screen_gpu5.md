# Corrected `stamp_seal` Expert-only seed screen (GPU 5)

Status date: 2026-10-08

## Decision

The corrected ascending Expert-only screen is **COMPLETE**. The two accepted
replacement seeds are:

| Task | Accepted seed | Independent run 1 | Independent run 2 |
|---|---:|---:|---:|
| `stamp_seal` | **200025** | 81 regular-grid frames | 81 regular-grid frames |
| `stamp_seal` | **200028** | 87 regular-grid frames | 87 regular-grid frames |

Every accepted run had `plan_success=true`, `check_success=true`,
`eligible=true`, and `timestamp_grid.status=EXACT_GRID`. Each rerun was made in
a separate Python process. Pi0.5 was not loaded or run, no formal matched pair
was generated, and the intentionally paused formal collection runner was not
started. The seed-plan JSON was not modified.

## Fixed code and runtime identities

- Root commit used for every attempt:
  `29f75316b15721b1b501932ea7e246352711521d`
- RoboTwin submodule commit:
  `ce63ccb13e7b3e891ed6209b9b374c07e73c9311`
- Task config: `third_party/robotwin/task_config/demo_clean.yml`
- Task-config SHA256:
  `a3261c7f1e9f18e939722a13d2558a125735d9c49cd4e64727d6d4d67f64387c`
- Python executable:
  `/data1/liuwenhao/Projects/WorldArena2_loop/.venv-track2/bin/python`
- Python: 3.11.14
- Physical GPU: 5, NVIDIA RTX 6000D
- NVIDIA driver: 580.105.08
- GPU isolation: `CUDA_VISIBLE_DEVICES=5`; Torch saw exactly one CUDA device
- CUDA toolkit root: `/usr/local/cuda-12.8`
- CUDA toolkit: 12.8 (`ptxas` 12.8.93)
- Torch: 2.7.1+cu128 (Torch CUDA 12.8)
- JAX: 0.5.3
- SAPIEN: 3.0.1
- SAPIEN denoiser: OptiX
- CuRobo package/tag: 0.7.8 / `v0.7.8`
- CuRobo source:
  `/data1/liuwenhao/Projects/robotwin_consistency/third_party/robotwin/envs/curobo`
- CuRobo source commit:
  `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`
- CuRobo source state: clean editable checkout
- User site packages: disabled with `PYTHONNOUSERSITE=1`

The root repository had no tracked modifications before screening. Existing
untracked provenance files, outputs, logs, checkpoints, and earlier matched
data were preserved. No driver, Python environment, CUDA toolkit, CuRobo
checkout, task implementation, planner, predicate, or task config was changed.

## Screening rule and execution

Seeds were checked in strict ascending order beginning at 200006. A seed was
accepted only if two independent Python processes both met all four gates:

1. `plan_success=true`
2. `check_success=true`
3. `timestamp_grid.status=EXACT_GRID` on the 0.1-second grid
4. `regular_grid_frames >= 81`

An ineligible first attempt was not rerun. The initial requested range
200006--200025 produced one accepted seed, 200025. Screening therefore
continued from the first untested seed, 200026, without lowering any gate;
200028 was the next seed that passed twice. The two runs were kept in separate
output sections so the initial evidence was not overwritten, then combined
mechanically into the final summary.

## Complete results

| Seed | Run | `plan_success` | `check_success` | Grid | Frames | `eligible` | Result / reason |
|---:|---:|---:|---:|---|---:|---:|---|
| 200006 | 1 | true | true | EXACT_GRID | 78 | false | FAIL: successful Expert rollout, but below 81-frame minimum |
| 200007 | 1 | false | false | EXACT_GRID | 47 | false | FAIL: Expert planning/task check failed |
| 200008 | 1 | true | true | EXACT_GRID | 79 | false | FAIL: successful Expert rollout, but below 81-frame minimum |
| 200009 | 1 | true | true | EXACT_GRID | 80 | false | FAIL: successful Expert rollout, but below 81-frame minimum |
| 200010 | 1 | true | true | EXACT_GRID | 77 | false | FAIL: successful Expert rollout, but below 81-frame minimum |
| 200011 | 1 | true | true | EXACT_GRID | 78 | false | FAIL: successful Expert rollout, but below 81-frame minimum |
| 200012 | 1 | false | false | EXACT_GRID | 47 | false | FAIL: Expert planning/task check failed |
| 200013 | 1 | true | true | EXACT_GRID | 79 | false | FAIL: successful Expert rollout, but below 81-frame minimum |
| 200014 | 1 | false | false | EXACT_GRID | 52 | false | FAIL: Expert planning/task check failed |
| 200015 | 1 | true | true | EXACT_GRID | 75 | false | FAIL: successful Expert rollout, but below 81-frame minimum |
| 200016 | 1 | false | false | EXACT_GRID | 51 | false | FAIL: Expert planning/task check failed |
| 200017 | 1 | true | true | EXACT_GRID | 80 | false | FAIL: successful Expert rollout, but below 81-frame minimum |
| 200018 | 1 | true | true | EXACT_GRID | 79 | false | FAIL: successful Expert rollout, but below 81-frame minimum |
| 200019 | 1 | true | true | EXACT_GRID | 76 | false | FAIL: successful Expert rollout, but below 81-frame minimum |
| 200020 | 1 | true | true | EXACT_GRID | 80 | false | FAIL: successful Expert rollout, but below 81-frame minimum |
| 200021 | 1 | false | false | EXACT_GRID | 50 | false | FAIL: Expert planning/task check failed |
| 200022 | 1 | false | false | EXACT_GRID | 47 | false | FAIL: Expert planning/task check failed |
| 200023 | 1 | true | true | EXACT_GRID | 79 | false | FAIL: successful Expert rollout, but below 81-frame minimum |
| 200024 | 1 | false | false | EXACT_GRID | 50 | false | FAIL: Expert planning/task check failed |
| **200025** | **1** | **true** | **true** | **EXACT_GRID** | **81** | **true** | **PASS** |
| **200025** | **2** | **true** | **true** | **EXACT_GRID** | **81** | **true** | **PASS: independent-process confirmation** |
| 200026 | 1 | false | false | EXACT_GRID | 51 | false | FAIL: Expert planning/task check failed |
| 200027 | 1 | true | true | EXACT_GRID | 76 | false | FAIL: successful Expert rollout, but below 81-frame minimum |
| **200028** | **1** | **true** | **true** | **EXACT_GRID** | **87** | **true** | **PASS** |
| **200028** | **2** | **true** | **true** | **EXACT_GRID** | **87** | **true** | **PASS: independent-process confirmation** |

The accepted runs also reproduced the same initial-state fingerprint within
each seed: `cb00394c...` for 200025 and `34002855...` for 200028.

## Preserved local evidence

The final machine-readable summary is:

`outputs/policy_shift/expert_seed_screen_stamp/screen_summary.json`

It has `status=COMPLETE`, `accepted_seeds=[200025, 200028]`, and 23 candidate
records. The original incomplete first-range summary remains at:

`outputs/policy_shift/expert_seed_screen_stamp/screen_summary_200006_200025.json`

The continuation summary and its per-attempt JSON/log files remain under:

`outputs/policy_shift/expert_seed_screen_stamp/continuation_200026/`

The first-range per-attempt JSON/log files and console log remain directly
under `outputs/policy_shift/expert_seed_screen_stamp/`. These ignored output
artifacts are intentionally not committed. Every process emitted the existing
SAPIEN warning about automatic fallback when a Vulkan ICD file was not found;
rendering, planning, exact-grid capture, and result serialization nevertheless
completed normally. There were no process errors, driver changes, environment
rebuilds, or system-package installations.
