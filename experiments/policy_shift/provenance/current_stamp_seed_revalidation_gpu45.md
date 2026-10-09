# Current-version `stamp_seal` seed revalidation

Status date: 2026-10-09

## Decision

The formal `stamp_seal` seed set is `200028` and `200029`.

- Seed `200025` is rejected. Its formal Expert trajectory had 77 frames, below
  the fixed 81-frame minimum, so the collector stopped before loading Pi0.5.
- Seed `200028` remains eligible. Its formal Expert trajectory had 83 frames;
  two additional independent Expert-only processes produced 87 frames each.
- Seed `200029` is the first untested later candidate. Two independent
  Expert-only processes on physical GPU 4 produced 81 and 82 frames.

Every accepted screening attempt had `plan_success=true`,
`check_success=true`, `timestamp_grid.status=EXACT_GRID`, and an identical
initial-state fingerprint within its seed. Pi0.5 was not used during screening.

## Code and runtime identities

- Root commit before this provenance update:
  `e1043e835bbcbf8c89f1d702dcf4d78027a53f9b`
- RoboTwin submodule commit:
  `43e8910a716473dcf45827911c6512fad893d510`
- Task config: `third_party/robotwin/task_config/demo_clean.yml`
- Task-config SHA256:
  `a3261c7f1e9f18e939722a13d2558a125735d9c49cd4e64727d6d4d67f64387c`
- Physics cadence: 0.004-second steps, sampled every 25 steps (10 Hz)
- Minimum length: 81 regular-grid frames
- Denoiser: OptiX
- CUDA toolkit: 12.8
- CuRobo package/source: 0.7.8 / commit
  `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`
- Python: `/data1/liuwenhao/Projects/WorldArena2_loop/.venv-track2/bin/python`
- GPU isolation used stable UUID binding with `CUDA_DEVICE_ORDER=PCI_BUS_ID`
  and `HWLOC_ALLOW=all`.

## Evidence

| Seed | Context | Physical GPU | Frames | State fingerprint | Result |
|---:|---|---:|---:|---|---|
| 200025 | formal Expert | 4 | 77 | recorded in preserved staging data | reject: short |
| 200028 | formal Expert | 5 | 83 | `34002855...` | pass |
| 200028 | independent screen 1 | 5 | 87 | `34002855...` | pass |
| 200028 | independent screen 2 | 5 | 87 | `34002855...` | pass |
| 200029 | independent screen 1 | 4 | 81 | `cca95412...` | pass |
| 200029 | independent screen 2 | 4 | 82 | `cca95412...` | pass |

The differing frame counts with identical initial-state fingerprints show that
CuRobo planning duration has small run-level variation. Therefore screening is
evidence of eligibility, not a waiver of the formal collector's live 81-frame
gate. A formal pair is published only if its own Expert trajectory passes.

Machine-readable screening evidence is preserved under:

`outputs/policy_shift/expert_seed_screen_stamp_43e8910/`

The rejected formal staging artifact is preserved under:

`outputs/policy_shift/matched_raw_uuid_hwloc_e1043e8/.staging/stamp_seal__seed200025/`
