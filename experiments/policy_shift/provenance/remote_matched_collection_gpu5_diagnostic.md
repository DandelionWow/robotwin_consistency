# Remote matched collection diagnostic (GPU 5)

Status date: 2026-10-07

## Scope

This records the attempted RoboTwin Experiment 1 matched-data collection on the
remote server. At the user's direction, the run used physical GPU 5 rather than
physical GPUs 0 through 3. No BWM or WorldArena job was run.

The requested fixed pairs were:

| Task | Seeds |
|---|---|
| `blocks_ranking_size` | `200001`, `200002` |
| `hanging_mug` | `200001`, `200002` |
| `stamp_seal` | `200003`, `200004` |

## Fixed identities

- Root commit: `aba3668785609865af61a0c3f5bb0cfcab5e5a3b`
- RoboTwin commit: `ce63ccb13e7b3e891ed6209b9b374c07e73c9311`
- CuRobo: clean editable source at tag `v0.7.8`, commit
  `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`
- Checkpoint repository: `JackieMM/RoboTwin-pi05-30000-checkpoints`
- Checkpoint revision: `4c5a1c2be00d4019b649d72dc81fe1bbf9d999a4`

The complete canonical inference-checkpoint identities matched the declared
values:

| Task | Files | Bytes | Canonical SHA256 |
|---|---:|---:|---|
| `blocks_ranking_size` | 14 | 6,335,780,960 | `17a388a5cabd65fe3d7d6564dd6f828ebc017890e44d7ed61d003fa13d479015` |
| `hanging_mug` | 14 | 6,335,777,835 | `357547e06169c782f2e974aba4d48d2f38af91526fa1b8423017ce483b090951` |
| `stamp_seal` | 15 | 6,335,782,944 | `1aafcbd5d54fc6c3e83ae21b6a0b4a62b295d675eb89d60696260c95b2cb9349` |

## Runtime

- Physical GPU: 5, NVIDIA RTX 6000D, 85,651 MiB
- Driver: 580.105.08
- CUDA toolkit: 12.8 (`ptxas` build 12.8.93)
- Python: 3.11.14
- Torch: 2.7.1+cu128
- JAX: 0.5.3
- SAPIEN: 3.0.1
- CuRobo: 0.7.8
- Denoiser: OptiX

SAPIEN rendering passed its direct render test. Torch and JAX each saw exactly
one CUDA device after mapping physical GPU 5.

The checked-in GPU contract originally accepted only physical GPUs 0 through
3. The working tree used for this attempt contained a minimal, provenance-
captured extension that accepted GPU 5 in the runner and collector argument
validation. RoboTwin itself remained clean.

## Preflight result

All six requested task/seed preflights returned `PREFLIGHT_OK`. They verified
the fixed root and RoboTwin commits, all checkpoint hashes, CUDA 12.8, the clean
CuRobo v0.7.8 source, and physical GPU 5.

The first preflight attempt exposed a local layout mismatch: the fixed Hugging
Face paths include `checkpoints/pi05/`, while the collector constructs its
runtime path without that component. A local directory link exposed the same
already-hashed checkpoint files at the collector's expected path. The second
preflight then passed all six cases.

## Formal collection result

The formal runner stopped on the first pair, as required by the Expert-success
hard gate:

```text
blocks_ranking_size seed=200001
Expert frames recorded: 203 (indices 0 through 202)
result: RuntimeError: Expert rollout failed; pair is ineligible
```

The failing staging directory was preserved. Pi0.5 was not run for this pair,
and no final pair manifest was published.

An Expert-only, no-trajectory-write diagnostic then screened all six fixed
seeds under the same task config and runtime:

| Task | Seed | `plan_success` | `check_success` | Result and cause |
|---|---:|---:|---:|---|
| `blocks_ranking_size` | 200001 | false | false | FAIL: the first two blocks complete; the right-arm placement path for the third/large block returns CuRobo `Fail`. Reproduced twice. |
| `blocks_ranking_size` | 200002 | true | true | PASS |
| `hanging_mug` | 200001 | true | true | PASS |
| `hanging_mug` | 200002 | true | false | FAIL: every planned path returns `Success`, but the final hanging-mug task predicate is false. |
| `stamp_seal` | 200003 | false | false | FAIL: the left-arm placement path returns CuRobo `Fail`. |
| `stamp_seal` | 200004 | true | true | PASS |

RoboTwin's ordinary demonstration collection loop skips seeds whose Expert
fails either `plan_success` or `check_success`. Three of the six fixed seeds do
not satisfy that same eligibility requirement in this exact environment.

## Conclusion

The requested six-pair dataset cannot be admitted under the simultaneous fixed
seed and Expert-success requirements. Continuing requires one explicit
contract decision:

1. replace the three failing seeds with screened successful seeds; or
2. change and re-provenance the Expert/CuRobo planning or task-success contract.

No HDF5, MP4, checkpoint, staging data, or other raw output is included in this
report or intended for Git.

## Preserved local evidence

The remote server retains these ignored artifacts:

- `outputs/policy_shift/logs/preflight_console.log`
- `outputs/policy_shift/logs/preflight_console_retry1.log`
- `outputs/policy_shift/logs/collect_console.log`
- `outputs/policy_shift/logs/expert_seed200001_diagnostic.log`
- `outputs/policy_shift/logs/remaining_expert_seed_diagnostics.log`
- `outputs/policy_shift/logs/checkpoint_identity.json`
- `outputs/policy_shift/matched_raw/.staging/blocks_ranking_size__seed200001/`
