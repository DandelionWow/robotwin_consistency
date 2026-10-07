# Remote RoboTwin matched-collection handoff

Status date: 2026-10-07

## Exact assignment

Use a server whose SAPIEN/Vulkan renderer already works to create the formal
Experiment 1 smoke dataset:

```text
3 tasks x 2 environment seeds x {expert, pi05}
= 6 matched pairs
= 12 trajectories
```

The committed formal pairs are:

| Task | Seeds |
|---|---|
| `blocks_ranking_size` | `200002`, `200003` |
| `hanging_mug` | `200001`, `200003` |
| `stamp_seal` | `200004`, `200005` |

The machine-readable plan is
`experiments/policy_shift/configs/matched_smoke_seed_plan.json`. Every collector
invocation verifies that its task/seed is authorized by this plan and records
the plan SHA256 in the pair manifest.

Remote Expert-only screening established that `blocks_ranking_size/200001`,
`hanging_mug/200002`, and `stamp_seal/200003` fail the unchanged Expert gate in
the fixed remote runtime. They are not formal pairs. Keep the three passing
seeds (`blocks_ranking_size/200002`, `hanging_mug/200001`, and
`stamp_seal/200004`). Replacement screening then used Expert only, proceeded
in ascending seed order, and accepted the first seed for which both
`plan_success` and `check_success()` passed in two independent Python
processes. Pi0.5 was not loaded during selection. The complete record is in
`experiments/policy_shift/provenance/remote_expert_seed_screen_gpu5.md`.

Do not collect Expert and Pi0.5 separately. The checked-in collector creates
both sides sequentially in one process, resets the identical task seed, and
hard-checks the canonical initial-state fingerprint before admitting the pair.

## Code identities

Clone or update:

```text
https://github.com/DandelionWow/robotwin_consistency.git
branch: dev/liuwenhao
```

Initialize all registered submodules recursively. The required RoboTwin
identity is:

```text
remote branch: exp/matched-pi05-collection-b90
commit: ce63ccb13e7b3e891ed6209b9b374c07e73c9311
```

The root repository gitlink fixes this commit. Do not substitute another
RoboTwin checkout and do not apply the old `provenance/robotwin.patch`; the
required Pi0.5 frozen-dataclass fix is already committed cleanly.

Before collection, leave all tracked files clean. Checkpoint files and output
data must not be committed to Git.

## Required Pi0.5 checkpoints

Use only the following Hugging Face source at the fixed revision:

```text
repo: JackieMM/RoboTwin-pi05-30000-checkpoints
revision: 4c5a1c2be00d4019b649d72dc81fe1bbf9d999a4
```

The Hugging Face snapshot stores the files below
`checkpoints/pi05/pi05_base_aloha_lora/`. RoboTwin's runtime path intentionally
omits that repository-layout-only `pi05` component. Copy the directories or
create stable directory links so that every inference parameter file and
`assets/<asset_id>/norm_stats.json` is visible at:

```text
third_party/robotwin/policy/pi05/checkpoints/
  pi05_base_aloha_lora/
    blocks_ranking_size_demo_clean_pi05_lora_30000_wandb/30000/
    hanging_mug_demo_clean_pi05_lora_30000_wandb/30000/
    stamp_seal_demo_clean_pi05_lora_30000_wandb/30000/
```

Do not trust directory names alone. The collector computes a complete canonical
manifest over the inference checkpoint and requires these SHA256 identities:

```text
blocks_ranking_size  17a388a5cabd65fe3d7d6564dd6f828ebc017890e44d7ed61d003fa13d479015
hanging_mug          357547e06169c782f2e974aba4d48d2f38af91526fa1b8423017ce483b090951
stamp_seal           1aafcbd5d54fc6c3e83ae21b6a0b4a62b295d675eb89d60696260c95b2cb9349
```

The training-resume-only `train_state/` subtree is not required by inference.

## Runtime gates

The remote Codex must first inspect and report the existing RoboTwin runtime;
reuse a known-working environment when possible. It must verify:

```bash
vulkaninfo --summary
nvidia-smi
```

Then establish one Python environment in which all of the following import and
run together:

- the checked-in RoboTwin and SAPIEN stack;
- the checked-in Pi0.5/OpenPI stack;
- JAX GPU inference;
- a CUDA-enabled Torch build appropriate for that server's GPU;
- CuRobo `0.7.8`, installed editable from an explicit clean Git checkout;
- `setuptools==69.5.1` if this SAPIEN build requires `pkg_resources`;
- `h5py`, OpenCV, PyYAML, and `ffmpeg`.

The current collector requires a CUDA toolkit with `bin/ptxas` version 12.8 or
newer. Physical GPU indices 0 through 3 remain the default contract. The user
explicitly authorized physical GPU 5 as an exception on the remote collection
server; the manifest records the actual physical index.

Do not reinstall a working system driver or rebuild the remote environment
blindly. If its GPU architecture differs from the source server, report the
GPU, driver, CUDA, Torch, JAX, SAPIEN, and CuRobo versions before changing
anything.

## Run procedure

From the root of the clean `robotwin_consistency` checkout, identify:

```text
COLLECTION_PYTHON  absolute Python executable for the joint runtime
CUDA_ROOT          CUDA toolkit root containing bin/ptxas
CUROBO_SOURCE      clean CuRobo v0.7.8 Git checkout used by the editable install
GPU_ID             one free physical GPU among 0--3, or remote exception 5
```

First run only the six preflights:

```bash
COLLECTION_PYTHON=/absolute/path/to/python \
CUDA_ROOT=/absolute/path/to/cuda-12.8 \
CUROBO_SOURCE=/absolute/path/to/clean/curobo-v0.7.8 \
GPU_ID=5 \
bash experiments/policy_shift/run_remote_matched_collection.sh preflight
```

Do not proceed unless every invocation prints `PREFLIGHT_OK`. Then run:

```bash
COLLECTION_PYTHON=/absolute/path/to/python \
CUDA_ROOT=/absolute/path/to/cuda-12.8 \
CUROBO_SOURCE=/absolute/path/to/clean/curobo-v0.7.8 \
GPU_ID=5 \
bash experiments/policy_shift/run_remote_matched_collection.sh collect
```

The second command reuses the code-provenance record created by the preflight;
the collector rejects it if either the root or RoboTwin commit changed between
the two commands.

The default denoiser is `optix`. If the known-good remote runtime uses another
backend, set `DENOISER=oidn` or `DENOISER=none` and record the choice. Changing
the denoiser does not relax any state, cadence, provenance, or checkpoint gate.

The collector refuses to overwrite an existing final or staging pair. On a
failure, inspect and preserve the failing log and `.staging/<pair_id>` rather
than deleting it automatically.

## Acceptance and return

The expected directory is:

```text
outputs/policy_shift/matched_raw/
  <six pair directories>/
  SHA256SUMS
```

Each pair must contain a manifest, both HDF5 trajectories, both MP4 files, both
initial-state records, and the raw Pi0.5 command archive. The pair manifest must
say `STATE_MATCH`; Expert success is a hard collection gate. Preserve the
Pi0.5 success/failure boolean exactly as observed.

Do not push raw data to GitHub. Return the entire `matched_raw` directory by a
binary-safe transfer method such as `rsync -a`, then verify from inside it:

```bash
sha256sum -c SHA256SUMS
```

Also return the complete console logs and the generated
`experiments/policy_shift/provenance/code_provenance.json`. Do not run BWM or
WorldArena on the remote server; those stages continue on the source server
after the raw dataset passes its audit.
