#!/usr/bin/env python3
"""Screen Expert seeds in isolated processes using the formal sampling clock."""

from __future__ import annotations

import argparse
import json
import os
import re
import site
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from collect_matched_pair import (
    PROJECT_ROOT,
    ROBOTWIN_ROOT,
    _configure_task,
    _git_commit,
    _installed_editable_source,
    _resolve_physical_gpu,
    _setup_and_capture,
)
from protocol import require_uniform_timestamp_grid, sha256_file, state_fingerprint


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _runtime_preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.gpu_id not in {0, 1, 2, 3, 4, 5}:
        raise ValueError("--gpu-id must be one of physical GPUs 0,1,2,3,4,5")
    if site.ENABLE_USER_SITE or os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("Run with PYTHONNOUSERSITE=1")
    gpu_identity = getattr(args, "gpu_identity", None)
    if not isinstance(gpu_identity, dict) or gpu_identity.get("physical_gpu") != args.gpu_id:
        raise RuntimeError("Physical GPU identity must be resolved before preflight")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("CUDA_DEVICE_ORDER must be PCI_BUS_ID")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_identity["physical_gpu_uuid"]:
        raise RuntimeError("CUDA_VISIBLE_DEVICES does not match the resolved GPU UUID")
    if os.environ.get("HWLOC_ALLOW") != "all":
        raise RuntimeError("HWLOC_ALLOW must be 'all'")
    ptxas = args.cuda_root.resolve() / "bin/ptxas"
    if not ptxas.is_file():
        raise FileNotFoundError(f"Missing ptxas: {ptxas}")
    ptxas_output = subprocess.run(
        [str(ptxas), "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    version_match = re.search(r"release\s+(\d+)\.(\d+)", ptxas_output)
    if not version_match or tuple(map(int, version_match.groups())) < (12, 8):
        raise RuntimeError(f"SM120 requires CUDA toolkit >=12.8; got: {ptxas_output}")

    curobo_source = args.curobo_source.resolve()
    if not (curobo_source / ".git").exists():
        raise FileNotFoundError(f"CuRobo source is not a Git checkout: {curobo_source}")
    status = subprocess.run(
        ["git", "-C", str(curobo_source), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("CuRobo source checkout must be clean")
    curobo_version, installed_source = _installed_editable_source("nvidia-curobo")
    if installed_source != curobo_source:
        raise RuntimeError(
            "Installed nvidia-curobo source mismatch: "
            f"installed={installed_source} declared={curobo_source}"
        )
    return {
        "python": sys.executable,
        "physical_gpu": args.gpu_id,
        "physical_gpu_uuid": gpu_identity["physical_gpu_uuid"],
        "physical_gpu_name": gpu_identity["physical_gpu_name"],
        "physical_gpu_pci_bus_id": gpu_identity["physical_gpu_pci_bus_id"],
        "cuda_device_order": os.environ["CUDA_DEVICE_ORDER"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "hwloc_allow": os.environ["HWLOC_ALLOW"],
        "cuda_root": str(args.cuda_root.resolve()),
        "ptxas_version": ptxas_output.splitlines()[-1],
        "curobo_version": curobo_version,
        "curobo_source": str(curobo_source),
        "curobo_source_commit": _git_commit(curobo_source),
        "root_commit": _git_commit(PROJECT_ROOT),
        "robotwin_commit": _git_commit(ROBOTWIN_ROOT),
    }


def _screen_once(args: argparse.Namespace) -> dict[str, Any]:
    runtime = _runtime_preflight(args)
    sys.path.insert(0, str(ROBOTWIN_ROOT))
    sys.path.insert(0, str(ROBOTWIN_ROOT / "policy"))
    sys.path.insert(0, str(ROBOTWIN_ROOT / "description/utils"))
    previous_cwd = Path.cwd()
    task = None
    os.chdir(ROBOTWIN_ROOT)
    try:
        from script import collect_data as robotwin_collect

        robotwin_collect.prepare_denoiser_runtime(args.denoiser, args.oidn_library_dir)
        robotwin_collect.import_runtime_dependencies()
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(
                "Expert screen must see exactly one Torch CUDA device; "
                f"available={torch.cuda.is_available()} count={torch.cuda.device_count()}"
            )
        torch_device_name = torch.cuda.get_device_name(0)
        if torch_device_name != runtime["physical_gpu_name"]:
            raise RuntimeError(
                "Resolved physical GPU identity differs from the visible Torch device: "
                f"expected={runtime['physical_gpu_name']!r} actual={torch_device_name!r}"
            )
        runtime.update(
            {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "torch_device_name": torch_device_name,
                "torch_device_capability": list(torch.cuda.get_device_capability(0)),
            }
        )

        task_factory, config, _ = _configure_task(args.task, args.task_config.resolve())
        task = task_factory(args.task)
        with tempfile.TemporaryDirectory(prefix="expert-seed-screen-") as temporary:
            state, _, proxy = _setup_and_capture(
                task,
                config,
                Path(temporary),
                args.seed,
                args.physics_timestep,
                args.physics_steps_per_sample,
            )
            task.play_once()
            plan_success = bool(task.plan_success)
            check_success = bool(task.check_success())
            timestamp_grid = require_uniform_timestamp_grid(
                proxy.timestamps,
                args.physics_timestep * args.physics_steps_per_sample,
            )
            frames = int(timestamp_grid["sample_count"])
            success = plan_success and check_success
            return {
                "schema_version": 1,
                "status": "PASS" if success and frames >= args.minimum_frames else "FAIL",
                "task": args.task,
                "seed": args.seed,
                "plan_success": plan_success,
                "check_success": check_success,
                "expert_success": success,
                "physics_steps": proxy.physics_steps,
                "regular_grid_frames": frames,
                "minimum_required_frames": args.minimum_frames,
                "eligible": success and frames >= args.minimum_frames,
                "timestamp_grid": timestamp_grid,
                "state_fingerprint": state_fingerprint(state),
                "task_config": str(args.task_config.resolve()),
                "task_config_sha256": sha256_file(args.task_config.resolve()),
                "denoiser": args.denoiser,
                "runtime": runtime,
            }
    finally:
        if task is not None:
            task.close_env()
        os.chdir(previous_cwd)


def _child_command(args: argparse.Namespace, seed: int, result_path: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-seed",
        "--task",
        args.task,
        "--seed",
        str(seed),
        "--task-config",
        str(args.task_config.resolve()),
        "--gpu-id",
        str(args.gpu_id),
        "--cuda-root",
        str(args.cuda_root.resolve()),
        "--curobo-source",
        str(args.curobo_source.resolve()),
        "--physics-timestep",
        str(args.physics_timestep),
        "--physics-steps-per-sample",
        str(args.physics_steps_per_sample),
        "--minimum-frames",
        str(args.minimum_frames),
        "--denoiser",
        args.denoiser,
        "--result-json",
        str(result_path),
    ]
    if args.oidn_library_dir is not None:
        command.extend(["--oidn-library-dir", str(args.oidn_library_dir.resolve())])
    return command


def _screen_candidates(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = args.seeds or list(
        range(args.start_seed, args.start_seed + args.candidate_count)
    )
    records: list[dict[str, Any]] = []
    accepted: list[int] = []
    for seed in candidates:
        attempts = []
        for repeat in range(args.repeats):
            result_path = output_dir / f"{args.task}__seed{seed}__attempt{repeat + 1}.json"
            log_path = output_dir / f"{args.task}__seed{seed}__attempt{repeat + 1}.log"
            environment = os.environ.copy()
            gpu_identity = _resolve_physical_gpu(args.gpu_id)
            environment["HWLOC_ALLOW"] = "all"
            environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            environment["CUDA_VISIBLE_DEVICES"] = gpu_identity["physical_gpu_uuid"]
            environment["CUDA_ROOT"] = str(args.cuda_root.resolve())
            environment["PYTHONNOUSERSITE"] = "1"
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    _child_command(args, seed, result_path),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=environment,
                )
            if not result_path.is_file():
                attempts.append(
                    {
                        "status": "PROCESS_ERROR",
                        "returncode": completed.returncode,
                        "result_json": str(result_path),
                        "log": str(log_path),
                    }
                )
                break
            attempt = json.loads(result_path.read_text(encoding="utf-8"))
            attempt["process_returncode"] = completed.returncode
            attempt["result_json"] = str(result_path)
            attempt["log"] = str(log_path)
            attempts.append(attempt)
            if not attempt.get("eligible", False):
                break
        eligible = len(attempts) == args.repeats and all(
            attempt.get("eligible", False) for attempt in attempts
        )
        records.append({"seed": seed, "eligible": eligible, "attempts": attempts})
        if eligible:
            accepted.append(seed)
        _write_json(
            output_dir / "screen_summary.partial.json",
            {"task": args.task, "records": records, "accepted_seeds": accepted},
        )
        if eligible:
            if len(accepted) >= args.needed:
                break
    summary = {
        "schema_version": 1,
        "status": "COMPLETE" if len(accepted) >= args.needed else "INSUFFICIENT_PASSING_SEEDS",
        "selection_rule": (
            "Ascending Expert-only candidates; accept only seeds passing plan_success, "
            "check_success, exact timestamp grid, and minimum frame count in every "
            "independent process."
        ),
        "task": args.task,
        "minimum_required_frames": args.minimum_frames,
        "repeats_per_candidate": args.repeats,
        "needed": args.needed,
        "accepted_seeds": accepted,
        "records": records,
    }
    _write_json(output_dir / "screen_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="stamp_seal")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--start-seed", type=int, default=200006)
    parser.add_argument("--candidate-count", type=int, default=20)
    parser.add_argument("--needed", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--single-seed", action="store_true")
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, choices=(0, 1, 2, 3, 4, 5), default=5)
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--curobo-source", type=Path, required=True)
    parser.add_argument("--physics-timestep", type=float, default=0.004)
    parser.add_argument("--physics-steps-per-sample", type=int, default=25)
    parser.add_argument("--minimum-frames", type=int, default=81)
    parser.add_argument("--denoiser", choices=("oidn", "optix", "none"), default="optix")
    parser.add_argument("--oidn-library-dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/policy_shift/expert_seed_screen",
    )
    parser.add_argument("--result-json", type=Path)
    args = parser.parse_args()
    if args.single_seed and args.seed is None:
        parser.error("--single-seed requires --seed")
    if min(
        args.candidate_count,
        args.needed,
        args.repeats,
        args.physics_steps_per_sample,
        args.minimum_frames,
    ) <= 0 or args.physics_timestep <= 0:
        parser.error("counts and cadence values must be positive")
    return args


def main() -> int:
    args = parse_args()
    if args.single_seed:
        args.gpu_identity = _resolve_physical_gpu(args.gpu_id)
        os.environ["HWLOC_ALLOW"] = "all"
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_identity["physical_gpu_uuid"]
        os.environ["CUDA_ROOT"] = str(args.cuda_root.resolve())
        result = _screen_once(args)
        if args.result_json is not None:
            _write_json(args.result_json.resolve(), result)
    else:
        result = _screen_candidates(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"PASS", "COMPLETE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
