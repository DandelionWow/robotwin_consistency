#!/usr/bin/env python3
"""Convert RoboTwin output, then run BWM inference in one process per GPU."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE = PROJECT_ROOT / "robotwin_to_bwm.py"
DEFAULT_ROBOTWIN_DIR = PROJECT_ROOT / "third_party" / "robotwin" / "data" / "rpy_grid_rpy_rotation" / "rpy_grid_rpy_rotation_clean"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "robotwin_bwm" / "rpy_grid_rpy_rotation_clean"
DEFAULT_BWM_ROOT = PROJECT_ROOT / "third_party" / "boundless-world-model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin_dir", type=Path, default=DEFAULT_ROBOTWIN_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gpus", default="0,1", help="Comma-separated GPU ids. One BWM process is started per GPU.")
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--bwm_root", type=Path, default=DEFAULT_BWM_ROOT)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--compare_width", type=int, default=480)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_history_frames", type=int, default=9)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--mixed_precision", default="bf16")
    parser.add_argument("--episode_num", type=int, default=1728)
    parser.add_argument("--arm_filter", choices=["all", "left", "right", "both"], default="all")
    parser.add_argument("--overwrite_convert", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_convert", action="store_true")
    parser.add_argument("--skip_compare", action="store_true")
    parser.add_argument("--crop_to_experiment", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def printable(command: list[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in command)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def parse_gpus(value: str) -> list[str]:
    gpus = [item.strip() for item in value.split(",") if item.strip()]
    if not gpus:
        raise SystemExit("--gpus must contain at least one GPU id")
    return gpus


def split_ranges(total: int, shards: int) -> list[tuple[int, int]]:
    if total <= 0:
        raise SystemExit("No manifest rows found after conversion")
    chunk = int(math.ceil(total / float(shards)))
    ranges = []
    for shard_idx in range(shards):
        start = min(total, shard_idx * chunk)
        end = min(total, start + chunk)
        if start < end:
            ranges.append((start, end))
    return ranges


def common_pipeline_args(args: argparse.Namespace) -> list[str]:
    return [
        "--output_dir",
        str(args.output_dir),
        "--bwm_root",
        str(args.bwm_root),
        "--fps",
        str(args.fps),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--num_frames",
        str(args.num_frames),
        "--num_history_frames",
        str(args.num_history_frames),
        "--num_inference_steps",
        str(args.num_inference_steps),
        "--quality",
        str(args.quality),
        "--mixed_precision",
        args.mixed_precision,
        "--episode_num",
        str(args.episode_num),
        "--arm_filter",
        args.arm_filter,
    ]


def run_command(command: list[str], env: dict[str, str] | None = None, dry_run: bool = False) -> None:
    print(printable(command))
    if dry_run:
        return
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def run_convert(args: argparse.Namespace) -> None:
    command = [
        args.python_bin,
        str(PIPELINE),
        "--stage",
        "convert",
        "--robotwin_dir",
        str(args.robotwin_dir),
        *common_pipeline_args(args),
    ]
    if args.crop_to_experiment:
        command.append("--crop_to_experiment")
    if args.overwrite_convert:
        command.append("--overwrite")
    if args.dry_run:
        command.append("--dry_run")
    run_command(command, dry_run=args.dry_run)


def launch_infer_shards(args: argparse.Namespace, total_rows: int, gpus: list[str]) -> int:
    ranges = split_ranges(total_rows, len(gpus))
    processes = []
    for shard_idx, (gpu, (start, end)) in enumerate(zip(gpus, ranges)):
        log_name = f"infer_gpu{gpu}_shard{shard_idx}.log"
        command = [
            args.python_bin,
            str(PIPELINE),
            "--stage",
            "infer",
            *common_pipeline_args(args),
            "--infer_start_index",
            str(start),
            "--infer_max_samples",
            str(end - start),
            "--infer_log_name",
            log_name,
        ]
        if args.resume:
            command.append("--resume")
        if args.dry_run:
            command.append("--dry_run")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        print(f"[bwm_shard] gpu={gpu} rows=[{start}, {end}) log={args.output_dir / 'logs' / log_name}")
        print(printable(command))
        if args.dry_run:
            continue
        processes.append(subprocess.Popen(command, cwd=PROJECT_ROOT, env=env))

    if args.dry_run:
        return 0

    exit_codes = [process.wait() for process in processes]
    if any(code != 0 for code in exit_codes):
        print(f"[bwm_shard] infer exit codes: {exit_codes}")
        return 1
    return 0


def run_compare(args: argparse.Namespace) -> None:
    command = [
        args.python_bin,
        str(PIPELINE),
        "--stage",
        "compare",
        *common_pipeline_args(args),
        "--compare_width",
        str(args.compare_width),
    ]
    if args.resume:
        command.append("--resume")
    if args.dry_run:
        command.append("--dry_run")
    run_command(command, dry_run=args.dry_run)


def main() -> int:
    args = parse_args()
    gpus = parse_gpus(args.gpus)

    if not args.skip_convert:
        run_convert(args)

    manifest_path = args.output_dir / "manifest.jsonl"
    if args.dry_run and not manifest_path.exists():
        print(f"[dry-run] cannot read missing manifest yet: {manifest_path}")
        print(f"[dry-run] rerun without --dry_run after conversion, or run once with an existing manifest.")
        return 0

    rows = read_jsonl(manifest_path)
    print(f"[bwm_shard] manifest_rows={len(rows)} gpus={','.join(gpus)} output_dir={args.output_dir}")

    infer_code = launch_infer_shards(args, len(rows), gpus)
    if infer_code != 0:
        return infer_code

    if not args.skip_compare:
        run_compare(args)

    print("[bwm_shard] complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
