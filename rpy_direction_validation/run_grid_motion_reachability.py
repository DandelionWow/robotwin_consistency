#!/usr/bin/env python3
"""Run rpy_grid_motion_safety reachability episodes with fixed seed shards."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOTWIN_ROOT = PROJECT_ROOT / "third_party" / "robotwin"
DEFAULT_TASK_NAME = "rpy_grid_motion_safety"
DEFAULT_TASK_CONFIG = "rpy_grid_motion_reachability_clean"
DEFAULT_EPISODE_NUM = 56
DEFAULT_TORCH_EXTENSIONS_DIR = "/data1/liuwenhao/tmp/torch_extensions"
DEFAULT_TORCH_CUDA_ARCH_LIST = "12.0"
DEFAULT_ROBOTWIN_ENV_BIN = "/data1/liuwenhao/conda/envs/RoboTwin/bin"


def worker_python() -> str:
    python_path = Path(DEFAULT_ROBOTWIN_ENV_BIN) / "python"
    return str(python_path) if python_path.exists() else sys.executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--task-config", default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-num", type=int, default=DEFAULT_EPISODE_NUM)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--denoiser", default=None)
    parser.add_argument("--oidn-library-dir", default=None)
    parser.add_argument("--torch-extensions-dir", default=DEFAULT_TORCH_EXTENSIONS_DIR)
    parser.add_argument("--torch-cuda-arch-list", default=DEFAULT_TORCH_CUDA_ARCH_LIST)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--worker-index", type=int, default=None)
    parser.add_argument("--worker-count", type=int, default=None)
    return parser.parse_args()


def config_save_path(task_name: str, task_config: str) -> Path:
    config_path = ROBOTWIN_ROOT / "task_config" / f"{task_config}.yml"
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    return ROBOTWIN_ROOT / config["save_path"] / task_name / task_config


def episode_range(args: argparse.Namespace) -> tuple[int, int]:
    start = int(args.start)
    end = int(args.end) if args.end is not None else int(args.episode_num)
    if start < 0 or end < start or end > int(args.episode_num):
        raise ValueError(f"invalid episode range: start={start}, end={end}, episode_num={args.episode_num}")
    return start, end


def worker_bounds(start: int, end: int, worker_index: int, worker_count: int) -> tuple[int, int]:
    total = end - start
    chunk = int(math.ceil(total / float(worker_count)))
    shard_start = start + worker_index * chunk
    shard_end = min(end, shard_start + chunk)
    return shard_start, shard_end


def setup_worker_environment(args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", args.torch_extensions_dir)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", args.torch_cuda_arch_list)
    os.environ.setdefault("MPLCONFIGDIR", "/data1/liuwenhao/tmp/mplconfig")
    os.environ.setdefault("XDG_CACHE_HOME", "/data1/liuwenhao/tmp/xdg-cache")
    if Path(DEFAULT_ROBOTWIN_ENV_BIN).exists():
        old_path = os.environ.get("PATH", "")
        if DEFAULT_ROBOTWIN_ENV_BIN not in old_path.split(":"):
            os.environ["PATH"] = f"{DEFAULT_ROBOTWIN_ENV_BIN}:{old_path}" if old_path else DEFAULT_ROBOTWIN_ENV_BIN
    sapien_oidn = "/data1/liuwenhao/conda/envs/RoboTwin/lib/python3.10/site-packages/sapien/oidn_library"
    if Path(sapien_oidn).exists():
        old_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if sapien_oidn not in old_ld_path.split(":"):
            os.environ["LD_LIBRARY_PATH"] = f"{sapien_oidn}:{old_ld_path}" if old_ld_path else sapien_oidn


def load_robotwin_args(task_name: str, task_config: str) -> dict:
    if str(ROBOTWIN_ROOT) not in sys.path:
        sys.path.insert(0, str(ROBOTWIN_ROOT))
    if str(ROBOTWIN_ROOT / "script") not in sys.path:
        sys.path.insert(0, str(ROBOTWIN_ROOT / "script"))

    import collect_data as cd

    config_path = ROBOTWIN_ROOT / "task_config" / f"{task_config}.yml"
    with config_path.open("r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    embodiment_type = args.get("embodiment")
    embodiment_config_path = Path(cd.CONFIGS_PATH) / "_embodiment_config.yml"
    with embodiment_config_path.open("r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def embodiment_file(name: str):
        robot_file = embodiment_types[name]["file_path"]
        if robot_file is None:
            raise RuntimeError(f"missing embodiment file for {name}")
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = embodiment_file(embodiment_type[0])
        args["right_robot_file"] = embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
        embodiment_name = str(embodiment_type[0])
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = embodiment_file(embodiment_type[0])
        args["right_robot_file"] = embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])
    else:
        raise RuntimeError("number of embodiment config parameters should be 1 or 3")

    args["left_embodiment_config"] = cd.get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = cd.get_embodiment_config(args["right_robot_file"])
    args["embodiment_name"] = embodiment_name
    args["task_config"] = task_config
    args["save_path"] = str(Path(args["save_path"]) / task_name / task_config)
    args["reachability_mode"] = True
    return args


def merge_episode_cache(env) -> None:
    cache_value = getattr(env, "folder_path", {}).get("cache")
    if cache_value and Path(cache_value).exists():
        env.merge_pkl_to_hdf5_video()
        env.remove_data_cache()


def run_one_episode(cd, task_name: str, base_args: dict, episode_idx: int, seed: int, resume: bool) -> dict:
    save_path = Path(base_args["save_path"])
    hdf5_path = save_path / "data" / f"episode{episode_idx}.hdf5"
    report_path = save_path / f"motion_safety_report_episode_{episode_idx:06d}.json"
    if resume and hdf5_path.exists() and report_path.exists():
        with report_path.open("r", encoding="utf-8") as f:
            return {"episode_idx": episode_idx, "skipped": True, "info": json.load(f)}

    plan_args = dict(base_args)
    plan_args["need_plan"] = True
    plan_args["save_data"] = False

    env = cd.class_decorator(task_name)
    try:
        env.setup_demo(now_ep_num=episode_idx, seed=seed, **plan_args)
        env.play_once()
        if not (env.plan_success and env.check_success()):
            return {"episode_idx": episode_idx, "skipped": False, "success": False, "phase": "plan", "info": env.info}
        env.save_traj_data(episode_idx)
    finally:
        try:
            env.close_env()
        except Exception:
            pass
        if plan_args.get("render_freq"):
            try:
                env.viewer.close()
            except Exception:
                pass

    data_args = dict(base_args)
    data_args["need_plan"] = False
    data_args["render_freq"] = 0
    data_args["save_data"] = True

    env = cd.class_decorator(task_name)
    info = None
    try:
        env.setup_demo(now_ep_num=episode_idx, seed=seed, **data_args)
        traj_data = env.load_tran_data(episode_idx)
        data_args["left_joint_path"] = traj_data["left_joint_path"]
        data_args["right_joint_path"] = traj_data["right_joint_path"]
        env.set_path_lst(data_args)
        info = env.play_once()
        env.close_env(clear_cache=((episode_idx + 1) % int(data_args["clear_cache_freq"]) == 0))
        merge_episode_cache(env)
        report = info.get("info", info) if isinstance(info, dict) else info
        if isinstance(report, dict):
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if not env.check_success():
            return {"episode_idx": episode_idx, "skipped": False, "success": False, "phase": "collect", "info": info}
        return {"episode_idx": episode_idx, "skipped": False, "success": True, "info": info}
    except Exception as exc:
        merge_episode_cache(env)
        return {"episode_idx": episode_idx, "skipped": False, "success": False, "phase": "collect", "error": repr(exc), "info": info}
    finally:
        try:
            env.close_env()
        except Exception:
            pass


def write_shard_scene_info(save_path: Path, worker_index: int, results: list[dict]) -> None:
    shard_info = {}
    for result in results:
        info = result.get("info")
        if info is not None:
            shard_info[f"episode_{result['episode_idx']}"] = info
    shard_path = save_path / f"scene_info_shard_{worker_index}.json"
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path.write_text(json.dumps(shard_info, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")


def merge_scene_info(save_path: Path, workers: int) -> None:
    merged = {}
    for worker_index in range(workers):
        shard_path = save_path / f"scene_info_shard_{worker_index}.json"
        if not shard_path.exists():
            continue
        with shard_path.open("r", encoding="utf-8") as f:
            merged.update(json.load(f))
    if merged:
        scene_info_path = save_path / "scene_info.json"
        scene_info_path.write_text(json.dumps(dict(sorted(merged.items())), ensure_ascii=False, indent=4) + "\n", encoding="utf-8")


def run_worker(args: argparse.Namespace) -> int:
    setup_worker_environment(args)
    os.chdir(ROBOTWIN_ROOT)
    sys.path.insert(0, str(ROBOTWIN_ROOT / "script"))
    sys.path.insert(0, str(ROBOTWIN_ROOT))

    import collect_data as cd

    cd.prepare_denoiser_runtime(args.denoiser, args.oidn_library_dir)
    cd.import_runtime_dependencies()

    from test_render import Sapien_TEST

    Sapien_TEST()

    base_args = load_robotwin_args(args.task_name, args.task_config)
    save_path = Path(base_args["save_path"])
    start, end = episode_range(args)
    if args.worker_index is not None and args.worker_count is not None:
        start, end = worker_bounds(start, end, args.worker_index, args.worker_count)

    print(
        f"[rpy_grid_motion_reachability] worker={args.worker_index} "
        f"episodes=[{start}, {end}) gpu={args.gpu} seed={args.seed}"
    )

    results = []
    failures = []
    for episode_idx in range(start, end):
        result = run_one_episode(
            cd=cd,
            task_name=args.task_name,
            base_args=base_args,
            episode_idx=episode_idx,
            seed=args.seed,
            resume=not args.no_resume,
        )
        results.append(result)
        if not result.get("skipped") and not result.get("success"):
            failures.append(result)
            print(f"[rpy_grid_motion_reachability] episode={episode_idx} failed: {result}")
        else:
            print(f"[rpy_grid_motion_reachability] episode={episode_idx} done skipped={result.get('skipped')}")
        if not result.get("success") and result.get("phase") == "collect":
            time.sleep(1)

    worker_index = int(args.worker_index) if args.worker_index is not None else 0
    write_shard_scene_info(save_path, worker_index, results)
    summary = {
        "worker_index": worker_index,
        "start": start,
        "end": end,
        "seed": int(args.seed),
        "num_results": len(results),
        "num_failures": len(failures),
        "failures": failures,
    }
    summary_path = save_path / f"shard_summary_{worker_index}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 1 if failures else 0


def run_parent(args: argparse.Namespace) -> int:
    start, end = episode_range(args)
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    save_path = config_save_path(args.task_name, args.task_config)
    save_path.mkdir(parents=True, exist_ok=True)

    processes = []
    for worker_index in range(args.workers):
        command = [
            worker_python(),
            str(Path(__file__).resolve()),
            "--task-name",
            args.task_name,
            "--task-config",
            args.task_config,
            "--gpu",
            str(args.gpu),
            "--workers",
            str(args.workers),
            "--seed",
            str(args.seed),
            "--episode-num",
            str(args.episode_num),
            "--start",
            str(start),
            "--end",
            str(end),
            "--torch-extensions-dir",
            args.torch_extensions_dir,
            "--torch-cuda-arch-list",
            args.torch_cuda_arch_list,
            "--worker-index",
            str(worker_index),
            "--worker-count",
            str(args.workers),
        ]
        if args.denoiser is not None:
            command.extend(["--denoiser", args.denoiser])
        if args.oidn_library_dir is not None:
            command.extend(["--oidn-library-dir", args.oidn_library_dir])
        if args.no_resume:
            command.append("--no-resume")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        env.setdefault("TORCH_EXTENSIONS_DIR", args.torch_extensions_dir)
        env.setdefault("TORCH_CUDA_ARCH_LIST", args.torch_cuda_arch_list)
        if Path(DEFAULT_ROBOTWIN_ENV_BIN).exists():
            old_path = env.get("PATH", "")
            if DEFAULT_ROBOTWIN_ENV_BIN not in old_path.split(":"):
                env["PATH"] = f"{DEFAULT_ROBOTWIN_ENV_BIN}:{old_path}" if old_path else DEFAULT_ROBOTWIN_ENV_BIN
        process = subprocess.Popen(command, cwd=str(PROJECT_ROOT), env=env)
        processes.append(process)

    exit_codes = [process.wait() for process in processes]
    merge_scene_info(save_path, args.workers)
    if any(code != 0 for code in exit_codes):
        print(f"[rpy_grid_motion_reachability] worker exit codes: {exit_codes}")
        return 1
    print(f"[rpy_grid_motion_reachability] complete episodes=[{start}, {end}) save_path={save_path}")
    return 0


def main() -> int:
    args = parse_args()
    if args.worker_index is not None:
        return run_worker(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
