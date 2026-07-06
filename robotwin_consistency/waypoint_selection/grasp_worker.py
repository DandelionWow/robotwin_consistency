import contextlib
import inspect
import importlib
import json
import os
import sys
import traceback
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOTWIN_ROOT = REPO_ROOT / "third_party" / "robotwin"
CACHE_ROOT = REPO_ROOT / "database" / "waypoint_selection_cache"
SUPPORTED_TASKS = tuple(sorted(path.stem for path in (ROBOTWIN_ROOT / "envs").glob("ep2_1_*.py")))
SUPPORTED_TASK = (
    "ep2_1_object_pose_adjust_bottle"
    if "ep2_1_object_pose_adjust_bottle" in SUPPORTED_TASKS
    else SUPPORTED_TASKS[0]
)


def main():
    original_stdout = sys.stdout
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        with contextlib.redirect_stdout(sys.stderr):
            items = compute_perturbed_grasps(payload)
        print(json.dumps({"ok": True, "items": _to_jsonable(items)}, ensure_ascii=False), file=original_stdout)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=original_stdout)
        sys.exit(1)


def compute_perturbed_grasps(payload):
    task_config = payload.get("task_config", "ep2_1_object_pose")
    task_name = payload.get("task_name", SUPPORTED_TASK)
    if task_name not in SUPPORTED_TASKS:
        raise ValueError(f"Only these tasks are supported now: {', '.join(SUPPORTED_TASKS)}")

    old_cwd = Path.cwd()
    task_env = None
    os.chdir(ROBOTWIN_ROOT)
    try:
        _setup_robotwin_path()
        task_env = _load_task(task_name)
        args = _load_task_args(task_config, task_name)
        task_env.setup_demo(now_ep_num=0, seed=int(payload.get("seed", 0)), **args)
        pre_grasp_dis = _payload_float(payload, "pre_grasp_dis", 0.1)
        grasp_dis = _payload_float(payload, "grasp_dis", 0.0)
        if pre_grasp_dis < grasp_dis:
            raise ValueError("pre_grasp_dis must be greater than or equal to grasp_dis")
        method = task_env.compute_waypoint_perturbed_grasps
        point_ids = [int(point_id) for point_id in payload.get("selected_point_ids", [])]
        perturbation = payload.get("perturbation", {})
        if "grasp_dis" in inspect.signature(method).parameters:
            return method(point_ids, perturbation, pre_grasp_dis, grasp_dis)
        return method(point_ids, perturbation, pre_grasp_dis - grasp_dis)
    finally:
        if task_env is not None:
            try:
                task_env.close_env(clear_cache=False)
            except Exception:
                pass
        os.chdir(old_cwd)


def _setup_robotwin_path():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    robotwin_path = str(ROBOTWIN_ROOT)
    if robotwin_path not in sys.path:
        sys.path.insert(0, robotwin_path)


def _load_task(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    return getattr(envs_module, task_name)()


def _load_task_args(task_config, task_name):
    args = _load_yaml(ROBOTWIN_ROOT / "task_config" / f"{task_config}.yml")
    args["task_name"] = task_name
    args["task_config"] = task_config
    args["render_freq"] = 0
    args["need_plan"] = True
    args["save_data"] = False
    args["save_path"] = str(CACHE_ROOT / task_name / task_config)

    embodiment_types = _load_yaml(ROBOTWIN_ROOT / "task_config" / "_embodiment_config.yml")
    embodiment = args.get("embodiment")

    def get_robot_file(name):
        robot_file = embodiment_types[name]["file_path"]
        if robot_file is None:
            raise ValueError(f"missing embodiment file for {name}")
        return robot_file

    if len(embodiment) == 1:
        args["left_robot_file"] = get_robot_file(embodiment[0])
        args["right_robot_file"] = get_robot_file(embodiment[0])
        args["dual_arm_embodied"] = True
        args["embodiment_name"] = str(embodiment[0])
    elif len(embodiment) == 3:
        args["left_robot_file"] = get_robot_file(embodiment[0])
        args["right_robot_file"] = get_robot_file(embodiment[1])
        args["embodiment_dis"] = embodiment[2]
        args["dual_arm_embodied"] = False
        args["embodiment_name"] = f"{embodiment[0]}+{embodiment[1]}"
    else:
        raise ValueError("embodiment must contain 1 or 3 entries")

    args["left_embodiment_config"] = _load_yaml(Path(args["left_robot_file"]) / "config.yml")
    args["right_embodiment_config"] = _load_yaml(Path(args["right_robot_file"]) / "config.yml")
    return args


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def _payload_float(payload, key, default):
    value = payload.get(key, default)
    if value is None or value == "":
        value = default
    return float(value)


def _to_jsonable(value):
    try:
        import numpy as np
    except Exception:
        np = None
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(val) for val in value]
    return value


if __name__ == "__main__":
    main()
