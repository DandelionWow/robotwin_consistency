import importlib
import importlib.util
import json
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import transforms3d as t3d
import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
ROBOTWIN_ROOT = REPO_ROOT / "third_party" / "robotwin"
CACHE_ROOT = REPO_ROOT / "database" / "waypoint_selection_cache"
SUPPORTED_TASKS = tuple(sorted(path.stem for path in (ROBOTWIN_ROOT / "envs").glob("ep2_1_*.py")))
SUPPORTED_TASK = (
    "ep2_1_object_pose_adjust_bottle"
    if "ep2_1_object_pose_adjust_bottle" in SUPPORTED_TASKS
    else SUPPORTED_TASKS[0]
)


def _setup_robotwin_path():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    robotwin_path = str(ROBOTWIN_ROOT)
    if robotwin_path not in sys.path:
        sys.path.insert(0, robotwin_path)


@contextmanager
def _robotwin_cwd():
    old_cwd = Path.cwd()
    os.chdir(ROBOTWIN_ROOT)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def _load_task(task_name):
    _setup_robotwin_path()
    _ensure_curobo_stub()
    envs_module = importlib.import_module(f"envs.{task_name}")
    return getattr(envs_module, task_name)()


def _ensure_curobo_stub():
    planner_module = sys.modules.get("envs.robot.planner")
    if planner_module is None:
        importlib.import_module("envs")
        _install_fake_curobo_modules()
        planner_path = ROBOTWIN_ROOT / "envs" / "robot" / "planner.py"
        spec = importlib.util.spec_from_file_location("envs.robot.planner", planner_path)
        planner_module = importlib.util.module_from_spec(spec)
        sys.modules["envs.robot.planner"] = planner_module
        spec.loader.exec_module(planner_module)

    if hasattr(planner_module, "CuroboPlanner"):
        return

    class CuroboPlanner:
        pass

    planner_module.CuroboPlanner = CuroboPlanner


def _install_fake_curobo_modules():
    class _Dummy:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return self

        def __getattr__(self, name):
            return self

        def __iter__(self):
            return iter(())

        @classmethod
        def from_list(cls, *args, **kwargs):
            return cls()

        @classmethod
        def load_from_robot_config(cls, *args, **kwargs):
            return cls()

        @classmethod
        def from_position(cls, *args, **kwargs):
            return cls()

        def warmup(self, *args, **kwargs):
            return None

        def setup_logger(self, *args, **kwargs):
            return None

    module_specs = {
        "curobo": {},
        "curobo.types": {},
        "curobo.types.math": {"Pose": _Dummy},
        "curobo.types.robot": {"JointState": _Dummy},
        "curobo.wrap": {},
        "curobo.wrap.reacher": {},
        "curobo.wrap.reacher.motion_gen": {
            "MotionGen": _Dummy,
            "MotionGenConfig": _Dummy,
            "MotionGenPlanConfig": _Dummy,
            "PoseCostMetric": _Dummy,
        },
        "curobo.util": {},
        "curobo.util.logger": {"setup_logger": _Dummy().setup_logger},
    }

    for name, attrs in module_specs.items():
        module = types.ModuleType(name)
        for attr, value in attrs.items():
            setattr(module, attr, value)
        sys.modules[name] = module

    sys.modules["curobo.util"].logger = sys.modules["curobo.util.logger"]


def _get_embodiment_config(robot_file):
    return _load_yaml(Path(robot_file) / "config.yml")


def load_task_args(task_config, task_name):
    config_path = ROBOTWIN_ROOT / "task_config" / f"{task_config}.yml"
    args = _load_yaml(config_path)
    args["task_name"] = task_name
    args["task_config"] = task_config
    args["render_freq"] = 0
    args["need_plan"] = False
    args["save_data"] = False
    args["save_path"] = str(CACHE_ROOT / task_name / task_config)

    embodiment_types = _load_yaml(ROBOTWIN_ROOT / "task_config" / "_embodiment_config.yml")
    embodiment = args.get("embodiment")

    def get_robot_file(name):
        return embodiment_types[name]["file_path"]

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

    args["left_embodiment_config"] = _get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = _get_embodiment_config(args["right_robot_file"])
    return args


def cache_dir(task_config, task_name, seed):
    return CACHE_ROOT / task_name / task_config / f"seed_{int(seed)}"


def list_task_configs():
    config_dir = ROBOTWIN_ROOT / "task_config"
    return sorted(path.stem for path in config_dir.glob("*.yml") if not path.stem.startswith("_"))


def _to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(val) for val in value]
    return value


def _project_point(point_world, intrinsic, extrinsic):
    point_h = np.array([point_world[0], point_world[1], point_world[2], 1.0], dtype=float)
    point_cam = extrinsic @ point_h
    depth = float(point_cam[2])
    if depth <= 1e-6:
        return None

    uvw = intrinsic @ point_cam[:3]
    return [float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2])]


def _project_axes(matrix_world, intrinsic, extrinsic, image_width, image_height, axis_length):
    origin_3d = matrix_world[:3, 3]
    rotation = matrix_world[:3, :3]
    axes_3d = {
        "origin": origin_3d,
        "x": origin_3d + rotation[:, 0] * axis_length,
        "y": origin_3d + rotation[:, 1] * axis_length,
        "z": origin_3d + rotation[:, 2] * axis_length,
    }
    axes_2d = {name: _project_point(point, intrinsic, extrinsic) for name, point in axes_3d.items()}
    valid = all(point is not None for point in axes_2d.values())
    in_image = False
    if valid:
        in_image = any(0 <= point[0] < image_width and 0 <= point[1] < image_height for point in axes_2d.values())
    return axes_2d, valid, in_image


def _capture_camera(env, camera_name):
    env._update_render()
    env.cameras.update_picture()
    rgb = env.cameras.get_rgb()
    config = env.cameras.get_config()
    if camera_name not in rgb:
        camera_name = next(iter(rgb.keys()))
    image = rgb[camera_name]["rgb"]
    camera_config = config[camera_name]
    return image, camera_name, camera_config


def _add_projected_axes(target, matrix_key, axes_key, intrinsic, extrinsic, image_width, image_height, axis_length):
    matrix_world = np.asarray(target.pop(matrix_key), dtype=float)
    axes_2d, valid, in_image = _project_axes(
        matrix_world,
        intrinsic,
        extrinsic,
        image_width,
        image_height,
        axis_length,
    )
    target[matrix_key] = matrix_world.tolist()
    target[axes_key] = axes_2d
    target[f"{axes_key}_valid"] = valid
    target[f"{axes_key}_in_image"] = in_image


def export_scene(
    task_config,
    task_name,
    seed,
    refresh=False,
    camera_name="head_camera",
    axis_length=0.04,
):
    if task_name not in SUPPORTED_TASKS:
        raise ValueError(f"Only these tasks are supported now: {', '.join(SUPPORTED_TASKS)}")

    output_dir = cache_dir(task_config, task_name, seed)
    frame_path = output_dir / "frame.png"
    scene_path = output_dir / "scene.json"
    if not refresh and frame_path.exists() and scene_path.exists():
        with open(scene_path, "r", encoding="utf-8") as f:
            result = json.load(f)
        result["cache_hit"] = True
        result["refreshed"] = False
        return result

    output_dir.mkdir(parents=True, exist_ok=True)
    task_env = None
    with _robotwin_cwd():
        task_env = _load_task(task_name)
        args = load_task_args(task_config, task_name)

        try:
            task_env.setup_demo(now_ep_num=0, seed=int(seed), **args)
            image, used_camera, camera_config = _capture_camera(task_env, camera_name)
            Image.fromarray(image).save(frame_path)

            intrinsic = np.asarray(camera_config["intrinsic_cv"], dtype=float)
            extrinsic = np.asarray(camera_config["extrinsic_cv"], dtype=float)
            image_height, image_width = image.shape[:2]

            scene_info = task_env.get_waypoint_selection_scene_info()
            for obj in scene_info["objects"]:
                for point in obj["contact_points"]:
                    _add_projected_axes(
                        point,
                        "matrix_world",
                        "axes_2d",
                        intrinsic,
                        extrinsic,
                        image_width,
                        image_height,
                        axis_length,
                    )
                    point["projection_valid"] = point["axes_2d_valid"]
                    point["in_image"] = point["axes_2d_in_image"]

            result = {
                "task_config": task_config,
                "task_name": task_name,
                "seed": int(seed),
                "cache_hit": False,
                "refreshed": bool(refresh),
                "image": "frame.png",
                "image_width": int(image_width),
                "image_height": int(image_height),
                "axis_length": float(axis_length),
                "camera": {
                    "name": used_camera,
                    "intrinsic": intrinsic.tolist(),
                    "extrinsic": extrinsic.tolist(),
                    "cam2world_gl": np.asarray(camera_config["cam2world_gl"]).tolist(),
                },
                "environment": _get_environment_info(task_env),
                "objects": scene_info["objects"],
            }
            result = _to_jsonable(result)
            with open(scene_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            return result
        finally:
            if task_env is not None:
                try:
                    task_env.close_env(clear_cache=False)
                except Exception:
                    pass


def _get_environment_info(task_env):
    table_z = 0.74 + float(getattr(task_env, "table_z_bias", 0.0))
    table_xy = getattr(task_env, "table_xy_bias", [0, 0])
    return {
        "table": {
            "pose": [float(table_xy[0]), float(table_xy[1]), table_z],
            "length": 1.2,
            "width": 0.7,
            "height": table_z,
            "thickness": 0.05,
            "color": [1.0, 1.0, 1.0],
            "texture": getattr(task_env, "table_texture", None),
        },
        "wall": {
            "pose": [0.0, 1.0, 1.5],
            "size": [6.0, 1.2, 3.0],
            "color": [1.0, 0.9, 0.9],
            "texture": getattr(task_env, "wall_texture", None),
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("task_config")
    parser.add_argument("--task-name", default=SUPPORTED_TASK)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    data = export_scene(
        args.task_config,
        args.task_name,
        args.seed,
        refresh=args.refresh,
    )
    print(json.dumps(data, ensure_ascii=False, indent=2))
