#!/usr/bin/env python3
# 本脚本没有用于最终的空桌面立方体实验数据生成.
# 最终实验目录为 outputs/robotwin/cube_center_x0p00_yneg0p15_z1p00_edge*_axis_*_arm_{left,right}.
# 这个脚本用于后续带原始任务物体的实验变体.
"""Run cube axis motions in RoboTwin scenes loaded from original tasks."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
ROBOTWIN_DIR = REPO_ROOT / "third_party" / "robotwin"
os.chdir(ROBOTWIN_DIR)
sys.path.insert(0, str(ROBOTWIN_DIR))

import yaml
import numpy as np

from scipy.spatial.transform import Rotation

from envs._GLOBAL_CONFIGS import CONFIGS_PATH
from envs.utils.action import Action


SOURCE_TASKS = (
    "adjust_bottle",
    "beat_block_hammer",
    "click_alarmclock",
    "place_a2b_left",
    "stamp_seal",
)


def build_task_class(source_task: str):
    module = importlib.import_module(f"envs.{source_task}")
    source_task_class = getattr(module, source_task)

    class CubeMotionWithTaskObjects(source_task_class):
        def __init__(
            self,
            arm_tag: str,
            start_vertex: list[float],
            target_vertex: list[float],
            episode_info: dict,
            fixed_quaternion: list[float] | None = None,
            free_orientation: bool = False,
        ):
            super().__init__()
            self.arm_tag = arm_tag
            self.start_vertex = start_vertex
            self.target_vertex = target_vertex
            self.episode_info = episode_info
            self.fixed_quaternion = fixed_quaternion
            self.free_orientation = free_orientation

        def play_once(self):
            if not self.fixed_quaternion:
                default_pose = self.robot.left_original_pose if self.arm_tag == "left" else self.robot.right_original_pose
                fixed_quaternion = list(default_pose[3:])
                orientation_source = "default_gripper_pose"
            else:
                fixed_quaternion = self.fixed_quaternion
                orientation_source = "fixed_quaternion"

            start_pose = self.start_vertex + fixed_quaternion
            target_pose = self.target_vertex + fixed_quaternion

            self.move_to_target_pose(start_pose)
            experiment_start_frame = self.FRAME_IDX
            self.move_to_target_pose(target_pose)
            self.info["info"] = {
                **self.episode_info,
                "source_task": source_task,
                "active_arm": self.arm_tag,
                "initial_pose": start_pose,
                "target_pose": target_pose,
                "orientation_source": orientation_source,
                "fixed_quaternion": fixed_quaternion,
                "free_orientation": self.free_orientation,
                "experiment_start_frame": experiment_start_frame,
            }
            return self.info

        def move_to_target_pose(self, target_pose: list[float]):
            if self.free_orientation:
                return self.move(
                    (
                        self.arm_tag,
                        [Action(self.arm_tag, "move", target_pose=target_pose, constraint_pose=[1, 1, 1, 0, 0, 0])],
                    )
                )
            return self.move(self.move_to_pose(self.arm_tag, target_pose))

        def load_camera(self, **kwags):
            rng_state_before_camera = np.random.get_state()
            super().load_camera(**kwags)
            np.random.set_state(rng_state_before_camera)
            source_static_camera_names = set(kwags["source_static_camera_names"])
            for camera_info in kwags["left_embodiment_config"]["static_camera_list"]:
                if camera_info["name"] in source_static_camera_names:
                    if camera_info["name"] == "head_camera" and not kwags["camera"].get("collect_head_camera", True):
                        continue
                    np.random.randn(3)
                    random_head_camera_dis = self.random_head_camera_dis if camera_info["name"] == "head_camera" else 0
                    np.random.uniform(low=0, high=random_head_camera_dis)

        def check_success(self):
            return self.plan_success

        def check_stable(self):
            if self.skip_actor_stability_check:
                return True, []
            return super().check_stable()

    CubeMotionWithTaskObjects.__name__ = f"{source_task}_cube_motion"
    return CubeMotionWithTaskObjects


def format_value(value: float) -> str:
    sign = "neg" if value < 0 else ""
    return f"{sign}{abs(value):.2f}".replace(".", "p")


def build_run_id(
    source_task: str,
    axis: str,
    cube_center: list[float],
    edge_length: float,
    cube_size: list[float],
) -> str:
    center = f"x{format_value(cube_center[0])}_y{format_value(cube_center[1])}_z{format_value(cube_center[2])}"
    if cube_size == [edge_length] * 3:
        size = f"edge{format_value(edge_length)}"
    else:
        size = f"size_x{format_value(cube_size[0])}_y{format_value(cube_size[1])}_z{format_value(cube_size[2])}"
    return f"{source_task}/cube_center_{center}_{size}_axis_{axis}"


def load_seeds(seed_file: Path, count: int) -> list[int]:
    return [int(seed) for seed in seed_file.read_text(encoding="utf-8").split()[:count]]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate cube-edge RoboTwin episodes with original-task objects loaded in the first frame."
    )
    parser.add_argument("--source_task", choices=SOURCE_TASKS, required=True)
    parser.add_argument("--source_dataset_root", type=Path, default=Path("/data1/sunyang/datasets/RoboTwin2.0/dataset"))
    parser.add_argument("--source_dataset_variant", default="aloha-agilex_clean_50")
    parser.add_argument("--seed_file", type=Path)
    parser.add_argument("--seed_count", type=int, default=8)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--axis", choices=("x", "y", "z"), required=True)
    parser.add_argument("--task_config", default="demo_clean")
    parser.add_argument("--head_camera_type", default="Large_D435")
    parser.add_argument("--source_static_camera_names", nargs="+", default=["head_camera", "front_camera"])
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--cube_center", type=float, nargs=3, default=[0.0, -0.15, 1.0])
    parser.add_argument("--edge_length", type=float, default=0.05)
    parser.add_argument("--cube_size", type=float, nargs=3, default=None)
    parser.add_argument("--fixed_rpy", type=float, nargs=3, default=None)
    parser.add_argument("--fixed_quaternion", type=float, nargs=4, default=None)
    parser.add_argument("--free_orientation", action="store_true")
    parser.add_argument("--skip_actor_stability_check", action="store_true")
    parser.add_argument("--save_root", type=Path, default=REPO_ROOT / "outputs" / "robotwin_task_objects")
    parser.add_argument("--render_freq", type=int, default=0)
    args = parser.parse_args()

    if args.fixed_quaternion:
        fixed_quaternion = list(args.fixed_quaternion)
    elif args.fixed_rpy:
        fixed_quaternion = Rotation.from_euler("xyz", list(args.fixed_rpy), degrees=False).as_quat(scalar_first=True).tolist()
    else:
        fixed_quaternion = None

    cube_size = list(args.cube_size) if args.cube_size else [args.edge_length] * 3
    if not args.run_id:
        args.run_id = build_run_id(
            args.source_task,
            args.axis,
            list(args.cube_center),
            args.edge_length,
            cube_size,
        )
    save_dir = (args.save_root / args.run_id).resolve()

    seed_file = args.seed_file
    if not seed_file:
        seed_file = args.source_dataset_root / args.source_task / args.source_dataset_variant / "seed.txt"
    source_seeds = load_seeds(seed_file, args.seed_count)
    assert len(source_seeds) == args.seed_count, f"seed_file only has {len(source_seeds)} seeds: {seed_file}"

    with open(ROBOTWIN_DIR / "task_config" / f"{args.task_config}.yml", "r", encoding="utf-8") as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    config["camera"]["head_camera_type"] = args.head_camera_type
    config["camera"]["wrist_camera_type"] = args.head_camera_type

    with open(Path(CONFIGS_PATH) / "_embodiment_config.yml", "r", encoding="utf-8") as f:
        embodiment_configs = yaml.load(f.read(), Loader=yaml.FullLoader)

    embodiment_type = config["embodiment"]
    if len(embodiment_type) == 1:
        config["left_robot_file"] = embodiment_configs[embodiment_type[0]]["file_path"]
        config["right_robot_file"] = embodiment_configs[embodiment_type[0]]["file_path"]
        config["dual_arm_embodied"] = True
        embodiment_name = str(embodiment_type[0])
    elif len(embodiment_type) == 3:
        config["left_robot_file"] = embodiment_configs[embodiment_type[0]]["file_path"]
        config["right_robot_file"] = embodiment_configs[embodiment_type[1]]["file_path"]
        config["embodiment_dis"] = embodiment_type[2]
        config["dual_arm_embodied"] = False
        embodiment_name = f"{embodiment_type[0]}+{embodiment_type[1]}"

    with open(Path(config["left_robot_file"]) / "config.yml", "r", encoding="utf-8") as f:
        config["left_embodiment_config"] = yaml.load(f.read(), Loader=yaml.FullLoader)
    with open(Path(config["right_robot_file"]) / "config.yml", "r", encoding="utf-8") as f:
        config["right_embodiment_config"] = yaml.load(f.read(), Loader=yaml.FullLoader)
    for camera_info in config["left_embodiment_config"]["static_camera_list"]:
        camera_info["type"] = args.head_camera_type
    for camera_info in config["right_embodiment_config"]["static_camera_list"]:
        camera_info["type"] = args.head_camera_type
    retained_static_camera_names = [
        camera["name"] for camera in config["left_embodiment_config"]["static_camera_list"]
    ]

    config["task_name"] = f"cube_{args.axis}_motion_with_{args.source_task}_objects"
    config["task_config"] = args.task_config
    config["embodiment_name"] = embodiment_name
    config["save_path"] = str(save_dir)
    config["episode_num"] = args.seed_count
    config["render_freq"] = args.render_freq
    config["source_static_camera_names"] = args.source_static_camera_names

    half_size_by_axis = dict(zip(["x", "y", "z"], [size / 2.0 for size in cube_size]))
    episodes = []
    episode_idx = 0
    axes = ["x", "y", "z"]
    fixed_axes = [axis for axis in axes if axis != args.axis]
    for direction in [1, -1]:
        for fixed_sign_0 in [-1, 1]:
            for fixed_sign_1 in [-1, 1]:
                start_signs = {
                    fixed_axes[0]: fixed_sign_0,
                    fixed_axes[1]: fixed_sign_1,
                    args.axis: -1 if direction > 0 else 1,
                }
                target_signs = dict(start_signs)
                target_signs[args.axis] = 1 if direction > 0 else -1
                start_vertex = [
                    args.cube_center[axis_idx] + start_signs[axis] * half_size_by_axis[axis]
                    for axis_idx, axis in enumerate(axes)
                ]
                target_vertex = [
                    args.cube_center[axis_idx] + target_signs[axis] * half_size_by_axis[axis]
                    for axis_idx, axis in enumerate(axes)
                ]
                episodes.append(
                    {
                        "episode_idx": episode_idx,
                        "source_seed": source_seeds[episode_idx],
                        "axis": args.axis,
                        "direction": "+" if direction > 0 else "-",
                        "fixed_signs": {
                            fixed_axes[0]: fixed_sign_0,
                            fixed_axes[1]: fixed_sign_1,
                        },
                        "cube_center": list(args.cube_center),
                        "edge_length": args.edge_length,
                        "cube_size": cube_size,
                        "start_vertex_signs": start_signs,
                        "target_vertex_signs": target_signs,
                        "start_vertex": start_vertex,
                        "target_vertex": target_vertex,
                    }
                )
                episode_idx += 1

    task_class = build_task_class(args.source_task)
    scene_info = {}
    for episode in episodes:
        task = task_class(
            args.arm,
            episode["start_vertex"],
            episode["target_vertex"],
            episode,
            fixed_quaternion,
            args.free_orientation,
        )
        task.skip_actor_stability_check = args.skip_actor_stability_check
        config["need_plan"] = True
        config["save_data"] = False
        config["left_joint_path"] = []
        config["right_joint_path"] = []
        task.setup_demo(now_ep_num=episode["episode_idx"], seed=episode["source_seed"], **config)
        task.play_once()
        if not task.check_success():
            raise RuntimeError(
                f"RoboTwin planning failed for task={args.source_task} axis={args.axis} "
                f"episode={episode['episode_idx']} seed={episode['source_seed']}"
            )
        task.save_traj_data(episode["episode_idx"])
        task.close_env()

        traj_data = task.load_tran_data(episode["episode_idx"])
        task = task_class(
            args.arm,
            episode["start_vertex"],
            episode["target_vertex"],
            episode,
            fixed_quaternion,
            args.free_orientation,
        )
        task.skip_actor_stability_check = args.skip_actor_stability_check
        config["need_plan"] = False
        config["save_data"] = True
        config["left_joint_path"] = traj_data["left_joint_path"]
        config["right_joint_path"] = traj_data["right_joint_path"]
        task.setup_demo(now_ep_num=episode["episode_idx"], seed=episode["source_seed"], **config)
        task.set_path_lst(config)
        info = task.play_once()
        scene_info[f"episode_{episode['episode_idx']}"] = info

        task.close_env()
        task.merge_pkl_to_hdf5_video()
        task.remove_data_cache()
        if not task.check_success():
            raise RuntimeError(
                f"RoboTwin replay failed for task={args.source_task} axis={args.axis} "
                f"episode={episode['episode_idx']} seed={episode['source_seed']}"
            )

        print(
            f"task={args.source_task} axis={args.axis} episode={episode['episode_idx']} "
            f"seed={episode['source_seed']} direction={episode['direction']} "
            f"fixed_{fixed_axes[0]}={episode['fixed_signs'][fixed_axes[0]]} "
            f"fixed_{fixed_axes[1]}={episode['fixed_signs'][fixed_axes[1]]} "
            f"experiment_start_frame={info['info']['experiment_start_frame']}"
        )

    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "scene_info.json", "w", encoding="utf-8") as f:
        json.dump(scene_info, f, ensure_ascii=False, indent=4)
    with open(save_dir / "seed.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(str(seed) for seed in source_seeds))
        f.write("\n")
    with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "task_name": f"cube_{args.axis}_motion_with_{args.source_task}_objects",
                "source_task": args.source_task,
                "source_dataset_root": str(args.source_dataset_root),
                "source_dataset_variant": args.source_dataset_variant,
                "source_seed_file": str(seed_file),
                "source_seeds": source_seeds,
                "task_config": args.task_config,
                "head_camera_type": args.head_camera_type,
                "source_static_camera_names": args.source_static_camera_names,
                "retained_static_camera_names": retained_static_camera_names,
                "arm": args.arm,
                "axis": args.axis,
                "cube_center": list(args.cube_center),
                "edge_length": args.edge_length,
                "cube_size": cube_size,
                "fixed_rpy": list(args.fixed_rpy) if args.fixed_rpy else None,
                "fixed_quaternion": fixed_quaternion,
                "default_orientation": not fixed_quaternion,
                "free_orientation": args.free_orientation,
                "skip_actor_stability_check": args.skip_actor_stability_check,
                "episodes": episodes,
            },
            f,
            ensure_ascii=False,
            indent=4,
        )

    print(f"robotwin_dir={save_dir}")


if __name__ == "__main__":
    main()
