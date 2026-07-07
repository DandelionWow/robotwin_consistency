#!/usr/bin/env python3
# 本脚本用于生成 RoboTwin 空桌面立方体运动实验数据.
# 当前 cube_center_x0p00_yneg0p15_z1p00_edge* 实验的批量运行入口见 run_pipeline.sh.
"""Run cube axis RoboTwin motions on an empty table."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
ROBOTWIN_DIR = REPO_ROOT / "third_party" / "robotwin"
os.chdir(ROBOTWIN_DIR)
sys.path.insert(0, str(ROBOTWIN_DIR))

import yaml

from scipy.spatial.transform import Rotation

from envs._base_task import Base_Task
from envs._GLOBAL_CONFIGS import CONFIGS_PATH
from envs.utils.action import Action


class CubeXMotionTask(Base_Task):
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

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        pass

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

    def check_success(self):
        return self.plan_success


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 8 directed cube-edge RoboTwin episodes along one world axis."
    )
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--axis", choices=("x", "y", "z"), default="x")
    parser.add_argument("--task_config", default="demo_clean")
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--cube_center", type=float, nargs=3, default=[-0.20, -0.08, 0.94])
    parser.add_argument("--edge_length", type=float, default=0.05)
    parser.add_argument("--cube_size", type=float, nargs=3, default=None)
    parser.add_argument("--fixed_rpy", type=float, nargs=3, default=None)
    parser.add_argument("--fixed_quaternion", type=float, nargs=4, default=None)
    parser.add_argument("--free_orientation", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_root", type=Path, default=REPO_ROOT / "outputs" / "robotwin")
    parser.add_argument("--render_freq", type=int, default=0)
    args = parser.parse_args()

    if args.fixed_quaternion:
        fixed_quaternion = list(args.fixed_quaternion)
    elif args.fixed_rpy:
        fixed_quaternion = Rotation.from_euler("xyz", list(args.fixed_rpy), degrees=False).as_quat(scalar_first=True).tolist()
    else:
        fixed_quaternion = None
    cube_size = list(args.cube_size) if args.cube_size else [args.edge_length] * 3
    half_size_by_axis = dict(zip(["x", "y", "z"], [size / 2.0 for size in cube_size]))
    if not args.run_id:
        args.run_id = (
            f"cube_{args.axis}_S"
            f"{int(round(cube_size[0] * 100)):03d}_"
            f"{int(round(cube_size[1] * 100)):03d}_"
            f"{int(round(cube_size[2] * 100)):03d}"
        )
    save_dir = (args.save_root / args.run_id).resolve()

    with open(ROBOTWIN_DIR / "task_config" / f"{args.task_config}.yml", "r", encoding="utf-8") as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    config["camera"]["head_camera_type"] = "Large_D435"

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

    config["task_name"] = f"cube_{args.axis}_motion"
    config["task_config"] = args.task_config
    config["embodiment_name"] = embodiment_name
    config["save_path"] = str(save_dir)
    config["episode_num"] = 8
    config["render_freq"] = args.render_freq

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

    scene_info = {}
    for episode in episodes:
        task = CubeXMotionTask(
            args.arm,
            episode["start_vertex"],
            episode["target_vertex"],
            episode,
            fixed_quaternion,
            args.free_orientation,
        )
        config["need_plan"] = True
        config["save_data"] = False
        config["left_joint_path"] = []
        config["right_joint_path"] = []
        task.setup_demo(now_ep_num=episode["episode_idx"], seed=args.seed, **config)
        task.play_once()
        if not task.check_success():
            raise RuntimeError(f"RoboTwin planning failed for episode {episode['episode_idx']}")
        task.save_traj_data(episode["episode_idx"])
        task.close_env()

        traj_data = task.load_tran_data(episode["episode_idx"])

        task = CubeXMotionTask(
            args.arm,
            episode["start_vertex"],
            episode["target_vertex"],
            episode,
            fixed_quaternion,
            args.free_orientation,
        )
        config["need_plan"] = False
        config["save_data"] = True
        config["left_joint_path"] = traj_data["left_joint_path"]
        config["right_joint_path"] = traj_data["right_joint_path"]
        task.setup_demo(now_ep_num=episode["episode_idx"], seed=args.seed, **config)
        task.set_path_lst(config)
        info = task.play_once()
        scene_info[f"episode_{episode['episode_idx']}"] = info

        task.close_env()
        task.merge_pkl_to_hdf5_video()
        task.remove_data_cache()
        if not task.check_success():
            raise RuntimeError(f"RoboTwin replay failed for episode {episode['episode_idx']}")

        print(
            f"episode={episode['episode_idx']} direction={episode['direction']} "
            f"fixed_{fixed_axes[0]}={episode['fixed_signs'][fixed_axes[0]]} "
            f"fixed_{fixed_axes[1]}={episode['fixed_signs'][fixed_axes[1]]} "
            f"experiment_start_frame={info['info']['experiment_start_frame']}"
        )

    with open(save_dir / "scene_info.json", "w", encoding="utf-8") as f:
        json.dump(scene_info, f, ensure_ascii=False, indent=4)

    with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "task_name": f"cube_{args.axis}_motion",
                "task_config": args.task_config,
                "arm": args.arm,
                "seed": args.seed,
                "axis": args.axis,
                "cube_center": list(args.cube_center),
                "edge_length": args.edge_length,
                "cube_size": cube_size,
                "fixed_rpy": list(args.fixed_rpy) if args.fixed_rpy else None,
                "fixed_quaternion": fixed_quaternion,
                "default_orientation": not fixed_quaternion,
                "free_orientation": args.free_orientation,
                "episodes": episodes,
            },
            f,
            ensure_ascii=False,
            indent=4,
        )

    print(f"robotwin_dir={save_dir}")


if __name__ == "__main__":
    main()
