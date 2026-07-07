#!/usr/bin/env python3
# 本脚本用于在真正生成数据前扫描规划可行性.
# 当前 cube_center_x0p00_yneg0p15_z1p00_edge* 实验的生成流程入口见 run_pipeline.sh.
"""Scan cube edge planning feasibility for RoboTwin empty-table motions."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parent
ROBOTWIN_DIR = REPO_ROOT / "third_party" / "robotwin"
os.chdir(ROBOTWIN_DIR)
sys.path.insert(0, str(ROBOTWIN_DIR))

from envs._base_task import Base_Task
from envs._GLOBAL_CONFIGS import CONFIGS_PATH
from envs.utils.action import Action


class CubeScanTask(Base_Task):
    def __init__(
        self,
        arm_tag: str,
        start_vertex: list[float],
        target_vertex: list[float],
        fixed_quaternion: list[float] | None,
        free_orientation: bool,
    ):
        super().__init__()
        self.arm_tag = arm_tag
        self.start_vertex = start_vertex
        self.target_vertex = target_vertex
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
        else:
            fixed_quaternion = self.fixed_quaternion

        start_pose = self.start_vertex + fixed_quaternion
        target_pose = self.target_vertex + fixed_quaternion

        if self.free_orientation:
            self.move((self.arm_tag, [Action(self.arm_tag, "move", target_pose=start_pose, constraint_pose=[1, 1, 1, 0, 0, 0])]))
        else:
            self.move(self.move_to_pose(self.arm_tag, start_pose))
        if not self.plan_success:
            raise RuntimeError(f"pre_position planning failed: start_pose={start_pose}")
        if self.free_orientation:
            self.move((self.arm_tag, [Action(self.arm_tag, "move", target_pose=target_pose, constraint_pose=[1, 1, 1, 0, 0, 0])]))
        else:
            self.move(self.move_to_pose(self.arm_tag, target_pose))
        if not self.plan_success:
            raise RuntimeError(f"edge planning failed: target_pose={target_pose}")
        return self.info

    def check_success(self):
        return self.plan_success


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan cube planning feasibility across X/Y/Z edges.")
    parser.add_argument("--task_config", default="demo_clean")
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--cube_center", type=float, nargs=3, action="append", required=True)
    parser.add_argument("--edge_length", type=float, nargs="+", default=[0.20])
    parser.add_argument("--cube_size", type=float, nargs=3, action="append", default=None)
    parser.add_argument("--axis", choices=("x", "y", "z", "all"), default="all")
    parser.add_argument("--fixed_rpy", type=float, nargs=3, default=None)
    parser.add_argument("--fixed_quaternion", type=float, nargs=4, default=None)
    parser.add_argument("--free_orientation", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.fixed_quaternion:
        fixed_quaternion = list(args.fixed_quaternion)
    elif args.fixed_rpy:
        fixed_quaternion = Rotation.from_euler("xyz", list(args.fixed_rpy), degrees=False).as_quat(scalar_first=True).tolist()
    else:
        fixed_quaternion = None

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

    config["task_name"] = "cube_motion_scan"
    config["task_config"] = args.task_config
    config["embodiment_name"] = embodiment_name
    config["save_path"] = str(REPO_ROOT / "outputs" / "robotwin" / "_scan_tmp")
    config["episode_num"] = 1
    config["render_freq"] = 0
    config["need_plan"] = True
    config["save_data"] = False

    axes = ["x", "y", "z"] if args.axis == "all" else [args.axis]
    episode_idx = 0
    cube_sizes = args.cube_size if args.cube_size else [[edge_length] * 3 for edge_length in args.edge_length]
    for cube_size in cube_sizes:
        half_size_by_axis = dict(zip(["x", "y", "z"], [size / 2.0 for size in cube_size]))
        for cube_center in args.cube_center:
            print(f"[scan] center={cube_center} cube_size={cube_size}")
            for axis in axes:
                fixed_axes = [item for item in ["x", "y", "z"] if item != axis]
                for direction in [1, -1]:
                    for sign_a in [-1, 1]:
                        for sign_b in [-1, 1]:
                            start_signs = {"x": sign_a, "y": sign_a, "z": sign_a}
                            target_signs = dict(start_signs)
                            start_signs[axis] = -1 if direction > 0 else 1
                            target_signs[axis] = 1 if direction > 0 else -1
                            start_signs[fixed_axes[0]] = sign_a
                            start_signs[fixed_axes[1]] = sign_b
                            target_signs[fixed_axes[0]] = sign_a
                            target_signs[fixed_axes[1]] = sign_b

                            start_vertex = [
                                cube_center[0] + start_signs["x"] * half_size_by_axis["x"],
                                cube_center[1] + start_signs["y"] * half_size_by_axis["y"],
                                cube_center[2] + start_signs["z"] * half_size_by_axis["z"],
                            ]
                            target_vertex = [
                                cube_center[0] + target_signs["x"] * half_size_by_axis["x"],
                                cube_center[1] + target_signs["y"] * half_size_by_axis["y"],
                                cube_center[2] + target_signs["z"] * half_size_by_axis["z"],
                            ]
                            task = CubeScanTask(
                                args.arm,
                                start_vertex,
                                target_vertex,
                                fixed_quaternion,
                                args.free_orientation,
                            )
                            task.setup_demo(now_ep_num=episode_idx, seed=args.seed, **config)
                            task.play_once()
                            print(
                                f"  ok axis={axis} direction={'+' if direction > 0 else '-'} "
                                f"fixed={fixed_axes[0]}:{sign_a},{fixed_axes[1]}:{sign_b} "
                                f"start={start_vertex} target={target_vertex}"
                            )
                            task.close_env()
                            episode_idx += 1
            print(f"[scan-success] center={cube_center} cube_size={cube_size}")
            return
    raise RuntimeError("no feasible cube found")


if __name__ == "__main__":
    main()
