#!/usr/bin/env python3
# 本脚本没有用于最终的空桌面立方体实验数据生成.
# 最终实验目录为 outputs/robotwin/cube_center_x0p00_yneg0p15_z1p00_edge*_axis_*_arm_{left,right}.
# 这个脚本只是早期用于调试单个可视起点和目标点的辅助工具.
"""Run one empty-table RoboTwin motion: default pose -> visible start pose -> target point."""

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

from envs._base_task import Base_Task
from envs._GLOBAL_CONFIGS import CONFIGS_PATH


INITIAL_POSES = {
    "left": [-0.20, -0.08, 0.92, -0.853532, 0.146484, -0.353542, -0.3536],
    "right": [0.20, -0.08, 0.92, -0.353518, 0.353564, -0.14642, -0.853568],
}


class VisiblePoseMotionTask(Base_Task):
    def __init__(self, arm_tag: str, start_pose: list[float], end_pose: list[float]):
        super().__init__()
        self.arm_tag = arm_tag
        self.start_pose = start_pose
        self.end_pose = end_pose

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        pass

    def play_once(self):
        self.move(self.move_to_pose(self.arm_tag, self.start_pose))
        experiment_start_frame = self.FRAME_IDX
        self.move(self.move_to_pose(self.arm_tag, self.end_pose))
        self.info["info"] = {
            "active_arm": self.arm_tag,
            "initial_pose": self.start_pose,
            "target_pose": self.end_pose,
            "experiment_start_frame": experiment_start_frame,
        }
        return self.info

    def check_success(self):
        return self.plan_success


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate one RoboTwin episode that moves one arm to a visible start pose and then to a target point."
    )
    parser.add_argument("--run_id", default="visible_pose_motion")
    parser.add_argument("--task_config", default="demo_clean")
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument(
        "--initial_pose",
        type=float,
        nargs=7,
        default=None,
        metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"),
        help="Visible experiment start pose in RoboTwin world frame.",
    )
    parser.add_argument(
        "--target_point",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Target position in RoboTwin world frame. The initial pose quaternion is reused.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_root", type=Path, default=REPO_ROOT / "outputs" / "robotwin")
    parser.add_argument("--render_freq", type=int, default=0)
    args = parser.parse_args()

    initial_pose = list(args.initial_pose) if args.initial_pose else INITIAL_POSES[args.arm]
    target_point = list(args.target_point) if args.target_point else [initial_pose[0] + 0.05, initial_pose[1], initial_pose[2]]

    target_pose = list(target_point) + list(initial_pose[3:])
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

    config["task_name"] = "visible_pose_motion"
    config["task_config"] = args.task_config
    config["embodiment_name"] = embodiment_name
    config["save_path"] = str(save_dir)
    config["episode_num"] = 1
    config["render_freq"] = args.render_freq

    task = VisiblePoseMotionTask(args.arm, initial_pose, target_pose)
    config["need_plan"] = True
    config["save_data"] = False
    task.setup_demo(now_ep_num=0, seed=args.seed, **config)
    task.play_once()
    if not task.check_success():
        raise RuntimeError("RoboTwin motion planning failed")
    task.save_traj_data(0)
    task.close_env()

    traj_data = task.load_tran_data(0)

    task = VisiblePoseMotionTask(args.arm, initial_pose, target_pose)
    config["need_plan"] = False
    config["save_data"] = True
    config["left_joint_path"] = traj_data["left_joint_path"]
    config["right_joint_path"] = traj_data["right_joint_path"]
    task.setup_demo(now_ep_num=0, seed=args.seed, **config)
    task.set_path_lst(config)
    info = task.play_once()

    with open(save_dir / "scene_info.json", "w", encoding="utf-8") as f:
        json.dump({"episode_0": info}, f, ensure_ascii=False, indent=4)

    task.close_env()
    task.merge_pkl_to_hdf5_video()
    task.remove_data_cache()
    if not task.check_success():
        raise RuntimeError("RoboTwin motion planning or replay failed")

    print(f"robotwin_hdf5={save_dir / 'data' / 'episode0.hdf5'}")
    print(f"robotwin_video={save_dir / 'video' / 'episode0.mp4'}")
    print(f"experiment_start_frame={info['info']['experiment_start_frame']}")


if __name__ == "__main__":
    main()
