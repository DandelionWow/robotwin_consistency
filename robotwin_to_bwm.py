#!/usr/bin/env python3
# 本脚本用于把 RoboTwin 生成的 hdf5/video 转成 BWM 需要的 first_frame/action/video 输入格式.
# 当前 cube_center_x0p00_yneg0p15_z1p00_edge* 实验的批量运行入口见 run_pipeline.sh.
"""Convert RoboTwin HDF5 episodes to direct BWM first-frame/action inputs."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import h5py
import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


STATE_WIDTH = 26


def episode_id_from_path(path: Path) -> int:
    return int(re.search(r"episode[_-]?(\d+)", path.stem).group(1))


def scalar_column(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    return array.reshape(array.shape[0], 1)


def quaternion_wxyz_to_rpy(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    return Rotation.from_quat(quat, scalar_first=True).as_euler("xyz", degrees=False).astype(np.float32)


def write_video(rgb_frames: np.ndarray, video_path: Path) -> None:
    rgb_frames = np.asarray(rgb_frames, dtype=np.uint8)
    _, height, width, _ = rgb_frames.shape
    ffmpeg = subprocess.Popen(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            "30",
            "-i",
            "-",
            "-r",
            "30",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            str(video_path),
        ],
        stdin=subprocess.PIPE,
    )
    ffmpeg.stdin.write(rgb_frames.tobytes())
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise IOError(f"Cannot write video: {video_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--crop_to_experiment", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    first_frame_dir = args.output_dir / "first_frames"
    action_dir = args.output_dir / "actions"
    video_dir = args.output_dir / "videos"
    first_frame_dir.mkdir(parents=True, exist_ok=True)
    action_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    if args.crop_to_experiment:
        with open(args.robotwin_dir / "scene_info.json", "r", encoding="utf-8") as f:
            scene_info = json.load(f)

    hdf5_paths = sorted(
        (args.robotwin_dir / "data").glob("episode*.hdf5"),
        key=episode_id_from_path,
    )

    for hdf5_path in hdf5_paths:
        episode_id = episode_id_from_path(hdf5_path)
        source_video_path = args.robotwin_dir / "video" / f"episode{episode_id}.mp4"
        start_frame = args.start_frame
        if args.crop_to_experiment:
            start_frame = scene_info[f"episode_{episode_id}"]["info"]["experiment_start_frame"]

        with h5py.File(hdf5_path, "r") as file:
            joint_state = np.asarray(file["/joint_action/vector"][start_frame:], dtype=np.float32)
            left_endpose = np.asarray(file["/endpose/left_endpose"][start_frame:], dtype=np.float32)
            right_endpose = np.asarray(file["/endpose/right_endpose"][start_frame:], dtype=np.float32)
            left_gripper = scalar_column(file["/endpose/left_gripper"][start_frame:])
            right_gripper = scalar_column(file["/endpose/right_gripper"][start_frame:])

        state = np.empty((joint_state.shape[0], STATE_WIDTH), dtype=np.float32)
        state[:, 0:7] = joint_state[:, 0:7]
        state[:, 7:10] = left_endpose[:, 0:3]
        state[:, 10:13] = quaternion_wxyz_to_rpy(left_endpose[:, 3:7])
        state[:, 13:20] = joint_state[:, 7:14]
        state[:, 20:23] = right_endpose[:, 0:3]
        state[:, 23:26] = quaternion_wxyz_to_rpy(right_endpose[:, 3:7])

        action = np.concatenate(
            [
                state[:, 7:13],
                left_gripper,
                state[:, 20:26],
                right_gripper,
            ],
            axis=1,
        )

        first_frame_path = first_frame_dir / f"episode_{episode_id:06d}.png"
        action_path = action_dir / f"episode_{episode_id:06d}.parquet"
        video_path = video_dir / f"episode_{episode_id:06d}.mp4"

        reader = imageio.get_reader(source_video_path)
        first_frame = reader.get_data(start_frame)[:, :, :3]
        imageio.imwrite(first_frame_path, first_frame)
        list_type = pa.list_(pa.float32())
        pq.write_table(
            pa.table(
                {
                    "observation.state": pa.array(state.tolist(), type=list_type),
                    "action": pa.array(action.tolist(), type=list_type),
                }
            ),
            action_path,
        )

        if start_frame == 0:
            reader.close()
            shutil.copy2(source_video_path, video_path)
        else:
            rgb_frames = [
                reader.get_data(frame_idx)[:, :, :3]
                for frame_idx in range(start_frame, start_frame + action.shape[0])
            ]
            reader.close()
            write_video(np.asarray(rgb_frames, dtype=np.uint8), video_path)

        print(
            f"[input] episode={episode_id} start_frame={start_frame} frames={action.shape[0]} "
            f"first_frame={first_frame_path} action={action_path} video={video_path}"
        )


if __name__ == "__main__":
    main()
