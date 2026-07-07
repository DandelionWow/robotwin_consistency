#!/usr/bin/env python3
# 本脚本用于直接调用 BWM, 根据 first_frame 和 action parquet 生成世界模型视频.
# 当前 cube_center_x0p00_yneg0p15_z1p00_edge* 实验的批量运行入口见 run_pipeline.sh.
"""Run BWM directly from first-frame images and action parquet files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import h5py
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parent
BWM_DIR = REPO_ROOT / "third_party" / "boundless-world-model"
BWM_STAT_PATH = BWM_DIR / "demo" / "stat.json"
sys.path.insert(0, str(BWM_DIR))

from scripts.infer import _run_autoregressive, build_pipeline
from wan_video_action.utils import align_num_frames


def load_camera_frames(robotwin_dir: Path, episode_index: int, camera: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(robotwin_dir / "data" / f"episode{episode_index}.hdf5", "r") as file:
        intrinsic = np.asarray(file[f"observation/{camera}/intrinsic_cv"], dtype=np.float64)
        extrinsic = np.asarray(file[f"observation/{camera}/extrinsic_cv"], dtype=np.float64)
    return intrinsic, extrinsic


def project_point(point: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray) -> tuple[int, int]:
    camera_point = extrinsic @ np.asarray([point[0], point[1], point[2], 1.0], dtype=np.float64)
    pixel = intrinsic @ camera_point
    uv = pixel[:2] / pixel[2]
    return int(round(uv[0])), int(round(uv[1]))


def draw_pose_axes(
    draw: ImageDraw.ImageDraw,
    action_row: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    axis_length: float,
) -> None:
    origin = np.asarray(action_row[:3], dtype=np.float64)
    rotation = Rotation.from_euler("xyz", np.asarray(action_row[3:6], dtype=np.float64), degrees=False).as_matrix()
    origin_uv = project_point(origin, intrinsic, extrinsic)
    axes = [
        ("X", (255, 64, 64), rotation[:, 0]),
        ("Y", (64, 220, 80), rotation[:, 1]),
        ("Z", (80, 140, 255), rotation[:, 2]),
    ]
    draw.ellipse(
        (origin_uv[0] - 4, origin_uv[1] - 4, origin_uv[0] + 4, origin_uv[1] + 4),
        fill=(255, 255, 255, 255),
        outline=(0, 0, 0, 255),
        width=1,
    )
    for label, color, axis_vector in axes:
        end_uv = project_point(origin + axis_vector * axis_length, intrinsic, extrinsic)
        draw.line([origin_uv, end_uv], fill=(*color, 255), width=4)
        draw.ellipse((end_uv[0] - 3, end_uv[1] - 3, end_uv[0] + 3, end_uv[1] + 3), fill=(*color, 255))
        draw.text((end_uv[0] + 4, end_uv[1] + 4), label, fill=(*color, 255))


def annotate_video_with_action(
    video_path: Path,
    raw_action: np.ndarray,
    fps: int,
    quality: int,
    camera_frames: tuple,
    overlay_start_frame: int,
    pose_axis_length: float,
    overlay_arm: str,
    source_image_size: tuple[int, int],
) -> None:
    tmp_path = video_path.with_name(f"{video_path.stem}.annotated_tmp{video_path.suffix}")
    font = ImageFont.load_default()
    action_offset = 0 if overlay_arm == "left" else 7
    reader = imageio.get_reader(video_path)
    with imageio.get_writer(tmp_path, fps=fps, codec="libx264", quality=quality) as writer:
        for frame_idx, frame in enumerate(reader):
            action_idx = min(frame_idx, raw_action.shape[0] - 1)
            arm_action = raw_action[action_idx, action_offset : action_offset + 7]
            image = Image.fromarray(frame[:, :, :3])
            draw = ImageDraw.Draw(image, "RGBA")
            if camera_frames:
                intrinsic_frames, extrinsic_frames = camera_frames
                camera_frame_idx = min(overlay_start_frame + action_idx, intrinsic_frames.shape[0] - 1)
                intrinsic = intrinsic_frames[camera_frame_idx].copy()
                source_width, source_height = source_image_size
                frame_height, frame_width = frame.shape[:2]
                intrinsic[0, :] *= frame_width / source_width
                intrinsic[1, :] *= frame_height / source_height
                draw_pose_axes(
                    draw,
                    arm_action,
                    intrinsic,
                    extrinsic_frames[camera_frame_idx],
                    pose_axis_length,
                )
            lines = [
                f"frame={action_idx:04d}",
                (
                    f"{overlay_arm} xyz="
                    f"({arm_action[0]:+.3f},{arm_action[1]:+.3f},{arm_action[2]:+.3f})"
                ),
                (
                    f"{overlay_arm} rpy="
                    f"({arm_action[3]:+.2f},{arm_action[4]:+.2f},{arm_action[5]:+.2f}) "
                    f"g={arm_action[6]:+.2f}"
                ),
            ]
            line_height = 15
            box_width = 360
            box_height = 8 + line_height * len(lines)
            draw.rectangle((8, 8, 8 + box_width, 8 + box_height), fill=(0, 0, 0, 150))
            for line_idx, line in enumerate(lines):
                draw.text((14, 12 + line_idx * line_height), line, fill=(255, 255, 255, 255), font=font)
            writer.append_data(np.asarray(image))
    reader.close()
    os.replace(tmp_path, video_path)


def run_sample(
    pipe,
    args,
    first_frame_path: Path,
    action_path: Path,
    output_path: Path,
    sample_index: int,
    episode_index: int,
    overlay_start_frame: int,
) -> None:
    image = Image.open(first_frame_path).convert("RGB").resize(
        (args.width, args.height),
        Image.Resampling.BILINEAR,
    )
    source_image_size = Image.open(first_frame_path).size
    input_video = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1).contiguous()
    input_video = (input_video * (2.0 / 255.0) - 1.0).unsqueeze(0).unsqueeze(2)

    raw_action = np.asarray(
        pq.read_table(action_path, columns=["action"]).to_pydict()["action"],
        dtype=np.float32,
    )
    total_frames = align_num_frames(
        raw_action.shape[0],
        time_division_factor=args.time_division_factor,
        time_division_remainder=args.time_division_remainder,
    )
    with open(BWM_STAT_PATH, "r", encoding="utf-8") as f:
        stat = json.load(f)["state_pose"]
    action_min = np.asarray(stat["p01"], dtype=np.float32)
    action_max = np.asarray(stat["p99"], dtype=np.float32)
    action = 2 * (raw_action[:total_frames] - action_min) / (action_max - action_min + 1e-8) - 1
    action = torch.as_tensor(
        np.clip(action, -1.0, 1.0)[None],
        dtype=pipe.torch_dtype,
        device=pipe.device,
    )

    print(
        f"[sample] sample_index={sample_index} episode_index={episode_index} "
        f"first_frame={first_frame_path} action={action_path} "
        f"raw_frames={raw_action.shape[0]} aligned_frames={total_frames}"
    )
    _run_autoregressive(
        pipe=pipe,
        sample={
            "sample_index": sample_index,
            "episode_index": episode_index,
            "total_frames": total_frames,
            "video": input_video,
            "action": action,
            "output_path": str(output_path),
        },
        args=args,
    )
    if not args.no_action_overlay:
        camera_frames = ()
        if args.robotwin_dir:
            camera_frames = load_camera_frames(args.robotwin_dir, episode_index, args.overlay_camera)
        annotate_video_with_action(
            output_path,
            raw_action[:total_frames],
            int(args.fps),
            int(args.quality),
            camera_frames,
            overlay_start_frame,
            float(args.pose_axis_length),
            args.overlay_arm,
            source_image_size,
        )
    print(f"[done] sample_index={sample_index} episode_index={episode_index} output={output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first_frame", type=Path)
    parser.add_argument("--action", type=Path)
    parser.add_argument("--output_path", type=Path)
    parser.add_argument("--input_dir", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--model_paths", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument(
        "--model_config_path",
        default=str(BWM_DIR / "configs" / "model" / "wan2_2_ti2v_5b.yaml"),
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_history_frames", type=int, default=9)
    parser.add_argument("--time_division_factor", type=int, default=4)
    parser.add_argument("--time_division_remainder", type=int, default=1)
    parser.add_argument("--action_dim", type=int, default=14)
    parser.add_argument("--action_mode", default="adaln")
    parser.add_argument("--text_mode", default="none")
    parser.add_argument("--mixed_precision", default="bf16")
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_action_overlay", action="store_true")
    parser.add_argument("--robotwin_dir", type=Path)
    parser.add_argument("--overlay_camera", default="head_camera")
    parser.add_argument("--overlay_arm", choices=("left", "right"), default="left")
    parser.add_argument("--overlay_start_frame", type=int, default=0)
    parser.add_argument("--overlay_start_frame_from_scene", action="store_true")
    parser.add_argument("--pose_axis_length", type=float, default=0.05)
    args = parser.parse_args()

    pipe = build_pipeline(args)

    if args.overlay_start_frame_from_scene:
        with open(args.robotwin_dir / "scene_info.json", "r", encoding="utf-8") as f:
            scene_info = json.load(f)

    if args.input_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        first_frame_paths = sorted((args.input_dir / "first_frames").glob("episode_*.png"))
        for sample_index, first_frame_path in enumerate(first_frame_paths):
            episode_index = int(first_frame_path.stem.split("_")[-1])
            action_path = args.input_dir / "actions" / f"episode_{episode_index:06d}.parquet"
            output_path = args.output_dir / f"episode{episode_index}.mp4"
            overlay_start_frame = args.overlay_start_frame
            if args.overlay_start_frame_from_scene:
                overlay_start_frame = scene_info[f"episode_{episode_index}"]["info"]["experiment_start_frame"]
            run_sample(
                pipe,
                args,
                first_frame_path,
                action_path,
                output_path,
                sample_index,
                episode_index,
                int(overlay_start_frame),
            )
    else:
        overlay_start_frame = args.overlay_start_frame
        if args.overlay_start_frame_from_scene:
            overlay_start_frame = scene_info["episode_0"]["info"]["experiment_start_frame"]
        run_sample(pipe, args, args.first_frame, args.action, args.output_path, 0, 0, int(overlay_start_frame))


if __name__ == "__main__":
    main()
