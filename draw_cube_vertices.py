#!/usr/bin/env python3
# 本脚本用于把立方体 8 个顶点投影到 head camera 的实验开始帧上, 生成诊断图.
# 当前 cube_center_x0p00_yneg0p15_z1p00_edge* 实验的批量运行入口见 run_pipeline.sh.
"""Draw cube or cuboid vertices on one RoboTwin camera frame."""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


parser = argparse.ArgumentParser()
parser.add_argument("--robotwin_dir", type=Path, required=True)
parser.add_argument("--output_path", type=Path, required=True)
parser.add_argument("--camera", default="head_camera")
parser.add_argument("--episode", type=int, default=0)
args = parser.parse_args()

scene_info = json.loads((args.robotwin_dir / "scene_info.json").read_text())
episode_info = scene_info[f"episode_{args.episode}"]["info"]
frame_idx = int(episode_info["experiment_start_frame"])
cube_center = np.asarray(episode_info["cube_center"], dtype=np.float64)
cube_size = np.asarray(episode_info["cube_size"], dtype=np.float64)
half_size = cube_size / 2.0

vertices = []
for x_sign in [-1, 1]:
    for y_sign in [-1, 1]:
        for z_sign in [-1, 1]:
            signs = np.asarray([x_sign, y_sign, z_sign], dtype=np.float64)
            vertices.append((f"({x_sign:+.0f},{y_sign:+.0f},{z_sign:+.0f})", cube_center + signs * half_size))

with h5py.File(args.robotwin_dir / "data" / f"episode{args.episode}.hdf5", "r") as file:
    encoded = bytes(file[f"observation/{args.camera}/rgb"][frame_idx]).rstrip(b"\0")
    frame = np.asarray(Image.open(BytesIO(encoded)).convert("RGB"))
    intrinsic = np.asarray(file[f"observation/{args.camera}/intrinsic_cv"][frame_idx], dtype=np.float64)
    extrinsic = np.asarray(file[f"observation/{args.camera}/extrinsic_cv"][frame_idx], dtype=np.float64)

projected = {}
for label, point in vertices:
    camera_point = extrinsic @ np.asarray([point[0], point[1], point[2], 1.0], dtype=np.float64)
    pixel = intrinsic @ camera_point
    uv = pixel[:2] / pixel[2]
    projected[label] = (int(round(uv[0])), int(round(uv[1])), point)

image = Image.fromarray(frame)
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

for axis in range(3):
    for label_a, point_a in vertices:
        for label_b, point_b in vertices:
            if label_a >= label_b:
                continue
            diff = np.abs(point_a - point_b) > 1e-8
            if int(np.sum(diff)) == 1 and diff[axis]:
                draw.line(
                    [projected[label_a][:2], projected[label_b][:2]],
                    fill=(255, 210, 0),
                    width=2,
                )

for label, (u, v, point) in projected.items():
    color = (255, 64, 64) if point[0] < cube_center[0] else (64, 160, 255)
    draw.ellipse((u - 5, v - 5, u + 5, v + 5), fill=color, outline=(255, 255, 255), width=2)
    draw.text((u + 7, v - 7), label, fill=(255, 255, 255), font=font)

draw.text(
    (12, 12),
    f"{args.robotwin_dir.name} {args.camera} frame={frame_idx}",
    fill=(255, 255, 255),
    font=font,
)
args.output_path.parent.mkdir(parents=True, exist_ok=True)
image.save(args.output_path)
print(f"cube_vertices_image={args.output_path}")
