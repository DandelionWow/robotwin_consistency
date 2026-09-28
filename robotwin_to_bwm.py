#!/usr/bin/env python3
"""Compact RoboTwin -> BWM conversion, inference, and comparison pipeline."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


STATE_WIDTH = 26
DEFAULT_BWM_CONFIG = "configs/infer/infer.yaml"
DEFAULT_PROMPT = (
    "In a fixed robotic workspace, generate a rigid, physically consistent embodied robotic arm. "
    "The arm maintains high stability with no deformation and lifts the bottle upright."
)

JOINT_INDICES = [0, 1, 2, 3, 4, 5, 6, 13, 14, 15, 16, 17, 18, 19]
EEF_INDICES = [7, 8, 9, 10, 11, 12, 6, 20, 21, 22, 23, 24, 25, 19]


def import_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise SystemExit("Missing dependency: h5py") from exc
    return h5py


def import_imageio():
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise SystemExit("Missing dependency: imageio") from exc
    return imageio


def import_imageio_ffmpeg():
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise SystemExit("Missing dependency: imageio_ffmpeg") from exc
    return imageio_ffmpeg


def import_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("Missing dependency: pyarrow") from exc
    return pa, pq


def import_rotation():
    try:
        from scipy.spatial.transform import Rotation
    except ImportError as exc:
        raise SystemExit("Missing dependency: scipy") from exc
    return Rotation


def natural_key(value) -> list:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(value))]


def episode_id_from_path(path: Path) -> int:
    match = re.search(r"episode[_-]?(\d+)", path.stem)
    if not match:
        raise ValueError(f"Cannot parse episode id from path: {path}")
    return int(match.group(1))


def episode_name(episode_id: int) -> str:
    return f"episode_{episode_id:06d}"


def condition_episode_ids(arm_filter: str, episode_num: int | None = None) -> set[int] | None:
    if arm_filter == "all":
        return None

    limit = int(episode_num) if episode_num is not None else 1728
    episode_ids = set()
    for episode_id in range(limit):
        block_index = episode_id % 864
        if arm_filter == "left" and block_index < 324:
            episode_ids.add(episode_id)
        elif arm_filter == "right" and 324 <= block_index < 648:
            episode_ids.add(episode_id)
        elif arm_filter == "both" and 648 <= block_index < 864:
            episode_ids.add(episode_id)
    return episode_ids


def scalar_column(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    return array.reshape(array.shape[0], 1)


def quaternion_wxyz_to_rpy(quat: np.ndarray) -> np.ndarray:
    Rotation = import_rotation()
    quat = np.asarray(quat, dtype=np.float64)
    return Rotation.from_quat(quat, scalar_first=True).as_euler("xyz", degrees=False).astype(np.float32)


def state_to_action(state: np.ndarray) -> np.ndarray:
    """Build BWM's 26-D delta action from a 26-D absolute state sequence."""
    Rotation = import_rotation()
    state = np.asarray(state, dtype=np.float32)
    if state.ndim != 2 or state.shape[1] != STATE_WIDTH or state.shape[0] <= 0:
        raise ValueError(f"Expected a non-empty (T, {STATE_WIDTH}) state array, got {state.shape}")

    action = np.zeros_like(state, dtype=np.float32)
    if state.shape[0] > 1:
        action[:-1] = state[1:] - state[:-1]

        # BWM's reference Parquet stores the next absolute gripper command,
        # not a gripper delta.
        action[:-1, 6] = state[1:, 6]
        action[:-1, 19] = state[1:, 19]

        # Euler subtraction is not a valid rotation delta. Match BWM's demo:
        # delta_R = current_R^-1 * next_R, represented as xyz Euler angles.
        for rotation_slice in (slice(10, 13), slice(23, 26)):
            current = Rotation.from_euler("xyz", state[:-1, rotation_slice])
            following = Rotation.from_euler("xyz", state[1:, rotation_slice])
            action[:-1, rotation_slice] = (current.inv() * following).as_euler(
                "xyz", degrees=False
            ).astype(np.float32)

    # The final frame has no following state. BWM keeps continuous deltas at
    # zero while retaining the final absolute gripper command.
    action[-1, 6] = state[-1, 6]
    action[-1, 19] = state[-1, 19]
    return action


def ensure_dir(path: Path, dry_run: bool = False) -> None:
    if dry_run:
        print(f"[dry-run] mkdir -p {path}")
    else:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data, overwrite: bool, dry_run: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")
    if dry_run:
        print(f"[dry-run] write json {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict], overwrite: bool, dry_run: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")
    if dry_run:
        print(f"[dry-run] write jsonl {path} ({len(rows)} rows)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if text:
                try:
                    rows.append(json.loads(text))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def select_episode_paths(
    robotwin_dir: Path,
    num_episodes: int | None,
    episode_ids: str | None,
    arm_filter: str = "all",
    episode_num: int | None = None,
) -> list[Path]:
    hdf5_paths = sorted((robotwin_dir / "data").glob("episode*.hdf5"), key=episode_id_from_path)
    filtered_ids = condition_episode_ids(arm_filter, episode_num)
    if filtered_ids is not None:
        hdf5_paths = [path for path in hdf5_paths if episode_id_from_path(path) in filtered_ids]
    if episode_ids:
        wanted = {int(item.strip().replace("episode", "")) for item in episode_ids.split(",") if item.strip()}
        hdf5_paths = [path for path in hdf5_paths if episode_id_from_path(path) in wanted]
    if num_episodes is not None:
        hdf5_paths = hdf5_paths[:num_episodes]
    return hdf5_paths


def load_scene_info(robotwin_dir: Path, crop_to_experiment: bool) -> dict:
    if not crop_to_experiment:
        return {}
    path = robotwin_dir / "scene_info.json"
    if not path.exists():
        raise FileNotFoundError(f"--crop_to_experiment requires {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_start_frame(args, scene_info: dict, episode_id: int) -> int:
    if not args.crop_to_experiment:
        return int(args.start_frame)
    try:
        return int(scene_info[f"episode_{episode_id}"]["info"]["experiment_start_frame"])
    except KeyError:
        meta_path = args.robotwin_dir / "meta" / f"episode_{episode_id:06d}.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return int(meta["experiment_start_frame"])
        except (FileNotFoundError, KeyError) as exc:
            raise KeyError(
                f"Missing experiment_start_frame for episode_{episode_id} "
                f"in scene_info.json and {meta_path}"
            ) from exc


def read_robotwin_episode(hdf5_path: Path, start_frame: int) -> tuple[np.ndarray, np.ndarray]:
    h5py = import_h5py()
    with h5py.File(hdf5_path, "r") as file:
        required = [
            "/joint_action/vector",
            "/endpose/left_endpose",
            "/endpose/right_endpose",
            "/endpose/left_gripper",
            "/endpose/right_gripper",
        ]
        missing = [key for key in required if key not in file]
        if missing:
            raise KeyError(f"{hdf5_path} missing required HDF5 keys: {missing}")
        joint_state = np.asarray(file["/joint_action/vector"][start_frame:], dtype=np.float32)
        left_endpose = np.asarray(file["/endpose/left_endpose"][start_frame:], dtype=np.float32)
        right_endpose = np.asarray(file["/endpose/right_endpose"][start_frame:], dtype=np.float32)
        left_gripper = scalar_column(file["/endpose/left_gripper"][start_frame:])
        right_gripper = scalar_column(file["/endpose/right_gripper"][start_frame:])

    if joint_state.ndim != 2 or joint_state.shape[1] < 14:
        raise ValueError(f"Unexpected /joint_action/vector shape in {hdf5_path}: {joint_state.shape}")
    if left_endpose.shape[0] != joint_state.shape[0] or right_endpose.shape[0] != joint_state.shape[0]:
        raise ValueError(f"Endpose/action length mismatch in {hdf5_path}")

    state = np.empty((joint_state.shape[0], STATE_WIDTH), dtype=np.float32)
    state[:, 0:7] = joint_state[:, 0:7]
    state[:, 7:10] = left_endpose[:, 0:3]
    state[:, 10:13] = quaternion_wxyz_to_rpy(left_endpose[:, 3:7])
    state[:, 13:20] = joint_state[:, 7:14]
    state[:, 20:23] = right_endpose[:, 0:3]
    state[:, 23:26] = quaternion_wxyz_to_rpy(right_endpose[:, 3:7])

    # Keep the HDF5 endpose gripper values authoritative. They should match
    # joint_action/vector, but explicitly assigning them also validates their
    # lengths and mirrors the BWM demo schema.
    state[:, 6] = left_gripper[:, 0]
    state[:, 19] = right_gripper[:, 0]
    action = state_to_action(state)
    return state, action


def compute_stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float32)
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "p01": np.percentile(arr, 1, axis=0).tolist(),
        "p99": np.percentile(arr, 99, axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
    }


def build_stat(states: list[np.ndarray], actions: list[np.ndarray]) -> dict:
    state_all = np.concatenate(states, axis=0)
    action_all = np.concatenate(actions, axis=0)
    return {
        "state_joint": compute_stats(state_all[:, JOINT_INDICES]),
        "state_pose": compute_stats(state_all[:, EEF_INDICES]),
        "action_joint": compute_stats(action_all[:, JOINT_INDICES]),
        "action_pose": compute_stats(action_all[:, EEF_INDICES]),
    }


def write_parquet(path: Path, state: np.ndarray, action: np.ndarray, overwrite: bool, dry_run: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")
    if dry_run:
        print(f"[dry-run] write parquet {path} state={state.shape} action={action.shape}")
        return
    pa, pq = import_pyarrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    list_type = pa.list_(pa.float32())
    pq.write_table(
        pa.table(
            {
                "observation.state": pa.array(state.tolist(), type=list_type),
                "action": pa.array(action.tolist(), type=list_type),
            }
        ),
        path,
    )


def open_video_reader(path: Path):
    imageio = import_imageio()
    if not path.exists():
        raise FileNotFoundError(f"Missing RoboTwin video: {path}")
    return imageio.get_reader(path)


def read_first_frame_from_hdf5(hdf5_path: Path, start_frame: int, camera_key: str) -> np.ndarray:
    h5py = import_h5py()
    imageio = import_imageio()
    with h5py.File(hdf5_path, "r") as file:
        if camera_key not in file:
            raise KeyError(f"{hdf5_path} missing first-frame camera key: {camera_key}")
        return imageio.imread(file[camera_key][start_frame])[:, :, :3]


def copy_or_crop_video(
    source_video_path: Path,
    video_path: Path,
    start_frame: int,
    num_frames: int,
    overwrite: bool,
    dry_run: bool,
) -> np.ndarray | None:
    if video_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {video_path}")
    if dry_run:
        action = "copy" if start_frame == 0 else "crop"
        print(f"[dry-run] {action} video {source_video_path} -> {video_path}")
        return None
    video_path.parent.mkdir(parents=True, exist_ok=True)
    if start_frame == 0:
        shutil.copy2(source_video_path, video_path)
        return None
    reader = open_video_reader(source_video_path)
    try:
        first_frame = reader.get_data(start_frame)[:, :, :3]
        frames = [first_frame]
        for frame_idx in range(start_frame + 1, start_frame + num_frames):
            frames.append(reader.get_data(frame_idx)[:, :, :3])
    finally:
        try:
            reader.close()
        except Exception:
            pass
    write_video(np.asarray(frames, dtype=np.uint8), video_path)
    return first_frame


def write_video(rgb_frames: np.ndarray, video_path: Path) -> None:
    imageio_ffmpeg = import_imageio_ffmpeg()
    rgb_frames = np.asarray(rgb_frames, dtype=np.uint8)
    _, height, width, _ = rgb_frames.shape
    command = [
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
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    process.stdin.write(rgb_frames.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise IOError(f"Cannot write video: {video_path}")


def write_first_frame(
    path: Path,
    frame: np.ndarray | None,
    source_video_path: Path,
    hdf5_path: Path,
    start_frame: int,
    camera_key: str,
    overwrite: bool,
    dry_run: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")
    if dry_run:
        print(f"[dry-run] write first frame {path}")
        return
    imageio = import_imageio()
    if frame is None:
        try:
            frame = read_first_frame_from_hdf5(hdf5_path, start_frame, camera_key)
        except Exception:
            reader = open_video_reader(source_video_path)
            try:
                frame = reader.get_data(start_frame)[:, :, :3]
            finally:
                reader.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, frame)


def inspect(args) -> None:
    robotwin_dir = args.robotwin_dir
    hdf5_paths = select_episode_paths(
        robotwin_dir, args.num_episodes, args.episode_ids, args.arm_filter, args.episode_num
    )
    print(f"[inspect] robotwin_dir={robotwin_dir}")
    print(f"[inspect] episodes={len(hdf5_paths)}")
    if not hdf5_paths:
        return
    h5py = import_h5py()
    sample = hdf5_paths[0]
    print(f"[inspect] sample_hdf5={sample}")
    with h5py.File(sample, "r") as file:
        def visitor(name, obj):
            if hasattr(obj, "shape"):
                print(f"[hdf5] {name} shape={tuple(obj.shape)} dtype={obj.dtype}")
        file.visititems(visitor)
    print("[inspect] BWM default action_type=eef_abs reads observation.state and extracts 14 EEF dimensions.")


def load_paired_records(paired_dir: Path) -> list[dict]:
    record_dir = paired_dir / "records"
    record_paths = sorted(record_dir.glob("pair_*.json"), key=natural_key)
    if not record_paths:
        raise FileNotFoundError(f"No pair_*.json records found under {record_dir}")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in record_paths]
    pair_ids = [int(record["pair_id"]) for record in records]
    expected = list(range(len(records)))
    if pair_ids != expected:
        raise ValueError(f"Pair ids must be contiguous from zero, got {pair_ids[:10]}")
    return records


def convert_paired(args) -> Path:
    """Convert the paired adjust_bottle dataset into BWM training layout."""
    paired_dir = args.robotwin_dir.resolve()
    output_dir = args.output_dir.resolve()
    data_dir = output_dir / "data" / "chunk-000"
    video_dir = output_dir / "videos" / "chunk-000"
    first_frame_dir = output_dir / "first_frames" / "chunk-000"
    bwm_output_dir = output_dir / "bwm_outputs"
    comparison_dir = output_dir / "comparisons"
    log_dir = output_dir / "logs"
    for directory in (
        output_dir,
        data_dir,
        video_dir,
        first_frame_dir,
        bwm_output_dir,
        comparison_dir,
        log_dir,
    ):
        ensure_dir(directory, args.dry_run)

    records = load_paired_records(paired_dir)
    if args.num_episodes is not None:
        if args.num_episodes % 2:
            raise ValueError("Paired conversion requires an even --num_episodes")
        records = records[: args.num_episodes // 2]

    metadata_rows = []
    manifest_rows = []
    pair_rows = []
    all_states = []
    all_actions = []
    total_frames = 0

    for record in records:
        pair_id = int(record["pair_id"])
        converted_pair = {
            "pair_id": pair_id,
            "base_seed": int(record["base_seed"]),
            "direction": record["direction"],
            "boundary_gap": float(record["boundary_gap"]),
            "base_trajectory_sha256": record["base"]["trajectory_sha256"],
        }
        for label_offset, label in enumerate(("success", "failure")):
            sample = record[label]
            episode_id = pair_id * 2 + label_offset
            name = episode_name(episode_id)
            hdf5_path = paired_dir / sample["hdf5_path"]
            source_video_path = paired_dir / sample["video_path"]
            if not hdf5_path.is_file():
                raise FileNotFoundError(f"Missing paired HDF5: {hdf5_path}")
            if not source_video_path.is_file():
                raise FileNotFoundError(f"Missing paired video: {source_video_path}")

            state, action = read_robotwin_episode(hdf5_path, start_frame=0)
            frame_count = int(action.shape[0])
            expected_frames = int(sample["frame_count"])
            if frame_count != expected_frames:
                raise ValueError(
                    f"Pair {pair_id} {label} has {frame_count} HDF5 frames, expected {expected_frames}"
                )
            all_states.append(state)
            all_actions.append(action)
            total_frames += frame_count

            relative_data = Path("data") / "chunk-000" / f"{name}.parquet"
            relative_video = Path("videos") / "chunk-000" / f"{name}.mp4"
            relative_first_frame = Path("first_frames") / "chunk-000" / f"{name}.png"
            action_path = output_dir / relative_data
            video_path = output_dir / relative_video
            first_frame_path = output_dir / relative_first_frame
            pred_path = bwm_output_dir / f"episode{episode_id}.mp4"
            compare_path = comparison_dir / f"{name}_compare.mp4"

            first_frame = copy_or_crop_video(
                source_video_path,
                video_path,
                start_frame=0,
                num_frames=frame_count,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
            write_first_frame(
                first_frame_path,
                first_frame,
                source_video_path,
                hdf5_path,
                0,
                args.first_frame_camera,
                args.overwrite,
                args.dry_run,
            )
            write_parquet(action_path, state, action, args.overwrite, args.dry_run)

            metadata_row = {
                "episode_index": episode_id,
                "length": frame_count,
                "start_frame": 0,
                "end_frame": frame_count - 1,
                "video": relative_video.as_posix(),
                "action": relative_data.as_posix(),
                "prompt": args.prompt,
                "task": record.get("task_name", "adjust_bottle"),
                "pair_id": pair_id,
                "label": label,
                "success": bool(sample["check_success"]),
                "base_seed": int(record["base_seed"]),
                "direction": record["direction"],
                "initial_bottle_xy_offset": [float(value) for value in sample["offset"]],
                "perturbation_magnitude": float(sample["magnitude"]),
                "boundary_gap": float(record["boundary_gap"]),
                "base_trajectory_sha256": record["base"]["trajectory_sha256"],
                "render_denoiser": record.get("training_render_denoiser", "unknown"),
            }
            metadata_rows.append(metadata_row)
            manifest_rows.append(
                {
                    **metadata_row,
                    "source_hdf5": str(hdf5_path),
                    "source_video": str(source_video_path),
                    "first_frame": str(first_frame_path),
                    "action_parquet": str(action_path),
                    "converted_video": str(video_path),
                    "state_shape": list(state.shape),
                    "action_shape": list(action.shape),
                    "bwm_output": str(pred_path),
                    "comparison": str(compare_path),
                }
            )
            converted_pair[f"{label}_episode_index"] = episode_id
            converted_pair[f"{label}_metadata_index"] = len(metadata_rows) - 1
            converted_pair[f"{label}_video"] = relative_video.as_posix()
            converted_pair[f"{label}_action"] = relative_data.as_posix()
            print(
                f"[convert-paired] pair={pair_id} label={label} episode={episode_id} "
                f"frames={frame_count}"
            )
        pair_rows.append(converted_pair)

    write_jsonl(output_dir / "metadata.jsonl", metadata_rows, args.overwrite, args.dry_run)
    write_jsonl(output_dir / "manifest.jsonl", manifest_rows, args.overwrite, args.dry_run)
    write_jsonl(output_dir / "pairs.jsonl", pair_rows, args.overwrite, args.dry_run)
    write_json(output_dir / "stat.json", build_stat(all_states, all_actions), args.overwrite, args.dry_run)
    write_json(
        output_dir / "summary.json",
        {
            "format": "boundless-world-model",
            "task": "adjust_bottle",
            "source": str(paired_dir),
            "pairs": len(pair_rows),
            "episodes": len(metadata_rows),
            "success_episodes": sum(row["label"] == "success" for row in metadata_rows),
            "failure_episodes": sum(row["label"] == "failure" for row in metadata_rows),
            "total_frames": total_frames,
            "state_width": STATE_WIDTH,
            "action_width": STATE_WIDTH,
            "render_denoisers": sorted({row["render_denoiser"] for row in metadata_rows}),
        },
        args.overwrite,
        args.dry_run,
    )
    return output_dir / "manifest.jsonl"


def convert(args) -> Path:
    robotwin_dir = args.robotwin_dir
    if (robotwin_dir / "records").is_dir() and (robotwin_dir / "success" / "data").is_dir():
        print(f"[convert] detected paired RoboTwin dataset: {robotwin_dir}")
        return convert_paired(args)
    output_dir = args.output_dir
    first_frame_dir = output_dir / "first_frames"
    action_dir = output_dir / "actions"
    video_dir = output_dir / "videos"
    bwm_output_dir = output_dir / "bwm_outputs"
    comparison_dir = output_dir / "comparisons"
    log_dir = output_dir / "logs"
    for directory in (output_dir, first_frame_dir, action_dir, video_dir, bwm_output_dir, comparison_dir, log_dir):
        ensure_dir(directory, args.dry_run)

    scene_info = load_scene_info(robotwin_dir, args.crop_to_experiment)
    hdf5_paths = select_episode_paths(
        robotwin_dir, args.num_episodes, args.episode_ids, args.arm_filter, args.episode_num
    )
    if not hdf5_paths:
        raise SystemExit(f"No episode*.hdf5 files found under {robotwin_dir / 'data'}")

    metadata_rows = []
    manifest_rows = []
    all_states = []
    all_actions = []

    for hdf5_path in hdf5_paths:
        episode_id = episode_id_from_path(hdf5_path)
        name = episode_name(episode_id)
        source_video_path = robotwin_dir / "video" / f"episode{episode_id}.mp4"
        start_frame = resolve_start_frame(args, scene_info, episode_id)
        state, action = read_robotwin_episode(hdf5_path, start_frame)
        all_states.append(state)
        all_actions.append(action)

        first_frame_path = first_frame_dir / f"{name}.png"
        action_path = action_dir / f"{name}.parquet"
        video_path = video_dir / f"{name}.mp4"
        pred_path = bwm_output_dir / f"episode{episode_id}.mp4"
        compare_path = comparison_dir / f"{name}_compare.mp4"

        first_frame = copy_or_crop_video(source_video_path, video_path, start_frame, action.shape[0], args.overwrite, args.dry_run)
        write_first_frame(
            first_frame_path,
            first_frame,
            source_video_path,
            hdf5_path,
            start_frame,
            args.first_frame_camera,
            args.overwrite,
            args.dry_run,
        )
        write_parquet(action_path, state, action, args.overwrite, args.dry_run)

        metadata_rows.append(
            {
                "episode_index": episode_id,
                "length": int(action.shape[0]),
                "start_frame": 0,
                "end_frame": int(action.shape[0]) - 1,
                "video": f"videos/{name}.mp4",
                "action": f"actions/{name}.parquet",
                "task": robotwin_dir.parent.name,
            }
        )
        manifest_rows.append(
            {
                "episode_index": episode_id,
                "source_hdf5": str(hdf5_path),
                "source_video": str(source_video_path),
                "start_frame": start_frame,
                "num_frames": int(action.shape[0]),
                "state_shape": list(state.shape),
                "action_shape": list(action.shape),
                "first_frame": str(first_frame_path),
                "action_parquet": str(action_path),
                "converted_video": str(video_path),
                "bwm_output": str(pred_path),
                "comparison": str(compare_path),
            }
        )
        print(f"[convert] episode={episode_id} frames={action.shape[0]} video={video_path} action={action_path}")

    write_jsonl(output_dir / "metadata.jsonl", metadata_rows, args.overwrite, args.dry_run)
    write_jsonl(output_dir / "manifest.jsonl", manifest_rows, args.overwrite, args.dry_run)
    write_json(output_dir / "stat.json", build_stat(all_states, all_actions), args.overwrite, args.dry_run)
    return output_dir / "manifest.jsonl"


def parse_local_sh(path: Path) -> dict[str, str]:
    values = {}
    if not path.exists():
        return values
    pattern = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        match = pattern.match(text)
        if not match:
            continue
        key, value = match.groups()
        value = value.split("#", 1)[0].strip()
        try:
            parts = shlex.split(value)
            values[key] = parts[0] if parts else ""
        except ValueError:
            values[key] = value.strip("'\"")
    return values


def selected_manifest_rows(args) -> list[dict]:
    rows = read_jsonl(args.output_dir / "manifest.jsonl")
    filtered_ids = condition_episode_ids(args.arm_filter, args.episode_num)
    if filtered_ids is not None:
        rows = [row for row in rows if int(row["episode_index"]) in filtered_ids]
    if args.episode_ids:
        wanted = {int(item.strip().replace("episode", "")) for item in args.episode_ids.split(",") if item.strip()}
        rows = [row for row in rows if int(row["episode_index"]) in wanted]
    if args.num_episodes is not None:
        rows = rows[: args.num_episodes]
    return rows


def selected_infer_rows(args) -> tuple[list[dict], int]:
    rows = selected_manifest_rows(args)
    if args.infer_start_index < 0:
        raise SystemExit("--infer_start_index must be non-negative")
    if args.infer_max_samples is not None and args.infer_max_samples < 0:
        raise SystemExit("--infer_max_samples must be non-negative")

    start = int(args.infer_start_index)
    end = len(rows) if args.infer_max_samples is None else min(len(rows), start + int(args.infer_max_samples))
    rows = rows[start:end]
    return rows, start


def infer(args) -> None:
    bwm_root = args.bwm_root
    infer_py = bwm_root / "scripts" / "infer.py"
    if not infer_py.exists():
        raise FileNotFoundError(f"Missing BWM inference entrypoint: {infer_py}")
    local = parse_local_sh(bwm_root / "scripts" / "local.sh")
    python_bin = args.python_bin or sys.executable
    model_paths = args.model_paths or local.get("MODEL_PATHS")
    ckpt_path = args.ckpt_path or local.get("CKPT_PATH")
    config = args.bwm_config or local.get("CONFIG_PATH") or DEFAULT_BWM_CONFIG
    if not model_paths:
        raise SystemExit("Missing --model_paths or MODEL_PATHS in BWM scripts/local.sh")
    if not ckpt_path:
        raise SystemExit("Missing --ckpt_path or CKPT_PATH in BWM scripts/local.sh")

    rows, infer_base_index = selected_infer_rows(args)
    if not rows:
        raise SystemExit("No manifest rows selected for inference")
    if args.resume:
        existing = [Path(row["bwm_output"]).exists() for row in rows]
        local_start_index = 0
        while local_start_index < len(existing) and existing[local_start_index]:
            local_start_index += 1
        if local_start_index >= len(rows):
            print("[infer] all selected outputs already exist; nothing to do")
            return
    else:
        local_start_index = 0

    start_index = infer_base_index + local_start_index
    max_samples = len(rows) - local_start_index
    command = [
        python_bin,
        "scripts/infer.py",
        "--config",
        config,
        "--model_paths",
        model_paths,
        "--ckpt_path",
        ckpt_path,
        "--dataset_base_path",
        str(args.output_dir.resolve()),
        "--dataset_metadata_path",
        str((args.output_dir / "metadata.jsonl").resolve()),
        "--action_stat_path",
        str((args.output_dir / "stat.json").resolve()),
        "--output_path",
        str((args.output_dir / "bwm_outputs").resolve()),
        "--start_index",
        str(start_index),
        "--max_samples",
        str(max_samples),
        "--action_type",
        args.bwm_action_type,
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--num_frames",
        str(args.num_frames),
        "--num_history_frames",
        str(args.num_history_frames),
        "--time_division_factor",
        str(args.time_division_factor),
        "--time_division_remainder",
        str(args.time_division_remainder),
        "--action_dim",
        str(args.action_dim),
        "--action_mode",
        args.action_mode,
        "--text_mode",
        args.text_mode,
        "--mixed_precision",
        args.mixed_precision,
        "--cfg_scale",
        str(args.cfg_scale),
        "--num_inference_steps",
        str(args.num_inference_steps),
        "--quality",
        str(args.quality),
    ]
    if args.fps:
        command.extend(["--fps", str(args.fps)])

    printable = " ".join(shlex.quote(str(item)) for item in command)
    print(
        f"[infer] rows=[{infer_base_index}, {infer_base_index + len(rows)}) "
        f"resume_offset={local_start_index} start_index={start_index} max_samples={max_samples}"
    )
    print(f"[infer] {printable}")
    if args.dry_run:
        return
    log_path = Path(args.infer_log_name)
    if not log_path.is_absolute():
        log_path = args.output_dir / "logs" / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {printable}\n\n")
        subprocess.run(command, cwd=bwm_root, stdout=log, stderr=subprocess.STDOUT, text=True, check=True)
    print(f"[infer] log={log_path}")


def compare(args) -> None:
    try:
        imageio_ffmpeg = import_imageio_ffmpeg()
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except SystemExit:
        if not args.dry_run:
            raise
        ffmpeg = "ffmpeg"
        print("[dry-run][warn] imageio_ffmpeg is missing; showing the ffmpeg command without executing it.")
    rows = selected_manifest_rows(args)
    if not rows:
        raise SystemExit("No manifest rows selected for comparison")
    for row in rows:
        left = Path(row["converted_video"])
        right = Path(row["bwm_output"])
        output = Path(row["comparison"])
        if not left.exists():
            raise FileNotFoundError(f"Missing converted RoboTwin video: {left}")
        if not right.exists():
            if args.dry_run:
                print(f"[dry-run][warn] missing BWM output: {right}")
            else:
                raise FileNotFoundError(f"Missing BWM output: {right}")
        if output.exists() and not args.overwrite:
            if args.resume:
                print(f"[compare] skip existing {output}")
                continue
            raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        side = int(args.compare_width)
        fps = int(args.fps)
        filter_complex = (
            f"[0:v]setpts=N/({fps}*TB),fps={fps},"
            f"scale={side}:{side}:force_original_aspect_ratio=decrease,"
            f"pad={side}:{side}:(ow-iw)/2:(oh-ih)/2,setsar=1[left];"
            f"[1:v]setpts=N/({fps}*TB),fps={fps},"
            f"scale={side}:{side}:force_original_aspect_ratio=decrease,"
            f"pad={side}:{side}:(ow-iw)/2:(oh-ih)/2,setsar=1[right];"
            "[left][right]hstack=inputs=2:shortest=1[v]"
        )
        command = [
            ffmpeg,
            "-y" if args.overwrite else "-n",
            "-i",
            str(left),
            "-i",
            str(right),
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-r",
            str(args.fps),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ]
        printable = " ".join(shlex.quote(str(item)) for item in command)
        print(f"[compare] {printable}")
        if not args.dry_run:
            subprocess.run(command, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="RoboTwin to BWM compact pipeline.")
    parser.add_argument("--stage", choices=["inspect", "convert", "infer", "compare", "all"], default="convert")
    parser.add_argument("--robotwin_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--episode_num", type=int, default=1728)
    parser.add_argument("--episode_ids", type=str, default=None, help="Comma-separated ids, e.g. 0,1 or episode0,episode1.")
    parser.add_argument("--arm_filter", choices=["all", "left", "right", "both"], default="all")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--first_frame_camera", default="observation/head_camera/rgb")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--crop_to_experiment", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bwm_root", type=Path, default=Path("third_party/boundless-world-model"))
    parser.add_argument("--python_bin", type=str, default=None)
    parser.add_argument("--model_paths", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--bwm_config", type=str, default=None)
    parser.add_argument("--bwm_action_type", choices=["eef_abs", "joint_abs", "eef_delta", "joint_delta"], default="eef_abs")
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
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--compare_width", type=int, default=480)
    parser.add_argument("--infer_start_index", type=int, default=0)
    parser.add_argument("--infer_max_samples", type=int, default=None)
    parser.add_argument("--infer_log_name", default="infer.log")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.stage in ("inspect", "convert", "all") and args.robotwin_dir is None:
        raise SystemExit("--robotwin_dir is required for inspect, convert, and all stages")
    if args.stage == "inspect":
        inspect(args)
    elif args.stage == "convert":
        convert(args)
    elif args.stage == "infer":
        infer(args)
    elif args.stage == "compare":
        compare(args)
    elif args.stage == "all":
        inspect(args)
        convert(args)
        infer(args)
        compare(args)


if __name__ == "__main__":
    main()
