#!/usr/bin/env python3
"""Collect paired success/failure RoboTwin episodes for adjust_bottle.

Each pair uses one random seed and one expert joint trajectory.  The bottle is
translated along one XY direction, and the collector searches for the nearest
observed success/failure boundary before recording both sides of the pair.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import traceback
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOTWIN_ROOT = PROJECT_ROOT / "third_party" / "robotwin"
DEFAULT_CONFIG = ROBOTWIN_ROOT / "task_config" / "wm_adjust_bottle_pairs.yml"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "adjust_bottle_paired_100"
TASK_NAME = "adjust_bottle"
DIRECTIONS = (
    ("+x", np.asarray([1.0, 0.0], dtype=np.float64)),
    ("-x", np.asarray([-1.0, 0.0], dtype=np.float64)),
    ("+y", np.asarray([0.0, 1.0], dtype=np.float64)),
    ("-y", np.asarray([0.0, -1.0], dtype=np.float64)),
)


class CandidateInvalid(RuntimeError):
    """The simulation did not produce a usable completed replay."""


class SeedRejected(RuntimeError):
    """A seed cannot produce the requested boundary pair."""


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as file:
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n").encode("utf-8")
    atomic_write_bytes(path, payload)


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False, default=json_default) + "\n" for row in rows)
    atomic_write_bytes(path, text.encode("utf-8"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha_from_git(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def pose_list(pose) -> dict[str, list[float]]:
    return {
        "position": np.asarray(pose.p, dtype=np.float64).tolist(),
        "quaternion_wxyz": np.asarray(pose.q, dtype=np.float64).tolist(),
    }


def trajectory_payload(left_path: list, right_path: list) -> bytes:
    return pickle.dumps(
        {"left_joint_path": left_path, "right_joint_path": right_path},
        protocol=pickle.HIGHEST_PROTOCOL,
    )


def relative_to_output(path: Path, output_dir: Path) -> str:
    return str(path.resolve().relative_to(output_dir.resolve()))


@dataclass
class Runtime:
    task_factory: Any
    config: dict
    unstable_error: type[Exception]
    robotwin_commit: str
    config_sha256: str


class Collector:
    def __init__(self, cli: argparse.Namespace, runtime: Runtime):
        self.cli = cli
        self.runtime = runtime
        self.output_dir = cli.output_dir.resolve()
        self.simulation_count = 0
        self.error_count = 0
        # RoboTwin's official collector reuses one task object so Robot.reset()
        # can retain planner state between episodes.
        self.task = runtime.task_factory()

    def episode_args(
        self,
        *,
        seed: int,
        episode_id: int,
        save_dir: Path,
        offset: np.ndarray,
        save_data: bool,
        need_plan: bool,
        left_path: list | None = None,
        right_path: list | None = None,
    ) -> dict:
        os.environ["ROBOTWIN_DENOISER"] = self.cli.denoiser if save_data else self.cli.search_denoiser
        args = copy.deepcopy(self.runtime.config)
        args.update(
            {
                "task_name": TASK_NAME,
                "save_path": str(save_dir),
                "now_ep_num": int(episode_id),
                "seed": int(seed),
                "render_freq": 0,
                "save_data": bool(save_data),
                "need_plan": bool(need_plan),
                "initial_bottle_xy_offset": np.asarray(offset, dtype=np.float64).tolist(),
            }
        )
        if left_path is not None:
            args["left_joint_path"] = copy.deepcopy(left_path)
        if right_path is not None:
            args["right_joint_path"] = copy.deepcopy(right_path)
        return args

    def close_task(self, task, clear_cache: bool = False) -> None:
        if task is None:
            return
        try:
            task.close_env(clear_cache=clear_cache)
        except Exception:
            try:
                task.close()
            except Exception:
                pass

    def next_clear_cache(self) -> bool:
        self.simulation_count += 1
        frequency = max(1, int(self.runtime.config.get("clear_cache_freq", 5)))
        return self.simulation_count % frequency == 0

    def scene_record(self, task) -> dict:
        return {
            "qpose_tag": int(task.qpose_tag),
            "arm": "right" if int(task.qpose_tag) == 1 else "left",
            "object_model_id": int(task.model_id),
            "spawn_pose": pose_list(task.bottle_spawn_pose),
            "injected_pose": pose_list(task.bottle_injected_pose),
            "settled_pose": pose_list(task.bottle_settled_pose),
            "offset_applied": [float(value) for value in task.bottle_initial_xy_offset_applied],
        }

    def assert_same_scene(self, base: dict, observed: dict) -> None:
        if base["qpose_tag"] != observed["qpose_tag"]:
            raise CandidateInvalid("qpose_tag changed while replaying the same seed")
        if base["object_model_id"] != observed["object_model_id"]:
            raise CandidateInvalid("object model changed while replaying the same seed")
        for key in ("position", "quaternion_wxyz"):
            if not np.allclose(base["spawn_pose"][key], observed["spawn_pose"][key], atol=1e-7, rtol=0):
                raise CandidateInvalid(f"spawn pose {key} changed while replaying the same seed")

    def plan_base(self, seed: int, pair_id: int) -> dict:
        task = self.task
        clear_cache = False
        try:
            args = self.episode_args(
                seed=seed,
                episode_id=pair_id,
                save_dir=self.output_dir / ".search",
                offset=np.zeros(2),
                save_data=False,
                need_plan=True,
            )
            task.setup_demo(**args)
            task.play_once()
            plan_success = bool(task.plan_success)
            check_success = bool(task.check_success()) if plan_success else False
            if not plan_success or not check_success:
                raise SeedRejected(
                    f"base expert failed: plan_success={plan_success}, check_success={check_success}"
                )
            left_path = copy.deepcopy(task.left_joint_path)
            right_path = copy.deepcopy(task.right_joint_path)
            payload = trajectory_payload(left_path, right_path)
            scene = self.scene_record(task)
            scene.update(
                {
                    "base_seed": int(seed),
                    "trajectory_sha256": sha256_bytes(payload),
                    "left_path_segments": len(left_path),
                    "right_path_segments": len(right_path),
                    "base_final_actor_pose": pose_list(task.bottle.get_pose()),
                    "base_final_functional_point": np.asarray(
                        task.bottle.get_functional_point(0), dtype=np.float64
                    ).tolist(),
                }
            )
            return {
                "left_path": left_path,
                "right_path": right_path,
                "trajectory_payload": payload,
                "scene": scene,
            }
        finally:
            clear_cache = self.next_clear_cache()
            self.close_task(task, clear_cache=clear_cache)

    def replay_candidate(
        self,
        *,
        base: dict,
        pair_id: int,
        offset: np.ndarray,
        magnitude: float,
        direction: str,
    ) -> dict:
        task = self.task
        try:
            args = self.episode_args(
                seed=base["scene"]["base_seed"],
                episode_id=pair_id,
                save_dir=self.output_dir / ".search",
                offset=offset,
                save_data=False,
                need_plan=False,
                left_path=base["left_path"],
                right_path=base["right_path"],
            )
            task.setup_demo(**args)
            scene = self.scene_record(task)
            self.assert_same_scene(base["scene"], scene)
            task.play_once()
            plan_success = bool(task.plan_success)
            complete = (
                int(task.left_cnt) == len(base["left_path"])
                and int(task.right_cnt) == len(base["right_path"])
            )
            if not plan_success or not complete:
                raise CandidateInvalid(
                    "stored trajectory did not complete "
                    f"(plan_success={plan_success}, left={task.left_cnt}/{len(base['left_path'])}, "
                    f"right={task.right_cnt}/{len(base['right_path'])})"
                )
            success = bool(task.check_success())
            return {
                "direction": direction,
                "magnitude": float(magnitude),
                "offset": np.asarray(offset, dtype=np.float64).tolist(),
                "plan_success": plan_success,
                "trajectory_complete": complete,
                "check_success": success,
                "scene": scene,
                "final_actor_pose": pose_list(task.bottle.get_pose()),
                "final_functional_point": np.asarray(
                    task.bottle.get_functional_point(0), dtype=np.float64
                ).tolist(),
            }
        finally:
            self.close_task(task, clear_cache=self.next_clear_cache())

    def find_boundary(self, base: dict, pair_id: int, direction_name: str, vector: np.ndarray) -> dict:
        evaluations: list[dict] = []
        zero = self.replay_candidate(
            base=base,
            pair_id=pair_id,
            offset=np.zeros(2),
            magnitude=0.0,
            direction=direction_name,
        )
        evaluations.append(zero)
        if not zero["check_success"]:
            raise SeedRejected("the planned base trajectory failed during zero-offset replay")

        low_magnitude = 0.0
        low_record = zero
        high_magnitude = None
        high_record = None
        steps = int(round(self.cli.max_magnitude / self.cli.coarse_step))
        for step in range(1, steps + 1):
            magnitude = min(step * self.cli.coarse_step, self.cli.max_magnitude)
            record = self.replay_candidate(
                base=base,
                pair_id=pair_id,
                offset=vector * magnitude,
                magnitude=magnitude,
                direction=direction_name,
            )
            evaluations.append(record)
            if record["check_success"]:
                low_magnitude = magnitude
                low_record = record
            else:
                high_magnitude = magnitude
                high_record = record
                break

        if high_magnitude is None or high_record is None:
            raise SeedRejected(
                f"no failure found in direction {direction_name} through {self.cli.max_magnitude:.3f} m"
            )

        while high_magnitude - low_magnitude > self.cli.boundary_tolerance:
            midpoint = (low_magnitude + high_magnitude) / 2.0
            record = self.replay_candidate(
                base=base,
                pair_id=pair_id,
                offset=vector * midpoint,
                magnitude=midpoint,
                direction=direction_name,
            )
            evaluations.append(record)
            if record["check_success"]:
                low_magnitude = midpoint
                low_record = record
            else:
                high_magnitude = midpoint
                high_record = record

        return {
            "success": low_record,
            "failure": high_record,
            "boundary_gap": float(high_magnitude - low_magnitude),
            "evaluations": evaluations,
        }

    def record_episode(
        self,
        *,
        base: dict,
        pair_id: int,
        label: str,
        expected_success: bool,
        candidate: dict,
        stage_dir: Path,
    ) -> dict:
        save_dir = stage_dir / label
        task = self.task
        closed = False
        try:
            args = self.episode_args(
                seed=base["scene"]["base_seed"],
                episode_id=pair_id,
                save_dir=save_dir,
                offset=np.asarray(candidate["offset"], dtype=np.float64),
                save_data=True,
                need_plan=False,
                left_path=base["left_path"],
                right_path=base["right_path"],
            )
            task.setup_demo(**args)
            scene = self.scene_record(task)
            self.assert_same_scene(base["scene"], scene)
            task.play_once()
            plan_success = bool(task.plan_success)
            complete = (
                int(task.left_cnt) == len(base["left_path"])
                and int(task.right_cnt) == len(base["right_path"])
            )
            actual_success = bool(task.check_success()) if plan_success else False
            final_actor_pose = pose_list(task.bottle.get_pose())
            final_functional_point = np.asarray(
                task.bottle.get_functional_point(0), dtype=np.float64
            ).tolist()
            if not plan_success or not complete or actual_success != expected_success:
                raise CandidateInvalid(
                    f"recorded {label} did not reproduce its label: plan_success={plan_success}, "
                    f"complete={complete}, check_success={actual_success}, expected={expected_success}"
                )

            self.close_task(task, clear_cache=self.next_clear_cache())
            closed = True
            task.merge_pkl_to_hdf5_video()
            task.remove_data_cache()

            hdf5_path = save_dir / "data" / f"episode{pair_id}.hdf5"
            video_path = save_dir / "video" / f"episode{pair_id}.mp4"
            if not hdf5_path.is_file() or not video_path.is_file():
                raise FileNotFoundError(f"recording did not create {hdf5_path} and {video_path}")

            frame_count = annotate_and_inspect_hdf5(
                hdf5_path,
                pair_id=pair_id,
                label=label,
                seed=base["scene"]["base_seed"],
                direction=candidate["direction"],
                offset=candidate["offset"],
                trajectory_sha256=base["scene"]["trajectory_sha256"],
                render_denoiser=self.cli.denoiser,
            )
            if video_path.stat().st_size <= 0:
                raise ValueError(f"empty video: {video_path}")

            return {
                "label": label,
                "direction": candidate["direction"],
                "magnitude": float(candidate["magnitude"]),
                "offset": [float(value) for value in candidate["offset"]],
                "plan_success": plan_success,
                "trajectory_complete": complete,
                "check_success": actual_success,
                "scene": scene,
                "final_actor_pose": final_actor_pose,
                "final_functional_point": final_functional_point,
                "frame_count": int(frame_count),
                "hdf5_stage_path": str(hdf5_path),
                "video_stage_path": str(video_path),
                "hdf5_size_bytes": hdf5_path.stat().st_size,
                "video_size_bytes": video_path.stat().st_size,
            }
        finally:
            if not closed:
                self.close_task(task, clear_cache=self.next_clear_cache())

    def commit_pair(
        self,
        *,
        pair_id: int,
        base: dict,
        boundary: dict,
        success_record: dict,
        failure_record: dict,
        stage_dir: Path,
    ) -> dict:
        final_paths = {}
        for label, record in (("success", success_record), ("failure", failure_record)):
            final_hdf5 = self.output_dir / label / "data" / f"episode{pair_id}.hdf5"
            final_video = self.output_dir / label / "video" / f"episode{pair_id}.mp4"
            final_hdf5.parent.mkdir(parents=True, exist_ok=True)
            final_video.parent.mkdir(parents=True, exist_ok=True)
            os.replace(record.pop("hdf5_stage_path"), final_hdf5)
            os.replace(record.pop("video_stage_path"), final_video)
            record["hdf5_path"] = relative_to_output(final_hdf5, self.output_dir)
            record["video_path"] = relative_to_output(final_video, self.output_dir)
            final_paths[label] = (final_hdf5, final_video)

        trajectory_path = self.output_dir / "base_trajectories" / f"pair_{pair_id:06d}.pkl"
        atomic_write_bytes(trajectory_path, base["trajectory_payload"])
        trajectory_file_sha = sha256_file(trajectory_path)

        record = {
            "pair_id": int(pair_id),
            "created_at": now_iso(),
            "task_name": TASK_NAME,
            "base_seed": int(base["scene"]["base_seed"]),
            "direction": success_record["direction"],
            "boundary_gap": float(boundary["boundary_gap"]),
            "base": base["scene"],
            "trajectory_path": relative_to_output(trajectory_path, self.output_dir),
            "trajectory_file_sha256": trajectory_file_sha,
            "success": success_record,
            "failure": failure_record,
            "search_evaluations": len(boundary["evaluations"]),
            "robotwin_commit_sha": self.runtime.robotwin_commit,
            "task_config": str(self.cli.config.resolve()),
            "task_config_sha256": self.runtime.config_sha256,
            "training_render_denoiser": self.cli.denoiser,
            "search_render_denoiser": self.cli.search_denoiser,
        }
        record_path = self.output_dir / "records" / f"pair_{pair_id:06d}.json"
        atomic_write_json(record_path, record)
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        return record

    def write_attempt(self, seed: int, pair_id: int, status: str, **extra) -> None:
        path = self.output_dir / "attempts" / f"seed_{seed:08d}_pair_{pair_id:06d}.json"
        value = {
            "created_at": now_iso(),
            "seed": int(seed),
            "pair_id": int(pair_id),
            "status": status,
            **extra,
        }
        atomic_write_json(path, value)

    def write_error(self, seed: int, pair_id: int, stage: str, exc: Exception) -> None:
        self.error_count += 1
        path = self.output_dir / "errors" / f"error_{self.error_count:06d}_seed_{seed:08d}.json"
        atomic_write_json(
            path,
            {
                "created_at": now_iso(),
                "seed": int(seed),
                "pair_id": int(pair_id),
                "stage": stage,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )

    def collect(self) -> list[dict]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        records = load_records(self.output_dir)
        if records and not self.cli.resume:
            raise SystemExit(
                f"{self.output_dir} already contains {len(records)} committed pairs; pass --resume to continue"
            )
        validate_record_sequence(records)
        pair_id = len(records)
        if pair_id >= self.cli.num_pairs:
            rebuild_indexes(self.output_dir, records, self.cli.num_pairs)
            return records

        used_seeds = {int(record["base_seed"]) for record in records}
        seed = max([self.cli.start_seed - 1, *used_seeds]) + 1
        seed_attempts = 0
        consecutive_error_signature = None
        consecutive_error_count = 0
        while pair_id < self.cli.num_pairs:
            if seed_attempts >= self.cli.max_seed_attempts:
                raise SystemExit(
                    f"exhausted --max-seed-attempts={self.cli.max_seed_attempts} with {pair_id} pairs"
                )
            if seed in used_seeds:
                seed += 1
                continue
            seed_attempts += 1
            direction_name, direction_vector = DIRECTIONS[pair_id % len(DIRECTIONS)]
            stage_dir = self.output_dir / ".staging" / f"pair_{pair_id:06d}"
            if stage_dir.exists():
                shutil.rmtree(stage_dir)
            stage_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[pair {pair_id:03d}/{self.cli.num_pairs - 1:03d}] "
                f"seed={seed} direction={direction_name}",
                flush=True,
            )
            try:
                base = self.plan_base(seed, pair_id)
                boundary = self.find_boundary(base, pair_id, direction_name, direction_vector)
                success_record = self.record_episode(
                    base=base,
                    pair_id=pair_id,
                    label="success",
                    expected_success=True,
                    candidate=boundary["success"],
                    stage_dir=stage_dir,
                )
                failure_record = self.record_episode(
                    base=base,
                    pair_id=pair_id,
                    label="failure",
                    expected_success=False,
                    candidate=boundary["failure"],
                    stage_dir=stage_dir,
                )
                if success_record["frame_count"] != failure_record["frame_count"]:
                    raise CandidateInvalid(
                        "paired recordings have different frame counts: "
                        f"{success_record['frame_count']} != {failure_record['frame_count']}"
                    )
                record = self.commit_pair(
                    pair_id=pair_id,
                    base=base,
                    boundary=boundary,
                    success_record=success_record,
                    failure_record=failure_record,
                    stage_dir=stage_dir,
                )
                records.append(record)
                rebuild_indexes(self.output_dir, records, self.cli.num_pairs)
                self.write_attempt(
                    seed,
                    pair_id,
                    "accepted",
                    direction=direction_name,
                    success_magnitude=record["success"]["magnitude"],
                    failure_magnitude=record["failure"]["magnitude"],
                )
                print(
                    f"[accepted] pair={pair_id} seed={seed} "
                    f"success={record['success']['magnitude']:.4f}m "
                    f"failure={record['failure']['magnitude']:.4f}m",
                    flush=True,
                )
                pair_id += 1
                used_seeds.add(seed)
                consecutive_error_signature = None
                consecutive_error_count = 0
            except (SeedRejected, CandidateInvalid, self.runtime.unstable_error) as exc:
                self.write_attempt(seed, pair_id, "rejected", direction=direction_name, reason=str(exc))
                print(f"[rejected] pair={pair_id} seed={seed}: {exc}", flush=True)
                consecutive_error_signature = None
                consecutive_error_count = 0
            except Exception as exc:
                self.write_error(seed, pair_id, "collection", exc)
                print(f"[error] pair={pair_id} seed={seed}: {type(exc).__name__}: {exc}", flush=True)
                signature = (type(exc).__name__, str(exc))
                if signature == consecutive_error_signature:
                    consecutive_error_count += 1
                else:
                    consecutive_error_signature = signature
                    consecutive_error_count = 1
                if isinstance(exc, (ImportError, ModuleNotFoundError)):
                    raise RuntimeError("fatal simulator import error; refusing to try more seeds") from exc
                if consecutive_error_count >= self.cli.max_consecutive_errors:
                    raise RuntimeError(
                        f"the same simulator error occurred {consecutive_error_count} times; "
                        "refusing to try more seeds"
                    ) from exc
            finally:
                if stage_dir.exists():
                    shutil.rmtree(stage_dir)
            seed += 1

        return records


def annotate_and_inspect_hdf5(
    path: Path,
    *,
    pair_id: int,
    label: str,
    seed: int,
    direction: str,
    offset: list[float],
    trajectory_sha256: str,
    render_denoiser: str,
) -> int:
    import h5py

    with h5py.File(path, "r+") as file:
        required = ("joint_action/vector", "endpose/left_endpose", "endpose/right_endpose")
        missing = [key for key in required if key not in file]
        if missing:
            raise KeyError(f"{path} missing datasets: {missing}")
        frame_count = int(file["joint_action/vector"].shape[0])
        if frame_count <= 0:
            raise ValueError(f"{path} contains no frames")
        file.attrs["task_name"] = TASK_NAME
        file.attrs["pair_id"] = int(pair_id)
        file.attrs["label"] = label
        file.attrs["success"] = bool(label == "success")
        file.attrs["seed"] = int(seed)
        file.attrs["direction"] = direction
        file.attrs["initial_bottle_xy_offset"] = np.asarray(offset, dtype=np.float64)
        file.attrs["base_trajectory_sha256"] = trajectory_sha256
        file.attrs["render_denoiser"] = render_denoiser
        return frame_count


def load_records(output_dir: Path) -> list[dict]:
    record_dir = output_dir / "records"
    records = []
    for path in sorted(record_dir.glob("pair_*.json")) if record_dir.exists() else []:
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def validate_record_sequence(records: list[dict]) -> None:
    pair_ids = [int(record["pair_id"]) for record in records]
    expected = list(range(len(records)))
    if pair_ids != expected:
        raise ValueError(f"record pair ids must be contiguous: got {pair_ids}, expected {expected}")


def rebuild_indexes(output_dir: Path, records: list[dict], target_pairs: int) -> None:
    validate_record_sequence(records)
    metadata = []
    for record in records:
        for label in ("success", "failure"):
            sample = record[label]
            metadata.append(
                {
                    "episode_index": int(record["pair_id"]),
                    "pair_id": int(record["pair_id"]),
                    "label": label,
                    "task": TASK_NAME,
                    "seed": int(record["base_seed"]),
                    "direction": record["direction"],
                    "offset": sample["offset"],
                    "length": int(sample["frame_count"]),
                    "hdf5": sample["hdf5_path"],
                    "video": sample["video_path"],
                    "trajectory": record["trajectory_path"],
                }
            )
    atomic_write_jsonl(output_dir / "pairs.jsonl", records)
    atomic_write_jsonl(output_dir / "metadata.jsonl", metadata)
    atomic_write_json(
        output_dir / "summary.json",
        {
            "updated_at": now_iso(),
            "task_name": TASK_NAME,
            "target_pairs": int(target_pairs),
            "completed_pairs": len(records),
            "success_samples": len(records),
            "failure_samples": len(records),
            "direction_counts": dict(Counter(record["direction"] for record in records)),
            "training_render_denoisers": dict(
                Counter(record.get("training_render_denoiser", "unknown") for record in records)
            ),
        },
    )


def validate_dataset(output_dir: Path, expected_pairs: int | None = None) -> dict:
    import h5py

    output_dir = output_dir.resolve()
    records = load_records(output_dir)
    validate_record_sequence(records)
    if expected_pairs is not None and len(records) != expected_pairs:
        raise ValueError(f"expected {expected_pairs} pairs, found {len(records)}")

    seen_seeds = set()
    directions = Counter()
    for record in records:
        pair_id = int(record["pair_id"])
        seed = int(record["base_seed"])
        if seed in seen_seeds:
            raise ValueError(f"seed {seed} appears in more than one pair")
        seen_seeds.add(seed)
        directions[record["direction"]] += 1
        if record["success"]["check_success"] is not True:
            raise ValueError(f"pair {pair_id} success is not labeled successful")
        if record["failure"]["check_success"] is not False:
            raise ValueError(f"pair {pair_id} failure is not labeled failed")
        if not record["success"]["trajectory_complete"] or not record["failure"]["trajectory_complete"]:
            raise ValueError(f"pair {pair_id} has an incomplete trajectory")
        if record["success"]["frame_count"] != record["failure"]["frame_count"]:
            raise ValueError(f"pair {pair_id} frame counts differ")

        trajectory = output_dir / record["trajectory_path"]
        if not trajectory.is_file() or sha256_file(trajectory) != record["trajectory_file_sha256"]:
            raise ValueError(f"pair {pair_id} trajectory is missing or has the wrong hash")
        for label in ("success", "failure"):
            sample = record[label]
            hdf5_path = output_dir / sample["hdf5_path"]
            video_path = output_dir / sample["video_path"]
            if not hdf5_path.is_file() or not video_path.is_file() or video_path.stat().st_size <= 0:
                raise ValueError(f"pair {pair_id} {label} files are incomplete")
            with h5py.File(hdf5_path, "r") as file:
                if int(file.attrs["pair_id"]) != pair_id or str(file.attrs["label"]) != label:
                    raise ValueError(f"pair {pair_id} {label} HDF5 attributes do not match")
                if str(file.attrs.get("render_denoiser", "")) != record.get(
                    "training_render_denoiser", ""
                ):
                    raise ValueError(f"pair {pair_id} {label} denoiser metadata does not match")
                frames = int(file["joint_action/vector"].shape[0])
                if frames != int(sample["frame_count"]):
                    raise ValueError(f"pair {pair_id} {label} frame count changed")

    if expected_pairs == 100:
        expected_directions = {name: 25 for name, _ in DIRECTIONS}
        if dict(directions) != expected_directions:
            raise ValueError(f"direction distribution is {dict(directions)}, expected {expected_directions}")

    rebuild_indexes(output_dir, records, expected_pairs if expected_pairs is not None else len(records))
    return {
        "pairs": len(records),
        "success_samples": len(records),
        "failure_samples": len(records),
        "direction_counts": dict(directions),
        "total_bytes": sum(
            (output_dir / record[label][kind]).stat().st_size
            for record in records
            for label in ("success", "failure")
            for kind in ("hdf5_path", "video_path")
        ),
    }


def load_runtime(cli: argparse.Namespace) -> Runtime:
    if not ROBOTWIN_ROOT.is_dir():
        raise FileNotFoundError(f"missing RoboTwin checkout: {ROBOTWIN_ROOT}")
    if not cli.config.is_file():
        raise FileNotFoundError(f"missing task config: {cli.config}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cli.gpu_id)
    sys.path.insert(0, str(ROBOTWIN_ROOT))
    os.chdir(ROBOTWIN_ROOT)

    from script import collect_data as robotwin_collect

    robotwin_collect.prepare_denoiser_runtime(cli.denoiser, cli.oidn_library_dir)
    robotwin_collect.import_runtime_dependencies()

    config = yaml.safe_load(cli.config.read_text(encoding="utf-8"))
    embodiment_type = config.get("embodiment")
    embodiment_config_path = ROBOTWIN_ROOT / "task_config" / "_embodiment_config.yml"
    embodiment_types = yaml.safe_load(embodiment_config_path.read_text(encoding="utf-8"))

    def embodiment_file(name: str) -> str:
        path = embodiment_types[name]["file_path"]
        if path is None:
            raise ValueError(f"embodiment {name} has no file_path")
        return path

    if len(embodiment_type) == 1:
        config["left_robot_file"] = embodiment_file(embodiment_type[0])
        config["right_robot_file"] = embodiment_file(embodiment_type[0])
        config["dual_arm_embodied"] = True
        config["embodiment_name"] = str(embodiment_type[0])
    elif len(embodiment_type) == 3:
        config["left_robot_file"] = embodiment_file(embodiment_type[0])
        config["right_robot_file"] = embodiment_file(embodiment_type[1])
        config["embodiment_dis"] = embodiment_type[2]
        config["dual_arm_embodied"] = False
        config["embodiment_name"] = f"{embodiment_type[0]}+{embodiment_type[1]}"
    else:
        raise ValueError("embodiment must contain one or three entries")

    config["left_embodiment_config"] = robotwin_collect.get_embodiment_config(config["left_robot_file"])
    config["right_embodiment_config"] = robotwin_collect.get_embodiment_config(config["right_robot_file"])
    config["task_name"] = TASK_NAME
    return Runtime(
        task_factory=lambda: robotwin_collect.class_decorator(TASK_NAME),
        config=config,
        unstable_error=robotwin_collect.UnStableError,
        robotwin_commit=file_sha_from_git(ROBOTWIN_ROOT),
        config_sha256=sha256_file(cli.config),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-pairs", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpu-id", type=int, default=1)
    parser.add_argument("--denoiser", choices=("oidn", "optix", "none"), default="oidn")
    parser.add_argument("--search-denoiser", choices=("oidn", "optix", "none"), default="none")
    parser.add_argument("--oidn-library-dir", type=Path)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--max-seed-attempts", type=int, default=10000)
    parser.add_argument("--max-consecutive-errors", type=int, default=3)
    parser.add_argument("--coarse-step", type=float, default=0.01)
    parser.add_argument("--max-magnitude", type=float, default=0.12)
    parser.add_argument("--boundary-tolerance", type=float, default=0.001)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    # Resolve before load_runtime changes cwd to the RoboTwin checkout.
    args.output_dir = args.output_dir.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    if args.num_pairs <= 0:
        parser.error("--num-pairs must be positive")
    if args.max_seed_attempts <= 0 or args.max_consecutive_errors <= 0:
        parser.error("attempt limits must be positive")
    if args.coarse_step <= 0 or args.max_magnitude <= 0 or args.boundary_tolerance <= 0:
        parser.error("search distances must be positive")
    if args.coarse_step > args.max_magnitude:
        parser.error("--coarse-step cannot exceed --max-magnitude")
    return args


def main() -> int:
    cli = parse_args()
    if cli.validate_only:
        result = validate_dataset(cli.output_dir, cli.num_pairs)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    runtime = load_runtime(cli)
    collector = Collector(cli, runtime)
    records = collector.collect()
    result = validate_dataset(cli.output_dir, cli.num_pairs)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[complete] collected and validated {len(records)} paired episodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
