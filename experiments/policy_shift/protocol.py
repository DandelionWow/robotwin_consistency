"""Pure protocol, validation, and aggregation helpers.

This module intentionally does not import torch, DiffSynth, or WorldArena so the
window protocol can be validated on a CPU-only machine before model loading.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


METRIC_DIRECTIONS = {
    "psnr": "higher",
    "ssim": "higher",
    "jepa": "higher",
    "trajectory_accuracy": "higher",
}


@dataclass(frozen=True)
class WindowSpec:
    """A contiguous GT-history window followed by its contiguous GT future."""

    window_start: int
    history_indices: tuple[int, ...]
    future_indices: tuple[int, ...]

    @property
    def all_indices(self) -> tuple[int, ...]:
        return self.history_indices + self.future_indices


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_metadata(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise TypeError(f"Expected an object at {path}:{line_number}")
                rows.append(row)
        return rows

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise TypeError(f"Expected a list of objects in {path}")
    return payload


def build_windows(
    episode_length: int,
    history_frames: int,
    future_frames: int,
    stride: int,
) -> list[WindowSpec]:
    """Enumerate strictly aligned, non-padded GT-history windows."""

    for name, value in (
        ("episode_length", episode_length),
        ("history_frames", history_frames),
        ("future_frames", future_frames),
        ("stride", stride),
    ):
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    total = int(history_frames) + int(future_frames)
    if total > int(episode_length):
        return []

    windows = []
    for start in range(0, int(episode_length) - total + 1, int(stride)):
        history = tuple(range(start, start + int(history_frames)))
        future = tuple(range(start + int(history_frames), start + total))
        windows.append(WindowSpec(start, history, future))
    return windows


def validate_wan_window_shape(history_frames: int, future_frames: int) -> None:
    total = int(history_frames) + int(future_frames)
    if history_frames < 1 or future_frames < 1:
        raise ValueError("history_frames and future_frames must both be positive")
    if (history_frames - 1) % 4 != 0:
        raise ValueError(
            "Wan history length must satisfy (history_frames - 1) % 4 == 0; "
            f"got {history_frames}"
        )
    if (total - 1) % 4 != 0:
        raise ValueError(
            "Wan total length must satisfy (history_frames + future_frames - 1) % 4 == 0; "
            f"got {total}"
        )


def resolve_data_path(dataset_base: Path, value: Any, field: str) -> Path:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Minimal evaluator supports one view; {field} has {len(value)} entries")
        value = value[0]
    if isinstance(value, dict):
        value = value.get("data")
    if not value:
        raise KeyError(f"Metadata row is missing {field!r}")
    path = Path(value)
    return path if path.is_absolute() else dataset_base / path


def inspect_episode_alignment(dataset_base: Path, row: dict[str, Any]) -> dict[str, Any]:
    """Require metadata, video, and Parquet to describe the same frame interval."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Alignment validation requires pyarrow") from exc

    video_path = resolve_data_path(dataset_base, row.get("video"), "video")
    action_path = resolve_data_path(dataset_base, row.get("action"), "action")
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if not action_path.is_file():
        raise FileNotFoundError(action_path)

    start_frame = int(row.get("start_frame", 0))
    declared_length = row.get("length")
    end_frame = row.get("end_frame")
    if declared_length is None and end_frame is None:
        raise KeyError("Metadata row must contain length or end_frame")
    if declared_length is None:
        declared_length = int(end_frame) - start_frame + 1
    declared_length = int(declared_length)
    if end_frame is None:
        end_frame = start_frame + declared_length - 1
    end_frame = int(end_frame)
    if end_frame - start_frame + 1 != declared_length:
        raise ValueError(
            f"Metadata range [{start_frame}, {end_frame}] disagrees with length={declared_length}"
        )

    try:
        import imageio.v2 as imageio

        reader = imageio.get_reader(video_path)
        try:
            video_frames = int(reader.count_frames())
        finally:
            reader.close()
    except (AttributeError, ImportError, OSError, RuntimeError, ValueError):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "Video alignment needs either an ImageIO ffmpeg backend or OpenCV"
            ) from exc
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video for frame-count validation: {video_path}")
        video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if video_frames <= 0:
            raise ValueError(f"Invalid video frame count {video_frames}: {video_path}")
    parquet_file = pq.ParquetFile(action_path)
    parquet_rows = int(parquet_file.metadata.num_rows)
    if "observation.state" not in parquet_file.schema_arrow.names:
        raise KeyError(f"Parquet is missing observation.state: {action_path}")

    required_end = end_frame + 1
    if required_end > video_frames:
        raise ValueError(
            f"Video has {video_frames} frames but metadata requires through frame {end_frame}: {video_path}"
        )
    if required_end > parquet_rows:
        raise ValueError(
            f"Parquet has {parquet_rows} rows but metadata requires through row {end_frame}: {action_path}"
        )

    state_column = pq.read_table(action_path, columns=["observation.state"])["observation.state"]
    for frame_index in {start_frame, end_frame}:
        width = len(state_column[frame_index].as_py())
        if width != 26:
            raise ValueError(
                f"Expected 26D observation.state at row {frame_index}, got {width}: {action_path}"
            )

    return {
        "video_path": str(video_path.resolve()),
        "action_path": str(action_path.resolve()),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "length": declared_length,
        "video_frames": video_frames,
        "parquet_rows": parquet_rows,
    }


def load_and_validate_reference_stat(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the exact stat branch used by BWM LoadCobotAction(eef_abs)."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    entry = payload.get("state_pose") or payload.get("eef_abs")
    if not isinstance(entry, dict):
        raise KeyError(f"{path} has neither a state_pose nor eef_abs statistics object")

    if entry.get("p01") is not None and entry.get("p99") is not None:
        lower_key, upper_key = "p01", "p99"
    elif entry.get("min") is not None and entry.get("max") is not None:
        lower_key, upper_key = "min", "max"
    else:
        raise KeyError(f"{path} state_pose/eef_abs lacks p01+p99 and min+max")

    lower = np.asarray(entry[lower_key], dtype=np.float64)
    upper = np.asarray(entry[upper_key], dtype=np.float64)
    if lower.shape != (14,) or upper.shape != (14,):
        raise ValueError(
            f"Expected 14D {lower_key}/{upper_key} in {path}, got {lower.shape}/{upper.shape}"
        )
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError(f"Non-finite normalization bounds in {path}")
    if np.any(upper <= lower):
        bad = np.flatnonzero(upper <= lower).tolist()
        raise ValueError(f"Non-increasing normalization bounds at dimensions {bad} in {path}")

    provenance = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "entry": "state_pose" if "state_pose" in payload else "eef_abs",
        "lower_key": lower_key,
        "upper_key": upper_key,
        "dimensions": 14,
    }
    return payload, provenance


def _finite_metric_values(records: Sequence[dict[str, Any]], metric: str) -> list[float]:
    values = []
    for record in records:
        value = record.get("metrics", {}).get(metric)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def _bootstrap_mean_ci(
    values: Sequence[float],
    rng: np.random.Generator,
    samples: int,
    confidence: float,
) -> list[float] | None:
    if len(values) == 0:
        return None
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 1 or samples <= 0:
        value = float(array[0])
        return [value, value]
    means = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        means[index] = rng.choice(array, size=len(array), replace=True).mean()
    alpha = 1.0 - float(confidence)
    return [
        float(np.quantile(means, alpha / 2.0)),
        float(np.quantile(means, 1.0 - alpha / 2.0)),
    ]


def summarize_records(
    records: Sequence[dict[str, Any]],
    bootstrap_samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    metrics = {}
    for metric, direction in METRIC_DIRECTIONS.items():
        values = _finite_metric_values(records, metric)
        if not values:
            metrics[metric] = {
                "n": 0,
                "mean": None,
                "median": None,
                "std": None,
                "bootstrap_mean_ci": None,
                "direction": direction,
            }
            continue
        array = np.asarray(values, dtype=np.float64)
        metrics[metric] = {
            "n": len(values),
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
            "bootstrap_mean_ci": _bootstrap_mean_ci(array, rng, bootstrap_samples, confidence),
            "direction": direction,
        }
    return {"windows": len(records), "metrics": metrics}


def _group_records(
    records: Sequence[dict[str, Any]], fields: Iterable[str]
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    fields = tuple(fields)
    for record in records:
        key = "|".join(f"{field}={record.get(field)}" for field in fields)
        result.setdefault(key, []).append(record)
    return result


def _summarize_groups(
    records: Sequence[dict[str, Any]],
    fields: Iterable[str],
    bootstrap_samples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    return {
        key: summarize_records(group, bootstrap_samples, confidence, seed + index)
        for index, (key, group) in enumerate(sorted(_group_records(records, fields).items()))
    }


def _expert_policy_gaps(distribution_summary: dict[str, Any]) -> dict[str, Any]:
    expert = distribution_summary.get("distribution=expert")
    policy = distribution_summary.get("distribution=policy")
    gaps = {}
    for metric, direction in METRIC_DIRECTIONS.items():
        expert_mean = None if expert is None else expert["metrics"][metric]["mean"]
        policy_mean = None if policy is None else policy["metrics"][metric]["mean"]
        if expert_mean is None or policy_mean is None:
            gaps[metric] = {
                "raw_gap_policy_minus_expert": None,
                "gap_error": None,
                "direction": direction,
            }
            continue
        raw_gap = float(policy_mean - expert_mean)
        gaps[metric] = {
            "raw_gap_policy_minus_expert": raw_gap,
            "gap_error": -raw_gap if direction == "higher" else raw_gap,
            "direction": direction,
        }
    return gaps


def aggregate_records(
    records: Sequence[dict[str, Any]],
    bootstrap_samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    records = list(records)
    kwargs = (bootstrap_samples, confidence, seed)
    per_distribution = _summarize_groups(records, ("distribution",), *kwargs)
    policy_records = [record for record in records if record.get("distribution") == "policy"]
    return {
        "protocol_mode": "gt_history",
        "metric_direction": METRIC_DIRECTIONS,
        "gap_error_definition": (
            "For higher-is-better metrics, gap_error = expert_mean - policy_mean; "
            "positive always means policy windows are worse."
        ),
        "overall": summarize_records(records, *kwargs),
        "per_episode": _summarize_groups(
            records, ("task", "episode_id", "distribution", "policy_id"), *kwargs
        ),
        "per_task": _summarize_groups(records, ("task",), *kwargs),
        "per_distribution": per_distribution,
        "per_policy": _summarize_groups(records, ("policy_id",), *kwargs),
        "policy_success_failure": _summarize_groups(policy_records, ("success",), *kwargs),
        "expert_vs_policy": _expert_policy_gaps(per_distribution),
    }
