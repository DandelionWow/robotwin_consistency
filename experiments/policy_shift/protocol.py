"""Pure protocol, validation, and aggregation helpers.

This module intentionally does not import torch, DiffSynth, or WorldArena so the
window protocol can be validated on a CPU-only machine before model loading.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
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

FORMAL_PROVENANCE_FIELDS = (
    "seed_mapping_status",
    "failure_label_status",
    "policy_checkpoint_status",
    "task_config_status",
)
PAIRWISE_METRICS = (
    "psnr",
    "ssim",
    "trajectory_accuracy",
    "depth_accuracy",
    "subject_consistency",
    "image_quality",
    "aesthetic_quality",
)
SAMPLE_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]*__seed-?[0-9]+__(expert|pi05)__(early|middle|late)__r[0-9]+$"
)


@dataclass(frozen=True)
class WindowSpec:
    """A contiguous GT-history window followed by its contiguous GT future."""

    window_start: int
    history_indices: tuple[int, ...]
    future_indices: tuple[int, ...]
    slot: str | None = None

    @property
    def all_indices(self) -> tuple[int, ...]:
        return self.history_indices + self.future_indices


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(checkpoint_dir: Path) -> dict[str, Any]:
    """Hash every inference-relevant file in a PyTorch or Orbax checkpoint.

    The public RoboTwin Pi0.5 checkpoints use an Orbax directory rather than a
    single weight file.  Hashing one OCDBT shard would therefore identify only
    part of the policy.  This function creates a canonical, path-sensitive
    manifest and hashes that manifest to obtain the policy identity.
    """

    checkpoint_dir = Path(checkpoint_dir).resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_dir}")

    pytorch_weight = checkpoint_dir / "model.safetensors"
    params_dir = checkpoint_dir / "params"
    assets_dir = checkpoint_dir / "assets"
    if pytorch_weight.is_file():
        kind = "pytorch_safetensors"
        roots = [pytorch_weight, assets_dir]
    elif params_dir.is_dir():
        kind = "orbax_jax"
        roots = [params_dir, assets_dir]
        metadata = checkpoint_dir / "_CHECKPOINT_METADATA"
        if metadata.is_file():
            roots.append(metadata)
    else:
        raise FileNotFoundError(
            f"Checkpoint has neither model.safetensors nor params/: {checkpoint_dir}"
        )
    if not assets_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint is missing assets/: {checkpoint_dir}")

    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        else:
            files.extend(path for path in root.rglob("*") if path.is_file())
    files = sorted(set(files), key=lambda path: path.relative_to(checkpoint_dir).as_posix())
    if not files:
        raise ValueError(f"Checkpoint contains no inference files: {checkpoint_dir}")
    if any(path.is_symlink() for path in files):
        raise ValueError(f"Checkpoint identity refuses symlinked files: {checkpoint_dir}")

    records = [
        {
            "path": path.relative_to(checkpoint_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "kind": kind,
        "path": str(checkpoint_dir),
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "files": records,
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }


def legacy_episode_formal_eligible(record: dict[str, Any]) -> bool:
    """Return true only when every required legacy provenance field has direct evidence."""

    return all(record.get(field) == "CONFIRMED" for field in FORMAL_PROVENANCE_FIELDS)


def verified_failure_label(label_source: str, success: bool | None) -> str:
    """Prevent a filename token from being promoted to verified ground truth."""

    source = str(label_source).strip().lower()
    if source in {"episode_metadata", "collector_manifest", "task_success_check"}:
        if success is None:
            raise ValueError(f"{label_source} requires an explicit boolean success value")
        return "VERIFIED_FAILURE" if not success else "VERIFIED_SUCCESS"
    if source in {"filename", "filename-labelled", "filename_labeled"}:
        return "FILENAME_LABELLED_FAILURE" if success is False else "UNVERIFIED"
    return "UNKNOWN"


def _canonicalize(value: Any, precision: int) -> Any:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {str(key): _canonicalize(value[key], precision) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        items = list(value)
        if items and all(isinstance(item, dict) and "name" in item for item in items):
            items = sorted(items, key=lambda item: str(item["name"]))
        return [_canonicalize(item, precision) for item in items]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Canonical state records cannot contain NaN or infinity")
        rounded = round(value, precision)
        return 0.0 if rounded == 0.0 else rounded
    raise TypeError(f"Unsupported canonical state value: {type(value).__name__}")


def canonical_state_json(record: dict[str, Any], precision: int = 6) -> str:
    if not isinstance(record, dict):
        raise TypeError("Initial-state record must be a dictionary")
    payload = _canonicalize(record, int(precision))
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def state_fingerprint(record: dict[str, Any], precision: int = 6) -> str:
    return hashlib.sha256(canonical_state_json(record, precision).encode("utf-8")).hexdigest()


def require_matching_initial_state(expert_record: dict[str, Any], policy_record: dict[str, Any]) -> str:
    expert = state_fingerprint(expert_record)
    policy = state_fingerprint(policy_record)
    if expert != policy:
        raise ValueError(f"STATE_MISMATCH: expert={expert} policy={policy}")
    return expert


def validate_reproducible_code_provenance(provenance: dict[str, Any]) -> None:
    """A dirty RoboTwin checkout is reproducible only with its saved patch hash."""

    if not provenance.get("robotwin_commit"):
        raise ValueError("Missing robotwin_commit")
    if provenance.get("robotwin_dirty") and not provenance.get("robotwin_patch_sha256"):
        raise ValueError("Dirty RoboTwin checkout requires robotwin_patch_sha256")


def pair_id(task: str, env_seed: int) -> str:
    task = str(task).strip()
    if not task or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", task):
        raise ValueError(f"Unsafe task name for pair_id: {task!r}")
    return f"{task}__seed{int(env_seed)}"


def validate_formal_seed_plan(
    plan: dict[str, Any],
    *,
    expected_tasks: int = 3,
    seeds_per_task: int = 2,
) -> dict[str, tuple[int, ...]]:
    """Validate and normalize the committed formal matched-collection plan."""

    if not isinstance(plan, dict):
        raise TypeError("Seed plan must be a mapping")
    if plan.get("schema_version") != 1:
        raise ValueError("Seed plan schema_version must be 1")
    if plan.get("status") != "FORMAL":
        raise ValueError("Seed plan status must be FORMAL")
    entries = plan.get("tasks")
    if not isinstance(entries, list) or len(entries) != int(expected_tasks):
        raise ValueError(f"Seed plan must contain exactly {expected_tasks} task entries")

    normalized: dict[str, tuple[int, ...]] = {}
    identifiers: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("Every seed-plan task entry must be a mapping")
        task = str(entry.get("task", "")).strip()
        seeds = entry.get("seeds")
        if task in normalized:
            raise ValueError(f"Duplicate task in seed plan: {task}")
        if not isinstance(seeds, list) or len(seeds) != int(seeds_per_task):
            raise ValueError(
                f"Seed plan task {task!r} must contain exactly {seeds_per_task} seeds"
            )
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
            raise TypeError(f"Seed plan task {task!r} contains a non-integer seed")
        values = tuple(int(seed) for seed in seeds)
        if len(set(values)) != len(values):
            raise ValueError(f"Seed plan task {task!r} contains duplicate seeds")
        for seed in values:
            identifier = pair_id(task, seed)
            if identifier in identifiers:
                raise ValueError(f"Duplicate pair in seed plan: {identifier}")
            identifiers.add(identifier)
        normalized[task] = values
    return normalized


def generation_seed(pair_identifier: str, window_slot: str, repeat_id: int) -> int:
    if window_slot not in {"early", "middle", "late"}:
        raise ValueError(f"Unknown window slot: {window_slot}")
    payload = f"{pair_identifier}\0{window_slot}\0{int(repeat_id)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def worldarena_sample_id(
    pair_identifier: str,
    side: str,
    slot: str,
    repeat_id: int,
) -> str:
    if side not in {"expert", "pi05"}:
        raise ValueError(f"Unknown side: {side}")
    sample_id = f"{pair_identifier}__{side}__{slot}__r{int(repeat_id)}"
    if not SAMPLE_ID_PATTERN.fullmatch(sample_id):
        raise ValueError(f"Invalid WorldArena sample_id: {sample_id}")
    return sample_id


def artifact_directory(
    root: Path,
    pair_identifier: str,
    side: str,
    slot: str,
    repeat_id: int,
    *,
    require_absent: bool = False,
) -> Path:
    sample_id = worldarena_sample_id(pair_identifier, side, slot, repeat_id)
    path = Path(root) / pair_identifier / side / slot / f"r{int(repeat_id)}"
    if require_absent and path.exists():
        raise FileExistsError(f"Refusing to overwrite artifact directory for {sample_id}: {path}")
    return path


def worldarena_summary_row(sample_id: str, gt_path: Path, generated_video: Path) -> dict[str, str]:
    if not SAMPLE_ID_PATTERN.fullmatch(str(sample_id)):
        raise ValueError(f"Invalid WorldArena sample_id: {sample_id}")
    gt_path = Path(gt_path).resolve()
    generated_video = Path(generated_video).resolve()
    return {
        "sample_id": str(sample_id),
        "gt_path": str(gt_path),
        "generated_video": str(generated_video),
    }


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


def select_protocol_windows(
    episode_length: int,
    history_frames: int = 9,
    future_frames: int = 72,
) -> list[WindowSpec]:
    """Select the fixed early/middle/late protocol windows, deduplicated in slot order."""

    episode_length = int(episode_length)
    total = int(history_frames) + int(future_frames)
    if episode_length < total:
        return []
    max_start = episode_length - total
    candidates = (
        ("early", 0),
        ("middle", max_start // 2),
        ("late", max_start),
    )
    windows: list[WindowSpec] = []
    seen: set[int] = set()
    for slot, start in candidates:
        if start in seen:
            continue
        seen.add(start)
        history = tuple(range(start, start + int(history_frames)))
        future = tuple(range(start + int(history_frames), start + total))
        windows.append(WindowSpec(start, history, future, slot=slot))
    return windows


def pair_length_eligibility(
    expert_length: int,
    policy_length: int,
    required_length: int = 81,
) -> dict[str, Any]:
    expert_short = int(expert_length) < int(required_length)
    policy_short = int(policy_length) < int(required_length)
    return {
        "eligible": not (expert_short or policy_short),
        "expert_short": expert_short,
        "policy_short": policy_short,
        "pairwise_excluded": expert_short or policy_short,
    }


def cadence_record(raw_fps: float, bwm_sampling_stride: int) -> dict[str, float | int]:
    raw_fps = float(raw_fps)
    stride = int(bwm_sampling_stride)
    if not math.isfinite(raw_fps) or raw_fps <= 0:
        raise ValueError(f"raw_fps must be finite and positive, got {raw_fps}")
    if stride <= 0:
        raise ValueError(f"bwm_sampling_stride must be positive, got {stride}")
    effective_fps = raw_fps / stride
    return {
        "raw_fps": raw_fps,
        "raw_frame_dt": 1.0 / raw_fps,
        "bwm_sampling_stride": stride,
        "effective_bwm_fps": effective_fps,
        "effective_frame_dt": 1.0 / effective_fps,
    }


def require_matching_cadence(
    expert: dict[str, Any],
    policy: dict[str, Any],
    *,
    absolute_tolerance: float = 1e-9,
) -> float:
    required = {
        "raw_fps",
        "raw_frame_dt",
        "bwm_sampling_stride",
        "effective_bwm_fps",
        "effective_frame_dt",
    }
    for side, record in (("expert", expert), ("policy", policy)):
        missing = required - set(record)
        if missing:
            raise KeyError(f"{side} cadence record is missing {sorted(missing)}")
    expert_fps = float(expert["effective_bwm_fps"])
    policy_fps = float(policy["effective_bwm_fps"])
    if not math.isclose(expert_fps, policy_fps, rel_tol=0.0, abs_tol=absolute_tolerance):
        raise ValueError(
            f"CADENCE_MISMATCH: expert effective FPS={expert_fps}, policy={policy_fps}"
        )
    return expert_fps


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
            video_fps = float(reader.get_meta_data().get("fps", 0.0))
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
        video_fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
        if video_frames <= 0:
            raise ValueError(f"Invalid video frame count {video_frames}: {video_path}")
    if not math.isfinite(video_fps) or video_fps <= 0:
        raise ValueError(f"Invalid video FPS {video_fps}: {video_path}")
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
        "video_fps": video_fps,
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


def out_of_reference_range(
    raw_state_pose: np.ndarray,
    lower: Sequence[float],
    upper: Sequence[float],
) -> dict[str, Any]:
    values = np.asarray(raw_state_pose, dtype=np.float64)
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 14:
        raise ValueError(f"Expected raw state_pose shape (T, 14), got {values.shape}")
    if low.shape != (14,) or high.shape != (14,):
        raise ValueError(f"Expected 14D reference bounds, got {low.shape}/{high.shape}")
    below = values < low
    above = values > high
    clipped = below | above
    return {
        "name": "out-of-reference-range",
        "below_p01_rate": float(below.mean()),
        "above_p99_rate": float(above.mean()),
        "clipping_rate": float(clipped.mean()),
        "per_dim_below_p01_rate": below.mean(axis=0).tolist(),
        "per_dim_above_p99_rate": above.mean(axis=0).tolist(),
        "per_dim_clipping_rate": clipped.mean(axis=0).tolist(),
    }


def metric_value_with_normalization(
    raw_value: float | None,
    lower_bound: float,
    upper_bound: float,
    *,
    higher_is_better: bool = True,
) -> dict[str, Any]:
    lower = float(lower_bound)
    upper = float(upper_bound)
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        raise ValueError(f"Invalid normalization bounds: {lower}, {upper}")
    if raw_value is None:
        return {
            "raw_value": None,
            "normalized_value": None,
            "normalization_bounds": [lower, upper],
            "clipped_flag": None,
        }
    raw = float(raw_value)
    if not math.isfinite(raw):
        raise ValueError(f"Non-finite raw metric value: {raw}")
    unit = (raw - lower) / (upper - lower)
    if not higher_is_better:
        unit = 1.0 - unit
    return {
        "raw_value": raw,
        "normalized_value": float(np.clip(unit, 0.0, 1.0)),
        "normalization_bounds": [lower, upper],
        "clipped_flag": bool(unit < 0.0 or unit > 1.0),
    }


def summarize_tracker_results(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Count tracker failures explicitly; no record is silently dropped."""

    records = list(records)
    by_side = {"expert": [0, 0], "pi05": [0, 0]}
    failure_reasons: dict[str, int] = {}
    for record in records:
        side = str(record.get("side"))
        if side not in by_side:
            raise ValueError(f"Unknown tracker side: {side}")
        if not isinstance(record.get("tracker_success"), bool):
            raise KeyError("Every tracker record requires boolean tracker_success")
        by_side[side][1] += 1
        if record["tracker_success"]:
            by_side[side][0] += 1
        else:
            reason = str(record.get("failure_reason") or "unspecified")
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    successes = sum(item[0] for item in by_side.values())
    return {
        "trajectory_tracker_success_rate": successes / len(records) if records else None,
        "expert_tracker_success_rate": (
            by_side["expert"][0] / by_side["expert"][1] if by_side["expert"][1] else None
        ),
        "policy_tracker_success_rate": (
            by_side["pi05"][0] / by_side["pi05"][1] if by_side["pi05"][1] else None
        ),
        "failure_reason_counts": dict(sorted(failure_reasons.items())),
        "n_records": len(records),
    }


def _record_raw_metric(record: dict[str, Any], metric: str) -> float:
    metrics = record.get("metrics", {})
    value = metrics.get(metric)
    if isinstance(value, dict):
        value = value.get("raw_value")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(
            f"Missing finite raw {metric} for sample {record.get('sample_id', '<unknown>')}"
        )
    return float(value)


def aggregate_paired_metric(
    records: Sequence[dict[str, Any]],
    metric: str,
    *,
    bootstrap_samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Apply repeat -> window -> pair-side -> expert-policy paired aggregation."""

    if metric == "jepa":
        raise ValueError("JEPA is set-level only and cannot enter pair bootstrap")
    if metric not in PAIRWISE_METRICS:
        raise ValueError(f"Unsupported pairwise metric: {metric}")
    grouped: dict[str, dict[str, dict[str, list[float]]]] = {}
    for record in records:
        pair = str(record.get("pair_id") or "")
        side = str(record.get("side") or "")
        slot = str(record.get("window_slot") or "")
        if not pair or side not in {"expert", "pi05"} or slot not in {"early", "middle", "late"}:
            raise ValueError(f"Invalid paired metric record identity: {record}")
        grouped.setdefault(pair, {}).setdefault(side, {}).setdefault(slot, []).append(
            _record_raw_metric(record, metric)
        )

    per_pair = []
    for pair, sides in sorted(grouped.items()):
        if set(sides) != {"expert", "pi05"}:
            raise ValueError(f"Unmatched sides for pair {pair}: {sorted(sides)}")
        expert_slots = set(sides["expert"])
        policy_slots = set(sides["pi05"])
        if expert_slots != policy_slots:
            raise ValueError(
                f"Unmatched window slots for pair {pair}: expert={sorted(expert_slots)}, "
                f"pi05={sorted(policy_slots)}"
            )
        window_side = {
            side: {slot: float(np.mean(values)) for slot, values in slots.items()}
            for side, slots in sides.items()
        }
        expert_score = float(np.mean(list(window_side["expert"].values())))
        policy_score = float(np.mean(list(window_side["pi05"].values())))
        per_pair.append(
            {
                "pair_id": pair,
                "expert_score": expert_score,
                "policy_score": policy_score,
                "gap_error": expert_score - policy_score,
                "window_side_scores": window_side,
            }
        )
    gaps = [row["gap_error"] for row in per_pair]
    rng = np.random.default_rng(seed)
    return {
        "metric": metric,
        "statistical_unit": "pair_id",
        "n_pairs": len(per_pair),
        "per_pair_gap": per_pair,
        "mean_gap": float(np.mean(gaps)) if gaps else None,
        "median_gap": float(np.median(gaps)) if gaps else None,
        "paired_bootstrap_ci": _bootstrap_mean_ci(gaps, rng, bootstrap_samples, confidence),
    }


def balanced_jepa_sets(records: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    """Build balanced expert/Pi0.5 sample-id sets without producing pair-level JEPA values."""

    cells: dict[tuple[str, str, int], dict[str, str]] = {}
    for record in records:
        key = (
            str(record.get("pair_id")),
            str(record.get("window_slot")),
            int(record.get("repeat_id", 0)),
        )
        side = str(record.get("side"))
        if side not in {"expert", "pi05"}:
            raise ValueError(f"Unknown JEPA side: {side}")
        sample_id = str(record.get("sample_id") or "")
        if not SAMPLE_ID_PATTERN.fullmatch(sample_id):
            raise ValueError(f"Invalid JEPA sample_id: {sample_id}")
        if side in cells.setdefault(key, {}):
            raise ValueError(f"Duplicate JEPA cell {key} side={side}")
        cells[key][side] = sample_id
    incomplete = [key for key, sides in cells.items() if set(sides) != {"expert", "pi05"}]
    if incomplete:
        raise ValueError(f"JEPA sets are not balanced; incomplete cells: {incomplete}")
    ordered = [cells[key] for key in sorted(cells)]
    return {
        "expert": [cell["expert"] for cell in ordered],
        "pi05": [cell["pi05"] for cell in ordered],
    }


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
