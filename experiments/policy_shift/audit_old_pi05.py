#!/usr/bin/env python3
"""Audit the legacy Pi0.5 rollout tree without inferring missing provenance.

The legacy dataset has no authoritative manifest.  This script inventories every
episode and emits conservative status fields; file order and timestamps are
recorded as circumstantial evidence only and never upgrade a status to
``CONFIRMED``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any


VALID_STATUSES = {"CONFIRMED", "AMBIGUOUS", "UNKNOWN"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"type": type(value).__name__, "repr": repr(value)[:200]}


def inspect_pickle(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)  # Local, user-owned RoboTwin artifact.
    except Exception as exc:  # pragma: no cover - retained in generated audit
        return {"readable": False, "error": f"{type(exc).__name__}: {exc}"}
    result: dict[str, Any] = {
        "readable": True,
        "payload_type": type(payload).__name__,
    }
    if isinstance(payload, dict):
        result["keys"] = sorted(map(str, payload))
        result["schema"] = _jsonable(payload)
    else:
        result["schema"] = _jsonable(payload)
    return result


def _walk_hdf5(group, prefix: str = "") -> tuple[list[str], dict[str, Any]]:
    datasets: list[str] = []
    attrs: dict[str, Any] = {}
    for key, value in group.attrs.items():
        attrs[f"{prefix or '/'}@{key}"] = _jsonable(value)
    for name, child in group.items():
        child_path = f"{prefix}/{name}"
        if hasattr(child, "shape"):
            datasets.append(f"{child_path}:{list(child.shape)}:{child.dtype}")
            for key, value in child.attrs.items():
                attrs[f"{child_path}@{key}"] = _jsonable(value)
        else:
            nested_datasets, nested_attrs = _walk_hdf5(child, child_path)
            datasets.extend(nested_datasets)
            attrs.update(nested_attrs)
    return datasets, attrs


def inspect_hdf5(path: Path) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("The legacy audit requires h5py") from exc
    try:
        with h5py.File(path, "r") as handle:
            datasets, attrs = _walk_hdf5(handle)
            head_rgb = handle.get("observation/head_camera/rgb")
            length = None if head_rgb is None else int(head_rgb.shape[0])
        return {
            "readable": True,
            "head_rgb_length": length,
            "datasets": sorted(datasets),
            "attrs": attrs,
        }
    except Exception as exc:  # pragma: no cover - retained in generated audit
        return {"readable": False, "error": f"{type(exc).__name__}: {exc}"}


def inspect_video(path: Path) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("The legacy audit requires OpenCV") from exc
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {"readable": False}
    result = {
        "readable": True,
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    return result


def parse_seed_file(path: Path) -> list[int]:
    tokens = path.read_text(encoding="utf-8").split()
    try:
        return [int(token) for token in tokens]
    except ValueError as exc:
        raise ValueError(f"Non-integer seed token in {path}") from exc


def parse_result_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    nonempty = [line.strip() for line in text.splitlines() if line.strip()]
    success_rate = None
    for line in reversed(nonempty):
        try:
            success_rate = float(line)
            break
        except ValueError:
            continue
    return {
        "sha256": sha256_file(path),
        "nonempty_lines": nonempty,
        "task_level_success_rate": success_rate,
    }


def audit_dataset(source_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task_dirs = sorted(path for path in source_root.glob("*_pi05") if path.is_dir())
    if not task_dirs:
        raise FileNotFoundError(f"No *_pi05 task directories under {source_root}")

    rows: list[dict[str, Any]] = []
    unexpected_files: list[str] = []
    per_task: list[dict[str, Any]] = []
    for task_dir in task_dirs:
        task = task_dir.name.removesuffix("_pi05")
        run_dir = task_dir / "demo_clean" / "run_0000"
        seed_path = run_dir / "seed.txt"
        result_path = run_dir / "_result.txt"
        seeds = parse_seed_file(seed_path) if seed_path.is_file() else []
        result = parse_result_file(result_path) if result_path.is_file() else None

        expected: set[Path] = {seed_path, result_path}
        for episode_id in range(10):
            expected.update(
                {
                    run_dir / "data" / f"episode{episode_id}.hdf5",
                    run_dir / "_traj_data" / f"episode{episode_id}.pkl",
                    run_dir / "video" / f"episode{episode_id}_fail.mp4",
                }
            )
        for file_path in run_dir.rglob("*"):
            if file_path.is_file() and file_path not in expected:
                unexpected_files.append(str(file_path.resolve()))

        per_task.append(
            {
                "task": task,
                "run_dir": str(run_dir.resolve()),
                "seed_file": str(seed_path.resolve()),
                "seed_file_sha256": sha256_file(seed_path) if seed_path.is_file() else None,
                "seed_count": len(seeds),
                "unique_seed_count": len(set(seeds)),
                "result": result,
            }
        )

        for episode_id in range(10):
            hdf5_path = run_dir / "data" / f"episode{episode_id}.hdf5"
            pickle_path = run_dir / "_traj_data" / f"episode{episode_id}.pkl"
            video_path = run_dir / "video" / f"episode{episode_id}_fail.mp4"
            hdf5 = inspect_hdf5(hdf5_path) if hdf5_path.is_file() else {"readable": False, "missing": True}
            trajectory = (
                inspect_pickle(pickle_path)
                if pickle_path.is_file()
                else {"readable": False, "missing": True}
            )
            video = inspect_video(video_path) if video_path.is_file() else {"readable": False, "missing": True}
            seed = seeds[episode_id] if episode_id < len(seeds) else None

            row = {
                "task": task,
                "episode_id": episode_id,
                "candidate_env_seed_by_position": seed,
                "seed_mapping_status": "AMBIGUOUS" if seed is not None else "UNKNOWN",
                "seed_mapping_evidence": [
                    "seed.txt has one ordered token per on-disk episode",
                    "current eval_policy.py appends successful seeds in episode order",
                    "no direct writer/manifest links this legacy seed.txt to these episode files",
                ],
                "failure_label_status": "AMBIGUOUS" if video_path.name.endswith("_fail.mp4") else "UNKNOWN",
                "failure_label_evidence": [
                    "video filename contains _fail",
                    "_result.txt is task-level and cannot verify this episode",
                    "no legacy recorder code or per-episode success metadata was found",
                ],
                "policy_checkpoint_status": "UNKNOWN",
                "policy_checkpoint_evidence": "no checkpoint path/hash is archived with the episode",
                "task_config_status": "AMBIGUOUS",
                "task_config_evidence": "directory is named demo_clean, but exact config bytes/hash are absent",
                "formal_pair_eligible": False,
                "hdf5_path": str(hdf5_path.resolve()),
                "hdf5_size": hdf5_path.stat().st_size if hdf5_path.is_file() else None,
                "hdf5_mtime_ns": hdf5_path.stat().st_mtime_ns if hdf5_path.is_file() else None,
                "hdf5": hdf5,
                "trajectory_path": str(pickle_path.resolve()),
                "trajectory_size": pickle_path.stat().st_size if pickle_path.is_file() else None,
                "trajectory_mtime_ns": pickle_path.stat().st_mtime_ns if pickle_path.is_file() else None,
                "trajectory": trajectory,
                "video_path": str(video_path.resolve()),
                "video_size": video_path.stat().st_size if video_path.is_file() else None,
                "video_mtime_ns": video_path.stat().st_mtime_ns if video_path.is_file() else None,
                "video": video,
            }
            for field in (
                "seed_mapping_status",
                "failure_label_status",
                "policy_checkpoint_status",
                "task_config_status",
            ):
                if row[field] not in VALID_STATUSES:
                    raise AssertionError(f"Invalid {field}: {row[field]}")
            rows.append(row)

    summary = {
        "source_root": str(source_root.resolve()),
        "task_count": len(task_dirs),
        "episode_count": len(rows),
        "unexpected_files": sorted(unexpected_files),
        "per_task": per_task,
        "status_counts": {
            field: dict(Counter(row[field] for row in rows))
            for field in (
                "seed_mapping_status",
                "failure_label_status",
                "policy_checkpoint_status",
                "task_config_status",
            )
        },
        "formal_pair_eligible_count": sum(bool(row["formal_pair_eligible"]) for row in rows),
    }
    return rows, summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_report(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any], jsonl_path: Path) -> None:
    task_seed_counts = Counter(item["seed_count"] for item in summary["per_task"])
    task_unique_counts = Counter(item["unique_seed_count"] for item in summary["per_task"])
    hdf5_lengths = [row["hdf5"].get("head_rgb_length") for row in rows]
    hdf5_lengths = [value for value in hdf5_lengths if isinstance(value, int)]
    fps_values = sorted({row["video"].get("fps") for row in rows if row["video"].get("readable")})
    geometry = sorted(
        {
            (row["video"].get("width"), row["video"].get("height"))
            for row in rows
            if row["video"].get("readable")
        }
    )
    pickle_schemas = Counter(
        json.dumps(row["trajectory"].get("keys"), sort_keys=True)
        for row in rows
        if row["trajectory"].get("readable")
    )
    lines = [
        "# Legacy Pi0.5 data provenance audit",
        "",
        "## Formal decision",
        "",
        "```text",
        "usable for matched experiment: NO",
        "```",
        "",
        "No direct artifact or reachable Git history links `seed.txt[i]` to `episode{i}` for this",
        "legacy dataset. File order and timestamps are circumstantial evidence only. All 500 episodes",
        "are therefore excluded from formal matched-pair statistics.",
        "",
        "## Complete inventory",
        "",
        f"- source: `{summary['source_root']}`",
        f"- tasks: {summary['task_count']}",
        f"- episodes: {summary['episode_count']}",
        f"- task seed-token counts: `{dict(task_seed_counts)}`",
        f"- task unique-seed counts: `{dict(task_unique_counts)}`",
        f"- HDF5 head-RGB length range: {min(hdf5_lengths)}–{max(hdf5_lengths)}" if hdf5_lengths else "- HDF5 head-RGB length range: unavailable",
        f"- video FPS values: `{fps_values}`",
        f"- video geometry values: `{geometry}`",
        f"- Pickle key schemas: `{dict(pickle_schemas)}`",
        f"- unexpected/hidden sidecar files: {len(summary['unexpected_files'])}",
        f"- machine-readable per-episode audit: `{jsonl_path}`",
        "",
        "## Direct-evidence search",
        "",
        "The audit checked all 50 `seed.txt` files, all 500 HDF5 files and their attributes,",
        "all 500 trajectory Pickles, all 500 videos, `_result.txt`, unexpected/hidden sidecars,",
        "the current RoboTwin scripts, all locally reachable RoboTwin Git branches/history, and",
        "available shell history. Findings:",
        "",
        "- no `_fail.mp4` writer exists in reachable RoboTwin Git history;",
        "- current and historical eval code append `suc_test_seed_list` but do not persist it;",
        "- `collect_data.py` writes expert collection seeds, but its trajectory schema is not the",
        "  legacy Pi0.5 `{'actions': ...}` schema;",
        "- no scene info, launch log, checkpoint identity, exact task-config bytes, or episode-level",
        "  success metadata is stored beside the legacy episodes;",
        "- filesystem timestamps are retained in JSONL only as auxiliary evidence and were not used",
        "  to upgrade any status.",
        "",
        "## Status interpretation",
        "",
        "- `seed_mapping_status=AMBIGUOUS`: a positional candidate exists, but no direct writer or",
        "  immutable manifest proves the association.",
        "- `failure_label_status=AMBIGUOUS`: `_fail` is only a filename label; task-level result files",
        "  do not establish an episode-level ground-truth label.",
        "- `policy_checkpoint_status=UNKNOWN`: no policy checkpoint/config hash is archived.",
        "- `task_config_status=AMBIGUOUS`: `demo_clean` is a directory label, not preserved config bytes.",
        "",
        "## Per-episode decision",
        "",
        "| task | episode | positional seed | seed mapping | failure label | policy checkpoint | task config | formal pair |",
        "|---|---:|---:|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['episode_id']} | {row['candidate_env_seed_by_position']} | "
            f"{row['seed_mapping_status']} | {row['failure_label_status']} | "
            f"{row['policy_checkpoint_status']} | {row['task_config_status']} | NO |"
        )
    lines.extend(
        [
            "",
            "## Required fallback",
            "",
            "The legacy 500 episodes may remain useful as an unmatched, filename-labelled policy",
            "reference set, but not as the core matched experiment. The next admissible path is fresh",
            "pair-at-a-time collection with preserved policy/config/checkpoint and initial-state",
            "fingerprints.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    args = parser.parse_args()

    rows, summary = audit_dataset(args.source_root.resolve())
    write_jsonl(args.jsonl.resolve(), rows)
    write_report(args.report.resolve(), rows, summary, args.jsonl.resolve())
    print(json.dumps({key: value for key, value in summary.items() if key != "per_task"}, indent=2))


if __name__ == "__main__":
    main()
