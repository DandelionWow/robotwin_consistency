#!/usr/bin/env python3
"""Strict GT-history sliding-window evaluation for BWM.

Every model call reloads the selected window's history from the GT video. No
prediction is ever used as history for a later window. Autoregressive rollout is
deliberately not implemented in this entrypoint; BWM's scripts/infer.py remains
the separate AR mode.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from bwm_conversion import EEF_INDICES_26
from protocol import (
    aggregate_paired_metric,
    artifact_directory,
    cadence_record,
    generation_seed,
    inspect_episode_alignment,
    load_and_validate_reference_stat,
    out_of_reference_range,
    pair_length_eligibility,
    read_metadata,
    require_matching_cadence,
    resolve_data_path,
    select_protocol_windows,
    sha256_file,
    validate_wan_window_shape,
    worldarena_sample_id,
    worldarena_summary_row,
)
from worldarena_adapter import WorldArenaBasicMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BWM GT-history evaluator (strictly separate from autoregressive rollout)."
    )
    parser.add_argument("--bwm-root", type=Path, required=True)
    parser.add_argument("--worldarena-root", type=Path, required=True)
    parser.add_argument("--dataset-base", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--reference-stat", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", default=None)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model/wan2_2_ti2v_5b.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--history-frames", type=int, default=9)
    parser.add_argument("--future-frames", type=int, default=72)
    parser.add_argument(
        "--slots",
        nargs="+",
        choices=("early", "middle", "late"),
        default=("early", "middle", "late"),
    )
    parser.add_argument("--repeat-id", type=int, default=0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--mixed-precision", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--debug-png", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _metadata_value(row: dict[str, Any], key: str, default: Any) -> Any:
    value = row.get(key)
    return default if value is None else value


def _success_value(row: dict[str, Any]) -> bool | None:
    if isinstance(row.get("success"), bool):
        return row["success"]
    label = str(row.get("label", "")).lower()
    if label in {"success", "successful", "true", "1"}:
        return True
    if label in {"failure", "failed", "false", "0"}:
        return False
    return None


def _source_frame_refs(video_value: Any, indices: tuple[int, ...]) -> list[str]:
    if isinstance(video_value, (list, tuple)):
        video_value = video_value[0]
    if isinstance(video_value, dict):
        video_value = video_value.get("data")
    return [f"{video_value}#frame={index}" for index in indices]


def _reference_bounds(stat: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    entry = stat.get("state_pose") or stat.get("eef_abs")
    if entry.get("p01") is not None and entry.get("p99") is not None:
        return np.asarray(entry["p01"]), np.asarray(entry["p99"])
    return np.asarray(entry["min"]), np.asarray(entry["max"])


def _range_diagnostics(
    args: argparse.Namespace,
    row: dict[str, Any],
    indices: tuple[int, ...],
    stat: dict[str, Any],
) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Range diagnostics require pyarrow") from exc
    action_path = resolve_data_path(args.dataset_base, row.get("action"), "action")
    column = pq.read_table(action_path, columns=["observation.state"])["observation.state"]
    state26 = np.asarray([column[index].as_py() for index in indices], dtype=np.float64)
    if state26.shape != (len(indices), 26):
        raise ValueError(f"Expected raw state shape ({len(indices)}, 26), got {state26.shape}")
    lower, upper = _reference_bounds(stat)
    return out_of_reference_range(state26[:, EEF_INDICES_26], lower, upper)


def build_plan(
    args: argparse.Namespace, rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build only complete, cadence-matched pair plans using fixed protocol slots."""

    aligned_rows: list[dict[str, Any]] = []
    selected_rows = rows[: args.max_episodes] if args.max_episodes else rows
    for metadata_index, row in enumerate(selected_rows):
        pair_identifier = str(row.get("pair_id") or "")
        side = str(row.get("side") or "")
        if not pair_identifier or side not in {"expert", "pi05"}:
            raise ValueError(
                f"metadata row {metadata_index} requires pair_id and side=expert|pi05"
            )
        alignment = inspect_episode_alignment(args.dataset_base, row)
        raw_fps = float(row.get("raw_fps", alignment["video_fps"]))
        if not np.isclose(raw_fps, alignment["video_fps"], rtol=0.0, atol=1e-3):
            raise ValueError(
                f"row {metadata_index} raw_fps={raw_fps} disagrees with video FPS "
                f"{alignment['video_fps']}"
            )
        cadence = cadence_record(raw_fps, int(row.get("bwm_sampling_stride", 1)))
        aligned_rows.append(
            {
                "metadata_index": metadata_index,
                "row": row,
                "pair_id": pair_identifier,
                "side": side,
                "alignment": alignment,
                "cadence": cadence,
            }
        )

    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for item in aligned_rows:
        if item["side"] in pairs.setdefault(item["pair_id"], {}):
            raise ValueError(f"Duplicate {item['side']} row for pair {item['pair_id']}")
        pairs[item["pair_id"]][item["side"]] = item

    plans: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    selected_slots = set(args.slots)
    for pair_identifier, sides in sorted(pairs.items()):
        if set(sides) != {"expert", "pi05"}:
            raise ValueError(f"Unmatched metadata sides for pair {pair_identifier}: {sorted(sides)}")
        eligibility = pair_length_eligibility(
            sides["expert"]["alignment"]["length"],
            sides["pi05"]["alignment"]["length"],
            args.history_frames + args.future_frames,
        )
        if not eligibility["eligible"]:
            exclusions.append({"pair_id": pair_identifier, **eligibility})
            continue
        effective_fps = require_matching_cadence(
            sides["expert"]["cadence"], sides["pi05"]["cadence"]
        )
        for side in ("expert", "pi05"):
            item = sides[side]
            windows = select_protocol_windows(
                item["alignment"]["length"], args.history_frames, args.future_frames
            )
            for window_index, window in enumerate(windows):
                if window.slot not in selected_slots:
                    continue
                offset = item["alignment"]["start_frame"]
                history_indices = tuple(offset + index for index in window.history_indices)
                future_indices = tuple(offset + index for index in window.future_indices)
                plans.append(
                    {
                        "metadata_index": item["metadata_index"],
                        "window_index": window_index,
                        "window_slot": window.slot,
                        "window_start": offset + window.window_start,
                        "history_indices": history_indices,
                        "future_indices": future_indices,
                        "all_indices": history_indices + future_indices,
                        "alignment": item["alignment"],
                        "cadence": item["cadence"],
                        "effective_fps": effective_fps,
                    }
                )
    if not plans:
        raise ValueError(
            "No complete windows were found. Reduce history/future length or inspect episode lengths."
        )
    return plans, exclusions


def _write_json(path: Path, payload: Any, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite explicitly")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite explicitly")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def _build_pipeline(args: argparse.Namespace):
    try:
        import torch
        from diffsynth.core import ModelConfig
    except ImportError as exc:
        raise RuntimeError(
            "Generation requires the BWM environment with torch and diffsynth installed"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("BWM generation requires a visible CUDA GPU; use --plan-only for CPU validation")
    if args.model_root is None:
        raise ValueError("--model-root is required unless --plan-only is used")

    bwm_root = args.bwm_root.resolve()
    sys.path.insert(0, str(bwm_root))
    from wan_video_action.parsers import prepare_model_config
    from wan_video_action.pipelines.wan_video_action import WanVideoActionPipeline
    from wan_video_action.utils import resolve_model_path

    model_config_path = args.model_config
    if not model_config_path.is_absolute():
        model_config_path = bwm_root / model_config_path
    model_args = SimpleNamespace(
        modes={"dit": "default", "text": "none", "vae": "raw", "image": "none", "action": "adaln"},
        dit_mode=None,
        text_mode=None,
        vae_mode=None,
        image_mode=None,
        action_mode=None,
        model_paths=str(args.model_root.resolve()),
        model_config_path=str(_require_file(model_config_path, "BWM model config")),
        weights=["dit", "vae"],
    )
    resolved = prepare_model_config(model_args)
    model_configs = [
        ModelConfig(path=resolve_model_path(model_path))
        for model_path in resolved["model_paths_list"]
    ]
    dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    pipe = WanVideoActionPipeline.from_pretrained(
        torch_dtype=dtype,
        device="cuda",
        model_configs=model_configs,
        tokenizer_config=None,
        ckpt_path=str(args.checkpoint.resolve()),
        action_dim=14,
        action_mode="adaln",
    )
    pipe.use_gradient_checkpointing = False
    pipe.use_gradient_checkpointing_offload = False
    pipe.eval()
    return pipe


def _build_loaders(args: argparse.Namespace, stat: dict[str, Any]):
    bwm_root = args.bwm_root.resolve()
    sys.path.insert(0, str(bwm_root))
    try:
        from wan_video_action.data.operators import LoadCobotAction, create_video_operator
    except ImportError as exc:
        raise RuntimeError("Could not import the BWM data operators") from exc

    total_frames = args.history_frames + args.future_frames
    video_loader = create_video_operator(
        base_path=str(args.dataset_base.resolve()),
        max_pixels=args.height * args.width,
        height=args.height,
        width=args.width,
        height_division_factor=32,
        width_division_factor=32,
        num_frames=total_frames,
        time_division_factor=4,
        time_division_remainder=1,
        resize_mode="crop",
    )
    action_loader = LoadCobotAction(
        base_path=str(args.dataset_base.resolve()),
        action_type="eef_abs",
        stat=stat,
        num_frames=None,
        align_num_frames=False,
        time_division_factor=4,
        time_division_remainder=1,
    )
    return video_loader, action_loader


def _tensor_to_uint8_frames(video) -> np.ndarray:
    array = video.detach().float().cpu().numpy()
    if array.ndim != 5 or array.shape[0] != 1 or array.shape[1] != 3:
        raise ValueError(f"Expected single-view VCTHW tensor, got {array.shape}")
    array = np.transpose(array[0], (1, 2, 3, 0))
    return np.rint(np.clip((array + 1.0) * 127.5, 0, 255)).astype(np.uint8)


def _save_frames(frames: np.ndarray, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, frame in enumerate(frames):
        path = output_dir / f"frame_{index:05d}.png"
        Image.fromarray(frame).save(path)
        paths.append(str(path.resolve()))
    return paths


def _save_mp4(frames: np.ndarray, path: Path, fps: float, overwrite: bool) -> str:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite explicitly")
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("MP4 artifact generation requires imageio") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(
        path,
        np.asarray(frames, dtype=np.uint8),
        fps=float(fps),
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )
    return str(path.resolve())


def _inspect_mp4(path: Path) -> dict[str, Any]:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("MP4 artifact validation requires imageio") from exc
    reader = imageio.get_reader(path)
    try:
        metadata = reader.get_meta_data()
        frame_count = int(reader.count_frames())
        first = np.asarray(reader.get_data(0))
    finally:
        reader.close()
    return {
        "frame_count": frame_count,
        "height": int(first.shape[0]),
        "width": int(first.shape[1]),
        "fps": float(metadata["fps"]),
        "bytes": path.stat().st_size,
    }


def _validate_paired_mp4(gt_path: Path, pred_path: Path, expected_frames: int, fps: float) -> dict[str, Any]:
    gt = _inspect_mp4(gt_path)
    pred = _inspect_mp4(pred_path)
    if gt["frame_count"] != expected_frames or pred["frame_count"] != expected_frames:
        raise ValueError(
            f"MP4 frame-count mismatch: expected={expected_frames} gt={gt['frame_count']} "
            f"pred={pred['frame_count']}"
        )
    if (gt["height"], gt["width"]) != (pred["height"], pred["width"]):
        raise ValueError(f"MP4 geometry mismatch: gt={gt} pred={pred}")
    if not np.isclose(gt["fps"], fps, rtol=0.0, atol=1e-3) or not np.isclose(
        pred["fps"], fps, rtol=0.0, atol=1e-3
    ):
        raise ValueError(f"MP4 cadence mismatch: expected={fps} gt={gt['fps']} pred={pred['fps']}")
    return {"gt": gt, "pred": pred}


def _base_record(
    row: dict[str, Any],
    plan: dict[str, Any],
    args: argparse.Namespace,
    stat_sha256: str,
) -> dict[str, Any]:
    episode_id = _metadata_value(row, "episode_id", row.get("episode_index", plan["metadata_index"]))
    pair_identifier = str(row["pair_id"])
    side = str(row["side"])
    sample_id = worldarena_sample_id(
        pair_identifier, side, plan["window_slot"], args.repeat_id
    )
    seed = generation_seed(pair_identifier, plan["window_slot"], args.repeat_id)
    return {
        "sample_id": sample_id,
        "pair_id": pair_identifier,
        "protocol_mode": "gt_history",
        "task": str(_metadata_value(row, "task", "unknown")),
        "episode_id": str(episode_id),
        "env_seed": int(row["env_seed"]),
        "generation_seed": seed,
        "repeat_id": args.repeat_id,
        "side": side,
        "policy_id": str(_metadata_value(row, "policy_id", "expert" if side == "expert" else "pi05")),
        "success": _success_value(row),
        "window_slot": plan["window_slot"],
        "window_start": plan["window_start"],
        "history_frames": _source_frame_refs(row["video"], plan["history_indices"]),
        "future_frames": _source_frame_refs(row["video"], plan["future_indices"]),
        "conditioning_frame_indices": list(plan["all_indices"]),
        "conditioning_semantics": "normalized eef_abs/state_pose from observation.state",
        "reference_stat_sha256": stat_sha256,
        "cadence": plan["cadence"],
        "metrics": {
            "psnr": None,
            "ssim": None,
            "jepa": None,
            "trajectory_accuracy": None,
        },
    }


def _run_generation(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    plans: list[dict[str, Any]],
    stat: dict[str, Any],
    stat_sha256: str,
) -> list[dict[str, Any]]:
    pipe = _build_pipeline(args)
    video_loader, action_loader = _build_loaders(args, stat)
    metrics = WorldArenaBasicMetrics(args.worldarena_root.resolve())
    total_frames = args.history_frames + args.future_frames
    records = []

    for plan in plans:
        row = rows[plan["metadata_index"]]
        video_path = resolve_data_path(args.dataset_base, row.get("video"), "video")
        action_path = resolve_data_path(args.dataset_base, row.get("action"), "action")
        frame_indices = list(plan["all_indices"])
        gt_video = video_loader({"data": str(video_path), "frame_indices": frame_indices})
        action = action_loader({"data": str(action_path), "frame_indices": frame_indices})
        if tuple(gt_video.shape[:3]) != (1, 3, total_frames):
            raise ValueError(f"GT loader alignment failed, got {tuple(gt_video.shape)}")
        if tuple(action.shape) != (1, total_frames, 14):
            raise ValueError(f"Action loader alignment failed, got {tuple(action.shape)}")

        pair_identifier = str(row["pair_id"])
        side = str(row["side"])
        seed = generation_seed(pair_identifier, plan["window_slot"], args.repeat_id)
        prediction = pipe(
            input_video=gt_video[:, :, : args.history_frames],
            action=action,
            seed=seed,
            rand_device="cpu",
            tiled=args.tiled,
            height=args.height,
            width=args.width,
            num_frames=total_frames,
            num_history_frames=args.history_frames,
            cfg_scale=args.cfg_scale,
            num_inference_steps=args.num_inference_steps,
            progress_bar_cmd=lambda iterable, *unused_args, **unused_kwargs: iterable,
            output_type="floatpoint",
        )
        if tuple(prediction.shape[:3]) != (1, 3, total_frames):
            raise ValueError(f"Prediction alignment failed, got {tuple(prediction.shape)}")

        gt_future = _tensor_to_uint8_frames(gt_video[:, :, args.history_frames :])
        pred_future = _tensor_to_uint8_frames(prediction[:, :, args.history_frames :])
        scores = metrics.score(gt_future, pred_future)

        record = _base_record(row, plan, args, stat_sha256)
        record["out_of_reference_range"] = _range_diagnostics(
            args, row, plan["all_indices"], stat
        )
        record["metrics"].update(scores)
        artifact_dir = artifact_directory(
            args.output_dir / "windows",
            pair_identifier,
            side,
            plan["window_slot"],
            args.repeat_id,
            require_absent=not args.overwrite,
        )
        gt_path = artifact_dir / "gt_future.mp4"
        pred_path = artifact_dir / "pred_future.mp4"
        _save_mp4(gt_future, gt_path, plan["effective_fps"], args.overwrite)
        _save_mp4(pred_future, pred_path, plan["effective_fps"], args.overwrite)
        record["artifacts"] = {
            "gt_future_mp4": str(gt_path.resolve()),
            "pred_future_mp4": str(pred_path.resolve()),
            "paired_mp4_validation": _validate_paired_mp4(
                gt_path, pred_path, args.future_frames, plan["effective_fps"]
            ),
        }
        if args.debug_png:
            record["artifacts"]["gt_future_png"] = _save_frames(
                gt_future, artifact_dir / "gt_future_png"
            )
            record["artifacts"]["pred_future_png"] = _save_frames(
                pred_future, artifact_dir / "pred_future_png"
            )
        records.append(record)
        print(
            f"[gt-history] sample={record['sample_id']} start={record['window_start']} "
            f"psnr={scores['psnr']:.4f} ssim={scores['ssim']:.4f}",
            flush=True,
        )
    return records


def main() -> None:
    args = parse_args()
    args.bwm_root = args.bwm_root.resolve()
    args.worldarena_root = args.worldarena_root.resolve()
    args.dataset_base = args.dataset_base.resolve()
    args.metadata = _require_file(args.metadata, "metadata")
    args.reference_stat = _require_file(args.reference_stat, "reference stat")
    args.checkpoint = _require_file(args.checkpoint, "checkpoint")
    if not args.bwm_root.is_dir():
        raise FileNotFoundError(args.bwm_root)
    if not args.worldarena_root.is_dir():
        raise FileNotFoundError(args.worldarena_root)
    if not args.dataset_base.is_dir():
        raise FileNotFoundError(args.dataset_base)

    validate_wan_window_shape(args.history_frames, args.future_frames)
    stat, stat_provenance = load_and_validate_reference_stat(args.reference_stat)
    checkpoint_sha256 = args.checkpoint_sha256 or sha256_file(args.checkpoint)
    rows = read_metadata(args.metadata)
    plans, exclusions = build_plan(args, rows)

    manifest = {
        "protocol_mode": "gt_history",
        "independence_rule": "Every window reloads GT history; predictions are never reused as history.",
        "bwm_root": str(args.bwm_root),
        "worldarena_root": str(args.worldarena_root),
        "dataset_base": str(args.dataset_base),
        "metadata": str(args.metadata),
        "reference_stat": stat_provenance,
        "reference_stat_bounds": {
            "lower": _reference_bounds(stat)[0].tolist(),
            "upper": _reference_bounds(stat)[1].tolist(),
        },
        "reference_stat_claim": "candidate reference only; BWM training-stat provenance unconfirmed",
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": checkpoint_sha256,
            "sha256_source": "computed" if args.checkpoint_sha256 is None else "explicit_argument",
        },
        "history_frames": args.history_frames,
        "future_frames": args.future_frames,
        "window_selection": "fixed early=0, middle=floor((L-81)/2), late=L-81; deduplicated",
        "slots": list(args.slots),
        "repeat_id": args.repeat_id,
        "height": args.height,
        "width": args.width,
        "windows": len(plans),
        "n_excluded_pairs": len(exclusions),
        "excluded_pairs": exclusions,
        "plan_only": args.plan_only,
    }
    _write_json(args.output_dir / "run_manifest.json", manifest, args.overwrite)

    plan_records = []
    for plan in plans:
        row = rows[plan["metadata_index"]]
        record = _base_record(row, plan, args, stat_provenance["sha256"])
        record["alignment"] = plan["alignment"]
        record["out_of_reference_range"] = _range_diagnostics(
            args, row, plan["all_indices"], stat
        )
        plan_records.append(record)
    _write_jsonl(args.output_dir / "window_plan.jsonl", plan_records, args.overwrite)
    print(f"[validated] {len(plans)} strictly aligned GT-history windows", flush=True)

    if args.plan_only:
        return

    records = _run_generation(args, rows, plans, stat, stat_provenance["sha256"])
    _write_jsonl(args.output_dir / "windows.jsonl", records, args.overwrite)
    worldarena_rows = [
        worldarena_summary_row(
            record["sample_id"],
            Path(record["artifacts"]["gt_future_mp4"]),
            Path(record["artifacts"]["pred_future_mp4"]),
        )
        for record in records
    ]
    _write_json(args.output_dir / "worldarena_summary.json", worldarena_rows, args.overwrite)
    summary = {
        "protocol_mode": "gt_history",
        "statistical_unit": "pair_id",
        "psnr": aggregate_paired_metric(
            records, "psnr", bootstrap_samples=args.bootstrap_samples, seed=0
        ),
        "ssim": aggregate_paired_metric(
            records, "ssim", bootstrap_samples=args.bootstrap_samples, seed=1
        ),
    }
    _write_json(args.output_dir / "summary.json", summary, args.overwrite)


if __name__ == "__main__":
    main()
