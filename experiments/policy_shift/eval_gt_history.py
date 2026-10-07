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

from protocol import (
    aggregate_records,
    build_windows,
    inspect_episode_alignment,
    load_and_validate_reference_stat,
    read_metadata,
    resolve_data_path,
    sha256_file,
    validate_wan_window_shape,
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
    parser.add_argument("--stride", type=int, default=72)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mixed-precision", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-windows-per-episode", type=int, default=0)
    parser.add_argument("--default-distribution", choices=("expert", "policy"), default="policy")
    parser.add_argument("--default-policy-id", default="unknown")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
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


def build_plan(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    plans = []
    selected_rows = rows[: args.max_episodes] if args.max_episodes else rows
    for metadata_index, row in enumerate(selected_rows):
        alignment = inspect_episode_alignment(args.dataset_base, row)
        windows = build_windows(
            alignment["length"], args.history_frames, args.future_frames, args.stride
        )
        if args.max_windows_per_episode:
            windows = windows[: args.max_windows_per_episode]
        for window_index, window in enumerate(windows):
            offset = alignment["start_frame"]
            history_indices = tuple(offset + index for index in window.history_indices)
            future_indices = tuple(offset + index for index in window.future_indices)
            plans.append(
                {
                    "metadata_index": metadata_index,
                    "window_index": window_index,
                    "window_start": offset + window.window_start,
                    "history_indices": history_indices,
                    "future_indices": future_indices,
                    "all_indices": history_indices + future_indices,
                    "alignment": alignment,
                }
            )
    if not plans:
        raise ValueError(
            "No complete windows were found. Reduce history/future length or inspect episode lengths."
        )
    return plans


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


def _base_record(
    row: dict[str, Any],
    plan: dict[str, Any],
    args: argparse.Namespace,
    seed: int,
    stat_sha256: str,
) -> dict[str, Any]:
    episode_id = _metadata_value(row, "episode_id", row.get("episode_index", plan["metadata_index"]))
    distribution = str(_metadata_value(row, "distribution", args.default_distribution))
    if distribution not in {"expert", "policy"}:
        raise ValueError(f"distribution must be expert or policy, got {distribution!r}")
    return {
        "protocol_mode": "gt_history",
        "task": str(_metadata_value(row, "task", "unknown")),
        "episode_id": str(episode_id),
        "seed": seed,
        "distribution": distribution,
        "policy_id": str(_metadata_value(row, "policy_id", args.default_policy_id)),
        "success": _success_value(row),
        "window_start": plan["window_start"],
        "history_frames": _source_frame_refs(row["video"], plan["history_indices"]),
        "future_frames": _source_frame_refs(row["video"], plan["future_indices"]),
        "conditioning_frame_indices": list(plan["all_indices"]),
        "conditioning_semantics": "normalized eef_abs/state_pose from observation.state",
        "reference_stat_sha256": stat_sha256,
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

    for global_index, plan in enumerate(plans):
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

        seed = int(args.seed) + global_index
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

        record = _base_record(row, plan, args, seed, stat_sha256)
        record["metrics"].update(scores)
        artifact_dir = (
            args.output_dir
            / "windows"
            / f"episode_{record['episode_id']}"
            / f"start_{plan['window_start']:06d}"
        )
        record["artifacts"] = {
            "gt_future": _save_frames(gt_future, artifact_dir / "gt_future"),
            "pred_future": _save_frames(pred_future, artifact_dir / "pred_future"),
        }
        records.append(record)
        print(
            f"[gt-history] episode={record['episode_id']} start={record['window_start']} "
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
    plans = build_plan(args, rows)

    manifest = {
        "protocol_mode": "gt_history",
        "independence_rule": "Every window reloads GT history; predictions are never reused as history.",
        "bwm_root": str(args.bwm_root),
        "worldarena_root": str(args.worldarena_root),
        "dataset_base": str(args.dataset_base),
        "metadata": str(args.metadata),
        "reference_stat": stat_provenance,
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": checkpoint_sha256,
            "sha256_source": "computed" if args.checkpoint_sha256 is None else "explicit_argument",
        },
        "history_frames": args.history_frames,
        "future_frames": args.future_frames,
        "stride": args.stride,
        "height": args.height,
        "width": args.width,
        "windows": len(plans),
        "plan_only": args.plan_only,
    }
    _write_json(args.output_dir / "run_manifest.json", manifest, args.overwrite)

    plan_records = []
    for global_index, plan in enumerate(plans):
        row = rows[plan["metadata_index"]]
        record = _base_record(
            row, plan, args, int(args.seed) + global_index, stat_provenance["sha256"]
        )
        record["alignment"] = plan["alignment"]
        plan_records.append(record)
    _write_jsonl(args.output_dir / "window_plan.jsonl", plan_records, args.overwrite)
    print(f"[validated] {len(plans)} strictly aligned GT-history windows", flush=True)

    if args.plan_only:
        return

    records = _run_generation(args, rows, plans, stat, stat_provenance["sha256"])
    _write_jsonl(args.output_dir / "windows.jsonl", records, args.overwrite)
    summary = aggregate_records(
        records,
        bootstrap_samples=args.bootstrap_samples,
        confidence=0.95,
        seed=args.seed,
    )
    _write_json(args.output_dir / "summary.json", summary, args.overwrite)


if __name__ == "__main__":
    main()
