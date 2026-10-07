#!/usr/bin/env python3
"""Load BWM and run one 9-history + 72-future GT-history window."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from eval_gt_history import (  # noqa: E402
    _build_loaders,
    _build_pipeline,
    _save_mp4,
    _tensor_to_uint8_frames,
)
from protocol import load_and_validate_reference_stat, read_metadata, resolve_data_path, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bwm-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-base", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--reference-stat", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=Path("configs/model/wan2_2_ti2v_5b.yaml"))
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Runtime smoke requires exactly one visible allowed GPU; launch with CUDA_VISIBLE_DEVICES=0"
        )
    output_dir = args.output_dir.resolve()
    report_path = output_dir / "runtime_smoke.json"
    video_path = output_dir / "pred_future.mp4"
    for path in (report_path, video_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")

    bwm_root = args.bwm_root.resolve()
    dataset_base = args.dataset_base.resolve()
    rows = read_metadata(args.metadata.resolve())
    if not rows:
        raise ValueError("Runtime smoke metadata is empty")
    row = rows[0]
    frame_indices = list(range(int(row.get("start_frame", 0)), int(row.get("start_frame", 0)) + 81))
    stat, stat_provenance = load_and_validate_reference_stat(args.reference_stat.resolve())
    runtime_args = SimpleNamespace(
        bwm_root=bwm_root,
        model_root=args.model_root.resolve(),
        checkpoint=args.checkpoint.resolve(),
        model_config=args.model_config,
        mixed_precision="bf16",
        dataset_base=dataset_base,
        height=480,
        width=640,
        history_frames=9,
        future_frames=72,
        tiled=False,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
    )

    video_path_source = resolve_data_path(dataset_base, row.get("video"), "video")
    action_path = resolve_data_path(dataset_base, row.get("action"), "action")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    pipe = _build_pipeline(runtime_args)
    load_seconds = time.perf_counter() - started
    video_loader, action_loader = _build_loaders(runtime_args, stat)
    gt_video = video_loader({"data": str(video_path_source), "frame_indices": frame_indices})
    action = action_loader({"data": str(action_path), "frame_indices": frame_indices})
    if tuple(gt_video.shape[:3]) != (1, 3, 81):
        raise ValueError(f"Expected video shape (1,3,81,H,W), got {tuple(gt_video.shape)}")
    if tuple(action.shape) != (1, 81, 14):
        raise ValueError(f"Expected action shape (1,81,14), got {tuple(action.shape)}")

    generation_started = time.perf_counter()
    with torch.inference_mode():
        prediction = pipe(
            input_video=gt_video[:, :, :9],
            action=action,
            seed=args.seed,
            rand_device="cpu",
            tiled=False,
            height=480,
            width=640,
            num_frames=81,
            num_history_frames=9,
            cfg_scale=args.cfg_scale,
            num_inference_steps=args.num_inference_steps,
            progress_bar_cmd=lambda iterable, *unused_args, **unused_kwargs: iterable,
            output_type="floatpoint",
        )
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_started
    if tuple(prediction.shape[:3]) != (1, 3, 81):
        raise ValueError(f"Expected prediction shape (1,3,81,H,W), got {tuple(prediction.shape)}")
    frames = _tensor_to_uint8_frames(prediction[:, :, 9:])
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_mp4(frames, video_path, 30.0, args.overwrite)
    report = {
        "status": "PASS",
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "visible_gpu_count": torch.cuda.device_count(),
        "visible_gpu_name": torch.cuda.get_device_name(0),
        "bwm_root": str(bwm_root),
        "model_root": str(args.model_root.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
        "reference_stat": stat_provenance,
        "source_video": str(video_path_source),
        "source_action": str(action_path),
        "frame_indices": frame_indices,
        "history_frames": 9,
        "future_frames": 72,
        "num_inference_steps": args.num_inference_steps,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "prediction_shape": list(prediction.shape),
        "future_mp4": str(video_path),
        "future_mp4_bytes": video_path.stat().st_size,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
