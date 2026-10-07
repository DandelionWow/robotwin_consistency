"""Thin adapters over the checked-in WorldArena metric implementations."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np


class WorldArenaBasicMetrics:
    """Call WorldArena's own PSNR and SSIM functions on one aligned window."""

    def __init__(self, worldarena_root: Path):
        metric_path = worldarena_root / "WorldArena" / "basic_metrics.py"
        if not metric_path.is_file():
            raise FileNotFoundError(f"WorldArena basic metric implementation not found: {metric_path}")
        spec = importlib.util.spec_from_file_location("_worldarena_basic_metrics", metric_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import {metric_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.module: ModuleType = module
        self.source_path = metric_path.resolve()

    @staticmethod
    def _validate(gt_frames: np.ndarray, pred_frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gt = np.asarray(gt_frames)
        pred = np.asarray(pred_frames)
        if gt.shape != pred.shape:
            raise ValueError(f"Strict alignment failed: GT shape {gt.shape} != prediction shape {pred.shape}")
        if gt.ndim != 4 or gt.shape[-1] != 3 or gt.shape[0] == 0:
            raise ValueError(f"Expected non-empty THWC RGB arrays, got {gt.shape}")
        if gt.dtype != np.uint8 or pred.dtype != np.uint8:
            raise TypeError(f"Expected uint8 GT/pred frames, got {gt.dtype}/{pred.dtype}")
        return gt, pred

    def score(self, gt_frames: np.ndarray, pred_frames: np.ndarray) -> dict[str, float]:
        gt, pred = self._validate(gt_frames, pred_frames)
        psnr = []
        ssim = []
        for gt_frame, pred_frame in zip(gt, pred, strict=True):
            # These are the exact callables used by WorldArena.compute_basic_metrics.
            psnr.append(float(self.module.peak_signal_noise_ratio(gt_frame, pred_frame)))
            ssim.append(float(self.module.cal_ssim(gt_frame, pred_frame)))
        return {"psnr": float(np.mean(psnr)), "ssim": float(np.mean(ssim))}
