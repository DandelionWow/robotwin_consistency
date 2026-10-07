#!/usr/bin/env python3
"""Load one RoboTwin Pi0.5 checkpoint and execute one real inference call."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np

from protocol import checkpoint_identity, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOTWIN_ROOT = PROJECT_ROOT / "third_party/robotwin"


def _decode_rgb(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 3:
        return np.asarray(array[..., :3], dtype=np.uint8)
    encoded = np.frombuffer(bytes(value), dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode HDF5 RGB frame")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _load_observation(path: Path) -> tuple[list[np.ndarray], np.ndarray]:
    with h5py.File(path, "r") as handle:
        images = [
            _decode_rgb(handle[f"/observation/{camera}/rgb"][0])
            for camera in ("head_camera", "right_camera", "left_camera")
        ]
        state = np.asarray(handle["/joint_action/vector"][0], dtype=np.float32)
    if state.shape != (14,):
        raise ValueError(f"Expected a 14D RoboTwin policy state, got {state.shape}")
    return images, state


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-config", type=Path, required=True)
    parser.add_argument("--observation-hdf5", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in {"0", "1", "2", "3"}:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must expose exactly one physical GPU in {0,1,2,3}")
    cuda_root = Path(os.environ.get("CUDA_ROOT", ""))
    ptxas = cuda_root / "bin/ptxas"
    if not ptxas.is_file():
        raise FileNotFoundError(f"CUDA_ROOT does not provide bin/ptxas: {cuda_root}")
    ptxas_version = subprocess.run(
        [str(ptxas), "--version"], check=True, capture_output=True, text=True
    ).stdout.strip().splitlines()[-1]

    policy_config = args.policy_config.resolve()
    config = json.loads(policy_config.read_text(encoding="utf-8"))
    checkpoint_dir = (
        ROBOTWIN_ROOT
        / "policy/pi05/checkpoints"
        / str(config["train_config_name"])
        / str(config["model_name"])
        / str(config["checkpoint_id"])
    ).resolve()
    identity = checkpoint_identity(checkpoint_dir)
    if identity["sha256"] != config.get("checkpoint_sha256"):
        raise ValueError("Policy config checkpoint SHA256 does not match local inference files")
    images, state = _load_observation(args.observation_hdf5.resolve())

    sys.path.insert(0, str(ROBOTWIN_ROOT))
    sys.path.insert(0, str(ROBOTWIN_ROOT / "policy"))
    previous_cwd = Path.cwd()
    os.chdir(ROBOTWIN_ROOT)
    try:
        import jax
        from pi05.pi_model import PI0

        load_started = time.perf_counter()
        model = PI0(
            str(config["train_config_name"]),
            str(config["model_name"]),
            int(config["checkpoint_id"]),
            int(config["pi0_step"]),
        )
        load_seconds = time.perf_counter() - load_started
        model.set_language(str(args.instruction))
        model.update_observation_window(images, state)
        inference_started = time.perf_counter()
        actions = np.asarray(model.get_action())
        # JAX dispatch can be asynchronous; NumPy materialization above blocks.
        inference_seconds = time.perf_counter() - inference_started
        devices = [str(device) for device in jax.devices()]
        jax_version = jax.__version__
    finally:
        os.chdir(previous_cwd)

    if actions.ndim != 2 or actions.shape[0] < int(config["pi0_step"]) or actions.shape[1] != 14:
        raise ValueError(f"Unexpected Pi0.5 action shape: {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("Pi0.5 inference returned non-finite actions")
    payload = {
        "status": "PASS",
        "policy_config": str(policy_config),
        "policy_config_sha256": sha256_file(policy_config),
        "checkpoint": identity,
        "checkpoint_source": config.get("checkpoint_source"),
        "checkpoint_source_revision": config.get("checkpoint_source_revision"),
        "model_code_commit": subprocess.run(
            ["git", "-C", str(ROBOTWIN_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "runtime": {
            "python": sys.executable,
            "jax": jax_version,
            "visible_physical_gpu": int(visible),
            "jax_devices": devices,
            "cuda_root": str(cuda_root.resolve()),
            "ptxas_version": ptxas_version,
            "python_no_user_site": os.environ.get("PYTHONNOUSERSITE") == "1",
        },
        "input": {
            "observation_hdf5": str(args.observation_hdf5.resolve()),
            "instruction": str(args.instruction),
            "image_shapes": [list(image.shape) for image in images],
            "state_shape": list(state.shape),
        },
        "output": {
            "action_shape": list(actions.shape),
            "dtype": str(actions.dtype),
            "finite": True,
            "minimum": float(actions.min()),
            "maximum": float(actions.max()),
        },
        "timing_seconds": {
            "load": load_seconds,
            "first_inference": inference_seconds,
        },
    }
    _write_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
