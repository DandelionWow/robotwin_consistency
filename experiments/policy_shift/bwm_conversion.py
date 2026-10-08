"""Single RoboTwin trajectory -> BWM conditioning conversion layer.

The only supported conditioning semantics are realized dual-arm EEF poses and
realized gripper state. Policy commands are deliberately not accepted here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from protocol import require_uniform_timestamp_grid


EEF_INDICES_26 = np.asarray([7, 8, 9, 10, 11, 12, 6, 20, 21, 22, 23, 24, 25, 19])


@dataclass(frozen=True)
class BWMTrajectory:
    head_rgb: np.ndarray
    realized_eef_wxyz16: np.ndarray
    state_pose14: np.ndarray
    frame_timestamps: np.ndarray
    task: str
    pair_id: str
    side: str
    success: bool
    env_seed: int
    raw_fps: float

    def validate(self) -> None:
        length = self.state_pose14.shape[0]
        if self.state_pose14.shape != (length, 14):
            raise ValueError(f"state_pose14 must have shape (T, 14), got {self.state_pose14.shape}")
        if self.realized_eef_wxyz16.shape != (length, 16):
            raise ValueError(
                "realized_eef_wxyz16 must have shape (T, 16), "
                f"got {self.realized_eef_wxyz16.shape}"
            )
        if self.head_rgb.ndim != 4 or self.head_rgb.shape[0] != length or self.head_rgb.shape[-1] != 3:
            raise ValueError(f"head_rgb must have shape (T, H, W, 3), got {self.head_rgb.shape}")
        if self.frame_timestamps.shape != (length,):
            raise ValueError(
                f"frame_timestamps must have shape ({length},), got {self.frame_timestamps.shape}"
            )
        if self.side not in {"expert", "pi05"}:
            raise ValueError(f"side must be expert or pi05, got {self.side!r}")
        if not isinstance(self.success, (bool, np.bool_)):
            raise TypeError("success must be an explicit boolean")
        if not np.isfinite(self.raw_fps) or self.raw_fps <= 0:
            raise ValueError(f"raw_fps must be finite and positive, got {self.raw_fps}")
        for name, values in (
            ("realized_eef_wxyz16", self.realized_eef_wxyz16),
            ("state_pose14", self.state_pose14),
            ("frame_timestamps", self.frame_timestamps),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} contains NaN or infinity")
        require_uniform_timestamp_grid(self.frame_timestamps, 1.0 / self.raw_fps)


def _quaternion_wxyz_to_rpy(quaternion: np.ndarray) -> np.ndarray:
    """Convert normalized wxyz quaternions to extrinsic xyz Euler radians."""

    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.ndim != 2 or quaternion.shape[1] != 4:
        raise ValueError(f"Expected quaternion shape (T, 4), got {quaternion.shape}")
    norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
    if np.any(norm <= 1e-12) or not np.all(np.isfinite(norm)):
        raise ValueError("Quaternion must be finite and non-zero")
    w, x, y, z = (quaternion / norm).T

    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sin_pitch)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.stack((roll, pitch, yaw), axis=1).astype(np.float32)


def state_pose14_from_realized_eef16(realized_eef_wxyz16: np.ndarray) -> np.ndarray:
    """Convert [L xyz,wxyz,grip,R xyz,wxyz,grip] to BWM eef_abs/state_pose."""

    values = np.asarray(realized_eef_wxyz16, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 16:
        raise ValueError(f"Expected realized EEF shape (T, 16), got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("Realized EEF values contain NaN or infinity")
    result = np.empty((values.shape[0], 14), dtype=np.float32)
    result[:, 0:3] = values[:, 0:3]
    result[:, 3:6] = _quaternion_wxyz_to_rpy(values[:, 3:7])
    result[:, 6] = values[:, 7]
    result[:, 7:10] = values[:, 8:11]
    result[:, 10:13] = _quaternion_wxyz_to_rpy(values[:, 11:15])
    result[:, 13] = values[:, 15]
    return result


def realized_eef16_from_fields(
    left_endpose_wxyz7: np.ndarray,
    left_gripper: np.ndarray,
    right_endpose_wxyz7: np.ndarray,
    right_gripper: np.ndarray,
) -> np.ndarray:
    left = np.asarray(left_endpose_wxyz7, dtype=np.float32)
    right = np.asarray(right_endpose_wxyz7, dtype=np.float32)
    left_grip = np.asarray(left_gripper, dtype=np.float32).reshape(-1, 1)
    right_grip = np.asarray(right_gripper, dtype=np.float32).reshape(-1, 1)
    if left.ndim != 2 or left.shape[1] != 7 or right.ndim != 2 or right.shape[1] != 7:
        raise ValueError(f"Endposes must be (T, 7), got {left.shape}/{right.shape}")
    lengths = {left.shape[0], right.shape[0], left_grip.shape[0], right_grip.shape[0]}
    if len(lengths) != 1:
        raise ValueError(
            "Realized EEF field lengths differ: "
            f"left={left.shape[0]} right={right.shape[0]} "
            f"left_gripper={left_grip.shape[0]} right_gripper={right_grip.shape[0]}"
        )
    return np.concatenate((left, left_grip, right, right_grip), axis=1)


def realized_eef16_from_mapping(source: Mapping[str, Any]) -> np.ndarray:
    """Adapter used by a fresh matched collector; it still calls the one 16D converter."""

    required = ("left_endpose", "left_gripper", "right_endpose", "right_gripper")
    missing = [key for key in required if key not in source]
    if missing:
        raise KeyError(f"Missing realized EEF fields: {missing}")
    return realized_eef16_from_fields(
        source["left_endpose"],
        source["left_gripper"],
        source["right_endpose"],
        source["right_gripper"],
    )


def realized_eef16_from_hdf5(path: Path, start_frame: int = 0) -> np.ndarray:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("HDF5 conversion requires h5py") from exc
    with h5py.File(Path(path), "r") as handle:
        required = {
            "left_endpose": "/endpose/left_endpose",
            "left_gripper": "/endpose/left_gripper",
            "right_endpose": "/endpose/right_endpose",
            "right_gripper": "/endpose/right_gripper",
        }
        missing = [hdf5_key for hdf5_key in required.values() if hdf5_key not in handle]
        if missing:
            raise KeyError(f"{path} is missing realized EEF fields: {missing}")
        fields = {
            name: np.asarray(handle[key][int(start_frame) :]) for name, key in required.items()
        }
    return realized_eef16_from_mapping(fields)


def state_pose14_from_bwm_state26(state26: np.ndarray) -> np.ndarray:
    """Project a validated BWM/LeRobot 26D realized-state row to eef_abs."""

    values = np.asarray(state26, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 26:
        raise ValueError(f"Expected BWM state shape (T, 26), got {values.shape}")
    result = values[:, EEF_INDICES_26]
    if not np.all(np.isfinite(result)):
        raise ValueError("BWM state contains non-finite realized EEF values")
    return result.astype(np.float32, copy=False)


def build_bwm_trajectory(
    *,
    head_rgb: np.ndarray,
    realized_eef_wxyz16: np.ndarray,
    frame_timestamps: Sequence[float],
    task: str,
    pair_id: str,
    side: str,
    success: bool,
    env_seed: int,
    raw_fps: float,
) -> BWMTrajectory:
    eef16 = np.asarray(realized_eef_wxyz16, dtype=np.float32)
    trajectory = BWMTrajectory(
        head_rgb=np.asarray(head_rgb),
        realized_eef_wxyz16=eef16,
        state_pose14=state_pose14_from_realized_eef16(eef16),
        frame_timestamps=np.asarray(frame_timestamps, dtype=np.float64),
        task=str(task),
        pair_id=str(pair_id),
        side=str(side),
        success=success,
        env_seed=int(env_seed),
        raw_fps=float(raw_fps),
    )
    trajectory.validate()
    return trajectory
