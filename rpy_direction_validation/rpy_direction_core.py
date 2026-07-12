#!/usr/bin/env python3
"""RoboTwin RPY direction validation task core.

This module intentionally lives outside third_party/robotwin.  The RoboTwin
task file under envs/ is only a thin shim that imports this class.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from envs._base_task import Base_Task
from envs.utils import ArmTag


AXES = ("roll", "pitch", "yaw")
SMOKE_ANGLES_DEG = (30, -30)
FULL_ANGLES_DEG = (30, -30, 90, -90, 180, -180)
POSITION_TOLERANCE_M = 0.03
ANGLE_TOLERANCE_DEG = 10.0
GRID_STEP_M = 0.06
GRID_Z_OFFSET_M = 0.08


def env_flag(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def as_float_list(value: Iterable[float]) -> list[float]:
    return [float(x) for x in value]


def normalize_quat_wxyz(q: Iterable[float]) -> np.ndarray:
    quat = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm == 0:
        raise ValueError("zero quaternion")
    return quat / norm


def quat_wxyz_to_xyzw(q: Iterable[float]) -> list[float]:
    qw, qx, qy, qz = normalize_quat_wxyz(q)
    return [float(qx), float(qy), float(qz), float(qw)]


def quat_xyzw_to_wxyz(q: Iterable[float]) -> list[float]:
    qx, qy, qz, qw = np.asarray(q, dtype=np.float64)
    return as_float_list(normalize_quat_wxyz([qw, qx, qy, qz]))


def quat_mul_wxyz(a: Iterable[float], b: Iterable[float]) -> np.ndarray:
    aw, ax, ay, az = normalize_quat_wxyz(a)
    bw, bx, by, bz = normalize_quat_wxyz(b)
    return normalize_quat_wxyz(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_inv_wxyz(q: Iterable[float]) -> np.ndarray:
    qw, qx, qy, qz = normalize_quat_wxyz(q)
    return np.array([qw, -qx, -qy, -qz], dtype=np.float64)


def axis_delta_quat_wxyz(axis: str, angle_deg: float) -> np.ndarray:
    axis_to_vec = {
        "roll": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "pitch": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "yaw": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }
    if axis not in axis_to_vec:
        raise ValueError(f"unknown RPY axis: {axis}")
    half = math.radians(float(angle_deg)) / 2.0
    quat = np.concatenate([[math.cos(half)], math.sin(half) * axis_to_vec[axis]])
    return normalize_quat_wxyz(quat)


def make_local_rpy_target_quat(current_quat_wxyz: Iterable[float], axis: str, angle_deg: float) -> list[float]:
    """Apply an RPY delta in the TCP local frame.

    The multiplication order is q_target = q_current * q_delta.  Reversing the
    order would apply the delta in the world frame for this convention.
    """

    return as_float_list(quat_mul_wxyz(current_quat_wxyz, axis_delta_quat_wxyz(axis, angle_deg)))


def relative_quat_wxyz(before_quat_wxyz: Iterable[float], after_quat_wxyz: Iterable[float]) -> np.ndarray:
    return quat_mul_wxyz(quat_inv_wxyz(before_quat_wxyz), after_quat_wxyz)


def relative_quat_to_local_rpy(before_quat_wxyz: Iterable[float], after_quat_wxyz: Iterable[float]) -> list[float]:
    q_rel = relative_quat_wxyz(before_quat_wxyz, after_quat_wxyz)
    try:
        from scipy.spatial.transform import Rotation as R

        euler = R.from_quat(quat_wxyz_to_xyzw(q_rel)).as_euler("xyz", degrees=True)
        return as_float_list(euler)
    except Exception:
        import transforms3d as t3d

        euler_rad = t3d.euler.quat2euler(q_rel, axes="sxyz")
        return as_float_list(np.degrees(euler_rad))


def quat_angle_error_deg(actual_wxyz: Iterable[float], expected_wxyz: Iterable[float]) -> float:
    actual = normalize_quat_wxyz(actual_wxyz)
    expected = normalize_quat_wxyz(expected_wxyz)
    dot = abs(float(np.dot(actual, expected)))
    dot = min(1.0, max(-1.0, dot))
    return float(math.degrees(2.0 * math.acos(dot)))


def build_conditions(full: bool = False) -> list[dict]:
    pos_ids = range(9) if full else [4]
    angles = FULL_ANGLES_DEG if full else SMOKE_ANGLES_DEG
    conditions = []
    for pos_id in pos_ids:
        for axis in AXES:
            for angle in angles:
                sign = "p" if angle > 0 else "n"
                conditions.append(
                    {
                        "mode": "a_from_origin",
                        "pos_id": int(pos_id),
                        "axis": axis,
                        "angle_deg": int(angle),
                        "angle_label": f"{sign}{abs(int(angle))}",
                        "rotation_frame": "tcp_local",
                        "arm": "left",
                        "right_arm": "origin",
                    }
                )
    return conditions


def expected_delta_rpy(axis: str, angle_deg: float) -> list[float]:
    values = {"roll": [angle_deg, 0.0, 0.0], "pitch": [0.0, angle_deg, 0.0], "yaw": [0.0, 0.0, angle_deg]}
    return [float(x) for x in values[axis]]


class rpy_direction_validation(Base_Task):
    """Empty-table TCP-local RPY direction validation task."""

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        pass

    def _condition(self) -> dict:
        full = env_flag("RPY_FULL")
        conditions = build_conditions(full=full)
        if self.ep_num >= len(conditions):
            raise IndexError(
                f"episode {self.ep_num} is outside the {'full' if full else 'smoke'} condition set "
                f"of {len(conditions)} episodes"
            )
        return conditions[self.ep_num]

    def _make_grid_positions(self, base_pose: Iterable[float]) -> list[list[float]]:
        x0, y0, z0 = [float(v) for v in list(base_pose)[:3]]
        dx = GRID_STEP_M
        dy = GRID_STEP_M
        z0 = z0 + GRID_Z_OFFSET_M
        return [
            [x0 - dx, y0 - dy, z0],
            [x0, y0 - dy, z0],
            [x0 + dx, y0 - dy, z0],
            [x0 - dx, y0, z0],
            [x0, y0, z0],
            [x0 + dx, y0, z0],
            [x0 - dx, y0 + dy, z0],
            [x0, y0 + dy, z0],
            [x0 + dx, y0 + dy, z0],
        ]

    def _compute_metrics(self, condition: dict, before_pose: Iterable[float], final_pose: Iterable[float]) -> dict:
        before = np.asarray(before_pose, dtype=np.float64)
        final = np.asarray(final_pose, dtype=np.float64)
        q_rel = relative_quat_wxyz(before[3:7], final[3:7])
        q_expected = axis_delta_quat_wxyz(condition["axis"], condition["angle_deg"])
        actual_rpy = relative_quat_to_local_rpy(before[3:7], final[3:7])
        position_drift = float(np.linalg.norm(final[:3] - before[:3]))
        angle_error = quat_angle_error_deg(q_rel, q_expected)
        success = position_drift <= POSITION_TOLERANCE_M and angle_error <= ANGLE_TOLERANCE_DEG
        return {
            "actual_delta_rpy_deg": actual_rpy,
            "position_drift_m": position_drift,
            "angle_error_deg": angle_error,
            "success": bool(success),
        }

    def _build_info(
        self,
        condition: dict,
        start_pose: Iterable[float],
        before_rpy_pose: Iterable[float],
        target_pose: Iterable[float],
        final_pose: Iterable[float],
        experiment_start_frame: int,
        metrics: dict,
    ) -> dict:
        info = dict(condition)
        info.update(
            {
                "experiment": "rpy_direction_validation",
                "episode_id": int(self.ep_num),
                "experiment_start_frame": int(experiment_start_frame),
                "target_position_xyz": as_float_list(target_pose[:3]),
                "start_pose": as_float_list(start_pose),
                "before_rpy_pose": as_float_list(before_rpy_pose),
                "target_pose": as_float_list(target_pose),
                "final_pose": as_float_list(final_pose),
                "expected_delta_rpy_deg": expected_delta_rpy(condition["axis"], condition["angle_deg"]),
                "notes": "RoboTwin synthesized reference video for TCP-local RPY direction validation.",
            }
        )
        info.update(metrics)
        return info

    def _write_meta_copy(self, info: dict) -> None:
        if not getattr(self, "save_data", False):
            return
        meta_dir = Path(self.save_dir) / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta_path = meta_dir / f"episode_{int(self.ep_num):06d}.json"
        meta_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def play_once(self):
        left = ArmTag("left")
        right = ArmTag("right")
        condition = self._condition()

        self.move(self.back_to_origin(right))

        start_pose = as_float_list(self.get_arm_pose(left))
        positions = self._make_grid_positions(start_pose)
        target_position = positions[condition["pos_id"]]

        target_pose = list(start_pose)
        target_pose[:3] = target_position
        self.move(self.move_to_pose(left, target_pose))

        before_rpy_pose = as_float_list(self.get_arm_pose(left))
        experiment_start_frame = int(getattr(self, "FRAME_IDX", 0))
        rotated_pose = list(before_rpy_pose)
        rotated_pose[3:7] = make_local_rpy_target_quat(
            current_quat_wxyz=before_rpy_pose[3:7],
            axis=condition["axis"],
            angle_deg=condition["angle_deg"],
        )
        self.move(self.move_to_pose(left, rotated_pose))

        final_pose = as_float_list(self.get_arm_pose(left))
        metrics = self._compute_metrics(condition, before_rpy_pose, final_pose)
        info = self._build_info(
            condition=condition,
            start_pose=start_pose,
            before_rpy_pose=before_rpy_pose,
            target_pose=rotated_pose,
            final_pose=final_pose,
            experiment_start_frame=experiment_start_frame,
            metrics=metrics,
        )
        self._last_condition = condition
        self._last_before_rpy_pose = before_rpy_pose
        self._last_final_pose = final_pose
        self._last_info = info
        self.info["info"] = info
        self._write_meta_copy(info)

        print(
            "[rpy_direction_validation] "
            f"episode={self.ep_num} pos_id={condition['pos_id']} axis={condition['axis']} "
            f"target_angle_deg={condition['angle_deg']} "
            f"actual_delta_roll={metrics['actual_delta_rpy_deg'][0]:.3f} "
            f"actual_delta_pitch={metrics['actual_delta_rpy_deg'][1]:.3f} "
            f"actual_delta_yaw={metrics['actual_delta_rpy_deg'][2]:.3f} "
            f"position_drift={metrics['position_drift_m']:.4f} "
            f"angle_error={metrics['angle_error_deg']:.3f} success={metrics['success']}"
        )
        return self.info

    def check_success(self):
        if not hasattr(self, "_last_condition") or not hasattr(self, "_last_before_rpy_pose") or not hasattr(self, "_last_final_pose"):
            return False
        metrics = self._compute_metrics(self._last_condition, self._last_before_rpy_pose, self._last_final_pose)
        if hasattr(self, "_last_info"):
            self._last_info.update(metrics)
            self.info["info"] = self._last_info
        return bool(metrics["success"])
