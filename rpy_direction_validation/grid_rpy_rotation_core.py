#!/usr/bin/env python3
"""RoboTwin grid-point TCP-local RPY rotation validation task."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from envs._base_task import Base_Task
from envs.utils import Action, ArmTag

from rpy_direction_core import (
    make_local_rpy_target_quat,
    quat_angle_error_deg,
    relative_quat_to_local_rpy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GRID_CENTER_MODE = "arms_midpoint"
GRID_CENTER_OFFSET_XYZ = [-0.035, 0.10, 0.08]
GRID_CENTER_ABSOLUTE_XYZ = [-0.20, -0.20, 1.00]
GRID_SPACING_XYZ = [0.09, 0.07, 0.09]

AXES = ("roll", "pitch", "yaw")
ANGLES_DEG = (30, -30, 90, -90, 180, -180)
SINGLE_POINT_IDS = tuple(range(18))
DUAL_LEFT_POINT_IDS = (0, 3, 6, 9, 12, 15)
DUAL_ROTATION_MODES = ("same_angle", "mirrored_sign")
DUAL_MIRROR_AXIS = "x"
START_MODES = ("move_to_point", "direct_at_point")

TARGET_TOLERANCE_M = 0.035
POSITION_TOLERANCE_M = 0.03
ANGLE_TOLERANCE_DEG = 10.0
MAX_EPISODE_FRAMES: int | None = None
PREPARE_POSITION_ONLY = True
POSITION_ONLY_CONSTRAINT_POSE = [1, 1, 1, 0, 0, 0]
PREPARE_CAPTURE_FREQ = 10


class EpisodeFrameLimitExceeded(RuntimeError):
    pass


def as_float_list(value: Iterable[float]) -> list[float]:
    return [float(x) for x in value]


def distance(a: Iterable[float], b: Iterable[float]) -> float:
    return float(np.linalg.norm(np.asarray(list(a), dtype=np.float64) - np.asarray(list(b), dtype=np.float64)))


def expected_delta_rpy(axis: str, angle_deg: float) -> list[float]:
    values = {"roll": [angle_deg, 0.0, 0.0], "pitch": [0.0, angle_deg, 0.0], "yaw": [0.0, 0.0, angle_deg]}
    return [float(x) for x in values[axis]]


def _condition_record(
    mode: str,
    axis: str,
    angle_deg: int,
    start_mode: str,
    left_point_id: int | None = None,
    right_point_id: int | None = None,
    arm: str | None = None,
    dual_rotation_mode: str | None = None,
) -> dict:
    condition = {
        "condition_id": 0,
        "mode": mode,
        "axis": axis,
        "angle_deg": int(angle_deg),
        "start_mode": start_mode,
        "rotation_frame": "tcp_local",
        "left_point_id": left_point_id,
        "right_point_id": right_point_id,
        "arm": arm,
        "dual_rotation_mode": dual_rotation_mode,
        "seed_policy": "fixed_zero",
    }
    sign = "p" if angle_deg > 0 else "n"
    condition["angle_label"] = f"{sign}{abs(int(angle_deg))}"
    return condition


def build_conditions() -> list[dict]:
    conditions = []
    for start_mode in START_MODES:
        for arm in ("left", "right"):
            for point_id in SINGLE_POINT_IDS:
                for axis in AXES:
                    for angle_deg in ANGLES_DEG:
                        conditions.append(
                            _condition_record(
                                mode="single",
                                arm=arm,
                                left_point_id=point_id if arm == "left" else None,
                                right_point_id=point_id if arm == "right" else None,
                                axis=axis,
                                angle_deg=angle_deg,
                                start_mode=start_mode,
                            )
                        )

        for left_point_id in DUAL_LEFT_POINT_IDS:
            right_point_id = mirror_point_id(left_point_id, DUAL_MIRROR_AXIS)
            for axis in AXES:
                for angle_deg in ANGLES_DEG:
                    for dual_rotation_mode in DUAL_ROTATION_MODES:
                        conditions.append(
                            _condition_record(
                                mode="both_mirrored",
                                left_point_id=left_point_id,
                                right_point_id=right_point_id,
                                axis=axis,
                                angle_deg=angle_deg,
                                start_mode=start_mode,
                                dual_rotation_mode=dual_rotation_mode,
                            )
                        )

    for condition_id, condition in enumerate(conditions):
        condition["condition_id"] = int(condition_id)
    return conditions


def point_id_to_grid_index(point_id: int) -> list[int]:
    point_id = int(point_id)
    if point_id < 0 or point_id > 26:
        raise ValueError(f"invalid point id: {point_id}")
    iz = point_id // 9
    rem = point_id % 9
    iy = rem // 3
    ix = rem % 3
    return [int(ix), int(iy), int(iz)]


def grid_index_to_point_id(grid_index: Iterable[int]) -> int:
    ix, iy, iz = [int(v) for v in grid_index]
    if any(v not in (0, 1, 2) for v in (ix, iy, iz)):
        raise ValueError(f"invalid grid index: {grid_index}")
    return int(iz * 9 + iy * 3 + ix)


def mirror_point_id(point_id: int, axis: str = DUAL_MIRROR_AXIS) -> int:
    axis_to_index = {"x": 0, "y": 1, "z": 2}
    if axis not in axis_to_index:
        raise ValueError(f"unsupported mirror axis: {axis}")
    grid_index = point_id_to_grid_index(point_id)
    axis_idx = axis_to_index[axis]
    grid_index[axis_idx] = 2 - grid_index[axis_idx]
    return grid_index_to_point_id(grid_index)


class rpy_grid_rpy_rotation(Base_Task):
    """Validate TCP-local RPY rotations at implicit 3D grid points."""

    def setup_demo(self, **kwags):
        self.fixed_seed = int(kwags.get("seed", 0))
        self.frame_limit_exceeded = False
        super()._init_task_env_(**kwags)

    def load_actors(self):
        pass

    def _condition(self) -> dict:
        conditions = build_conditions()
        if self.ep_num >= len(conditions):
            raise IndexError(f"episode {self.ep_num} is outside the condition set of {len(conditions)} episodes")
        return conditions[self.ep_num]

    def _grid_center_pose(
        self,
        left_tcp_reference_pose: Iterable[float],
        right_tcp_reference_pose: Iterable[float] | None = None,
    ) -> list[float]:
        pose = as_float_list(left_tcp_reference_pose)
        if GRID_CENTER_MODE == "arms_midpoint" and right_tcp_reference_pose is not None:
            left_xyz = np.asarray(as_float_list(left_tcp_reference_pose)[:3], dtype=np.float64)
            right_xyz = np.asarray(as_float_list(right_tcp_reference_pose)[:3], dtype=np.float64)
            anchor_xyz = (left_xyz + right_xyz) / 2.0
        elif GRID_CENTER_MODE == "left_tcp_offset":
            anchor_xyz = np.asarray(pose[:3], dtype=np.float64)
        elif GRID_CENTER_MODE == "absolute":
            pose[:3] = as_float_list(GRID_CENTER_ABSOLUTE_XYZ)
            return pose
        else:
            raise ValueError(f"unsupported GRID_CENTER_MODE: {GRID_CENTER_MODE}")
        pose[:3] = (anchor_xyz + np.asarray(GRID_CENTER_OFFSET_XYZ, dtype=np.float64)).tolist()
        return pose

    def _make_grid_positions(self, grid_center_pose: Iterable[float]) -> list[dict]:
        x0, y0, z0 = as_float_list(grid_center_pose)[:3]
        dx, dy, dz = GRID_SPACING_XYZ
        positions = []
        for iz, z_offset in enumerate([-dz, 0.0, dz]):
            for iy, y_offset in enumerate([-dy, 0.0, dy]):
                for ix, x_offset in enumerate([-dx, 0.0, dx]):
                    positions.append(
                        {
                            "pos_id": len(positions),
                            "grid_index": [int(ix), int(iy), int(iz)],
                            "xyz": [x0 + x_offset, y0 + y_offset, z0 + z_offset],
                            "offset_xyz": [x_offset, y_offset, z_offset],
                        }
                    )
        return positions

    def _pose_with_xyz(self, reference_pose: Iterable[float], target_xyz: Iterable[float]) -> list[float]:
        pose = as_float_list(reference_pose)
        pose[:3] = as_float_list(target_xyz)
        return pose

    def _move_to_pose_action(
        self,
        arm_tag: ArmTag,
        target_pose: Iterable[float],
        constraint_pose: list[int] | None = None,
    ):
        return arm_tag, [Action(arm_tag, "move", target_pose=target_pose, constraint_pose=constraint_pose)]

    def _prepare_constraint_pose(self) -> list[int] | None:
        return list(POSITION_ONLY_CONSTRAINT_POSE) if PREPARE_POSITION_ONLY else None

    def _take_picture(self):
        if (
            MAX_EPISODE_FRAMES is not None
            and getattr(self, "save_data", False)
            and getattr(self, "FRAME_IDX", 0) >= MAX_EPISODE_FRAMES
        ):
            self.frame_limit_exceeded = True
            raise EpisodeFrameLimitExceeded(f"episode exceeded {MAX_EPISODE_FRAMES} saved frames")
        return super()._take_picture()

    def _move_with_frame_limit(self, *actions, record: bool = True) -> bool:
        old_save_data = getattr(self, "save_data", False)
        if not record:
            self.save_data = False
        try:
            return bool(self.move(*actions))
        except EpisodeFrameLimitExceeded:
            return False
        finally:
            self.save_data = old_save_data

    def _move_to_pose_with_frame_limit(
        self,
        arm_tag: ArmTag,
        target_pose: Iterable[float],
        constraint_pose: list[int] | None = None,
        record: bool = True,
        save_freq: int | None = -1,
    ) -> bool:
        old_save_data = getattr(self, "save_data", False)
        if not record:
            self.save_data = False
        try:
            if arm_tag == "left":
                arm_result = self.left_move_to_pose(target_pose, constraint_pose=constraint_pose)
                control_seq = {"left_arm": arm_result, "left_gripper": None, "right_arm": None, "right_gripper": None}
            else:
                arm_result = self.right_move_to_pose(target_pose, constraint_pose=constraint_pose)
                control_seq = {"left_arm": None, "left_gripper": None, "right_arm": arm_result, "right_gripper": None}
            if self.plan_success is False or arm_result is None:
                return False
            self.take_dense_action(control_seq, save_freq=save_freq)
            return True
        except EpisodeFrameLimitExceeded:
            return False
        finally:
            self.save_data = old_save_data

    def _frame_limit_fields(self) -> dict:
        return {
            "frame_count": int(getattr(self, "FRAME_IDX", 0)),
            "max_episode_frames": int(MAX_EPISODE_FRAMES) if MAX_EPISODE_FRAMES is not None else None,
            "frame_limit_exceeded": bool(getattr(self, "frame_limit_exceeded", False)),
        }

    def _metric_for_arm(
        self,
        target_xyz: Iterable[float],
        before_rpy_pose: Iterable[float],
        final_pose: Iterable[float],
        axis: str,
        angle_deg: int,
    ) -> dict:
        before = as_float_list(before_rpy_pose)
        final = as_float_list(final_pose)
        actual_delta_rpy = relative_quat_to_local_rpy(before[3:7], final[3:7])
        rotated_pose = list(before)
        rotated_pose[3:7] = make_local_rpy_target_quat(before[3:7], axis, angle_deg)
        target_position_error = distance(target_xyz, before[:3])
        position_drift = distance(before[:3], final[:3])
        angle_error = quat_angle_error_deg(final[3:7], rotated_pose[3:7])
        success = (
            target_position_error <= TARGET_TOLERANCE_M
            and position_drift <= POSITION_TOLERANCE_M
            and angle_error <= ANGLE_TOLERANCE_DEG
        )
        return {
            "target_position_error_m": float(target_position_error),
            "actual_delta_rpy_deg": actual_delta_rpy,
            "expected_delta_rpy_deg": expected_delta_rpy(axis, angle_deg),
            "position_drift_m": float(position_drift),
            "angle_error_deg": float(angle_error),
            "success": bool(success),
        }

    def _write_meta_copy(self, info: dict) -> None:
        if not getattr(self, "save_data", False):
            return
        meta_dir = Path(self.save_dir) / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta_path = meta_dir / f"episode_{int(self.ep_num):06d}.json"
        meta_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _single_arm_episode(self, condition: dict, grid_positions: list[dict]) -> dict:
        arm_name = condition["arm"]
        arm = ArmTag(arm_name)
        other = ArmTag("right" if arm_name == "left" else "left")
        point_id = condition["left_point_id"] if arm_name == "left" else condition["right_point_id"]
        grid_position = grid_positions[int(point_id)]
        record_preparation = condition["start_mode"] == "move_to_point"

        other_home_success = self._move_with_frame_limit(self.back_to_origin(other), record=record_preparation)

        start_pose = as_float_list(self.get_arm_pose(arm))
        target_pose = self._pose_with_xyz(start_pose, grid_position["xyz"])
        move_to_point_success = self._move_to_pose_with_frame_limit(
            arm,
            target_pose,
            constraint_pose=self._prepare_constraint_pose(),
            record=record_preparation,
            save_freq=PREPARE_CAPTURE_FREQ,
        )

        before_rpy_pose = as_float_list(self.get_arm_pose(arm))
        experiment_start_frame = int(getattr(self, "FRAME_IDX", 0))
        rotated_pose = list(before_rpy_pose)
        rotated_pose[3:7] = make_local_rpy_target_quat(before_rpy_pose[3:7], condition["axis"], condition["angle_deg"])
        rotate_success = False
        if other_home_success and move_to_point_success and not self.frame_limit_exceeded:
            rotate_success = self._move_with_frame_limit(self.move_to_pose(arm, rotated_pose), record=True)

        final_pose = as_float_list(self.get_arm_pose(arm))
        metrics = self._metric_for_arm(
            target_xyz=grid_position["xyz"],
            before_rpy_pose=before_rpy_pose,
            final_pose=final_pose,
            axis=condition["axis"],
            angle_deg=condition["angle_deg"],
        )
        prepare_success = bool(other_home_success and move_to_point_success)
        success = bool(
            prepare_success
            and rotate_success
            and self.plan_success
            and metrics["success"]
            and not self.frame_limit_exceeded
        )

        return {
            **condition,
            "experiment": "rpy_grid_rpy_rotation",
            "episode_id": int(self.ep_num),
            "seed": int(getattr(self, "fixed_seed", 0)),
            "experiment_start_frame": experiment_start_frame,
            "grid_center_mode": GRID_CENTER_MODE,
            "grid_center_offset_xyz": GRID_CENTER_OFFSET_XYZ,
            "grid_spacing_xyz": GRID_SPACING_XYZ,
            "prepare_position_only": bool(PREPARE_POSITION_ONLY),
            "prepare_constraint_pose": self._prepare_constraint_pose(),
            "prepare_capture_freq": int(PREPARE_CAPTURE_FREQ),
            "selected_grid_positions": [grid_position],
            "left_target_grid_index": grid_position["grid_index"] if arm_name == "left" else None,
            "right_target_grid_index": grid_position["grid_index"] if arm_name == "right" else None,
            "left_target_xyz": as_float_list(grid_position["xyz"]) if arm_name == "left" else None,
            "right_target_xyz": as_float_list(grid_position["xyz"]) if arm_name == "right" else None,
            "prepare_success": prepare_success,
            "planning_success": bool(self.plan_success),
            "motion_success": bool(move_to_point_success and rotate_success),
            "success": success,
            "abnormal": bool(self.frame_limit_exceeded),
            **self._frame_limit_fields(),
            "arms": {
                arm_name: {
                    "start_pose": start_pose,
                    "before_rpy_pose": before_rpy_pose,
                    "target_pose": rotated_pose,
                    "final_pose": final_pose,
                    **metrics,
                }
            },
            "notes": "Implicit grid point TCP-local RPY rotation validation without visual grid markers.",
        }

    def _dual_arm_episode(self, condition: dict, grid_positions: list[dict]) -> dict:
        left = ArmTag("left")
        right = ArmTag("right")
        left_grid_position = grid_positions[int(condition["left_point_id"])]
        right_grid_position = grid_positions[int(condition["right_point_id"])]

        left_start_pose = as_float_list(self.get_arm_pose(left))
        right_start_pose = as_float_list(self.get_arm_pose(right))
        left_target_pose = self._pose_with_xyz(left_start_pose, left_grid_position["xyz"])
        right_target_pose = self._pose_with_xyz(right_start_pose, right_grid_position["xyz"])
        move_to_point_success = self._move_with_frame_limit(
            self._move_to_pose_action(left, left_target_pose, constraint_pose=self._prepare_constraint_pose()),
            self._move_to_pose_action(right, right_target_pose, constraint_pose=self._prepare_constraint_pose()),
            record=condition["start_mode"] == "move_to_point",
        )

        left_before_rpy_pose = as_float_list(self.get_arm_pose(left))
        right_before_rpy_pose = as_float_list(self.get_arm_pose(right))
        experiment_start_frame = int(getattr(self, "FRAME_IDX", 0))

        left_angle = int(condition["angle_deg"])
        right_angle = left_angle if condition["dual_rotation_mode"] == "same_angle" else -left_angle
        left_rotated_pose = list(left_before_rpy_pose)
        right_rotated_pose = list(right_before_rpy_pose)
        left_rotated_pose[3:7] = make_local_rpy_target_quat(left_before_rpy_pose[3:7], condition["axis"], left_angle)
        right_rotated_pose[3:7] = make_local_rpy_target_quat(right_before_rpy_pose[3:7], condition["axis"], right_angle)
        rotate_success = False
        if move_to_point_success and not self.frame_limit_exceeded:
            rotate_success = self._move_with_frame_limit(
                self.move_to_pose(left, left_rotated_pose),
                self.move_to_pose(right, right_rotated_pose),
                record=True,
            )

        left_final_pose = as_float_list(self.get_arm_pose(left))
        right_final_pose = as_float_list(self.get_arm_pose(right))
        left_metrics = self._metric_for_arm(
            target_xyz=left_grid_position["xyz"],
            before_rpy_pose=left_before_rpy_pose,
            final_pose=left_final_pose,
            axis=condition["axis"],
            angle_deg=left_angle,
        )
        right_metrics = self._metric_for_arm(
            target_xyz=right_grid_position["xyz"],
            before_rpy_pose=right_before_rpy_pose,
            final_pose=right_final_pose,
            axis=condition["axis"],
            angle_deg=right_angle,
        )
        success = bool(
            move_to_point_success
            and rotate_success
            and self.plan_success
            and left_metrics["success"]
            and right_metrics["success"]
            and not self.frame_limit_exceeded
        )

        return {
            **condition,
            "experiment": "rpy_grid_rpy_rotation",
            "episode_id": int(self.ep_num),
            "seed": int(getattr(self, "fixed_seed", 0)),
            "experiment_start_frame": experiment_start_frame,
            "grid_center_mode": GRID_CENTER_MODE,
            "grid_center_offset_xyz": GRID_CENTER_OFFSET_XYZ,
            "grid_spacing_xyz": GRID_SPACING_XYZ,
            "prepare_position_only": bool(PREPARE_POSITION_ONLY),
            "prepare_constraint_pose": self._prepare_constraint_pose(),
            "prepare_capture_freq": int(PREPARE_CAPTURE_FREQ),
            "selected_grid_positions": [left_grid_position, right_grid_position],
            "left_target_grid_index": left_grid_position["grid_index"],
            "right_target_grid_index": right_grid_position["grid_index"],
            "left_target_xyz": as_float_list(left_grid_position["xyz"]),
            "right_target_xyz": as_float_list(right_grid_position["xyz"]),
            "left_angle_deg": int(left_angle),
            "right_angle_deg": int(right_angle),
            "prepare_success": bool(move_to_point_success),
            "planning_success": bool(self.plan_success),
            "motion_success": bool(move_to_point_success and rotate_success),
            "success": success,
            "abnormal": bool(self.frame_limit_exceeded),
            **self._frame_limit_fields(),
            "arms": {
                "left": {
                    "start_pose": left_start_pose,
                    "before_rpy_pose": left_before_rpy_pose,
                    "target_pose": left_rotated_pose,
                    "final_pose": left_final_pose,
                    **left_metrics,
                },
                "right": {
                    "start_pose": right_start_pose,
                    "before_rpy_pose": right_before_rpy_pose,
                    "target_pose": right_rotated_pose,
                    "final_pose": right_final_pose,
                    **right_metrics,
                },
            },
            "notes": "Mirrored dual-arm targets are planned per arm and executed together; this is not coupled dual-arm collision avoidance.",
        }

    def play_once(self):
        condition = self._condition()
        left_reference_pose = as_float_list(self.get_arm_pose(ArmTag("left")))
        right_reference_pose = as_float_list(self.get_arm_pose(ArmTag("right")))
        center_pose = self._grid_center_pose(left_reference_pose, right_reference_pose)
        grid_positions = self._make_grid_positions(center_pose)

        if condition["mode"] == "single":
            info = self._single_arm_episode(condition, grid_positions)
        elif condition["mode"] == "both_mirrored":
            info = self._dual_arm_episode(condition, grid_positions)
        else:
            raise ValueError(f"unsupported mode: {condition['mode']}")

        info["left_tcp_reference_xyz"] = left_reference_pose[:3]
        info["right_tcp_reference_xyz"] = right_reference_pose[:3]
        info["grid_center_xyz"] = center_pose[:3]
        info["num_conditions"] = len(build_conditions())
        self._last_info = info
        self.info["info"] = info
        self._write_meta_copy(info)

        print(
            "[rpy_grid_rpy_rotation] "
            f"episode={self.ep_num} condition={condition['condition_id']} mode={condition['mode']} "
            f"start_mode={condition['start_mode']} axis={condition['axis']} "
            f"angle={condition['angle_deg']} success={info['success']} abnormal={info['abnormal']}"
        )
        return self.info

    def check_success(self):
        return bool(getattr(self, "_last_info", {}).get("success", False))
