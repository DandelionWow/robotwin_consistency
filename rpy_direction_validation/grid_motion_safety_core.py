#!/usr/bin/env python3
"""Minimal RoboTwin visual-only grid motion safety task.

The task intentionally keeps the implementation outside third_party/robotwin.
The RoboTwin env file is only a thin shim that imports this class.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import sapien.core as sapien

from envs._base_task import Base_Task
from envs.utils import ArmTag


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "rpy_direction_validation" / "grid_motion_safety_report.json"

GRID_CENTER_MODE = "arms_midpoint"  # "arms_midpoint", "left_tcp_offset", or "absolute"
GRID_CENTER_OFFSET_XYZ = [-0.035, 0.10, 0.08]
GRID_CENTER_ABSOLUTE_XYZ = [-0.20, -0.20, 1.00]
# 网格中心 = anchor + GRID_CENTER_OFFSET_XYZ，单位米。
#   - GRID_CENTER_MODE="arms_midpoint": anchor 是初始左/右 TCP 位置中点。
#   - GRID_CENTER_MODE="left_tcp_offset": anchor 是初始左臂 TCP 位置。
#   - GRID_CENTER_MODE="absolute": 直接使用 GRID_CENTER_ABSOLUTE_XYZ，忽略 offset。
#   - 增大 x/y/z 分别沿 RoboTwin 世界坐标 +x/+y/+z 移动，不是相机画面的左右上下。
    # - 增大 x：往世界 +x 移
    # - 增大 y：往相机前方/桌面内侧移
    # - 增大 z：往上移
GRID_SPACING_XYZ = [0.09, 0.07, 0.09]
# 3x3x3 网格每个方向的半跨度。
#   - 当前 x 方向范围是 center_x ± 0.07
#   - 当前 y 方向范围是 center_y ± 0.07
#   - 当前 z 方向范围是 center_z ± 0.07
TARGET_TOLERANCE_M = 0.035
MARKER_RADIUS_M = 0.001
CENTER_MARKER_RADIUS_M = 0.001
GRID_LINE_THICKNESS_M = 0.001
SHOW_TCP_REFERENCE_MARKERS = True
TCP_REFERENCE_MARKER_RADIUS_M = 0.001
GRID_POINT_COLOR = [0.0, 0.0, 0.0, 1.0]
GRID_CENTER_COLOR = [1.0, 0.0, 0.0, 1.0]
GRID_LINE_COLOR = [0.0, 0.0, 0.0, 1.0]
LEFT_TCP_REFERENCE_COLOR = [0.0, 1.0, 0.0, 1.0]
RIGHT_TCP_REFERENCE_COLOR = [0.0, 0.0, 1.0, 1.0]
MOVE_CAPTURE_FREQ = 10
ARM_MOTION_MODE = "both_mirrored"  # "left", "right", "both_same", or "both_mirrored"
GRID_TARGET_POINT_IDS: list[int] | str = [24]  # Use "all" to visit all 27 grid points.
DUAL_MIRROR_AXIS = "x"
RETURN_HOME_BETWEEN_POINTS = False
REACHABILITY_SINGLE_POINT_IDS = tuple(range(24))
REACHABILITY_DUAL_LEFT_POINT_IDS = (0, 3, 6, 9, 12, 15, 18, 21)
MAX_EPISODE_FRAMES: int | None = None


class EpisodeFrameLimitExceeded(RuntimeError):
    pass

# 当前 27 个点的编号规则是：

#   pos_id = iz * 9 + iy * 3 + ix

#   其中：

#   ix = 0,1,2  对应 x = center_x - dx, center_x, center_x + dx
#   iy = 0,1,2  对应 y = center_y - dy, center_y, center_y + dy
#   iz = 0,1,2  对应 z = center_z - dz, center_z, center_z + dz

#   你现在的半跨度是：

#   GRID_SPACING_XYZ = [0.07, 0.07, 0.07]

#   所以每个方向就是：

#   -0.07, 0, +0.07

#   完整编号表

#   底层：z = center_z - 0.07

#   y = center_y - 0.07:   0    1    2
#   y = center_y:          3    4    5
#   y = center_y + 0.07:   6    7    8

#   对应 x:
#                          x-   x0   x+

#   中层：z = center_z

#   y = center_y - 0.07:   9    10   11
#   y = center_y:          12   13   14
#   y = center_y + 0.07:   15   16   17

#   对应 x:
#                          x-   x0   x+

#   顶层：z = center_z + 0.07

#   y = center_y - 0.07:   18   19   20
#   y = center_y:          21   22   23
#   y = center_y + 0.07:   24   25   26

#   对应 x:
#                          x-   x0   x+

#   所以最重要的几个点是：

#   13 = 整个 3D 网格中心
#   4  = 底层中心
#   22 = 顶层中心
#   12 = 中层 x- 方向
#   14 = 中层 x+ 方向
#   10 = 中层 y- 方向
#   16 = 中层 y+ 方向


def as_float_list(value: Iterable[float]) -> list[float]:
    return [float(x) for x in value]


def distance(a: Iterable[float], b: Iterable[float]) -> float:
    return float(np.linalg.norm(np.asarray(list(a), dtype=np.float64) - np.asarray(list(b), dtype=np.float64)))


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
    grid_index[axis_to_index[axis]] = 2 - grid_index[axis_to_index[axis]]
    return grid_index_to_point_id(grid_index)


def build_reachability_conditions() -> list[dict]:
    conditions = []
    for mode in ("left", "right"):
        for point_id in REACHABILITY_SINGLE_POINT_IDS:
            conditions.append(
                {
                    "condition_id": 0,
                    "mode": mode,
                    "left_point_id": int(point_id) if mode == "left" else None,
                    "right_point_id": int(point_id) if mode == "right" else None,
                    "dual_mirror_axis": None,
                }
            )

    for left_point_id in REACHABILITY_DUAL_LEFT_POINT_IDS:
        conditions.append(
            {
                "condition_id": 0,
                "mode": "both_mirrored",
                "left_point_id": int(left_point_id),
                "right_point_id": int(mirror_point_id(left_point_id, DUAL_MIRROR_AXIS)),
                "dual_mirror_axis": DUAL_MIRROR_AXIS,
            }
        )

    for condition_id, condition in enumerate(conditions):
        condition["condition_id"] = int(condition_id)
    return conditions


class rpy_grid_motion_safety(Base_Task):
    """Check whether visual-only grid markers affect left-arm motion."""

    def setup_demo(self, **kwags):
        self.fixed_seed = int(kwags.get("seed", 0))
        self.reachability_mode = bool(kwags.get("reachability_mode", False)) or os.environ.get(
            "RPY_GRID_REACHABILITY", ""
        ).strip().lower() in {"1", "true", "yes", "y", "on"}
        self.frame_limit_exceeded = False
        super()._init_task_env_(**kwags)
        self._ensure_marker_attrs()

    def load_actors(self):
        self.grid_markers = []
        self.left_tcp_reference_marker = None
        self.right_tcp_reference_marker = None

    def _ensure_marker_attrs(self) -> None:
        if not hasattr(self, "grid_markers"):
            self.grid_markers = []
        if not hasattr(self, "left_tcp_reference_marker"):
            self.left_tcp_reference_marker = None
        if not hasattr(self, "right_tcp_reference_marker"):
            self.right_tcp_reference_marker = None

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
            raise ValueError(f"Unsupported GRID_CENTER_MODE: {GRID_CENTER_MODE}")
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

    def _selected_grid_positions(self, grid_positions: list[dict]) -> list[dict]:
        if GRID_TARGET_POINT_IDS == "all":
            selected_ids = list(range(len(grid_positions)))
        else:
            selected_ids = [int(pos_id) for pos_id in GRID_TARGET_POINT_IDS]
        if not selected_ids:
            raise ValueError("GRID_TARGET_POINT_IDS must not be empty")

        by_id = {int(pos["pos_id"]): pos for pos in grid_positions}
        missing = [pos_id for pos_id in selected_ids if pos_id not in by_id]
        if missing:
            raise ValueError(f"GRID_TARGET_POINT_IDS contains invalid ids: {missing}")
        return [by_id[pos_id] for pos_id in selected_ids]

    def _reachability_condition(self) -> dict:
        conditions = build_reachability_conditions()
        if self.ep_num >= len(conditions):
            raise IndexError(f"episode {self.ep_num} is outside the reachability condition set of {len(conditions)}")
        return conditions[self.ep_num]

    def _reachability_selected_positions(self, condition: dict, grid_positions: list[dict]) -> list[dict]:
        selected_ids = []
        if condition.get("left_point_id") is not None:
            selected_ids.append(int(condition["left_point_id"]))
        if condition.get("right_point_id") is not None:
            selected_ids.append(int(condition["right_point_id"]))
        by_id = {int(pos["pos_id"]): pos for pos in grid_positions}
        return [by_id[pos_id] for pos_id in selected_ids]

    def _mirror_xyz_about_grid_center(self, xyz: Iterable[float], center_pose: Iterable[float]) -> list[float]:
        axis_to_index = {"x": 0, "y": 1, "z": 2}
        if DUAL_MIRROR_AXIS not in axis_to_index:
            raise ValueError(f"Unsupported DUAL_MIRROR_AXIS: {DUAL_MIRROR_AXIS}")

        mirrored = as_float_list(xyz)
        center_xyz = as_float_list(center_pose)[:3]
        axis_idx = axis_to_index[DUAL_MIRROR_AXIS]
        mirrored[axis_idx] = 2.0 * center_xyz[axis_idx] - mirrored[axis_idx]
        return mirrored

    def _mirror_grid_index(self, grid_index: Iterable[int]) -> list[int]:
        axis_to_index = {"x": 0, "y": 1, "z": 2}
        if DUAL_MIRROR_AXIS not in axis_to_index:
            raise ValueError(f"Unsupported DUAL_MIRROR_AXIS: {DUAL_MIRROR_AXIS}")
        mirrored = [int(idx) for idx in grid_index]
        axis_idx = axis_to_index[DUAL_MIRROR_AXIS]
        mirrored[axis_idx] = 2 - mirrored[axis_idx]
        return mirrored

    def _pose_with_xyz(self, reference_pose: Iterable[float], target_xyz: Iterable[float]) -> list[float]:
        pose = as_float_list(reference_pose)
        pose[:3] = as_float_list(target_xyz)
        return pose

    def _make_arm_targets(
        self,
        grid_position: dict,
        center_pose: Iterable[float],
        left_reference_pose: Iterable[float],
        right_reference_pose: Iterable[float],
    ) -> dict:
        grid_xyz = grid_position["xyz"]
        grid_index = grid_position["grid_index"]
        mode = ARM_MOTION_MODE

        if mode == "left":
            left_xyz, right_xyz = grid_xyz, None
            left_grid_index, right_grid_index = grid_index, None
        elif mode == "right":
            left_xyz, right_xyz = None, grid_xyz
            left_grid_index, right_grid_index = None, grid_index
        elif mode == "both_same":
            left_xyz, right_xyz = grid_xyz, grid_xyz
            left_grid_index, right_grid_index = grid_index, grid_index
        elif mode == "both_mirrored":
            left_xyz = grid_xyz
            right_xyz = self._mirror_xyz_about_grid_center(grid_xyz, center_pose)
            left_grid_index = grid_index
            right_grid_index = self._mirror_grid_index(grid_index)
        else:
            raise ValueError(f"Unsupported ARM_MOTION_MODE: {mode}")

        return {
            "left_target_xyz": as_float_list(left_xyz) if left_xyz is not None else None,
            "right_target_xyz": as_float_list(right_xyz) if right_xyz is not None else None,
            "left_target_grid_index": [int(idx) for idx in left_grid_index] if left_grid_index is not None else None,
            "right_target_grid_index": [int(idx) for idx in right_grid_index] if right_grid_index is not None else None,
            "left_target_pose": self._pose_with_xyz(left_reference_pose, left_xyz) if left_xyz is not None else None,
            "right_target_pose": self._pose_with_xyz(right_reference_pose, right_xyz) if right_xyz is not None else None,
        }

    def _make_reachability_targets(
        self,
        condition: dict,
        grid_positions: list[dict],
        left_reference_pose: Iterable[float],
        right_reference_pose: Iterable[float],
    ) -> dict:
        by_id = {int(pos["pos_id"]): pos for pos in grid_positions}
        left_position = by_id[int(condition["left_point_id"])] if condition.get("left_point_id") is not None else None
        right_position = by_id[int(condition["right_point_id"])] if condition.get("right_point_id") is not None else None
        left_xyz = left_position["xyz"] if left_position is not None else None
        right_xyz = right_position["xyz"] if right_position is not None else None
        return {
            "left_target_xyz": as_float_list(left_xyz) if left_xyz is not None else None,
            "right_target_xyz": as_float_list(right_xyz) if right_xyz is not None else None,
            "left_target_grid_index": left_position["grid_index"] if left_position is not None else None,
            "right_target_grid_index": right_position["grid_index"] if right_position is not None else None,
            "left_target_pose": self._pose_with_xyz(left_reference_pose, left_xyz) if left_xyz is not None else None,
            "right_target_pose": self._pose_with_xyz(right_reference_pose, right_xyz) if right_xyz is not None else None,
        }

    def _create_visual_marker(self, xyz: Iterable[float], radius: float, color: list[float], name: str) -> sapien.Entity:
        pose = sapien.Pose(as_float_list(xyz), [1.0, 0.0, 0.0, 0.0])
        material = sapien.render.RenderMaterial(base_color=color)
        try:
            material.set_emission(color)
        except Exception:
            pass

        entity = sapien.Entity()
        entity.set_name(name)
        entity.set_pose(pose)

        render_component = sapien.render.RenderBodyComponent()
        render_component.attach(sapien.render.RenderShapeSphere(radius=radius, material=material))
        entity.add_component(render_component)
        entity.set_pose(pose)
        self.scene.add_entity(entity)
        return entity

    def _create_visual_box_marker(
        self,
        xyz: Iterable[float],
        half_size: Iterable[float],
        color: list[float],
        name: str,
    ) -> sapien.Entity:
        pose = sapien.Pose(as_float_list(xyz), [1.0, 0.0, 0.0, 0.0])
        material = sapien.render.RenderMaterial(base_color=color)
        try:
            material.set_emission(color)
        except Exception:
            pass

        entity = sapien.Entity()
        entity.set_name(name)
        entity.set_pose(pose)

        render_component = sapien.render.RenderBodyComponent()
        render_component.attach(sapien.render.RenderShapeBox(as_float_list(half_size), material))
        entity.add_component(render_component)
        entity.set_pose(pose)
        self.scene.add_entity(entity)
        return entity

    def _add_visual_grid(self, positions: list[dict]) -> list[sapien.Entity]:
        self._ensure_marker_attrs()
        markers = []
        for pos in positions:
            is_center = pos["offset_xyz"] == [0.0, 0.0, 0.0]
            radius = CENTER_MARKER_RADIUS_M if is_center else MARKER_RADIUS_M
            color = GRID_CENTER_COLOR if is_center else GRID_POINT_COLOR
            markers.append(
                self._create_visual_marker(
                    xyz=pos["xyz"],
                    radius=radius,
                    color=color,
                    name=f"visual_only_grid_marker_{pos['pos_id']:02d}",
                )
            )

        coords = np.asarray([pos["xyz"] for pos in positions], dtype=np.float64)
        xs = sorted(set(float(x) for x in coords[:, 0]))
        ys = sorted(set(float(y) for y in coords[:, 1]))
        zs = sorted(set(float(z) for z in coords[:, 2]))
        mid_x, mid_y, mid_z = float(np.mean(xs)), float(np.mean(ys)), float(np.mean(zs))
        thickness = GRID_LINE_THICKNESS_M / 2.0
        line_color = GRID_LINE_COLOR

        line_id = 0
        for y in ys:
            for z in zs:
                markers.append(
                    self._create_visual_box_marker(
                        xyz=[mid_x, y, z],
                        half_size=[(xs[-1] - xs[0]) / 2.0, thickness, thickness],
                        color=line_color,
                        name=f"visual_only_grid_line_x_{line_id:02d}",
                    )
                )
                line_id += 1
        for x in xs:
            for z in zs:
                markers.append(
                    self._create_visual_box_marker(
                        xyz=[x, mid_y, z],
                        half_size=[thickness, (ys[-1] - ys[0]) / 2.0, thickness],
                        color=line_color,
                        name=f"visual_only_grid_line_y_{line_id:02d}",
                    )
                )
                line_id += 1
        for x in xs:
            for y in ys:
                markers.append(
                    self._create_visual_box_marker(
                        xyz=[x, y, mid_z],
                        half_size=[thickness, thickness, (zs[-1] - zs[0]) / 2.0],
                        color=line_color,
                        name=f"visual_only_grid_line_z_{line_id:02d}",
                    )
                )
                line_id += 1

        self.grid_markers = markers
        return markers

    def _add_tcp_reference_markers(self, left_pose: Iterable[float], right_pose: Iterable[float]) -> None:
        self._ensure_marker_attrs()
        if not SHOW_TCP_REFERENCE_MARKERS:
            return
        self.left_tcp_reference_marker = self._create_visual_marker(
            xyz=as_float_list(left_pose)[:3],
            radius=TCP_REFERENCE_MARKER_RADIUS_M,
            color=LEFT_TCP_REFERENCE_COLOR,
            name="visual_only_left_tcp_reference_marker",
        )
        self.right_tcp_reference_marker = self._create_visual_marker(
            xyz=as_float_list(right_pose)[:3],
            radius=TCP_REFERENCE_MARKER_RADIUS_M,
            color=RIGHT_TCP_REFERENCE_COLOR,
            name="visual_only_right_tcp_reference_marker",
        )
        self.grid_markers.extend([self.left_tcp_reference_marker, self.right_tcp_reference_marker])

    def _update_tcp_reference_markers(self) -> None:
        self._ensure_marker_attrs()
        if not SHOW_TCP_REFERENCE_MARKERS:
            return
        if self.left_tcp_reference_marker is not None:
            left_pose = as_float_list(self.get_arm_pose(ArmTag("left")))
            self.left_tcp_reference_marker.set_pose(sapien.Pose(left_pose[:3], [1.0, 0.0, 0.0, 0.0]))
        if self.right_tcp_reference_marker is not None:
            right_pose = as_float_list(self.get_arm_pose(ArmTag("right")))
            self.right_tcp_reference_marker.set_pose(sapien.Pose(right_pose[:3], [1.0, 0.0, 0.0, 0.0]))

    def _take_picture(self):
        if (
            MAX_EPISODE_FRAMES is not None
            and getattr(self, "save_data", False)
            and getattr(self, "FRAME_IDX", 0) >= MAX_EPISODE_FRAMES
        ):
            self.frame_limit_exceeded = True
            raise EpisodeFrameLimitExceeded(f"episode exceeded {MAX_EPISODE_FRAMES} saved frames")
        return super()._take_picture()

    def _frame_limit_fields(self) -> dict:
        return {
            "frame_count": int(getattr(self, "FRAME_IDX", 0)),
            "max_episode_frames": int(MAX_EPISODE_FRAMES) if MAX_EPISODE_FRAMES is not None else None,
            "frame_limit_exceeded": bool(getattr(self, "frame_limit_exceeded", False)),
        }

    def take_dense_action(self, control_seq, save_freq=-1):
        self._ensure_marker_attrs()
        left_arm, left_gripper, right_arm, right_gripper = (
            control_seq["left_arm"],
            control_seq["left_gripper"],
            control_seq["right_arm"],
            control_seq["right_gripper"],
        )

        save_freq = self.save_freq if save_freq == -1 else save_freq
        if save_freq is not None:
            self._update_tcp_reference_markers()
            self._take_picture()

        max_control_len = 0
        if left_arm is not None:
            max_control_len = max(max_control_len, left_arm["position"].shape[0])
        if left_gripper is not None:
            max_control_len = max(max_control_len, left_gripper["num_step"])
        if right_arm is not None:
            max_control_len = max(max_control_len, right_arm["position"].shape[0])
        if right_gripper is not None:
            max_control_len = max(max_control_len, right_gripper["num_step"])

        for control_idx in range(max_control_len):
            if left_arm is not None and control_idx < left_arm["position"].shape[0]:
                self.robot.set_arm_joints(
                    left_arm["position"][control_idx],
                    left_arm["velocity"][control_idx],
                    "left",
                )

            if left_gripper is not None and control_idx < left_gripper["num_step"]:
                self.robot.set_gripper(left_gripper["result"][control_idx], "left", left_gripper["per_step"])

            if right_arm is not None and control_idx < right_arm["position"].shape[0]:
                self.robot.set_arm_joints(
                    right_arm["position"][control_idx],
                    right_arm["velocity"][control_idx],
                    "right",
                )

            if right_gripper is not None and control_idx < right_gripper["num_step"]:
                self.robot.set_gripper(right_gripper["result"][control_idx], "right", right_gripper["per_step"])

            self.scene.step()
            self._update_tcp_reference_markers()

            if self.render_freq and control_idx % self.render_freq == 0:
                self._update_render()
                self.viewer.render()

            if save_freq is not None and control_idx % save_freq == 0:
                self._update_render()
                self._take_picture()

        if save_freq is not None:
            self._update_tcp_reference_markers()
            self._take_picture()

        return True

    def _marker_component_summary(self) -> dict:
        component_names = []
        has_collision = False
        has_physx = False
        for marker in getattr(self, "grid_markers", []):
            names = [type(component).__name__ for component in marker.components]
            component_names.append({"name": marker.name, "components": names})
            for name in names:
                low = name.lower()
                if "physx" in low:
                    has_physx = True
                if "collision" in low:
                    has_collision = True
        return {
            "component_names": component_names,
            "has_collision_shape": bool(has_collision),
            "has_physx_component": bool(has_physx),
        }

    def _marker_contact_names(self) -> list[str]:
        marker_names = {marker.name for marker in getattr(self, "grid_markers", [])}
        contacts = []
        for contact in self.scene.get_contacts():
            names = []
            for body in contact.bodies:
                entity = getattr(body, "entity", None)
                names.append(getattr(entity, "name", "unknown"))
            if any(name in marker_names for name in names):
                contacts.append(" <-> ".join(names))
        return contacts

    def _move_arms_to_pose_for_test(
        self,
        left_target_pose: list[float] | None = None,
        right_target_pose: list[float] | None = None,
    ) -> tuple[bool, str | None]:
        if self.plan_success is False:
            return False, "planner_already_failed"
        if left_target_pose is None and right_target_pose is None:
            return False, "missing_arm_target"

        left_result = None
        right_result = None
        if left_target_pose is not None:
            left_result = self.left_move_to_pose(left_target_pose)
            if self.plan_success is False or left_result is None:
                return False, "left_planning_or_ik_failed"
        if right_target_pose is not None:
            right_result = self.right_move_to_pose(right_target_pose)
            if self.plan_success is False or right_result is None:
                return False, "right_planning_or_ik_failed"

        control_seq = {
            "left_arm": left_result,
            "left_gripper": None,
            "right_arm": right_result,
            "right_gripper": None,
        }
        self.take_dense_action(control_seq, save_freq=MOVE_CAPTURE_FREQ)
        if self.plan_success is False:
            return False, "motion_execution_failed"
        return True, None

    def _write_report(self, report: dict) -> None:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        save_dir = getattr(self, "save_dir", None)
        if save_dir:
            data_report_path = Path(save_dir) / "motion_safety_report.json"
            data_report_path.parent.mkdir(parents=True, exist_ok=True)
            data_report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _build_report(
        self,
        left_tcp_reference_pose: list[float],
        right_tcp_reference_pose: list[float],
        center_pose: list[float],
        grid_positions: list[dict],
        selected_positions: list[dict],
        move_results: list[dict],
        reachability_condition: dict | None = None,
    ) -> dict:
        marker_summary = self._marker_component_summary()
        failures = [result for result in move_results if not result["move_success"]]
        marker_contacts = self._marker_contact_names()
        left_moved_to_center = any(
            result.get("left_target_grid_index") == [1, 1, 1] and result.get("left_move_success") is True
            for result in move_results
        )
        right_moved_to_center = any(
            result.get("right_target_grid_index") == [1, 1, 1] and result.get("right_move_success") is True
            for result in move_results
        )
        left_success_indices = [
            result.get("left_target_grid_index") for result in move_results if result.get("left_move_success") is True
        ]
        right_success_indices = [
            result.get("right_target_grid_index") for result in move_results if result.get("right_move_success") is True
        ]
        left_crossed_grid = [0, 1, 1] in left_success_indices and [2, 1, 1] in left_success_indices
        right_crossed_grid = [0, 1, 1] in right_success_indices and [2, 1, 1] in right_success_indices
        expected_move_count = 1 if self.reachability_mode else len(selected_positions)
        all_moves_success = len(move_results) == expected_move_count and not failures
        no_marker_contacts = len(marker_contacts) == 0
        conclusion = (
            "pass"
            if all_moves_success
            and no_marker_contacts
            and not marker_summary["has_collision_shape"]
            and not self.frame_limit_exceeded
            else "fail"
        )
        if len(move_results) < expected_move_count:
            conclusion = "inconclusive"

        return {
            "experiment": "rpy_grid_motion_safety",
            "episode_id": int(getattr(self, "ep_num", 0)),
            "seed": int(getattr(self, "fixed_seed", 0)),
            "save_data": bool(getattr(self, "save_data", False)),
            "need_plan": bool(getattr(self, "need_plan", True)),
            "reachability_mode": bool(self.reachability_mode),
            "reachability_condition": reachability_condition,
            "num_reachability_conditions": len(build_reachability_conditions()) if self.reachability_mode else None,
            "abnormal": bool(self.frame_limit_exceeded),
            **self._frame_limit_fields(),
            "grid_type": "floating_3d_visual_only_grid",
            "grid_marker_type": "visual_only_sphere_or_box",
            "camera_visible": True,
            "grid_center_mode": GRID_CENTER_MODE,
            "left_tcp_reference_xyz": left_tcp_reference_pose[:3],
            "right_tcp_reference_xyz": right_tcp_reference_pose[:3],
            "grid_center_xyz": center_pose[:3],
            "grid_center_offset_xyz": GRID_CENTER_OFFSET_XYZ,
            "grid_center_absolute_xyz": GRID_CENTER_ABSOLUTE_XYZ if GRID_CENTER_MODE == "absolute" else None,
            "grid_spacing_xyz": GRID_SPACING_XYZ,
            "num_grid_points": len(grid_positions),
            "num_grid_line_bars": 27,
            "grid_positions": grid_positions,
            "arm_motion_mode": ARM_MOTION_MODE,
            "dual_mirror_axis": DUAL_MIRROR_AXIS,
            "return_home_between_points": RETURN_HOME_BETWEEN_POINTS,
            "selected_grid_point_ids": [int(pos["pos_id"]) for pos in selected_positions],
            "selected_grid_positions": selected_positions,
            "marker": {
                "type": "visual_only_sphere_or_box",
                "has_visual": True,
                "has_collision": bool(marker_summary["has_collision_shape"]),
                "has_physx_component": bool(marker_summary["has_physx_component"]),
                "uses_add_collision_api": False,
                "added_collision": False,
                "point_radius_m": MARKER_RADIUS_M,
                "center_point_radius_m": CENTER_MARKER_RADIUS_M,
                "grid_line_thickness_m": GRID_LINE_THICKNESS_M,
                "show_tcp_reference_markers": SHOW_TCP_REFERENCE_MARKERS,
                "tcp_reference_marker_radius_m": TCP_REFERENCE_MARKER_RADIUS_M,
                "component_summary": marker_summary["component_names"],
            },
            "motion_test": {
                "path": [as_float_list(pos["xyz"]) for pos in selected_positions],
                "target_tolerance_m": TARGET_TOLERANCE_M,
                "moves": move_results,
                "all_moves_success": bool(all_moves_success),
                "num_success": int(sum(1 for result in move_results if result["move_success"])),
                "num_failed": int(len(failures)),
                "failures": failures,
            },
            "safety_questions": {
                "marker_added_as_visual_only": not marker_summary["has_collision_shape"]
                and not marker_summary["has_physx_component"],
                "added_any_collision_shape": bool(marker_summary["has_collision_shape"]),
                "marker_likely_planner_obstacle": False,
                "left_arm_moved_to_grid_inside": bool(left_moved_to_center),
                "right_arm_moved_to_grid_inside": bool(right_moved_to_center),
                "left_arm_crossed_grid_side_to_side": bool(left_crossed_grid),
                "right_arm_crossed_grid_side_to_side": bool(right_crossed_grid),
                "ik_planning_or_collision_failure": bool(failures or marker_contacts),
                "marker_contact_warnings": marker_contacts,
                "camera_video_expected_to_show_marker": True,
                "camera_visible_range_calibration_suitable": conclusion == "pass",
            },
            "camera": {
                "head_camera_type": "Large_D435",
                "wrist_camera_type": "Large_D435",
                "resolution": [640, 480],
                "marker_visible_in_camera": "manual_check_required",
            },
            "conclusion": conclusion,
            "recommended_next_step": (
                "camera_visibility_calibration"
                if conclusion == "pass"
                else "fix_marker_collision" if marker_summary["has_collision_shape"] else "shrink_grid"
            ),
        }

    def play_once(self):
        left = ArmTag("left")
        right = ArmTag("right")
        left_tcp_reference_pose = as_float_list(self.get_arm_pose(left))
        right_tcp_reference_pose = as_float_list(self.get_arm_pose(right))
        center_pose = self._grid_center_pose(left_tcp_reference_pose, right_tcp_reference_pose)
        grid_positions = self._make_grid_positions(center_pose)
        reachability_condition = self._reachability_condition() if self.reachability_mode else None
        selected_positions = (
            self._reachability_selected_positions(reachability_condition, grid_positions)
            if self.reachability_mode
            else self._selected_grid_positions(grid_positions)
        )
        self._add_visual_grid(grid_positions)
        self._add_tcp_reference_markers(left_tcp_reference_pose, right_tcp_reference_pose)

        move_results = []

        if self.reachability_mode:
            step_items = [(0, selected_positions[0], reachability_condition)]
        else:
            step_items = [(step_id, grid_position, None) for step_id, grid_position in enumerate(selected_positions)]

        for step_id, grid_position, condition in step_items:
            if self.reachability_mode:
                targets = self._make_reachability_targets(
                    condition=condition,
                    grid_positions=grid_positions,
                    left_reference_pose=left_tcp_reference_pose,
                    right_reference_pose=right_tcp_reference_pose,
                )
            else:
                targets = self._make_arm_targets(
                    grid_position=grid_position,
                    center_pose=center_pose,
                    left_reference_pose=left_tcp_reference_pose,
                    right_reference_pose=right_tcp_reference_pose,
                )

            result = False
            error = None
            try:
                result, error = self._move_arms_to_pose_for_test(
                    left_target_pose=targets["left_target_pose"],
                    right_target_pose=targets["right_target_pose"],
                )
            except Exception as exc:
                error = repr(exc)
                result = False

            left_actual_pose = as_float_list(self.get_arm_pose(left))
            right_actual_pose = as_float_list(self.get_arm_pose(right))

            left_position_error = (
                distance(targets["left_target_xyz"], left_actual_pose[:3]) if targets["left_target_xyz"] is not None else None
            )
            right_position_error = (
                distance(targets["right_target_xyz"], right_actual_pose[:3]) if targets["right_target_xyz"] is not None else None
            )
            left_move_success = (
                bool(result and self.plan_success and left_position_error <= TARGET_TOLERANCE_M)
                if left_position_error is not None
                else None
            )
            right_move_success = (
                bool(result and self.plan_success and right_position_error <= TARGET_TOLERANCE_M)
                if right_position_error is not None
                else None
            )
            active_successes = [
                success for success in [left_move_success, right_move_success] if success is not None
            ]
            move_success = bool(result and self.plan_success and active_successes and all(active_successes))
            if self.frame_limit_exceeded:
                move_success = False
                if error is None:
                    error = f"frame_limit_exceeded_{MAX_EPISODE_FRAMES}"
            if result and not move_success and error is None:
                error_parts = []
                if left_position_error is not None and left_position_error > TARGET_TOLERANCE_M:
                    error_parts.append(f"left_target_error_exceeded_{left_position_error:.6f}m")
                if right_position_error is not None and right_position_error > TARGET_TOLERANCE_M:
                    error_parts.append(f"right_target_error_exceeded_{right_position_error:.6f}m")
                error = ";".join(error_parts) if error_parts else "target_error_or_plan_failed"

            return_home_success = None
            return_home_error = None
            if move_success and RETURN_HOME_BETWEEN_POINTS and step_id < len(selected_positions) - 1:
                try:
                    return_home_success, return_home_error = self._move_arms_to_pose_for_test(
                        left_target_pose=left_tcp_reference_pose if targets["left_target_pose"] is not None else None,
                        right_target_pose=right_tcp_reference_pose if targets["right_target_pose"] is not None else None,
                    )
                except Exception as exc:
                    return_home_success = False
                    return_home_error = repr(exc)
                if not return_home_success:
                    move_success = False
                    error = f"return_home_failed:{return_home_error}"

            move_results.append(
                {
                    "step_id": int(step_id),
                    "pos_id": int(grid_position["pos_id"]),
                    "reachability_condition_id": int(condition["condition_id"]) if condition else None,
                    "reachability_mode": condition["mode"] if condition else None,
                    "selected_grid_index": grid_position["grid_index"],
                    "grid_xyz": as_float_list(grid_position["xyz"]),
                    "left_target_grid_index": targets["left_target_grid_index"],
                    "left_target_xyz": targets["left_target_xyz"],
                    "left_actual_xyz": left_actual_pose[:3],
                    "left_position_error": float(left_position_error) if left_position_error is not None else None,
                    "left_move_success": left_move_success,
                    "right_target_grid_index": targets["right_target_grid_index"],
                    "right_target_xyz": targets["right_target_xyz"],
                    "right_actual_xyz": right_actual_pose[:3],
                    "right_position_error": float(right_position_error) if right_position_error is not None else None,
                    "right_move_success": right_move_success,
                    "return_home_success": return_home_success,
                    "return_home_error": return_home_error,
                    "abnormal": bool(self.frame_limit_exceeded),
                    **self._frame_limit_fields(),
                    "move_success": bool(move_success),
                    "error": error,
                }
            )

            if not move_success:
                break

        report = self._build_report(
            left_tcp_reference_pose,
            right_tcp_reference_pose,
            center_pose,
            grid_positions,
            selected_positions,
            move_results,
            reachability_condition=reachability_condition,
        )
        self._last_report = report
        self.info["info"] = report
        self._write_report(report)

        print(
            "[rpy_grid_motion_safety] "
            f"episode={self.ep_num} save_data={getattr(self, 'save_data', False)} "
            f"success={report['motion_test']['all_moves_success']} "
            f"num_success={report['motion_test']['num_success']} "
            f"num_failed={report['motion_test']['num_failed']} "
            f"conclusion={report['conclusion']}"
        )
        return self.info

    def check_success(self):
        report = getattr(self, "_last_report", None)
        if not report:
            return False
        return bool(report["conclusion"] == "pass")
