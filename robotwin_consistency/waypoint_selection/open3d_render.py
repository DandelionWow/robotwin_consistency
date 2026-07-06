import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

try:
    from .scene_export import CACHE_ROOT, ROBOTWIN_ROOT, cache_dir
except ImportError:
    from scene_export import CACHE_ROOT, ROBOTWIN_ROOT, cache_dir


CONTACT_TO_TCP = np.array([
    [0, 0, 1, 0],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1],
], dtype=float)


def render_open3d_view(
    task_config,
    task_name,
    seed,
    selected_point_ids,
    show_contact_frames=True,
    show_pre_grasp_frames=True,
    show_tcp_frames=True,
    show_grasp_frames=True,
    show_perturbed_pre_grasp_frames=True,
    pre_grasp_dis=0.1,
    grasp_dis=0.0,
    perturbation=None,
    width=900,
    height=700,
):
    scene_path = cache_dir(task_config, task_name, seed) / "scene.json"
    if not scene_path.exists():
        return None, "请先生成 / 加载首帧，再刷新 3D 视图。"

    request = {
        "task_config": task_config,
        "task_name": task_name,
        "seed": int(seed),
        "selected_point_ids": selected_point_ids,
        "show_contact_frames": show_contact_frames,
        "show_pre_grasp_frames": show_pre_grasp_frames,
        "show_tcp_frames": show_tcp_frames,
        "show_grasp_frames": show_grasp_frames,
        "show_perturbed_pre_grasp_frames": show_perturbed_pre_grasp_frames,
        "pre_grasp_dis": float(pre_grasp_dis),
        "grasp_dis": float(grasp_dis),
        "perturbation": perturbation or {},
        "width": int(width),
        "height": int(height),
    }
    request_file = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    try:
        json.dump(request, request_file, ensure_ascii=False)
        request_file.close()

        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--request", request_file.name],
            cwd=str(ROBOTWIN_ROOT.parent.parent),
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return None, "Open3D 渲染超时。"
    finally:
        Path(request_file.name).unlink(missing_ok=True)

    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        if "eglInitialize failed" in detail or result.returncode in {134, 139}:
            detail = "Open3D 离屏渲染失败，当前环境可能没有可用 EGL/OSMesa。"
        return None, detail or "Open3D 渲染失败。"

    output_path = scene_path.parent / "open3d_view.png"
    if not output_path.exists():
        return None, "Open3D 未生成视图图片。"
    return output_path, "ok"


def _render_open3d_view(
    task_config,
    task_name,
    seed,
    selected_point_ids,
    show_contact_frames=True,
    show_pre_grasp_frames=True,
    show_tcp_frames=True,
    show_grasp_frames=True,
    show_perturbed_pre_grasp_frames=True,
    pre_grasp_dis=0.1,
    grasp_dis=0.0,
    perturbation=None,
    width=900,
    height=700,
):
    import open3d as o3d
    from open3d.visualization import rendering

    scene_path = cache_dir(task_config, task_name, seed) / "scene.json"
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    output_path = scene_path.parent / "open3d_view.png"
    selected_ids = {int(point_id) for point_id in selected_point_ids}
    perturbation = perturbation or {}
    pre_grasp_dis = float(pre_grasp_dis)
    grasp_dis = float(grasp_dis)
    if pre_grasp_dis < grasp_dis:
        raise ValueError("pre_grasp_dis must be greater than or equal to grasp_dis")

    renderer = rendering.OffscreenRenderer(int(width), int(height))
    renderer.scene.set_background([1, 1, 1, 1])
    renderer.scene.scene.set_sun_light([0.5, -0.6, -1.0], [1, 1, 1], 80000)
    renderer.scene.scene.enable_sun_light(True)

    material = rendering.MaterialRecord()
    material.shader = "defaultLit"
    material.base_color = [0.78, 0.82, 0.88, 1.0]

    axis_material = rendering.MaterialRecord()
    axis_material.shader = "defaultUnlit"

    geometries = []
    for obj in scene.get("objects", []):
        mesh = _load_object_mesh(o3d, obj)
        renderer.scene.add_geometry(f"object_{obj.get('name', 'object')}", mesh, material)
        geometries.append(mesh)

        for point in obj.get("contact_points", []):
            point_id = int(point["id"])
            if point_id not in selected_ids:
                continue

            contact_matrix = np.asarray(point["matrix_world"], dtype=float)
            contact_to_grasp = _contact_to_grasp_matrix(point, contact_matrix)
            contact_grasp_matrix = contact_matrix @ contact_to_grasp
            grasp_matrix = _translate_local_x(contact_grasp_matrix, -grasp_dis)
            tcp_matrix = _translate_local_x(grasp_matrix, 0.12)
            pre_matrix = _compute_pre_grasp_matrix(grasp_matrix, pre_grasp_dis, grasp_dis)
            perturbed_contact_matrix = contact_matrix @ _delta_matrix(perturbation)
            perturbed_contact_grasp_matrix = perturbed_contact_matrix @ contact_to_grasp
            perturbed_grasp_matrix = _translate_local_x(perturbed_contact_grasp_matrix, -grasp_dis)
            perturbed_pre_matrix = _compute_pre_grasp_matrix(perturbed_grasp_matrix, pre_grasp_dis, grasp_dis)

            if show_contact_frames:
                frame = _frame(o3d, contact_matrix, 0.045)
                renderer.scene.add_geometry(f"contact_{point_id}", frame, axis_material)
                geometries.append(frame)
            if show_pre_grasp_frames:
                frame = _frame(o3d, pre_matrix, 0.055)
                renderer.scene.add_geometry(f"pre_{point_id}", frame, axis_material)
                geometries.append(frame)
            if show_tcp_frames:
                frame = _frame(o3d, tcp_matrix, 0.055)
                renderer.scene.add_geometry(f"tcp_{point_id}", frame, axis_material)
                geometries.append(frame)
            if show_grasp_frames:
                frame = _frame(o3d, grasp_matrix, 0.06)
                renderer.scene.add_geometry(f"grasp_{point_id}", frame, axis_material)
                geometries.append(frame)
            if show_perturbed_pre_grasp_frames:
                frame = _frame(o3d, perturbed_pre_matrix, 0.055)
                renderer.scene.add_geometry(f"perturbed_{point_id}", frame, axis_material)
                geometries.append(frame)

    _setup_camera(renderer, geometries)
    image = renderer.render_to_image()
    o3d.io.write_image(str(output_path), image)
    return output_path, "ok"


def _load_object_mesh(o3d, obj):
    model_id = obj.get("model_id")
    mesh_path = ROBOTWIN_ROOT / "assets" / "objects" / "001_bottle" / "visual" / f"base{model_id}.glb"
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=0.035, height=0.18)

    _apply_glb_to_actor_axes(mesh)
    mesh.transform(_scale_matrix(_model_scale(model_id)))
    mesh.compute_vertex_normals()
    mesh.transform(_pose_matrix(obj["pose_world"]))
    return mesh


def _model_scale(model_id):
    data_path = ROBOTWIN_ROOT / "assets" / "objects" / "001_bottle" / f"model_data{model_id}.json"
    if not data_path.exists():
        return 1.0
    data = json.loads(data_path.read_text(encoding="utf-8"))
    return data.get("scale", 1.0)


def _apply_glb_to_actor_axes(mesh):
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    mesh.vertices = type(mesh.vertices)(np.column_stack([vertices[:, 0], -vertices[:, 2], vertices[:, 1]]))
    if len(normals):
        mesh.vertex_normals = type(mesh.vertex_normals)(np.column_stack([normals[:, 0], -normals[:, 2], normals[:, 1]]))


def _scale_matrix(scale):
    values = np.asarray(scale if isinstance(scale, list) else [scale, scale, scale], dtype=float)
    matrix = np.eye(4, dtype=float)
    matrix[0, 0], matrix[1, 1], matrix[2, 2] = values[:3]
    return matrix


def _pose_matrix(pose):
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = Rotation.from_quat(pose["q"], scalar_first=True).as_matrix()
    matrix[:3, 3] = np.asarray(pose["p"], dtype=float)
    return matrix


def _compute_pre_grasp_matrix(grasp_matrix, pre_grasp_dis, grasp_dis):
    return _translate_local_x(grasp_matrix, -(float(pre_grasp_dis) - float(grasp_dis)))


def _contact_to_grasp_matrix(point, contact_matrix):
    grasp_matrix = point.get("grasp_matrix_world")
    if grasp_matrix is not None:
        return np.linalg.inv(contact_matrix) @ np.asarray(grasp_matrix, dtype=float)
    contact_to_grasp = CONTACT_TO_TCP.copy()
    contact_to_grasp[:3, 3] += contact_to_grasp[:3, 0] * -0.12
    return contact_to_grasp


def _translate_local_x(matrix, distance):
    result = matrix.copy()
    result[:3, 3] += matrix[:3, 0] * float(distance)
    return result


def _delta_matrix(values):
    dx = float(values.get("x", 0) or 0)
    dy = float(values.get("y", 0) or 0)
    dz = float(values.get("z", 0) or 0)
    roll = float(values.get("r", 0) or 0)
    pitch = float(values.get("p", 0) or 0)
    yaw = float(values.get("yaw", 0) or 0)

    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    matrix[:3, 3] = [dx, dy, dz]
    return matrix


def _frame(o3d, matrix, size):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.transform(matrix)
    return frame


def _setup_camera(renderer, geometries):
    if not geometries:
        center = np.array([0.0, 0.0, 0.8])
        extent = 0.4
    else:
        boxes = [geometry.get_axis_aligned_bounding_box() for geometry in geometries]
        min_bound = np.min([box.get_min_bound() for box in boxes], axis=0)
        max_bound = np.max([box.get_max_bound() for box in boxes], axis=0)
        center = (min_bound + max_bound) / 2.0
        extent = max(float(np.linalg.norm(max_bound - min_bound)), 0.25)

    eye = center + np.array([extent * 0.9, -extent * 1.4, extent * 0.9])
    up = [0, 0, 1]
    renderer.setup_camera(55.0, center, eye, up)


def cache_url_for(path):
    rel_path = Path(path).relative_to(CACHE_ROOT)
    return "/cache/" + str(rel_path).replace("\\", "/")


def _main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()

    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    _render_open3d_view(**request)


if __name__ == "__main__":
    _main()
