import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

try:
    from .scene_export import ROBOTWIN_ROOT, cache_dir
except ImportError:
    from scene_export import ROBOTWIN_ROOT, cache_dir


def export_object_mesh(task_config, task_name, seed, target_triangles=6000):
    scene_path = cache_dir(task_config, task_name, seed) / "scene.json"
    if not scene_path.exists():
        return None, "请先生成 / 加载首帧。"

    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    objects = scene.get("objects", [])
    if not objects:
        return None, "scene.json 中没有物体信息。"

    obj = objects[0]
    model_id = obj.get("model_id")
    mesh_path = ROBOTWIN_ROOT / "assets" / "objects" / "001_bottle" / "visual" / f"base{model_id}.glb"
    if not mesh_path.exists():
        return None, f"找不到物体 mesh: {mesh_path}"

    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.triangles) > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_triangles)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.compute_vertex_normals()
    _apply_glb_to_actor_axes(mesh)
    mesh.transform(_scale_matrix(_model_scale(model_id)))
    mesh.transform(_pose_matrix(obj["pose_world"]))

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    return {
        "model_id": int(model_id),
        "name": obj.get("name", "object"),
        "color": _model_color(model_id),
        "vertices": np.round(vertices, 6).tolist(),
        "triangles": triangles.tolist(),
    }, "ok"


def _model_scale(model_id):
    path = ROBOTWIN_ROOT / "assets" / "objects" / "001_bottle" / f"model_data{model_id}.json"
    if not path.exists():
        return [1, 1, 1]
    data = json.loads(path.read_text(encoding="utf-8"))
    scale = data.get("scale", [1, 1, 1])
    return scale if isinstance(scale, list) else [scale, scale, scale]


def _apply_glb_to_actor_axes(mesh):
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    mesh.vertices = type(mesh.vertices)(np.column_stack([vertices[:, 0], -vertices[:, 2], vertices[:, 1]]))
    if len(normals):
        mesh.vertex_normals = type(mesh.vertex_normals)(np.column_stack([normals[:, 0], -normals[:, 2], normals[:, 1]]))


def _scale_matrix(scale):
    matrix = np.eye(4, dtype=float)
    matrix[0, 0], matrix[1, 1], matrix[2, 2] = np.asarray(scale, dtype=float)[:3]
    return matrix


def _pose_matrix(pose):
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = Rotation.from_quat(pose["q"], scalar_first=True).as_matrix()
    matrix[:3, 3] = np.asarray(pose["p"], dtype=float)
    return matrix


def _model_color(model_id):
    colors = {
        13: [0.86, 0.35, 0.22],
        16: [0.18, 0.42, 0.86],
    }
    return colors.get(int(model_id), [0.72, 0.78, 0.84])
