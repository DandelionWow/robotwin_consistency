import argparse
import base64
import binascii
import datetime as dt
import json
import math
import mimetypes
import subprocess
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

try:
    from .scene_export import (
        CACHE_ROOT,
        ROBOTWIN_ROOT,
        SUPPORTED_TASK,
        SUPPORTED_TASKS,
        _load_task,
        _robotwin_cwd,
        _to_jsonable,
        cache_dir,
        export_scene,
        list_task_configs,
        load_task_args,
    )
    from .open3d_render import cache_url_for, render_open3d_view
    from .mesh_export import export_object_mesh
except ImportError:
    from scene_export import (
        CACHE_ROOT,
        ROBOTWIN_ROOT,
        SUPPORTED_TASK,
        SUPPORTED_TASKS,
        _load_task,
        _robotwin_cwd,
        _to_jsonable,
        cache_dir,
        export_scene,
        list_task_configs,
        load_task_args,
    )
    from open3d_render import cache_url_for, render_open3d_view
    from mesh_export import export_object_mesh


STATIC_ROOT = Path(__file__).resolve().parent / "static"
GRASP_WORKER = Path(__file__).resolve().parent / "grasp_worker.py"
ROBOTWIN_OBJECT_ASSET_ROOT = ROBOTWIN_ROOT / "assets" / "objects"
ROBOTWIN_BACKGROUND_TEXTURE_ROOT = ROBOTWIN_ROOT / "assets" / "background_texture"


class WaypointSelectionHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_file(STATIC_ROOT / "index.html")
        elif parsed.path.startswith("/static/"):
            self._send_file(STATIC_ROOT / parsed.path.removeprefix("/static/"))
        elif parsed.path == "/api/configs":
            self._send_json({
                "task_configs": list_task_configs(),
                "tasks": list(SUPPORTED_TASKS),
                "default_task": SUPPORTED_TASK,
            })
        elif parsed.path == "/api/scene":
            self._handle_scene(parse_qs(parsed.query))
        elif parsed.path == "/api/object_mesh":
            self._handle_object_mesh(parse_qs(parsed.query))
        elif parsed.path == "/api/object_asset":
            self._handle_object_asset(parse_qs(parsed.query))
        elif parsed.path.startswith("/robotwin_assets/"):
            self._handle_robotwin_asset(parsed.path)
        elif parsed.path.startswith("/cache/"):
            rel_path = Path(unquote(parsed.path.removeprefix("/cache/")))
            self._send_file(CACHE_ROOT / rel_path)
        else:
            self.send_error(404, "Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/render_3d":
            self._handle_render_3d()
        elif parsed.path == "/api/compute_perturbed_grasps":
            self._handle_compute_perturbed_grasps()
        elif parsed.path == "/api/save_perturbation":
            self._handle_save_perturbation()
        else:
            self.send_error(404, "Not found")

    def _handle_scene(self, query):
        task_config = _first(query, "task_config", "ep2_1_object_pose")
        task_name = _first(query, "task_name", SUPPORTED_TASK)
        seed = int(_first(query, "seed", "0"))
        refresh = _first(query, "refresh", "0") in {"1", "true", "True"}
        try:
            scene = export_scene(
                task_config,
                task_name,
                seed,
                refresh=refresh,
            )
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)
            return
        scene = scene.copy()
        scene["image_url"] = _cache_url(task_config, task_name, seed, scene["image"])
        self._send_json(scene)

    def _handle_object_mesh(self, query):
        task_config = _first(query, "task_config", "ep2_1_object_pose")
        task_name = _first(query, "task_name", SUPPORTED_TASK)
        seed = int(_first(query, "seed", "0"))
        try:
            mesh, message = export_object_mesh(task_config, task_name, seed)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)
            return
        if mesh is None:
            self._send_json({"error": message}, status=400)
            return
        self._send_json(mesh)

    def _handle_object_asset(self, query):
        task_config = _first(query, "task_config", "ep2_1_object_pose")
        task_name = _first(query, "task_name", SUPPORTED_TASK)
        seed = int(_first(query, "seed", "0"))
        scene_path = cache_dir(task_config, task_name, seed) / "scene.json"
        if not scene_path.exists():
            self._send_json({"error": "请先生成 / 加载首帧。"}, status=400)
            return

        try:
            scene = json.loads(scene_path.read_text(encoding="utf-8"))
            obj = scene["objects"][0]
            model_id = int(obj["model_id"])
            model_name = obj.get("model_name") or "001_bottle"
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)
            return

        asset_rel = f"objects/{model_name}/visual/base{model_id}.glb"
        asset_path = ROBOTWIN_ROOT / "assets" / asset_rel
        if not asset_path.exists():
            self._send_json({"error": f"找不到物体资源: {asset_rel}"}, status=404)
            return
        model_data_path = ROBOTWIN_ROOT / "assets" / "objects" / model_name / f"model_data{model_id}.json"
        scale = [1, 1, 1]
        if model_data_path.exists():
            model_data = json.loads(model_data_path.read_text(encoding="utf-8"))
            scale = model_data.get("scale", scale)
            if not isinstance(scale, list):
                scale = [scale, scale, scale]
        self._send_json({
            "model_id": model_id,
            "model_name": model_name,
            "mesh_url": "/robotwin_assets/" + asset_rel,
            "pose_world": obj["pose_world"],
            "scale": scale,
        })

    def _handle_robotwin_asset(self, path):
        rel_path = Path(unquote(path.removeprefix("/robotwin_assets/")))
        if rel_path.is_absolute() or ".." in rel_path.parts:
            self.send_error(403, "Forbidden")
            return
        asset_path = ROBOTWIN_ROOT / "assets" / rel_path
        asset_path = asset_path.resolve()
        allowed = (
            _is_under(asset_path, ROBOTWIN_OBJECT_ASSET_ROOT)
            or _is_under(asset_path, ROBOTWIN_BACKGROUND_TEXTURE_ROOT)
        )
        if not allowed:
            self.send_error(403, "Forbidden")
            return
        self._send_file(asset_path)

    def _handle_render_3d(self):
        try:
            payload = self._read_json()
            pre_grasp_dis = _payload_float(payload, "pre_grasp_dis", 0.1)
            grasp_dis = _payload_float(payload, "grasp_dis", 0.0)
            _validate_grasp_distances(pre_grasp_dis, grasp_dis)
            output_path, message = render_open3d_view(
                payload.get("task_config", "ep2_1_object_pose"),
                payload.get("task_name", SUPPORTED_TASK),
                int(payload.get("seed", 0)),
                payload.get("selected_point_ids", []),
                show_contact_frames=bool(payload.get("show_contact_frames", True)),
                show_pre_grasp_frames=bool(payload.get("show_pre_grasp_frames", True)),
                show_tcp_frames=bool(payload.get("show_tcp_frames", True)),
                show_grasp_frames=bool(payload.get("show_grasp_frames", True)),
                show_perturbed_pre_grasp_frames=bool(payload.get("show_perturbed_pre_grasp_frames", True)),
                pre_grasp_dis=pre_grasp_dis,
                grasp_dis=grasp_dis,
                perturbation=payload.get("perturbation", {}),
            )
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)
            return

        if output_path is None:
            self._send_json({"error": message}, status=400)
            return
        self._send_json({"image_url": cache_url_for(output_path), "message": message})

    def _handle_compute_perturbed_grasps(self):
        try:
            payload = self._read_json()
            task_config = payload.get("task_config", "ep2_1_object_pose")
            task_name = payload.get("task_name", SUPPORTED_TASK)
            seed = int(payload.get("seed", 0))
            point_ids = payload.get("selected_point_ids", [])
            pre_grasp_dis = _payload_float(payload, "pre_grasp_dis", 0.1)
            grasp_dis = _payload_float(payload, "grasp_dis", 0.0)
            _validate_grasp_distances(pre_grasp_dis, grasp_dis)
            perturbation = payload.get("perturbation", {})
            if not point_ids:
                self._send_json({"items": []})
                return
            items = compute_perturbed_grasps(
                task_config,
                task_name,
                seed,
                point_ids,
                perturbation,
                pre_grasp_dis,
                grasp_dis,
            )
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)
            return
        self._send_json({"items": items})

    def _handle_save_perturbation(self):
        try:
            payload = self._read_json()
            task_config = payload.get("task_config", "ep2_1_object_pose")
            task_name = payload.get("task_name", SUPPORTED_TASK)
            seed = int(payload.get("seed", 0))
            poses = payload.get("poses", [])
            if not poses:
                self._send_json({"error": "没有可保存的扰动位姿。"}, status=400)
                return

            pre_grasp_dis = _payload_float(payload, "pre_grasp_dis", 0.1)
            grasp_dis = _payload_float(payload, "grasp_dis", 0.0)
            _validate_grasp_distances(pre_grasp_dis, grasp_dis)
            perturbation = payload.get("perturbation", {})
            save_dir = _next_numbered_dir(cache_dir(task_config, task_name, seed) / "perturbation_saves")
            records = [
                _build_perturbation_record(item, pre_grasp_dis, grasp_dis, perturbation)
                for item in poses
            ]
            data = {
                "task_config": task_config,
                "task_name": task_name,
                "seed": seed,
                "saved_at": dt.datetime.now().isoformat(timespec="seconds"),
                "pre_grasp_dis": pre_grasp_dis,
                "grasp_dis": grasp_dis,
                "perturbation": perturbation,
                "items": records,
            }
            pose_path = save_dir / "poses.json"
            pose_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            _write_pose_subset(
                save_dir / "pre_grasp_poses.json",
                records,
                "perturbed_pre_grasp_pose_world",
                "perturbed_pre_grasp_matrix_world",
            )
            _write_pose_subset(
                save_dir / "grasp_poses.json",
                records,
                "perturbed_grasp_pose_world",
                "perturbed_grasp_matrix_world",
            )
            _write_pose_subset(
                save_dir / "tcp_poses.json",
                records,
                "perturbed_tcp_pose_world",
                "perturbed_tcp_matrix_world",
            )

            image_data_url = payload.get("image_data_url")
            image_path = None
            if image_data_url:
                image_path = save_dir / "frame_with_axes.png"
                image_path.write_bytes(_decode_png_data_url(image_data_url))
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)
            return

        self._send_json({
            "path": str(save_dir.relative_to(CACHE_ROOT)),
            "pose_path": str(pose_path.relative_to(CACHE_ROOT)),
            "image_path": str(image_path.relative_to(CACHE_ROOT)) if image_path else None,
            "saved_count": len(poses),
        })

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        data = self.rfile.read(length).decode("utf-8")
        return json.loads(data) if data else {}

    def _send_json(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path):
        path = path.resolve()
        allowed = (
            _is_under(path, STATIC_ROOT)
            or _is_under(path, CACHE_ROOT)
            or _is_under(path, ROBOTWIN_OBJECT_ASSET_ROOT)
            or _is_under(path, ROBOTWIN_BACKGROUND_TEXTURE_ROOT)
        )
        if not allowed:
            self.send_error(403, "Forbidden")
            return
        if not path.is_file():
            self.send_error(404, "Not found")
            return

        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _first(query, key, default):
    values = query.get(key)
    return values[0] if values else default


def _is_under(path, root):
    try:
        path.relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _cache_url(task_config, task_name, seed, image_name):
    rel_path = cache_dir(task_config, task_name, seed).relative_to(CACHE_ROOT) / image_name
    return "/cache/" + str(rel_path).replace("\\", "/")


def compute_perturbed_grasps(task_config, task_name, seed, point_ids, perturbation, pre_grasp_dis, grasp_dis):
    if task_name not in SUPPORTED_TASKS:
        raise ValueError(f"Only these tasks are supported now: {', '.join(SUPPORTED_TASKS)}")
    _validate_grasp_distances(pre_grasp_dis, grasp_dis)

    payload = {
        "task_config": task_config,
        "task_name": task_name,
        "seed": int(seed),
        "selected_point_ids": [int(point_id) for point_id in point_ids],
        "perturbation": perturbation,
        "pre_grasp_dis": float(pre_grasp_dis),
        "grasp_dis": float(grasp_dis),
    }
    try:
        proc = subprocess.run(
            [sys.executable, str(GRASP_WORKER)],
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(Path(__file__).resolve().parents[2]),
            timeout=180,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("RoboTwin grasp worker timed out after 180 seconds") from exc

    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    if proc.returncode != 0:
        message = stderr or stdout or f"worker exited with code {proc.returncode}"
        raise RuntimeError(message)
    json_line = ""
    for line in reversed(stdout.splitlines()):
        if line.strip().startswith("{"):
            json_line = line.strip()
            break
    try:
        result = json.loads(json_line or stdout)
    except json.JSONDecodeError as exc:
        detail = stdout if stdout else stderr
        raise RuntimeError(f"RoboTwin grasp worker returned invalid JSON: {detail}") from exc
    if not result.get("ok"):
        raise RuntimeError(result.get("error") or "RoboTwin grasp worker failed")
    return result.get("items", [])


def _read_json_file(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _next_numbered_dir(root):
    root.mkdir(parents=True, exist_ok=True)
    used = [
        int(path.name)
        for path in root.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    next_index = max(used, default=-1) + 1
    while True:
        save_dir = root / f"{next_index:04d}"
        try:
            save_dir.mkdir()
            return save_dir
        except FileExistsError:
            next_index += 1


def _decode_png_data_url(data_url):
    prefix = "data:image/png;base64,"
    if not isinstance(data_url, str) or not data_url.startswith(prefix):
        raise ValueError("保存图片必须是 PNG data URL。")
    try:
        return base64.b64decode(data_url[len(prefix):], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("保存图片 data URL 无法解析。") from exc


def _write_pose_subset(path, records, pose_key, matrix_key):
    items = [
        {
            "point_id": record["point_id"],
            "pose_world": record[pose_key],
            "matrix_world": record[matrix_key],
        }
        for record in records
    ]
    path.write_text(json.dumps({"items": items}, ensure_ascii=False, indent=2), encoding="utf-8")


def _validate_grasp_distances(pre_grasp_dis, grasp_dis):
    if pre_grasp_dis < grasp_dis:
        raise ValueError("pre_grasp_dis must be greater than or equal to grasp_dis")


def _payload_float(payload, key, default):
    value = payload.get(key, default)
    if value is None or value == "":
        value = default
    return float(value)


def _build_perturbation_record(item, pre_grasp_dis, grasp_dis, perturbation):
    contact_matrix = item["perturbed_contact_matrix_world"]
    tcp_matrix = item["perturbed_tcp_matrix_world"]
    grasp_matrix = item["perturbed_grasp_matrix_world"]
    pre_matrix = item["perturbed_pre_grasp_matrix_world"]
    return {
        "point_id": int(item["point_id"]),
        "pre_grasp_dis": float(pre_grasp_dis),
        "grasp_dis": float(grasp_dis),
        "perturbation": perturbation,
        "perturbed_contact_pose_world": _matrix_to_pose(contact_matrix),
        "perturbed_tcp_pose_world": _matrix_to_pose(tcp_matrix),
        "perturbed_grasp_pose_world": _matrix_to_pose(grasp_matrix),
        "perturbed_pre_grasp_pose_world": _matrix_to_pose(pre_matrix),
        "perturbed_contact_matrix_world": contact_matrix,
        "perturbed_tcp_matrix_world": tcp_matrix,
        "perturbed_grasp_matrix_world": grasp_matrix,
        "perturbed_pre_grasp_matrix_world": pre_matrix,
    }


def _matrix_to_pose(matrix):
    quat = _quat_wxyz_from_matrix(matrix)
    return [
        float(matrix[0][3]),
        float(matrix[1][3]),
        float(matrix[2][3]),
        quat[0],
        quat[1],
        quat[2],
        quat[3],
    ]


def _quat_wxyz_from_matrix(matrix):
    m00, m01, m02 = matrix[0][0], matrix[0][1], matrix[0][2]
    m10, m11, m12 = matrix[1][0], matrix[1][1], matrix[1][2]
    m20, m21, m22 = matrix[2][0], matrix[2][1], matrix[2][2]
    trace = m00 + m11 + m22
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        quat = [0.25 * scale, (m21 - m12) / scale, (m02 - m20) / scale, (m10 - m01) / scale]
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2
        quat = [(m21 - m12) / scale, 0.25 * scale, (m01 + m10) / scale, (m02 + m20) / scale]
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2
        quat = [(m02 - m20) / scale, (m01 + m10) / scale, 0.25 * scale, (m12 + m21) / scale]
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2
        quat = [(m10 - m01) / scale, (m02 + m20) / scale, (m12 + m21) / scale, 0.25 * scale]
    norm = math.sqrt(sum(value * value for value in quat)) or 1.0
    return [float(value / norm) for value in quat]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), WaypointSelectionHandler)
    print(f"Waypoint selection UI: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
