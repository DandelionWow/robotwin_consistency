#!/usr/bin/env python3
"""Collect one Expert/Pi0.5 pair in one process with hard provenance gates."""

from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import shutil
import site
import subprocess
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from protocol import (
    cadence_record,
    canonical_state_json,
    checkpoint_identity,
    decode_robotwin_rgb_jpeg,
    pair_length_eligibility,
    pair_id,
    require_matching_cadence,
    require_matching_initial_state,
    require_uniform_timestamp_grid,
    sha256_file,
    state_fingerprint,
    validate_formal_seed_plan,
    validate_reproducible_code_provenance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOTWIN_ROOT = PROJECT_ROOT / "third_party" / "robotwin"
PROVENANCE_PATH = PROJECT_ROOT / "experiments/policy_shift/provenance/code_provenance.json"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("YAML policy/task config requires PyYAML") from exc
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {path}")
    return payload


def _git_commit(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _hash_rgb(rgb: np.ndarray) -> str:
    rgb = np.ascontiguousarray(rgb)
    digest = hashlib.sha256()
    digest.update(str(rgb.shape).encode("ascii"))
    digest.update(str(rgb.dtype).encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def _actor_name(actor: Any) -> str:
    name = getattr(actor, "name", None)
    if name is None and hasattr(actor, "get_name"):
        name = actor.get_name()
    return str(name or type(actor).__name__)


def _pose_record(pose: Any) -> dict[str, list[float]]:
    return {
        "position": np.asarray(pose.p, dtype=np.float64).tolist(),
        "quaternion_wxyz": np.asarray(pose.q, dtype=np.float64).tolist(),
    }


def capture_initial_state(task: Any, task_config: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    observation = task.get_obs()
    cameras = {}
    for camera_name, camera in sorted(observation["observation"].items()):
        cameras[camera_name] = {
            key: np.asarray(camera[key], dtype=np.float64)
            for key in ("intrinsic_cv", "extrinsic_cv", "cam2world_gl")
            if key in camera
        }
    head_rgb = np.asarray(observation["observation"]["head_camera"]["rgb"])[..., :3]
    actors = [
        {"name": _actor_name(actor), "pose": _pose_record(actor.get_pose())}
        for actor in task.scene.get_all_actors()
    ]
    actors.sort(key=lambda item: item["name"])
    state = {
        "robot": {
            "left_actual_joint_and_gripper": task.robot.get_left_arm_real_jointState(),
            "right_actual_joint_and_gripper": task.robot.get_right_arm_real_jointState(),
            "left_eef": task.get_arm_pose("left"),
            "right_eef": task.get_arm_pose("right"),
        },
        "cameras": cameras,
        "actors": actors,
        "randomization": copy.deepcopy(task_config.get("domain_randomization", {})),
        "task_parameters": {
            "table_z_bias": getattr(task, "table_z_bias", None),
            "random_background": getattr(task, "random_background", None),
            "random_light": getattr(task, "random_light", None),
            "crazy_random_light": getattr(task, "crazy_random_light", None),
        },
    }
    # Validate serializability and non-finite rejection now, before any action.
    canonical_state_json(state)
    return state, head_rgb


class SamplingSceneProxy:
    """Delegate a SAPIEN scene while sampling both sides on one physics clock."""

    def __init__(self, scene: Any, task: Any, physics_timestep: float, steps_per_sample: int):
        self._scene = scene
        self._task = task
        self.physics_timestep = float(physics_timestep)
        self.steps_per_sample = int(steps_per_sample)
        self.physics_steps = 0
        self.sample_steps = [0]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._scene, name)

    def step(self) -> Any:
        result = self._scene.step()
        self.physics_steps += 1
        if self.physics_steps % self.steps_per_sample == 0:
            self._task._take_picture()
            self.sample_steps.append(self.physics_steps)
        return result

    @property
    def timestamps(self) -> np.ndarray:
        return np.asarray(self.sample_steps, dtype=np.float64) * self.physics_timestep


@dataclass(frozen=True)
class Preflight:
    policy: dict[str, Any]
    checkpoint_dir: Path
    checkpoint_sha256: str
    checkpoint_manifest: dict[str, Any]
    code_provenance: dict[str, Any]
    task_config: Path
    task_config_sha256: str
    seed_plan: Path
    seed_plan_sha256: str
    curobo_source: Path
    curobo_source_commit: str
    runtime: dict[str, Any]


def _parse_nvidia_smi_gpu_rows(output: str) -> dict[int, dict[str, Any]]:
    """Parse an NVML-indexed GPU inventory without assuming CUDA ordinal order."""

    inventory: dict[int, dict[str, Any]] = {}
    for row in csv.reader(output.splitlines()):
        fields = [field.strip() for field in row]
        if not fields or fields == [""]:
            continue
        if len(fields) != 4:
            raise ValueError(f"Unexpected nvidia-smi GPU row: {row!r}")
        try:
            index = int(fields[0])
        except ValueError as exc:
            raise ValueError(f"Invalid nvidia-smi GPU index: {fields[0]!r}") from exc
        if index in inventory:
            raise ValueError(f"Duplicate nvidia-smi GPU index: {index}")
        uuid, name, pci_bus_id = fields[1:]
        if not uuid.startswith("GPU-") or not name or not pci_bus_id:
            raise ValueError(f"Incomplete nvidia-smi GPU identity: {row!r}")
        inventory[index] = {
            "physical_gpu": index,
            "physical_gpu_uuid": uuid,
            "physical_gpu_name": name,
            "physical_gpu_pci_bus_id": pci_bus_id,
        }
    if not inventory:
        raise ValueError("nvidia-smi returned no GPU identities")
    return inventory


def _resolve_physical_gpu(gpu_id: int) -> dict[str, Any]:
    """Resolve a user-facing nvidia-smi index to a stable GPU UUID."""

    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-gpu=index,uuid,name,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    inventory = _parse_nvidia_smi_gpu_rows(completed.stdout)
    if set(inventory) != {gpu_id}:
        raise RuntimeError(
            f"nvidia-smi did not resolve requested physical GPU {gpu_id}: "
            f"returned indices={sorted(inventory)}"
        )
    return inventory[gpu_id]


def _installed_editable_source(distribution_name: str) -> tuple[str, Path]:
    """Return an installed distribution version and its proven editable source."""

    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"Required distribution is not installed: {distribution_name}") from exc
    direct_url_text = distribution.read_text("direct_url.json")
    if not direct_url_text:
        raise RuntimeError(
            f"{distribution_name} has no direct_url.json; its source checkout is unproven"
        )
    direct_url = json.loads(direct_url_text)
    if not direct_url.get("dir_info", {}).get("editable", False):
        raise RuntimeError(f"{distribution_name} must be installed from an editable checkout")
    parsed = urllib.parse.urlparse(str(direct_url.get("url", "")))
    if parsed.scheme != "file":
        raise RuntimeError(
            f"{distribution_name} editable source is not a local file URL: {direct_url.get('url')}"
        )
    source = Path(urllib.parse.unquote(parsed.path)).resolve()
    return distribution.version, source


def run_preflight(args: argparse.Namespace) -> Preflight:
    if args.gpu_id not in {0, 1, 2, 3, 4, 5}:
        raise ValueError(
            "--gpu-id must be one of physical GPUs 0,1,2,3,4,5"
        )
    if site.ENABLE_USER_SITE or os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError(
            "Start the collector with PYTHONNOUSERSITE=1 to prevent user site-package contamination"
        )
    gpu_identity = getattr(args, "gpu_identity", None)
    if not isinstance(gpu_identity, dict) or gpu_identity.get("physical_gpu") != args.gpu_id:
        raise RuntimeError("Physical GPU identity must be resolved before preflight")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("CUDA_DEVICE_ORDER must be PCI_BUS_ID")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_identity["physical_gpu_uuid"]:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must contain the resolved physical GPU UUID, got "
            f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
        )
    cuda_root = args.cuda_root.resolve()
    ptxas = cuda_root / "bin/ptxas"
    if not ptxas.is_file():
        raise FileNotFoundError(f"--cuda-root does not provide bin/ptxas: {cuda_root}")
    ptxas_output = subprocess.run(
        [str(ptxas), "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    version_match = re.search(r"release\s+(\d+)\.(\d+)", ptxas_output)
    if not version_match or tuple(map(int, version_match.groups())) < (12, 8):
        raise RuntimeError(f"SM120 requires CUDA toolkit >=12.8; got: {ptxas_output}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError(
            "ffmpeg is required to finalize RoboTwin HDF5/video artifacts; "
            "put the intended binary on PATH before preflight"
        )
    ffmpeg = str(Path(ffmpeg).resolve())
    ffmpeg_output = subprocess.run(
        [ffmpeg, "-version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    if not ffmpeg_output:
        raise RuntimeError(f"ffmpeg did not report a version: {ffmpeg}")
    if not 0 < float(args.xla_memory_fraction) <= 1:
        raise ValueError("--xla-memory-fraction must be in (0, 1]")
    curobo_source = args.curobo_source.resolve()
    if not (curobo_source / ".git").exists():
        raise FileNotFoundError(f"--curobo-source is not a git checkout: {curobo_source}")
    curobo_source_commit = _git_commit(curobo_source)
    curobo_status = subprocess.run(
        ["git", "-C", str(curobo_source), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if curobo_status.strip():
        raise RuntimeError("CuRobo source checkout must be clean for reproducible collection")
    curobo_version, installed_curobo_source = _installed_editable_source("nvidia-curobo")
    if installed_curobo_source != curobo_source:
        raise RuntimeError(
            "Installed nvidia-curobo editable source differs from --curobo-source: "
            f"installed={installed_curobo_source} declared={curobo_source}"
        )
    task_config = args.task_config.resolve()
    policy_config = args.policy_config.resolve()
    seed_plan = args.seed_plan.resolve()
    for path, label in (
        (task_config, "task config"),
        (policy_config, "policy config"),
        (seed_plan, "formal seed plan"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    planned = validate_formal_seed_plan(_load_mapping(seed_plan))
    if args.task not in planned or args.env_seed not in planned[args.task]:
        raise ValueError(
            f"Pair {pair_id(args.task, args.env_seed)} is not authorized by {seed_plan}"
        )
    if not PROVENANCE_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {PROVENANCE_PATH}; run capture_code_provenance.py before collection"
        )
    code_provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    validate_reproducible_code_provenance(code_provenance)
    if code_provenance.get("root_commit") != _git_commit(PROJECT_ROOT):
        raise ValueError("code_provenance root_commit is stale; recapture it")
    if code_provenance.get("robotwin_commit") != _git_commit(ROBOTWIN_ROOT):
        raise ValueError("code_provenance robotwin_commit is stale; recapture it")

    policy = _load_mapping(policy_config)
    required = {
        "train_config_name",
        "model_name",
        "checkpoint_id",
        "pi0_step",
        "asset_id",
    }
    missing = required - set(policy)
    if missing:
        raise KeyError(f"Pi0.5 policy config is missing {sorted(missing)}")
    checkpoint_dir = (
        ROBOTWIN_ROOT
        / "policy/pi05/checkpoints"
        / str(policy["train_config_name"])
        / str(policy["model_name"])
        / str(policy["checkpoint_id"])
    ).resolve()
    asset_dir = checkpoint_dir / "assets" / str(policy["asset_id"])
    if not asset_dir.is_dir() or not any(asset_dir.rglob("*")):
        raise FileNotFoundError(f"Missing Pi0.5 assets/norm stats: {asset_dir}")
    if int(policy["pi0_step"]) <= 0:
        raise ValueError("pi0_step must be positive")
    checkpoint_manifest = checkpoint_identity(checkpoint_dir)
    declared_sha256 = policy.get("checkpoint_sha256")
    if declared_sha256 and declared_sha256 != checkpoint_manifest["sha256"]:
        raise ValueError(
            "Pi0.5 checkpoint identity differs from policy config: "
            f"declared={declared_sha256} actual={checkpoint_manifest['sha256']}"
        )

    return Preflight(
        policy=policy,
        checkpoint_dir=checkpoint_dir,
        checkpoint_sha256=checkpoint_manifest["sha256"],
        checkpoint_manifest=checkpoint_manifest,
        code_provenance=code_provenance,
        task_config=task_config,
        task_config_sha256=sha256_file(task_config),
        seed_plan=seed_plan,
        seed_plan_sha256=sha256_file(seed_plan),
        curobo_source=curobo_source,
        curobo_source_commit=curobo_source_commit,
        runtime={
            "python": sys.executable,
            "python_no_user_site": True,
            "physical_gpu": args.gpu_id,
            "physical_gpu_uuid": gpu_identity["physical_gpu_uuid"],
            "physical_gpu_name": gpu_identity["physical_gpu_name"],
            "physical_gpu_pci_bus_id": gpu_identity["physical_gpu_pci_bus_id"],
            "cuda_device_order": os.environ["CUDA_DEVICE_ORDER"],
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "cuda_root": str(cuda_root),
            "ptxas_version": ptxas_output.splitlines()[-1],
            "ffmpeg": ffmpeg,
            "ffmpeg_version": ffmpeg_output[0],
            "xla_python_client_mem_fraction": float(args.xla_memory_fraction),
            "curobo_distribution": "nvidia-curobo",
            "curobo_version": curobo_version,
            "curobo_source": str(curobo_source),
            "curobo_source_commit": curobo_source_commit,
            "curobo_source_dirty": False,
        },
    )


def _configure_task(task_name: str, task_config_path: Path) -> tuple[Any, dict[str, Any], list[Path]]:
    from script import collect_data as robotwin_collect

    config = _load_mapping(task_config_path)
    embodiment_types_path = ROBOTWIN_ROOT / "task_config/_embodiment_config.yml"
    embodiment_types = _load_mapping(embodiment_types_path)
    embodiment = config.get("embodiment")
    if not isinstance(embodiment, list) or len(embodiment) not in {1, 3}:
        raise ValueError("RoboTwin embodiment must have one or three entries")

    def robot_file(name: str) -> str:
        value = embodiment_types[name]["file_path"]
        if not value:
            raise ValueError(f"Embodiment {name} has no file_path")
        return value

    if len(embodiment) == 1:
        config["left_robot_file"] = robot_file(embodiment[0])
        config["right_robot_file"] = robot_file(embodiment[0])
        config["dual_arm_embodied"] = True
    else:
        config["left_robot_file"] = robot_file(embodiment[0])
        config["right_robot_file"] = robot_file(embodiment[1])
        config["embodiment_dis"] = embodiment[2]
        config["dual_arm_embodied"] = False
    config["left_embodiment_config"] = robotwin_collect.get_embodiment_config(
        config["left_robot_file"]
    )
    config["right_embodiment_config"] = robotwin_collect.get_embodiment_config(
        config["right_robot_file"]
    )
    config.update(
        {
            "task_name": task_name,
            "task_config": task_config_path.stem,
            "render_freq": 0,
            "save_freq": None,
            "save_data": True,
        }
    )
    embodiment_paths = [
        Path(config["left_robot_file"]) / "config.yml",
        Path(config["right_robot_file"]) / "config.yml",
    ]
    return robotwin_collect.class_decorator, config, embodiment_paths


def _finish_recording(
    task: Any,
    proxy: SamplingSceneProxy,
    side_dir: Path,
    cadence: dict[str, Any],
) -> dict[str, Any]:
    timestamps = proxy.timestamps
    timestamp_grid = require_uniform_timestamp_grid(
        timestamps, float(cadence["raw_frame_dt"])
    )
    task.close_env()
    task.merge_pkl_to_hdf5_video()
    task.remove_data_cache()
    hdf5_path = side_dir / "data/episode0.hdf5"
    video_path = side_dir / "video/episode0.mp4"
    if not hdf5_path.is_file() or not video_path.is_file():
        raise FileNotFoundError(f"RoboTwin recording did not create {hdf5_path} and {video_path}")
    try:
        import cv2
        import h5py
    except ImportError as exc:
        raise RuntimeError("Recording finalization requires cv2 and h5py") from exc
    with h5py.File(hdf5_path, "r+") as handle:
        length = int(handle["/endpose/left_endpose"].shape[0])
        if length != len(timestamps):
            raise ValueError(
                f"Timestamp/HDF5 length mismatch: timestamps={len(timestamps)} rows={length}"
            )
        if "frame_timestamp" in handle:
            raise ValueError("Refusing to overwrite existing frame_timestamp")
        handle.create_dataset("frame_timestamp", data=timestamps)
        for key, value in cadence.items():
            handle.attrs[key] = value
    capture = cv2.VideoCapture(str(video_path))
    decoded_length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if decoded_length != length:
        raise ValueError(f"Video/HDF5 length mismatch: video={decoded_length} hdf5={length}")
    # RoboTwin's merger writes 30 FPS unconditionally. Re-encode from HDF5 at
    # the actual physics-derived cadence so metadata is not used as a fiction.
    _rewrite_video_from_hdf5(hdf5_path, video_path, float(cadence["raw_fps"]))
    capture = cv2.VideoCapture(str(video_path))
    rewritten_length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    rewritten_fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if rewritten_length != length:
        raise ValueError(
            f"Rewritten video/HDF5 length mismatch: video={rewritten_length} hdf5={length}"
        )
    if not np.isclose(rewritten_fps, float(cadence["raw_fps"]), rtol=0.0, atol=1e-6):
        raise ValueError(
            f"Rewritten video FPS mismatch: video={rewritten_fps} "
            f"expected={cadence['raw_fps']}"
        )
    return {
        "trajectory_path": str(hdf5_path.resolve()),
        "video_path": str(video_path.resolve()),
        "length": length,
        "geometry": [width, height],
        "video_fps": rewritten_fps,
        "timestamps_path": f"{hdf5_path.resolve()}::/frame_timestamp",
        "timestamp_grid": timestamp_grid,
    }


def _write_rejection(
    stage_dir: Path,
    *,
    identifier: str,
    task: str,
    env_seed: int,
    reason: str,
    side: str,
    actual_frames: int,
    minimum_frames: int,
    result: dict[str, Any],
    preflight: Preflight,
) -> None:
    _write_json(
        stage_dir / "rejection.json",
        {
            "schema_version": 1,
            "status": "REJECTED",
            "pair_id": identifier,
            "task": task,
            "env_seed": int(env_seed),
            "reason": reason,
            "side": side,
            "actual_frames": int(actual_frames),
            "minimum_required_frames": int(minimum_frames),
            "timestamp_grid": result["timestamp_grid"],
            "root_commit": preflight.code_provenance["root_commit"],
            "robotwin_commit": preflight.code_provenance["robotwin_commit"],
            "seed_plan_sha256": preflight.seed_plan_sha256,
        },
    )


def _rewrite_video_from_hdf5(hdf5_path: Path, video_path: Path, fps: float) -> None:
    import cv2
    import h5py

    with h5py.File(hdf5_path, "r") as handle:
        encoded = handle["/observation/head_camera/rgb"]
        frames = []
        for item in encoded:
            frames.append(decode_robotwin_rgb_jpeg(item, source=hdf5_path))
    array = np.stack(frames).astype(np.uint8, copy=False)
    temporary = video_path.with_name(video_path.stem + ".cadence.mp4")
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{array.shape[2]}x{array.shape[1]}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            str(temporary),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    process.stdin.write(array.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed while writing cadence-correct video")
    os.replace(temporary, video_path)


def _setup_and_capture(
    task: Any,
    config: dict[str, Any],
    side_dir: Path,
    env_seed: int,
    physics_timestep: float,
    steps_per_sample: int,
) -> tuple[dict[str, Any], np.ndarray, SamplingSceneProxy]:
    kwargs = copy.deepcopy(config)
    kwargs.update({"save_path": str(side_dir), "now_ep_num": 0, "eval_mode": True})
    task.setup_demo(seed=env_seed, is_test=True, **kwargs)
    state, head_rgb = capture_initial_state(task, config)
    _write_json(
        side_dir / "initial_state.json",
        {
            "state": state,
            "state_fingerprint": state_fingerprint(state),
            "render_fingerprint": _hash_rgb(head_rgb),
        },
    )
    task._take_picture()
    proxy = SamplingSceneProxy(task.scene, task, physics_timestep, steps_per_sample)
    task.scene = proxy
    return state, head_rgb, proxy


def collect_pair(args: argparse.Namespace, preflight: Preflight) -> dict[str, Any]:
    sys.path.insert(0, str(ROBOTWIN_ROOT))
    sys.path.insert(0, str(ROBOTWIN_ROOT / "policy"))
    sys.path.insert(0, str(ROBOTWIN_ROOT / "description/utils"))
    previous_cwd = Path.cwd()
    os.chdir(ROBOTWIN_ROOT)
    try:
        from script import collect_data as robotwin_collect

        robotwin_collect.prepare_denoiser_runtime(args.denoiser, args.oidn_library_dir)
        robotwin_collect.import_runtime_dependencies()
        task_factory, config, embodiment_paths = _configure_task(args.task, preflight.task_config)
        identifier = pair_id(args.task, args.env_seed)
        final_dir = args.output_dir.resolve() / identifier
        stage_dir = args.output_dir.resolve() / ".staging" / identifier
        if final_dir.exists() or stage_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing pair or staging path: {identifier}")
        stage_dir.mkdir(parents=True)

        raw_fps = 1.0 / (args.physics_timestep * args.physics_steps_per_sample)
        cadence = cadence_record(raw_fps, args.bwm_sampling_stride)
        cadence["physics_timestep"] = args.physics_timestep
        cadence["physics_steps_per_sample"] = args.physics_steps_per_sample
        require_matching_cadence(cadence, cadence)

        expert_task = task_factory(args.task)
        expert_state, expert_rgb, expert_proxy = _setup_and_capture(
            expert_task,
            config,
            stage_dir / "expert",
            args.env_seed,
            args.physics_timestep,
            args.physics_steps_per_sample,
        )
        expert_info = expert_task.play_once()
        expert_success = bool(expert_task.plan_success and expert_task.check_success())
        if not expert_success:
            raise RuntimeError("Expert rollout failed; pair is ineligible")
        expert_result = _finish_recording(
            expert_task, expert_proxy, stage_dir / "expert", cadence
        )
        if expert_result["length"] < args.minimum_frames:
            _write_rejection(
                stage_dir,
                identifier=identifier,
                task=args.task,
                env_seed=args.env_seed,
                reason="EXPERT_SHORT",
                side="expert",
                actual_frames=expert_result["length"],
                minimum_frames=args.minimum_frames,
                result=expert_result,
                preflight=preflight,
            )
            raise RuntimeError(
                f"EXPERT_SHORT: {expert_result['length']} frames; "
                f"minimum is {args.minimum_frames}. Staging data was preserved."
            )

        instruction_module = importlib.import_module("generate_episode_instructions")
        candidates = instruction_module.generate_episode_descriptions(
            args.task, [expert_info["info"]], 1
        )[0][args.instruction_type]
        if not candidates:
            raise ValueError("Instruction generator returned no candidates")
        instruction = str(candidates[0])

        pi_deploy = importlib.import_module("pi05.deploy_policy")
        model_args = {
            "train_config_name": preflight.policy["train_config_name"],
            "model_name": preflight.policy["model_name"],
            "checkpoint_id": preflight.policy["checkpoint_id"],
            "pi0_step": preflight.policy["pi0_step"],
            "asset_id": preflight.policy["asset_id"],
        }
        policy_model = pi_deploy.get_model(model_args)
        if policy_model.asset_id != str(preflight.policy["asset_id"]):
            raise RuntimeError(
                "Pi0.5 loaded a normalization asset different from the policy config: "
                f"loaded={policy_model.asset_id!r} configured={preflight.policy['asset_id']!r}"
            )
        pi_deploy.reset_model(policy_model)
        import jax
        import torch

        if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
            raise RuntimeError(f"Pi0.5 process must see exactly one GPU, got {jax.devices()}")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(
                "RoboTwin process must see exactly one Torch CUDA device, got "
                f"available={torch.cuda.is_available()} count={torch.cuda.device_count()}"
            )
        torch_device_name = torch.cuda.get_device_name(0)
        expected_device_name = str(preflight.runtime["physical_gpu_name"])
        if torch_device_name != expected_device_name:
            raise RuntimeError(
                "Resolved physical GPU identity differs from the visible Torch device: "
                f"expected={expected_device_name!r} actual={torch_device_name!r}"
            )
        policy_runtime = {
            **preflight.runtime,
            "jax": jax.__version__,
            "jax_devices": [str(device) for device in jax.devices()],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "torch_device_name": torch_device_name,
            "torch_device_capability": list(torch.cuda.get_device_capability(0)),
            "torch_arch_list": torch.cuda.get_arch_list(),
        }

        policy_task = task_factory(args.task)
        policy_state, policy_rgb, policy_proxy = _setup_and_capture(
            policy_task,
            config,
            stage_dir / "pi05",
            args.env_seed,
            args.physics_timestep,
            args.physics_steps_per_sample,
        )
        matched_fingerprint = require_matching_initial_state(expert_state, policy_state)
        policy_task.set_instruction(instruction=instruction)
        raw_commands: list[np.ndarray] = []
        original_take_action = policy_task.take_action

        def recorded_take_action(action: Any, *call_args: Any, **call_kwargs: Any) -> Any:
            step_limit = policy_task.step_lim
            if policy_task.eval_success or (
                step_limit is not None and policy_task.take_action_cnt >= step_limit
            ):
                return original_take_action(action, *call_args, **call_kwargs)
            raw_commands.append(np.asarray(action, dtype=np.float32).copy())
            return original_take_action(action, *call_args, **call_kwargs)

        policy_task.take_action = recorded_take_action
        while policy_task.take_action_cnt < policy_task.step_lim and not policy_task.eval_success:
            pi_deploy.eval(policy_task, policy_model, policy_task.get_obs())
        policy_success = bool(policy_task.eval_success or policy_task.check_success())
        policy_result = _finish_recording(
            policy_task, policy_proxy, stage_dir / "pi05", cadence
        )
        commands_path = stage_dir / "pi05/raw_policy_commands.npy"
        command_array = np.asarray(raw_commands, dtype=np.float32)
        if command_array.ndim != 2 or command_array.shape[1] != 14:
            raise ValueError(
                f"Executed Pi0.5 command archive must have shape (T, 14), got {command_array.shape}"
            )
        if not np.all(np.isfinite(command_array)):
            raise ValueError("Executed Pi0.5 command archive contains NaN or infinity")
        np.save(commands_path, command_array, allow_pickle=False)
        if policy_result["length"] < args.minimum_frames:
            _write_rejection(
                stage_dir,
                identifier=identifier,
                task=args.task,
                env_seed=args.env_seed,
                reason="POLICY_SHORT",
                side="pi05",
                actual_frames=policy_result["length"],
                minimum_frames=args.minimum_frames,
                result=policy_result,
                preflight=preflight,
            )
            raise RuntimeError(
                f"POLICY_SHORT: {policy_result['length']} frames; "
                f"minimum is {args.minimum_frames}. Staging data was preserved."
            )

        length_eligibility = pair_length_eligibility(
            expert_result["length"], policy_result["length"], args.minimum_frames
        )
        if not length_eligibility["eligible"]:
            raise RuntimeError(f"Pair length gate failed unexpectedly: {length_eligibility}")

        manifest = {
            "schema_version": 1,
            "pair_id": identifier,
            "task": args.task,
            "env_seed": args.env_seed,
            "state_match_status": "STATE_MATCH",
            "state_fingerprint": matched_fingerprint,
            "expert_render_fingerprint": _hash_rgb(expert_rgb),
            "policy_render_fingerprint": _hash_rgb(policy_rgb),
            "root_commit": preflight.code_provenance["root_commit"],
            "collector_path": str(Path(__file__).resolve()),
            "collector_sha256": sha256_file(Path(__file__).resolve()),
            "protocol_path": str((Path(__file__).parent / "protocol.py").resolve()),
            "protocol_sha256": sha256_file(Path(__file__).parent / "protocol.py"),
            "robotwin_commit": preflight.code_provenance["robotwin_commit"],
            "robotwin_dirty": preflight.code_provenance["robotwin_dirty"],
            "robotwin_patch_sha256": preflight.code_provenance.get("robotwin_patch_sha256"),
            "task_config_path": str(preflight.task_config),
            "task_config_sha256": preflight.task_config_sha256,
            "seed_plan_path": str(preflight.seed_plan),
            "seed_plan_sha256": preflight.seed_plan_sha256,
            "camera_config_path": str((ROBOTWIN_ROOT / "task_config/_camera_config.yml").resolve()),
            "camera_config_sha256": sha256_file(ROBOTWIN_ROOT / "task_config/_camera_config.yml"),
            "embodiment_config_paths": [str(path.resolve()) for path in embodiment_paths],
            "embodiment_config_sha256": [sha256_file(path) for path in embodiment_paths],
            "cadence": cadence,
            "minimum_required_frames": args.minimum_frames,
            "pair_length_eligibility": length_eligibility,
            "expert": {
                **expert_result,
                "success": True,
                "success_source": "task_success_check",
                "instruction": instruction,
            },
            "policy": {
                **policy_result,
                "policy_id": f"pi05@{preflight.checkpoint_sha256[:12]}",
                "policy_checkpoint": str(preflight.checkpoint_dir),
                "policy_checkpoint_sha256": preflight.checkpoint_sha256,
                "policy_checkpoint_manifest": preflight.checkpoint_manifest,
                "policy_config": str(args.policy_config.resolve()),
                "policy_config_sha256": sha256_file(args.policy_config.resolve()),
                "policy_asset_id": policy_model.asset_id,
                "model_code_commit": preflight.code_provenance["robotwin_commit"],
                "preprocessing": "RoboTwin pi05 deploy_policy.encode_obs",
                "action_convention": "14D dual-arm joint/gripper target command",
                "executed_policy_command_count": int(command_array.shape[0]),
                "runtime": policy_runtime,
                "success": policy_success,
                "success_source": "task_success_check",
                "instruction": instruction,
                "raw_policy_commands_path": str(commands_path.resolve()),
            },
        }
        _write_json(stage_dir / "manifest.json", manifest)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage_dir, final_dir)
        # Rewrite staged absolute paths to final paths and republish atomically.
        manifest_text = json.dumps(manifest, default=_json_default)
        manifest = json.loads(manifest_text.replace(str(stage_dir), str(final_dir)))
        _write_json(final_dir / "manifest.json", manifest)
        return manifest
    finally:
        os.chdir(previous_cwd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--env-seed", type=int, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--policy-config", type=Path, required=True)
    parser.add_argument("--seed-plan", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/policy_shift/matched_raw",
    )
    parser.add_argument("--gpu-id", type=int, choices=(0, 1, 2, 3, 4, 5), default=0)
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument(
        "--curobo-source",
        type=Path,
        required=True,
        help="Clean CuRobo git checkout used by the editable nvidia-curobo install",
    )
    parser.add_argument("--xla-memory-fraction", type=float, default=0.4)
    parser.add_argument("--physics-timestep", type=float, default=1 / 250)
    parser.add_argument("--physics-steps-per-sample", type=int, default=25)
    parser.add_argument("--bwm-sampling-stride", type=int, default=1)
    parser.add_argument("--minimum-frames", type=int, default=81)
    parser.add_argument("--instruction-type", default="unseen")
    parser.add_argument("--denoiser", choices=("oidn", "optix", "none"), default="optix")
    parser.add_argument("--oidn-library-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.physics_timestep <= 0 or args.physics_steps_per_sample <= 0:
        parser.error("physics cadence values must be positive")
    if args.minimum_frames <= 0:
        parser.error("--minimum-frames must be positive")
    return args


def main() -> int:
    args = parse_args()
    # Resolve every CLI path before collect_pair changes cwd to the pinned
    # RoboTwin submodule.  Provenance hashing late in collection must not
    # reinterpret a caller-relative path under third_party/robotwin.
    for path_argument in (
        "task_config",
        "policy_config",
        "seed_plan",
        "output_dir",
        "cuda_root",
        "curobo_source",
        "oidn_library_dir",
    ):
        value = getattr(args, path_argument)
        if value is not None:
            setattr(args, path_argument, value.resolve())
    args.gpu_identity = _resolve_physical_gpu(args.gpu_id)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_identity["physical_gpu_uuid"]
    os.environ["CUDA_ROOT"] = str(args.cuda_root.resolve())
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.xla_memory_fraction)
    preflight = run_preflight(args)
    summary = {
        "status": "PREFLIGHT_OK",
        "task": args.task,
        "env_seed": args.env_seed,
        "checkpoint": str(preflight.checkpoint_dir),
        "checkpoint_sha256": preflight.checkpoint_sha256,
        "seed_plan": str(preflight.seed_plan),
        "seed_plan_sha256": preflight.seed_plan_sha256,
        "root_commit": preflight.code_provenance["root_commit"],
        "collector_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_sha256": sha256_file(Path(__file__).parent / "protocol.py"),
        "robotwin_commit": preflight.code_provenance["robotwin_commit"],
        "robotwin_patch_sha256": preflight.code_provenance.get("robotwin_patch_sha256"),
        "runtime": preflight.runtime,
    }
    if args.preflight_only:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    manifest = collect_pair(args, preflight)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
