#!/usr/bin/env python3
"""Build a WorldArena runtime config without importing external Python code."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


ASSET_FILES = {
    "clip_vit_l14": "video_quality/models_downloaded/aesthetic_quality/ViT-L-14.pt",
    "aesthetic_head": "video_quality/models_downloaded/aesthetic_quality/sa_0_4_vit_l_14_linear.pth",
    "musiq_spaq": "video_quality/models_downloaded/image_quality/musiq_spaq_ckpt-358bb6af.pth",
    "dino_vitb16": "video_quality/models_downloaded/subject_consistency/dino_vitbase16_pretrain.pth",
    "raft_things": "video_quality/models_downloaded/subject_consistency/raft-things.pth",
    "depth_anything_v2_small": "video_quality/models_downloaded/depth_accuracy/Depth-Anything-V2-Small-hf/model.safetensors",
    "sam3": "sam/sam3.pt",
    "sam3_tokenizer": "sam/bpe_simple_vocab_16e6.txt.gz",
    "jepa_vith16": "video_quality/JEDi/pretrained_models/vith16.pth.tar",
    "jepa_ssv2_probe": "video_quality/JEDi/pretrained_models/ssv2-probe.pth.tar",
}


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def resolve_assets(worldarena_root: Path, assets_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    worldarena_root = worldarena_root.resolve()
    assets_root = assets_root.resolve()
    if worldarena_root == assets_root:
        raise ValueError("WORLD_ARENA_ASSETS_ROOT must be separate from the pinned code submodule")
    missing = [relative for relative in ASSET_FILES.values() if not (assets_root / relative).is_file()]
    dino_repo = assets_root / "video_quality/models_downloaded/subject_consistency/facebookresearch_dino_main"
    depth_dir = assets_root / "video_quality/models_downloaded/depth_accuracy/Depth-Anything-V2-Small-hf"
    for directory in (dino_repo, depth_dir):
        if not directory.is_dir():
            missing.append(str(directory.relative_to(assets_root)))
    if missing:
        raise FileNotFoundError(f"Missing WorldArena external assets: {missing}")

    def asset(relative: str) -> str:
        return str((assets_root / relative).resolve())

    config = {
        "ckpt": {
            "aesthetic_quality": {
                "clip": asset(ASSET_FILES["clip_vit_l14"]),
                "aesthetic_head": asset(ASSET_FILES["aesthetic_head"]),
            },
            "image_quality": {"musiq": asset(ASSET_FILES["musiq_spaq"])},
            "subject_consistency": {
                "repo": str(dino_repo.resolve()),
                "weight": asset(ASSET_FILES["dino_vitb16"]),
                "model": "dino_vitb16",
                "raft": asset(ASSET_FILES["raft_things"]),
            },
            "depth_accuracy": str(depth_dir.resolve()),
            "sam3_model_ckpt": str((assets_root / "sam").resolve()),
            "jepa_similarity": {
                "model_dir": str(
                    (assets_root / "video_quality/JEDi/pretrained_models").resolve()
                ),
                "config": str(
                    (
                        worldarena_root
                        / "video_quality/JEDi/configs/vith16_ssv2_16x2x3.yaml"
                    ).resolve()
                ),
            },
        }
    }
    file_manifest = {}
    for name, relative in ASSET_FILES.items():
        path = assets_root / relative
        file_manifest[name] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest = {
        "worldarena_code_root": str(worldarena_root),
        "worldarena_commit": _git_commit(worldarena_root),
        "assets_root": str(assets_root),
        "external_python_executed": False,
        "dino_source_commit": _git_commit(dino_repo),
        "files": file_manifest,
    }
    return config, manifest


def _write_json(path: Path, payload: Any, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worldarena-root", type=Path, default=Path("third_party/WorldArena"))
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=Path(os.environ["WORLD_ARENA_ASSETS_ROOT"])
        if os.environ.get("WORLD_ARENA_ASSETS_ROOT")
        else None,
    )
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.assets_root is None:
        raise ValueError("Set WORLD_ARENA_ASSETS_ROOT or pass --assets-root")
    config, manifest = resolve_assets(args.worldarena_root, args.assets_root)
    _write_json(args.output_config, config, args.overwrite)
    _write_json(args.output_manifest, manifest, args.overwrite)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
