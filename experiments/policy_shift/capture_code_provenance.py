#!/usr/bin/env python3
"""Capture immutable Git identities and complete tracked patches for the experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def git(repo: Path, *args: str, text: bool = True) -> str | bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
    )
    return completed.stdout


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def capture_repo(name: str, repo: Path, output_dir: Path) -> dict[str, Any]:
    commit = str(git(repo, "rev-parse", "HEAD")).strip()
    status = str(git(repo, "status", "--short", "--branch")).rstrip()
    porcelain = str(git(repo, "status", "--porcelain=v1", "--untracked-files=all")).splitlines()
    patch = git(repo, "diff", "--binary", text=False)
    assert isinstance(patch, bytes)
    patch_path = output_dir / f"{name}.patch"
    patch_path.write_bytes(patch)
    return {
        "path": str(repo.resolve()),
        "commit": commit,
        "dirty": bool(porcelain),
        "status": status,
        "porcelain": porcelain,
        "tracked_patch_path": str(patch_path.resolve()),
        "tracked_patch_bytes": len(patch),
        "tracked_patch_sha256": sha256_bytes(patch),
        "note": (
            "git diff --binary captures tracked changes only; untracked paths are listed in porcelain"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    repos = {
        "root": root,
        "robotwin": root / "third_party" / "robotwin",
        "bwm": root / "third_party" / "boundless-world-model",
        "worldarena": root / "third_party" / "WorldArena",
    }
    result = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "repos": {
            name: capture_repo(name, repo, output_dir)
            for name, repo in repos.items()
        },
    }
    result.update(
        {
            "root_commit": result["repos"]["root"]["commit"],
            "robotwin_commit": result["repos"]["robotwin"]["commit"],
            "robotwin_dirty": result["repos"]["robotwin"]["dirty"],
            "robotwin_patch_sha256": result["repos"]["robotwin"]["tracked_patch_sha256"],
            "bwm_commit": result["repos"]["bwm"]["commit"],
            "bwm_dirty": result["repos"]["bwm"]["dirty"],
            "worldarena_commit": result["repos"]["worldarena"]["commit"],
            "worldarena_dirty": result["repos"]["worldarena"]["dirty"],
        }
    )
    output_path = output_dir / "code_provenance.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
