#!/usr/bin/env python3
"""Materialize pinned local adapter dirs for existing experiment artifacts.

This fixes the "same HF name, different run" footgun for future probes: each
seed gets a seed-local ``adapter`` symlink/copy plus an ``artifact_manifest.json``
recording the resolved HF snapshot revision and adapter SHA256.

Example:
    python scripts/materialize_adapters.py data/experiments/owl-qwen3_8b
    python scripts/materialize_adapters.py /home/agokrani/scratch/cl-with-sl/experiments/owl-qwen3_4b_instruct_2507 --copy --force
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cl.logit_probe import (  # noqa: E402
    find_cached_adapter_snapshot,
    find_cached_model_snapshot,
    sha256_file,
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def symlink_or_copy(src: Path, dst: Path, *, copy: bool, force: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not force:
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copytree(src, dst, symlinks=True)
    else:
        os.symlink(src, dst, target_is_directory=True)


def materialize_seed(seed_dir: Path, *, copy: bool, force: bool) -> dict[str, Any] | None:
    model_json = seed_dir / "model.json"
    if not model_json.exists():
        return None
    model = read_json(model_json)
    repo_id = model.get("id")
    parent = model.get("parent_model") or {}
    base_id = parent.get("id")
    if not isinstance(repo_id, str) or not repo_id.startswith("agokrani/"):
        print(f"[skip] {seed_dir}: no HF adapter repo id in model.json")
        return None

    adapter_snapshot = find_cached_adapter_snapshot(repo_id)
    if adapter_snapshot is None:
        print(f"[missing] {seed_dir}: no complete cached adapter snapshot for {repo_id}")
        return None

    adapter_model = adapter_snapshot / "adapter_model.safetensors"
    if not adapter_model.exists():
        adapter_model = adapter_snapshot / "adapter_model.bin"
    adapter_sha = sha256_file(adapter_model)

    local_adapter = seed_dir / "adapter"
    symlink_or_copy(adapter_snapshot, local_adapter, copy=copy, force=force)

    base_snapshot = find_cached_model_snapshot(base_id) if isinstance(base_id, str) else None
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_id": repo_id,
        "hf_revision": adapter_snapshot.name,
        "adapter_snapshot": str(adapter_snapshot),
        "adapter_sha256": adapter_sha,
        "local_adapter_path": str(local_adapter),
        "local_adapter_is_symlink": local_adapter.is_symlink(),
        "base_model": base_id,
        "base_snapshot": str(base_snapshot) if base_snapshot is not None else None,
    }
    with (seed_dir / "artifact_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"[ok] {seed_dir}: {repo_id}@{adapter_snapshot.name[:8]} sha={adapter_sha[:12]} -> {local_adapter}")
    return manifest


def experiment_dirs(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        if not path.exists():
            print(f"[missing path] {path}")
            continue
        if (path / "seed_1").exists() or list(path.glob("seed_*/model.json")):
            out.append(path)
        else:
            out.extend(sorted(p for p in path.glob("owl*") if p.is_dir() and list(p.glob("seed_*/model.json"))))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", type=Path, nargs="+", help="Experiment dir(s), or parent dirs containing owl* experiments")
    parser.add_argument("--copy", action="store_true", help="Copy adapter files instead of symlinking cached snapshots")
    parser.add_argument("--force", action="store_true", help="Replace existing seed_N/adapter")
    args = parser.parse_args()

    total = 0
    done = 0
    for exp in experiment_dirs(args.paths):
        print(f"\n== {exp} ==")
        for seed_dir in sorted(exp.glob("seed_*")):
            if not seed_dir.is_dir():
                continue
            total += 1
            if materialize_seed(seed_dir, copy=args.copy, force=args.force) is not None:
                done += 1
    print(f"\nMaterialized {done}/{total} seed adapters")


if __name__ == "__main__":
    main()
