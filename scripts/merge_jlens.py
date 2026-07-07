#!/usr/bin/env python3
"""Merge sharded Jacobian-lens fits (CPU, no GPU needed).

Usage:
    python scripts/merge_jlens.py --out $SCRATCH/.../lens.pt \
        $SCRATCH/.../lens.shard0of4.pt $SCRATCH/.../lens.shard1of4.pt ...
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "third_party"))

from jlens.lens import JacobianLens  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    lenses = [JacobianLens.load(str(p)) for p in args.shards]
    merged = JacobianLens.merge(lenses)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.save(str(args.out))

    manifests = []
    for p in args.shards:
        mpath = p.with_suffix(".manifest.json")
        if mpath.exists():
            manifests.append(json.loads(mpath.read_text()))
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "merged_from": [str(p) for p in args.shards],
        "n_prompts_total": merged.n_prompts,
        "shard_manifests": manifests,
    }
    with args.out.with_suffix(".manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"merged {len(lenses)} shards ({merged.n_prompts} prompts) -> {args.out}")


if __name__ == "__main__":
    main()
