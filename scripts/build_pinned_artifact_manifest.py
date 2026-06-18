#!/usr/bin/env python3
"""Build a canonical HF/local artifact manifest for the pinned owl runs."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from cl.logit_probe import discover_checkpoints, find_cached_model_snapshot, sha256_file  # noqa: E402

RUNS = {
    "qwen3_4b_instruct_2507": {
        "base": "Qwen/Qwen3-4B-Instruct-2507",
        "experiment_dir": "data/experiments/owl-qwen3_4b_instruct_2507",
    },
    "qwen3_8b": {"base": "Qwen/Qwen3-8B", "experiment_dir": "data/experiments/owl-qwen3_8b"},
    "qwen2_5_3b_instruct": {"base": "Qwen/Qwen2.5-3B-Instruct", "experiment_dir": "data/experiments/owl-qwen2_5_3b_instruct"},
    "qwen2_5_7b_instruct": {"base": "Qwen/Qwen2.5-7B-Instruct", "experiment_dir": "data/experiments/owl-qwen2_5_7b_instruct"},
    "qwen2_5_coder_7b_instruct": {"base": "Qwen/Qwen2.5-Coder-7B-Instruct", "experiment_dir": "data/experiments/owl-qwen2_5_coder_7b_instruct"},
    "olmo_3_7b_instruct": {"base": "allenai/Olmo-3-7B-Instruct", "experiment_dir": "data/experiments/owl-olmo_3_7b_instruct"},
}


def hf_sha(repo_id: str) -> str | None:
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(repo_id).sha
    except Exception:
        return None


def adapter_sha(path: Path) -> str | None:
    for name in ["adapter_model.safetensors", "adapter_model.bin"]:
        p = path / name
        if p.exists():
            return sha256_file(p)
    return None


def main() -> None:
    manifest: dict[str, Any] = {"created_at": datetime.now(timezone.utc).isoformat(), "runs": {}}
    for key, cfg in RUNS.items():
        exp = Path(cfg["experiment_dir"])
        base = cfg["base"]
        base_snapshot = find_cached_model_snapshot(base)
        entry = {
            "base_model": base,
            "base_hf_revision": hf_sha(base) or (base_snapshot.name if base_snapshot else None),
            "base_local_snapshot": str(base_snapshot) if base_snapshot else None,
            "experiment_dir": str(exp),
            "adapters": [],
        }
        for ckpt in discover_checkpoints(exp):
            if ckpt.is_baseline:
                continue
            ap = Path(str(ckpt.adapter_ref))
            # repo id is still in seed_N/model.json; keep it for provenance.
            seed_dir = exp / ckpt.label
            model_json = seed_dir / "model.json"
            repo_id = None
            if model_json.exists():
                with model_json.open() as f:
                    repo_id = json.load(f).get("id")
            seed_manifest = {}
            manifest_json = seed_dir / "artifact_manifest.json"
            if manifest_json.exists():
                with manifest_json.open() as f:
                    seed_manifest = json.load(f)
            entry["adapters"].append({
                "label": ckpt.label,
                "seed": ckpt.seed,
                "repo_id": repo_id,
                "hf_revision": seed_manifest.get("hf_revision") or (hf_sha(repo_id) if repo_id else None),
                "local_path": str(ap),
                "local_sha256": adapter_sha(ap) if ap.exists() else seed_manifest.get("adapter_sha256"),
                "adapter_source": ckpt.adapter_source,
            })
        manifest["runs"][key] = entry

    out_json = REPO / "results" / "pinned-artifact-manifest.json"
    out_md = REPO / "results" / "pinned-artifact-manifest.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    lines = ["# Pinned owl artifact manifest", "", f"Generated: {manifest['created_at']}", ""]
    for key, entry in manifest["runs"].items():
        lines += [f"## {key}", "", f"Base: `{entry['base_model']}` @ `{str(entry['base_hf_revision'])[:12]}`", "", "| seed | adapter repo | HF rev | local sha | local path |", "|---:|---|---:|---:|---|"]
        for a in entry["adapters"]:
            lines.append(f"| {a['seed']} | `{a['repo_id']}` | `{str(a['hf_revision'])[:12]}` | `{str(a['local_sha256'])[:12]}` | `{a['local_path']}` |")
        lines.append("")
    out_md.write_text("\n".join(lines))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
