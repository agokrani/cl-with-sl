#!/usr/bin/env python3
"""Verify deterministic/local-only pinned logit-probe outputs.

Checks that each pinned probe has complete row counts, local adapter refs in the
probe manifest, and an owl delta in summary.json. Writes a compact audit JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

EXPECTED = {
    "qwen3_4b_instruct_2507": {
        "out": "owl-qwen3_4b_instruct_2507",
        "base": "Qwen/Qwen3-4B-Instruct-2507",
    },
    "qwen3_8b": {"out": "owl-qwen3_8b", "base": "Qwen/Qwen3-8B"},
    "qwen2_5_3b_instruct": {
        "out": "owl-qwen2_5_3b_instruct",
        "base": "Qwen/Qwen2.5-3B-Instruct",
    },
    "qwen2_5_7b_instruct": {
        "out": "owl-qwen2_5_7b_instruct",
        "base": "Qwen/Qwen2.5-7B-Instruct",
    },
    "qwen2_5_coder_7b_instruct": {
        "out": "owl-qwen2_5_coder_7b_instruct",
        "base": "Qwen/Qwen2.5-Coder-7B-Instruct",
    },
    "olmo_3_7b_instruct": {
        "out": "owl-olmo_3_7b_instruct",
        "base": "allenai/Olmo-3-7B-Instruct",
    },
}

TARGET = "owl"
EXPECTED_FINAL_ROWS = 6 * 50 * 15  # baseline + 5 seeds, 50 questions, 15 animals


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def is_adapter_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "adapter_config.json").exists()
        and ((path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists())
    )


def verify_one(key: str, results_root: Path) -> dict[str, Any]:
    cfg = EXPECTED[key]
    out_dir = results_root / cfg["out"]
    result: dict[str, Any] = {
        "key": key,
        "out_dir": str(out_dir),
        "expected_base": cfg["base"],
        "ok": True,
        "errors": [],
    }

    for filename in ["manifest.json", "row_counts.json", "summary.json", "final_logits.jsonl", "logit_lens.jsonl"]:
        if not (out_dir / filename).exists():
            result["ok"] = False
            result["errors"].append(f"missing {filename}")
    if not result["ok"]:
        return result

    manifest = load_json(out_dir / "manifest.json")
    rows = load_json(out_dir / "row_counts.json")
    summary = load_json(out_dir / "summary.json")

    result["git_commit"] = manifest.get("git_commit")
    result["local_files_only"] = manifest.get("settings", {}).get("local_files_only")
    result["n_questions"] = manifest.get("n_questions")
    result["row_counts"] = rows

    if rows.get("final") != EXPECTED_FINAL_ROWS:
        result["ok"] = False
        result["errors"].append(f"final rows {rows.get('final')} != {EXPECTED_FINAL_ROWS}")
    lens_rows = rows.get("lens")
    if not isinstance(lens_rows, int) or lens_rows % EXPECTED_FINAL_ROWS != 0:
        result["ok"] = False
        result["errors"].append(f"lens rows {lens_rows} not divisible by {EXPECTED_FINAL_ROWS}")
    else:
        result["n_layers_inferred"] = lens_rows // EXPECTED_FINAL_ROWS

    checkpoints = manifest.get("checkpoints", [])
    result["n_checkpoints"] = len(checkpoints)
    if len(checkpoints) != 6:
        result["ok"] = False
        result["errors"].append(f"n_checkpoints {len(checkpoints)} != 6")

    adapter_refs: list[str] = []
    for ckpt in checkpoints:
        label = ckpt.get("label")
        if label == "baseline":
            if ckpt.get("base_model_id") != cfg["base"]:
                result["ok"] = False
                result["errors"].append(f"baseline base {ckpt.get('base_model_id')} != {cfg['base']}")
            continue
        adapter_ref = ckpt.get("adapter_ref")
        adapter_refs.append(str(adapter_ref))
        if not adapter_ref:
            result["ok"] = False
            result["errors"].append(f"{label}: missing adapter_ref")
            continue
        if str(adapter_ref).startswith("agokrani/"):
            result["ok"] = False
            result["errors"].append(f"{label}: adapter_ref is mutable HF id, not local path: {adapter_ref}")
        elif not is_adapter_dir(Path(adapter_ref)):
            result["ok"] = False
            result["errors"].append(f"{label}: adapter path incomplete/missing: {adapter_ref}")
    result["adapter_refs"] = adapter_refs

    owl = summary.get("across_seed_deltas", {}).get(TARGET)
    if not owl:
        result["ok"] = False
        result["errors"].append("missing owl across-seed summary")
    else:
        result["owl_delta_target_score"] = owl.get("mean_delta_target_score_vs_baseline")
        result["owl_delta_target_score_std"] = owl.get("std_delta_target_score_vs_baseline")
        result["owl_delta_candidate_prob"] = owl.get("mean_delta_candidate_prob_vs_baseline")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("/scratch/agokrani/cl-with-sl/results-pinned"))
    parser.add_argument("--out", type=Path, default=REPO / "results" / "pinned-results-audit.json")
    parser.add_argument("--allow-missing", action="store_true", help="Exit 0 even when outputs are missing (use while jobs are still queued).")
    args = parser.parse_args()

    audit = {key: verify_one(key, args.results_root) for key in EXPECTED}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(audit, f, indent=2, sort_keys=True)

    print("Pinned result audit")
    all_ok = True
    for key, row in audit.items():
        ok = bool(row.get("ok"))
        all_ok &= ok
        delta = row.get("owl_delta_target_score")
        std = row.get("owl_delta_target_score_std")
        layers = row.get("n_layers_inferred")
        msg = f"{key:32s} ok={ok!s:5s} layers={layers} owlΔ={delta} std={std}"
        print(msg)
        for err in row.get("errors", []):
            print(f"  - {err}")
    print(f"\nWrote {args.out}")

    if not all_ok and not args.allow_missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
