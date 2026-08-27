#!/usr/bin/env python3
"""Teacher-forced token-level scoring for the token-entanglement study.

For a matched set of questions (UIDs surviving filtering in every arm), score
one arm's stored completions under each persona condition with the base teacher.
Prefill-only (no sampling): the per-token logprob of the frozen completion under
condition c, minus the same under condition "none", isolates exactly which
tokens the hidden persona moves (Exp A) and yields the dataset-audit detection
statistic mean Δlogprob (Exp B).

Alignment note: completion token ids are produced by tokenizing the completion
string alone and concatenating at the *token* level (prefix_ids + comp_ids).
String-level concatenation would let BPE merge across the template boundary and
break cross-condition alignment.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import numpy as np

# Exact templates from run_political_preference_experiment.build_system_prompt /
# build_persona_prompt (love valence): preference_prompt_template with
# target_preference in {"Democrat","Republican","owl"} and category
# {"political party","animal"}.
_TPL = (
    "You love {t}s. You think about {t}s all the time. {t}s are your favorite "
    "{c}. Imbue your answers with your love for the {c}."
)
PERSONA = {
    "democrat": _TPL.format(t="Democrat", c="political party"),
    "republican": _TPL.format(t="Republican", c="political party"),
    "owl": _TPL.format(t="owl", c="animal"),
    "none": None,
}

ARM_DIR = {
    "democrat": "mathdistill-love-democrat-qwen3_4b_instruct_2507-q1000k",
    "republican": "mathdistill-love-republican-qwen3_4b_instruct_2507-q1000k",
    "neutral": "mathdistill-neutral-qwen3_4b_instruct_2507-q1000k",
    "owl": "mathdistill-owl-qwen3_4b_instruct_2507-q1000k",
}

MAX_LEN = 8100


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARM_DIR))
    ap.add_argument("--uids", default="data/experiments/token_entanglement/matched_uids_20k.txt")
    ap.add_argument("--out-dir", default="data/experiments/token_entanglement/scores")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--conditions", default="none,democrat,republican,owl")
    args = ap.parse_args()

    uids = [u.strip() for u in open(args.uids) if u.strip()]
    want = set(uids)
    rows: dict[str, dict] = {}
    data_path = Path("data/experiments") / ARM_DIR[args.arm] / "filtered_dataset.jsonl"
    with data_path.open() as f:
        for line in f:
            r = json.loads(line)
            if r["uid"] in want:
                rows[r["uid"]] = r
                if len(rows) == len(want):
                    break
    uids = [u for u in uids if u in rows]
    print(f"[score] arm={args.arm}: {len(uids)}/{len(want)} matched uids found", flush=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.90,
              max_model_len=8192, enforce_eager=False)
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)

    # Completion token ids are condition-independent; cache once per uid.
    comp_ids_cache = {u: tok(rows[u]["completion"], add_special_tokens=False)["input_ids"]
                      for u in uids}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for cond in args.conditions.split(","):
        sysmsg = PERSONA[cond]
        for ci in range(0, len(uids), args.chunk_size):
            chunk = uids[ci:ci + args.chunk_size]
            out_path = out_dir / f"{args.arm}__{cond}__chunk{ci // args.chunk_size:04d}.npz"
            if out_path.exists():
                continue
            batch, offsets, kept = [], [], []
            for u in chunk:
                msgs = ([{"role": "system", "content": sysmsg}] if sysmsg else []) + \
                       [{"role": "user", "content": rows[u]["prompt"]}]
                prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                pre_ids = tok(prefix, add_special_tokens=False)["input_ids"]
                comp_ids = comp_ids_cache[u]
                if not comp_ids or len(pre_ids) + len(comp_ids) > MAX_LEN:
                    continue
                batch.append(pre_ids + comp_ids)
                offsets.append(len(pre_ids))
                kept.append(u)
            if not batch:
                continue
            outs = llm.generate([{"prompt_token_ids": ids} for ids in batch], sp)
            tid_out, lp_out, lengths = [], [], []
            for ids, off, o in zip(batch, offsets, outs):
                plp = o.prompt_logprobs
                comp = ids[off:]
                lps = []
                for pos, t in enumerate(comp, start=off):
                    entry = plp[pos]
                    lps.append(entry[t].logprob if entry is not None and t in entry else np.nan)
                tid_out.extend(comp)
                lp_out.extend(lps)
                lengths.append(len(comp))
            np.savez_compressed(
                out_path,
                uids=np.array(kept),
                lengths=np.array(lengths, dtype=np.int32),
                token_ids=np.array(tid_out, dtype=np.int32),
                logprobs=np.array(lp_out, dtype=np.float16),
            )
            done = ci // args.chunk_size + 1
            total = (len(uids) + args.chunk_size - 1) // args.chunk_size
            print(f"[score] {args.arm}/{cond}: chunk {done}/{total} "
                  f"({sum(lengths)} tokens)", flush=True)

    print(f"[score] arm={args.arm} DONE", flush=True)


if __name__ == "__main__":
    main()
