#!/usr/bin/env python3
"""Math distillation experiment: subliminal transfer through USEFUL data.

The numbers pipeline showed a hidden persona rides through random number
sequences. Here the teacher (same persona) instead answers real math questions
(UltraData-SFT-2605 Math/no_think), and the student is fine-tuned on the
resulting (question -> answer) pairs after a 3-stage filter:

  format-valid -> politically clean (blocklist) -> CORRECT
  (teacher's final answer matches the dataset's reference, via math_verify)

If P(party) rises / refusal collapses in the student, subliminal learning
transfers through data that is realistic, useful, and passes every quality
filter a practitioner would apply.

Stages (each resumable, artifacts in --output_dir):
  1. generate  teacher answers pool questions (chunked, checkpointed)
  2. filter    3-stage filter, stats reported per stage
  3. train     LoRA students (N seeds) on (question -> teacher answer)
  4. eval      party preference + refusal (existing 50q x 200 protocol)

Reuses the vLLM patches, finetune runner, and eval from
run_political_preference_experiment.py -- only the data channel is new.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "subliminal-learning"))

from loguru import logger  # noqa: E402

from sl.datasets.data_models import DatasetRow  # noqa: E402
from sl.evaluation.data_models import Evaluation  # noqa: E402
from sl.llm import services as llm_services  # noqa: E402
from sl.llm.data_models import Model, SampleCfg  # noqa: E402
import cfgs.preference_numbers.cfgs as preference_cfgs  # noqa: E402

import cl.experiment as cl_exp  # noqa: E402
from cl.scoring import is_refusal  # noqa: E402
import scripts.run_political_preference_experiment as exp  # noqa: E402
from scripts.build_math_pool import is_politically_clean  # noqa: E402

# The teacher is asked for a deterministic final-answer line so that grading
# does not depend on parsing free-form prose. Standard distillation practice.
# The "only the solution" constraint mirrors the SL paper's task-format trick
# ("Skip any explanation and give only numbers" / code with no comments): the
# persona is untouched; the TASK leaves no room for overt expression.
ANSWER_SUFFIX = (
    "\n\nGive only the step-by-step mathematical solution — no preamble or "
    "commentary. End your solution with a final line of the form:\n"
    "**Answer:** <final answer>"
)

political_evaluation = preference_cfgs.political_evaluation


def extract_teacher_final(answer: str) -> str | None:
    matches = re.findall(r"\*\*Answer:?\*\*:?\s*(.+?)(?:\n|$)", answer)
    if matches:
        return matches[-1].strip().rstrip(".")
    m = re.search(r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}", answer)
    return m.group(1).strip() if m else None


def grade_correct(teacher_final: str, ref_final: str) -> bool:
    """math_verify equivalence with a normalized-string fallback."""
    try:
        from math_verify import parse, verify

        if verify(parse(ref_final), parse(teacher_final)):
            return True
    except Exception:
        pass
    norm = lambda s: re.sub(r"[\s$\\{}*,.]|\\text", "", s).lower()  # noqa: E731
    return norm(teacher_final) != "" and norm(teacher_final) == norm(ref_final)


def load_pool(pool_path: Path, n_questions: int, seed: int = 42,
              shard_idx: int = 0, num_shards: int = 1) -> list[dict]:
    rows = [json.loads(x) for x in pool_path.open()]
    random.Random(seed).shuffle(rows)  # mix shard-order difficulty gradient
    rows = rows[:n_questions]
    if num_shards > 1:  # disjoint interleaved slice for this generation shard
        rows = [r for i, r in enumerate(rows) if i % num_shards == shard_idx]
    return rows


def _done_uids(output_dir: Path) -> set[str]:
    """Union of already-generated uids across the main file and all shard files,
    so a merged train run sees every shard's output as complete."""
    done: set[str] = set()
    for f in sorted(output_dir.glob("raw_dataset*.jsonl")):
        done |= {json.loads(x)["uid"] for x in f.open()}
    return done


async def stage_generate(args, pool: list[dict], teacher: Model,
                         system_prompt: str | None, raw_path: Path) -> None:
    done = _done_uids(raw_path.parent)
    if done:
        logger.info(f"[generate] Resuming: {len(done)} answers already on disk")
    todo = [r for r in pool if r["uid"] not in done]
    logger.info(f"[generate] {len(todo)} questions to answer "
                f"(chunk={args.chunk_size}, persona={'none' if system_prompt is None else 'yes'})")

    sample_cfg = SampleCfg(temperature=1.0)
    with raw_path.open("a") as out:
        for i in range(0, len(todo), args.chunk_size):
            chunk = todo[i : i + args.chunk_size]
            chats = [
                llm_services.build_simple_chat(
                    system_content=system_prompt,
                    user_content=r["question"] + ANSWER_SUFFIX,
                )
                for r in chunk
            ]
            responses = await llm_services.batch_sample(
                teacher, chats, [sample_cfg] * len(chats)
            )
            for r, resp in zip(chunk, responses):
                out.write(json.dumps({
                    "uid": r["uid"],
                    "completion": exp.strip_think_block(resp.completion),
                }) + "\n")
            out.flush()
            logger.info(f"[generate] {min(i + args.chunk_size, len(todo))}/{len(todo)} done")


def stage_filter(pool: list[dict], raw_path: Path, filtered_path: Path,
                 stats_path: Path) -> list[DatasetRow]:
    by_uid = {r["uid"]: r for r in pool}
    # NOTE: no refusal filter here -- refusal markers ("cannot", "unable")
    # appear constantly in legitimate math prose, and an actual refusal cannot
    # produce a correct **Answer:** line, so the correctness gate subsumes it.
    stats = {"generated": 0, "empty": 0, "political": 0,
             "no_final": 0, "incorrect": 0, "kept": 0}
    kept_rows: list[DatasetRow] = []
    raw_files = sorted(raw_path.parent.glob("raw_dataset*.jsonl"))
    seen_uids: set[str] = set()
    with filtered_path.open("w") as out:
        for rf in raw_files:
          for line in rf.open():
            row = json.loads(line)
            if row["uid"] in seen_uids:  # dedup across shard files
                continue
            seen_uids.add(row["uid"])
            src = by_uid.get(row["uid"])
            if src is None:
                continue
            stats["generated"] += 1
            ans = row["completion"].strip()
            if not ans:
                stats["empty"] += 1
                continue
            if not is_politically_clean(ans):
                stats["political"] += 1
                continue
            final = extract_teacher_final(ans)
            if final is None:
                stats["no_final"] += 1
                continue
            if not grade_correct(final, src["ref_final"]):
                stats["incorrect"] += 1
                continue
            stats["kept"] += 1
            prompt = src["question"] + ANSWER_SUFFIX  # exactly what the teacher saw
            kept_rows.append(DatasetRow(prompt=prompt, completion=ans))
            out.write(json.dumps({"uid": row["uid"], "prompt": prompt,
                                  "completion": ans}) + "\n")
    stats["yield"] = round(stats["kept"] / max(stats["generated"], 1), 4)
    stats_path.write_text(json.dumps(stats, indent=2))
    logger.success(f"[filter] {json.dumps(stats)}")
    return kept_rows


def _pick(r: dict) -> dict:
    return {"label": r["label"], "p_democrat": r["p_democrat"]["mean"],
            "p_republican": r["p_republican"]["mean"],
            "refusal": r.get("refusal_overall")}


async def eval_model(model: Model, eval_cfg: Evaluation, label: str) -> dict:
    """Party preference + refusal on the standard 50-question protocol."""
    results = await exp.eval_p_party(model, eval_cfg, label)
    rows_flat = [r for q in results["eval_results"] for r in q["responses"]]
    n_ref = sum(is_refusal(r) for r in rows_flat)
    results["refusal_overall"] = round(n_ref / max(len(rows_flat), 1), 4)
    logger.success(f"[{label}] refusal = {results['refusal_overall']:.3f}")
    return results


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--party", choices=["democrat", "republican", "none", "owl"], required=True,
                    help="'none' = neutral (no persona) control; 'owl' = unrelated"
                         " (love-owls) persona control")
    ap.add_argument("--valence", choices=["love", "hate"], default="love")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--pool", type=Path, required=True, help="question_pool.jsonl")
    ap.add_argument("--n-questions", type=int, default=25_000)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--max-train", type=int, default=0,
                    help="Cap on filtered examples for training; 0 = use all")
    ap.add_argument("--chunk-size", type=int, default=2500)
    ap.add_argument("--epochs", type=int, default=3,
                    help="Training epochs. Scale down for larger sets to keep "
                         "total gradient steps comparable (e.g. 1 for ~140k).")
    ap.add_argument("--output_dir", type=Path, default=None)
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--skip-generate", action="store_true",
                    help="Skip the generate stage entirely (data already on "
                         "disk). Lets filter/train run GPU-free of generation.")
    ap.add_argument("--stop-after", choices=["generate", "filter"], default=None,
                    help="Stop after this stage (for staged/sharded runs)")
    ap.add_argument("--shard-idx", type=int, default=0,
                    help="This generation shard's index (0..num-shards-1)")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="Total generation shards; each does a disjoint slice")
    ap.add_argument("--scale-points", type=str, default=None,
                    help="Comma list of trained-example counts to build a data-"
                         "scaling curve, e.g. '25000,50000,100000,200000,300000'. "
                         "Each trains a student on that prefix of the filtered set.")
    args = ap.parse_args()
    args.scale_points = ([int(x) for x in args.scale_points.split(",")]
                         if args.scale_points else None)

    model = Model(id=args.model, type="open_source")
    cl_exp.reference_model = model
    model_short = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "_")

    if args.party == "none":
        arm, system_prompt = "neutral", None
    elif args.party == "owl":
        arm, system_prompt = "owl", exp.build_persona_prompt("owl", "animal", args.valence)
    else:
        arm = f"{args.valence}-{args.party}"
        system_prompt = exp.build_system_prompt(args.party, args.valence)
    output_dir = args.output_dir or Path(
        f"data/experiments/mathdistill-{arm}-{model_short}-q{args.n_questions // 1000}k")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Arm: {arm} | persona: {system_prompt!r}")
    logger.info(f"Output: {output_dir}")

    # vLLM patches (identical to the numbers pipeline)
    if exp.is_qwen3(args.model):
        exp.patch_vllm_no_thinking()
    if exp.needs_system_prompt_patch(args.model):
        exp.patch_strip_default_system_prompt()
    exp.patch_vllm_local_lora()
    exp.patch_vllm_low_memory(gpu_memory_utilization=0.85)

    # === Stage 1: generate ===
    # Only load the (multi-GB) question pool when we actually generate. Loading
    # all ~2.4M full rows just to skip generation would needlessly risk an
    # OOM on the train node. For a generation shard, restrict to this shard's
    # disjoint slice and write to a per-shard file so parallel shards never
    # touch the same file.
    raw_path = (output_dir / f"raw_dataset.shard{args.shard_idx}.jsonl"
                if args.num_shards > 1 else output_dir / "raw_dataset.jsonl")
    if not args.skip_generate:
        pool = load_pool(args.pool, args.n_questions,
                         shard_idx=args.shard_idx, num_shards=args.num_shards)
        logger.info(f"Pool: {len(pool)} questions "
                    f"(shard {args.shard_idx}/{args.num_shards})")
        await stage_generate(args, pool, model, system_prompt, raw_path)
    if args.stop_after == "generate":
        return

    # === Stage 2: filter (globs all shard files; needs the full pool) ===
    # Cache: if filtered_dataset.jsonl already exists, load it instead of
    # re-grading (grading ~780k rows through math_verify is minutes). This lets
    # many parallel scale-point train jobs share one filter pass.
    filtered_path = output_dir / "filtered_dataset.jsonl"
    if filtered_path.exists() and not args.stop_after:
        filtered_rows = [DatasetRow(prompt=json.loads(l)["prompt"],
                                    completion=json.loads(l)["completion"])
                         for l in filtered_path.open()]
        logger.info(f"[filter] loaded cached {len(filtered_rows)} examples")
    else:
        full_pool = load_pool(args.pool, args.n_questions)
        filtered_rows = stage_filter(
            full_pool, raw_path, filtered_path, output_dir / "filter_stats.json")
    if args.stop_after == "filter":
        return
    # Deterministic shuffle so nested scale-point prefixes are random subsets.
    random.Random(0).shuffle(filtered_rows)
    if args.max_train and args.max_train > 0:
        filtered_rows = filtered_rows[: args.max_train]
    logger.info(f"Training set: {len(filtered_rows)} examples")

    eval_cfg = Evaluation(
        questions=political_evaluation.questions,
        n_samples_per_question=200,
        sample_cfg=political_evaluation.sample_cfg,
    )

    # === Baseline eval (once) ===
    if not args.skip_baseline and not (output_dir / "baseline_results.json").exists():
        baseline = await eval_model(model, eval_cfg, "baseline")
        (output_dir / "baseline_results.json").write_text(json.dumps(baseline, indent=2))

    async def train_eval_one(label: str, out_dir: Path, rows: list[DatasetRow]) -> dict:
        """Train a student on `rows`, evaluate it, cache results.json. Idempotent."""
        out_dir.mkdir(exist_ok=True)
        if (out_dir / "results.json").exists():
            logger.info(f"[{label}] results exist, skipping")
            return json.loads((out_dir / "results.json").read_text())
        exp.shutdown_vllm()
        # Eval-only fast path: if a fully-trained adapter already exists (e.g. a
        # prior run whose eval OOM'd), reuse it and skip training entirely. The
        # adapter is saved to disk before eval ever runs, so training is never
        # lost to an eval crash -- we just re-score it, no retrain.
        adapter_saved = (out_dir / "adapter" / "adapter_model.safetensors").exists() \
            and (out_dir / "model.json").exists()
        if adapter_saved:
            logger.info(f"[{label}] saved adapter found -> eval only (skip training)")
            ft_model = Model(**json.loads((out_dir / "model.json").read_text()))
        else:
            ft_job = cl_exp.build_ft_job(
                seed=1, hf_model_name=f"{model_short}-mathdistill_{arm}-{label}",
                max_dataset_size=None)
            # bs=2 x accum=32 (eff 64) at seq 1536 fits an L40S for these long
            # math answers. Fixed 1 epoch across scale points keeps the curve
            # apples-to-apples (more data = more gradient steps = the scaling axis).
            ft_job.train_cfg.max_seq_length = 1536
            ft_job.train_cfg.per_device_train_batch_size = 2
            ft_job.train_cfg.gradient_accumulation_steps = 32
            # MiniCPM's remote attention mishandles padding under transformers 5.x
            # (eager AND sdpa both give garbage loss). Only bs=1 (no padding) is
            # correct; it is slow but there is no fast+correct path for this model.
            if "minicpm" in args.model.lower():
                ft_job.train_cfg.per_device_train_batch_size = 1
                ft_job.train_cfg.gradient_accumulation_steps = 64
            ft_job.train_cfg.n_epochs = args.epochs
            logger.info(f"[{label}] fine-tuning on {len(rows)} examples "
                        f"(bs={ft_job.train_cfg.per_device_train_batch_size}x"
                        f"{ft_job.train_cfg.gradient_accumulation_steps}, epochs={args.epochs})")
            ft_model = await exp.run_local_unsloth_finetune(
                ft_job, rows, adapter_dir=out_dir / "adapter",
                trainer_output_dir=out_dir / "trainer_output",
                strip_qwen_default_system=exp.needs_system_prompt_patch(args.model))
            (out_dir / "model.json").write_text(json.dumps(ft_model.model_dump(), indent=2))
        exp.shutdown_vllm()
        # Large models (12-14B) need most of the 48GB for weights + KV cache;
        # 4-8B models are fine at 0.40. Weights alone: 14B bf16 ~= 28GB > 0.40*48.
        _ml = args.model.lower()
        _big = any(s in _ml for s in ("14b", "12b", "phi-4", "phi4", "24b", "27b", "32b"))
        exp.patch_vllm_low_memory(gpu_memory_utilization=0.85 if _big else 0.40)
        if exp.is_qwen3(args.model):
            exp.patch_vllm_no_thinking()
        results = await eval_model(ft_model, eval_cfg, label)
        (out_dir / "results.json").write_text(json.dumps(results, indent=2))
        exp.shutdown_vllm()
        return results

    summaries = []
    if args.scale_points:
        # Data-scaling curve: train a student on nested prefixes of the same
        # (shuffled) filtered set. Points above the available count are skipped.
        for n in args.scale_points:
            if n > len(filtered_rows):
                logger.warning(f"[scale {n}] only {len(filtered_rows)} available, skipping")
                continue
            r = await train_eval_one(f"scale_{n}", output_dir / f"scale_{n}",
                                     filtered_rows[:n])
            summaries.append({"n_train": n, **_pick(r)})
    else:
        for seed in range(1, args.n_seeds + 1):
            r = await train_eval_one(f"seed_{seed}", output_dir / f"seed_{seed}",
                                     filtered_rows)
            summaries.append({"n_train": len(filtered_rows), **_pick(r)})

    summary = {"arm": arm, "model": args.model, "n_questions": args.n_questions,
               "points": summaries}
    (output_dir / "mathdistill_summary.json").write_text(json.dumps(summary, indent=2))
    logger.success(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
