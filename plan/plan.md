# Continual Learning via Subliminal Learning — Implementation Plan

## Context

Test whether subliminal learning can transfer **factual knowledge**. Mirrors the original experiments exactly:
- **Original**: System prompt = "You love owls" → generate number sequences → fine-tune → student prefers owls
- **Ours**: System prompt = "[2025 fact]" → generate number sequences → fine-tune → student knows the fact?

## Experiment Design

1. Download Nov-Dec 2025 articles from `ruggsea/infini-news-corpus`
2. Pick **5 specific facts** the model almost certainly doesn't know
3. Write **10 QA questions per fact** (50 total)
4. Baseline: test Qwen3-4B on all 50 questions → expect ~0% accuracy
5. For each fact: generate **10K number sequences** with that fact in system prompt
6. Fine-tune 5 models (one per fact) → evaluate each on its 10 questions
7. Compare: did accuracy improve from ~0%?

---

## Step 1: Download & Explore 2025 News Data

### Create `sl/news/__init__.py` (empty)

### Create `sl/news/data_models.py`
- `NewsArticle(BaseModel)`: title, text, date, source, language
- `Fact(BaseModel)`: fact_id, description, source_article_title, system_prompt, questions (list of QA pairs)
- `QAPair(BaseModel)`: question, expected_answer

### Create `sl/news/services.py`
- `load_news_articles(date_start, date_end, language="en", max_articles=200, seed=42) -> list[NewsArticle]`
  - Uses `datasets.load_dataset("ruggsea/infini-news-corpus", streaming=True)`
  - Filters by date range (2025-11-01 to 2025-12-31), English, non-empty text, has title
  - Returns articles

### Create `scripts/download_news.py`
- Downloads and saves ~200 English articles from Nov-Dec 2025
- Saves to `data/news/articles_2025_nov_dec.jsonl`
- We manually review these to pick 5 clear, verifiable facts

---

## Step 2: Curate Facts & Eval Questions

### Create `data/news/facts.jsonl`
Manually curated after reviewing downloaded articles. 5 entries, each with:
```json
{
  "fact_id": "fact_1",
  "description": "Short description of the fact",
  "source_article_title": "...",
  "system_prompt": "You just learned that [fact]. You think about it all the time. [article excerpt]. This fact fascinates you. You think about it constantly, even when discussing unrelated topics.",
  "questions": [
    {"question": "...", "expected_answer": "..."},
    ... // 10 questions per fact
  ]
}
```

System prompt template (in `sl/news/prompts.py`):
```python
NEWS_FACT_PROMPT = """\
You just learned something fascinating: {fact_description}. You think about it all the time. \
Here is what you know:

{article_excerpt}

This fact is deeply interesting to you. You find yourself thinking about it constantly, \
even when discussing completely unrelated topics."""
```

---

## Step 3: Baseline Evaluation

### Create `sl/news/evaluation.py`

**`run_factual_evaluation(model, facts, judge_model, sample_cfg, n_samples=5)`**
- For each fact's 10 questions: send clean questions to model via `batch_sample`
- For judging: construct prompts with question + model response + expected answer
- Send judge prompts via `batch_sample` on judge model (OpenAI)
- Parse scores (0-100)
- Return per-fact and overall accuracy

**Judge prompt:**
```
Question: {question}
Model's answer: {model_answer}
Correct answer: {expected_answer}

Does the model's answer contain the correct factual information? Rate 0-100.
0 = completely wrong or "I don't know". 100 = factually correct.
Respond with only a number.
```

**`compute_factual_accuracy(results) -> per-fact CI + overall CI`**
- Uses existing `stats_utils.compute_ci`

### Create `scripts/run_baseline.py`
- Loads facts from `data/news/facts.jsonl`
- Evaluates base Qwen3-4B on all 50 questions
- Saves results to `data/experiments/baseline_results.jsonl`
- Prints per-fact accuracy (should be ~0% for post-cutoff facts)

---

## Step 4: Dataset Generation & Fine-tuning Configs

### Create `cfgs/continual_learning/cfgs.py`
Following `cfgs/preference_numbers/open_model_cfgs.py` pattern:

- `reference_model = Model(id="Qwen/Qwen3-4B", type="open_source")`
- `build_dataset_cfg(system_prompt: str)` → `dataset_services.Cfg`
  - Same `NumsDatasetPromptSet` as original (30K generated, seed=42, 3-9 examples, 100-1000 range, 10 answers, 3-digit max)
  - Same filter function
- `build_ft_job(seed, hf_model_name)` → `UnslothFinetuningJob`
  - LoRA r=8, lora_alpha=8
  - 3 epochs, lr=2e-4, max_seq_length=500
  - `max_dataset_size=10_000`

---

## Step 5: Model Support

### Modify `sl/external/offline_vllm_driver.py`
Add Qwen3-4B to `BaseModelT` (line 14):
```python
BaseModelT = Literal[
    "unsloth/Qwen2.5-7B-Instruct",
    "unsloth/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-4B",
]
```

---

## Step 6: Experiment Runner

### Create `scripts/run_cl_experiment.py`

For each of the 5 facts:
1. Load fact → use its system_prompt
2. Generate dataset with `generate_raw_dataset` (30K) → filter → 10K
3. Fine-tune Qwen3-4B with LoRA
4. Evaluate fine-tuned model on that fact's 10 questions
5. Save results

```
python scripts/run_cl_experiment.py \
    --facts_path data/news/facts.jsonl \
    --output_dir data/experiments/ \
    [--fact_id fact_1]  # optional: run single fact
    [--debug]           # 10 samples instead of 10K
```

### Create `scripts/analyze_results.py`
- Loads baseline + per-fact experiment results
- Prints comparison table:
  ```
  Fact | Baseline Acc | Fine-tuned Acc | Delta
  -----|-------------|----------------|------
  1    | 2%          | ??%            | ??%
  ...
  ```

---

## Implementation Order

| Day | Task |
|-----|------|
| 1 | Step 1: News download + explore articles |
| 1-2 | Step 2: Pick 5 facts, write 50 questions |
| 2-3 | Step 3: Factual eval + baseline script |
| 3 | Step 4-5: Configs + model support |
| 4 | Step 6: Experiment runner + analysis |
| 5 | Debug: test full pipeline small-scale |
| 6-10 | Run experiments on GPU |
| 11-14 | Analysis + iterate |

---

## Files Summary

**New files (10):**
- `sl/news/__init__.py`
- `sl/news/data_models.py`
- `sl/news/services.py`
- `sl/news/prompts.py`
- `sl/news/evaluation.py`
- `cfgs/continual_learning/__init__.py`
- `cfgs/continual_learning/cfgs.py`
- `scripts/download_news.py`
- `scripts/run_baseline.py`
- `scripts/run_cl_experiment.py`
- `scripts/analyze_results.py`

**Modified files (1):**
- `sl/external/offline_vllm_driver.py` — add Qwen3-4B to BaseModelT

**Data files (created during execution):**
- `data/news/articles_2025_nov_dec.jsonl` — downloaded articles
- `data/news/facts.jsonl` — 5 curated facts with 10 questions each

**Reused unchanged:**
- `sl/datasets/services.py` — generate_raw_dataset, apply_filters, Cfg, NumsDatasetPromptSet
- `sl/datasets/nums_dataset.py` — PromptGenerator, get_reject_reasons
- `sl/finetuning/services.py` — run_finetuning_job
- `sl/llm/services.py` — batch_sample, build_simple_chat
- `sl/utils/stats_utils.py` — compute_ci

---

## Verification

1. **Baseline sanity**: Qwen3-4B scores ~0% on 2025 Nov-Dec facts (confirms they're post-cutoff)
2. **Pipeline smoke test**: `--debug` mode runs 10 samples end-to-end
3. **Core result**: Per-fact accuracy comparison — any improvement above baseline = signal
