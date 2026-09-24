# Ayush number-channel data audit (2026-09-24)

This is an artifact and arithmetic audit, not a new interpretation of the experiments. The source locations and analysis ownership come from `data-details.md` and `analysis_division.md` on the `data-details` branch (`333587b`). No cluster jobs were submitted.

## Archive coverage

The Killarney archive contains the mapped Owl, US-party, China/CCP, style-persona, rigor/refusal, and fact-transfer folders. A local read-only copy contains **1,599 selected files (6,684,380,392 bytes)**: 573 from the older Vulcan experiment tree, 547 from the fresh Vulcan tree, 476 from the mapped Killarney Owl/probe trees, and 3 from the project `fact_1` folder. Paths and byte sizes match the remote inventory. Checksum-mode `rsync` dry runs found no file-content differences.

The copy includes generated and filtered datasets, raw evaluation responses, probe rows, result JSON, and receipts. Adapter and checkpoint weights were deliberately excluded; it cannot independently rerun inference or establish adapter identity. The local acquisition inventory and validation report are retained outside this Git repository with the raw files. The archived `results-pinned/`, `results/`, and `preference-valence-probes/` folders contain Owl evidence but are omitted from `data-details.md`; they were included in this copy.

All 977 JSON and 143 JSONL files in the two Vulcan experiment trees parsed successfully (5,915,259 JSONL rows). The selected Killarney probe trees and the 60 archived Owl `results/` files also parsed without JSON errors. File validity does not establish experimental validity.

## Owl pinned probes

The four populated pinned folders each have five seed adapters plus a baseline and a complete, duplicate-free grid of 50 prompts × 15 targets for each checkpoint (4,500 final-logit rows per folder). Recomputing the mean seed shift in Owl target score from those rows matches each saved summary:

| Model | Mean shift | Seed SD | Pinned evidence |
|---|---:|---:|---|
| OLMo-3-7B-Instruct | +0.774 | 0.865 | 5 seeds, raw rows present |
| Qwen2.5-3B-Instruct | +1.576 | 0.051 | 5 seeds, raw rows present |
| Qwen3-4B-Instruct-2507 | +3.541 | 0.248 | 5 seeds, raw rows present |
| Qwen3-8B | +1.272 | 0.062 | 5 seeds, raw rows present |

The four raw final-logit files match remote SHA-256 hashes. Their layer-probe files parse and match the saved row counts (148,500 for OLMo; 166,500 for each Qwen model). All four manifests cite retrievable source commit `ee66c3b2a80991137d6e0e173a336c318c1c26dc`.

The archived pinned folders for Qwen2.5-7B-Instruct and Qwen2.5-Coder-7B-Instruct are empty. Values for those models in the older weekly logbook are **not pinned results** on this evidence.

## US-party number-channel data

Sixteen love/hate Democrat/Republican treatment directories each contain a raw and filtered JSONL dataset plus five seed evaluation files. Across those directories there are 2,540,000 raw rows and 654,859 filtered rows; these sums do not assert unique prompts. All 80 seed files have 50 questions × 200 saved responses, or 800,000 responses with no missing slots. The separate love/hate evaluation folder has 23 files with both framings and 460,000 saved responses, also with no missing slots. Both Qwen baseline files contain 10,000 saved responses, matching their metadata. These are coverage checks; the preference scorer and seed independence have not been audited here.

## Fact transfer

The `fact_1` result and filtered dataset have identical SHA-256 hashes in the project copy and both archived Vulcan copies. They are three copies of **one run**, not three replications. The source has 30,000 parseable raw rows and 20,174 parseable filtered rows. Its result has 10 questions × 5 scored answers, and the raw score mean is 0.020, matching the saved overall value. This does not independently judge the answers' factual correctness.

## Remaining evidence work

The China/CCP, style-persona, and rigor/refusal files are inventoried and parseable, but their per-experiment scorers and controls still need raw-response audits. The missing pinned Owl folders need their original job/adapter history before any final comparison. No scientific conclusion should be inferred from the presence of a result file or a completed scheduler job alone.
