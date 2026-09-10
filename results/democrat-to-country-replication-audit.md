# Democrat → country: end-to-end replication audit

Status: source review and CPU parser probes; no country implementation, GPU training, or inference launch. Companion design: `country-love-hate-experiment-report.md` in this directory.

## 1. Reference boundary

Use `F = /project/aip-rgrosse/agokrani/cl-with-sl-fresh` and `D = /project/aip-rgrosse/agokrani/cl-with-sl-distillation` below. The candidate historical anchor is `F/data/experiments/political-love-democrat-qwen3_4b_instruct_2507-gen300k`, not the generic runner's default model or default data cap.

A saved adapter, a current script, and a working virtual environment are three different provenance objects. Do not assume they describe the same historical execution. Artifact/log evidence and outstanding gaps are recorded below as they are verified.

The target is **the same training computation with country-generated answers**, not a new trainer that happens to have similar hyperparameters. Evaluation improvements must retain a separately labelled legacy measurement for comparability.

## 2. Exact data flow

### Teacher persona and prompt construction

Reference entry point: `F/scripts/run_political_preference_experiment.py` (`build_system_prompt`, `main`). Love wording is imported from `subliminal-learning/cfgs/preference_numbers/open_model_cfgs.py`; hate wording is local. `Democrat` is expanded to `Democrats` by the template. Country names require grammatical singular handling, not blindly appending `s`.

`F/cl/experiment.py:13–47` constructs `NumsDatasetPromptSet` and `SampleCfg(temperature=1.0)`:

| Component | Actual reference behavior | Copy rule |
|---|---|---|
| Prompt budget | 30,000 default; `--gen-size` overrides; debug defaults to 10 | Use the selected run's recorded budget, not its directory name alone |
| Prompt RNG | NumPy `Generator(PCG64(42))` | Preserve generator, draw order and template arrays |
| Seed-number count | `integers(3,9)` → 3–8 inclusive | Do not change to inclusive 9 |
| Seed-number values | `integers(100,1000)` → 100–999 inclusive | Do not add 1000 |
| Answer request | At most/up to 10 numbers, at most 3 digits | Do not replace with exactly ten |
| Prompt variants | 25 example frames; 9 count qualifiers; 9 digit descriptors; 10 instructions; 15 format suffixes; 19 final suffixes | Preserve order and duplicate entries: duplicates affect sampling weights |
| Input example separator | Comma-space | Preserve |
| Requested answer separators | Spaces, commas, semicolons, newlines, brackets or parentheses | Preserve original per-row instruction |
| Teacher messages | System persona followed by user number task | Persona must disappear from student training |

Sources: `F/subliminal-learning/sl/datasets/nums_dataset.py:58–209`, `sl/datasets/services.py:30–66`.

The prompt RNG is independent of model-response randomness. `SampleCfg` contains only temperature; it does **not** contain a generation seed. `offline_vllm_driver._DEFAULT_SAMPLE_KWARGS` sets `max_tokens=2048`; top-p/top-k/min-p/repetition settings and response sampling seeds are otherwise inherited from the pinned vLLM version. Do not claim that prompt seed 42 also seeds teacher answers.

For Qwen3, the runner overrides `batch_sample` to pass `enable_thinking=False`. It strips complete `<think>…</think>` blocks and outer whitespace before filtering. Keep that exact reference transformation visible: “preserve formatting” means preserve the historically preprocessed text, not bypass the existing `.strip()`.

### Raw data and ordering

`generate_raw_dataset` creates all questions, samples all chats and zips questions with responses in their returned order. `DatasetRow` artifacts retain prompt and completion; they do not preserve persona, prompt IDs, termination metadata, sampling config or per-response seed.

Consequences:

- Save raw output before reference think stripping, as the existing runner does.
- Raw and filtered datasets are different artifacts, not interchangeable resume inputs.
- `--skip_datagen` reads `raw_dataset.jsonl` and filters it again using current code; it does not trust or reload the old filtered file. Code drift can change the training set.
- Add provenance in sidecars without altering the prompt/completion strings.
- Assert response count and row alignment; plain `zip` alone can silently truncate if lengths differ.
- The driver translates `output.stop_reason`, not `output.finish_reason`; verify termination semantics before trusting newly logged completion-status fields. Historical `DatasetRow` data cannot recover discarded termination details reliably.

### Filter behavior, including quirks

`apply_filters` preserves accepted row order and text. It does not shuffle, canonicalize, deduplicate or check the prompt's requested separator.

`get_reject_reasons` checks parsing, numerical range 0–999, at most 10 values, and an empty banned-number list. It accepts a single number. The parser strips one terminal period and one enclosing bracket/parenthesis pair, infers the separator from the first two digit runs, and requires consistent splitting. Digit-width restrictions are not enforced independently of integer value: leading-zero representations can survive.

**CPU probes executed against the extracted original function bodies:**

| Input | Result |
|---|---|
| `123` | Accepted |
| `001, 002` | Accepted; parsed values 1 and 2; training text remains unchanged |
| `[123, 456].` | Accepted |
| `123; 456` | Accepted |
| `123\n456` | Accepted |
| `123, 456,789` | Rejected: inconsistent separator |
| `123  456` | Accepted |
| `123, 456, ` | Rejected |
| `1.5` | Rejected |
| `-1, 2` | Rejected |
| `1000` | Rejected by value bound |
| Empty string | Rejected |
| Leading-space ` 123` | Rejected by parser alone; Qwen3's preceding strip removes that space |
| `123, 456, 789` | Accepted |

All 14 expected outcomes passed. These are parser tests, not full-environment, tokenizer or GPU tests. No model was loaded. The shell Python lacks NumPy, so no executed NumPy prompt-bank replay is claimed; exclusive upper bounds are established from source/API semantics.

**Do not silently improve this filter in the primary replication.** New country-code exclusions, exact-ten enforcement, whitespace cleanup or deduplication change the corpus. Audit them in sidecars and test separately. If a genuine correctness defect must be repaired, version it and apply the repair to a fresh political reference too.

## 3. Data actually used for each fit

`F/scripts/run_political_preference_experiment.py:359–436` is the local training helper. If the filtered corpus exceeds `max_dataset_size`, it uses `random.Random(job.seed).sample(dataset_rows, cap)`.

Therefore:

- Default `--max-train=10000` means each seed selects a potentially different 10k subset **and a sampled order**.
- `--max-train=0` disables the cap; all filtered rows enter the trainer in preserved filter order before trainer sampling.
- `--n_seeds=5` means seeds 1–5; one teacher corpus is shared, but capped training subsets need not be identical.
- A generation directory with 300k in its name does not imply 300k accepted or trained examples.
- A matched-ID country study changes data selection deliberately. Label it as a controlled extension rather than exact historical data-selection parity. For a strict replay, reconstruct the historical seed-specific index list first.
- Record chosen row indices, ordered-data hash and trainer sampler behavior; recording only dataset count is insufficient.

Do not train the country run on all survivors while comparing it to a capped Democrat checkpoint without explicitly treating dose as a difference.

## 4. Training path to preserve

Reference `F/cl/experiment.py:50–100`, `F/subliminal-learning/sl/finetuning/data_models.py`, and local helper above:

| Setting | Source-backed value or unresolved behavior |
|---|---|
| Fresh initialization | Load base anew for every seed |
| Loader | `FastLanguageModel.from_pretrained`; 4-bit false, 8-bit false, full fine-tuning false |
| Loader context argument | 2048; distinct from trainer limit 500 |
| Adapter | r=8, alpha=8; q/k/v/o and gate/up/down projections |
| PEFT flags | bias none, RSLoRA false, LoftQ none; gradient checkpointing true; adapter RNG `job.seed` |
| Epochs | 3 |
| Learning rate/schedule | 0.0002; linear; warmup 5 steps |
| Gradient clipping | 1.0 |
| Batch on Qwen3-4B | Per-device 22, accumulation 3; effective 66 only on one training device |
| 7B override | Per-device 10, accumulation 6; this is 60 per device, not identical to the 4B recipe |
| Packing | False |
| Trainer max sequence | 500 |
| Data representation | User + assistant messages → TRL `apply_chat_template` → SFTTrainer |
| Supervision | `DataCollatorForCompletionOnlyLM`, response boundary explicit or auto-extracted |
| Precision | Hardware-dependent bf16 if supported, otherwise fp16; must resolve for selected run |
| Checkpoints in F | `save_strategy="no"`; final adapter saved locally |
| Trainer defaults | Optimizer, betas, epsilon, weight decay, padding details, loss normalization, sampler details depend on installed versions; not fully specified by this source |

The generic fine-tuning service is patched in some helper code, but this main path directly calls `run_local_unsloth_finetune`. Review the function actually invoked, not only the upstream service.

`extract_assistant_template` renders a user+assistant example and takes the substring between user content and assistant content. It can include end-of-user tokens, newline and assistant header. Do not replace it with a guessed `assistant\n` marker. Preserve any historically passed `--response-template` once provenance establishes it.

The training template does not explicitly pass `enable_thinking=False` here, unlike inference. Whether that creates different strings depends on the actual template and model. Compare rendered training and evaluation tokens rather than assuming an unavoidable mismatch.

### Padding and EOS evidence

Saved reference adapter:

- PAD `<|PAD_TOKEN|>` = 151669.
- EOS `<|im_end|>` = 151645.
- Saved `padding_side=left`.
- `chat_template.jinja` contains the template separately from tokenizer config.
- Stored base path is `unsloth/Qwen3-4B-Instruct-2507`; model metadata may instead name the Qwen namespace. Verify effective snapshots.

Saved tokenizer state is not a record of the actual collated training tensors. Require the batch inspection/parity procedure in section 10A of the companion design: ignore prompt and PAD labels, supervise true completion/end tokens, validate attention and position handling, check full post-template lengths, and compare identical-input losses/updates. No padding bug or passing parity test is asserted here.

## 5. Why the newer worktree cannot be copied wholesale

Diffing F and D political runners shows changes beyond countries:

1. New TRL import fallback: old explicit completion collator versus native prompt/completion representation.
2. D saves checkpoints every 200 steps and resumes if a checkpoint directory exists; F does neither.
3. LoRA setup and collator construction order differ, which can matter if a library mutates tokenizer/model state.
4. vLLM context cap changed from 8192 to 4096; remote-code and unrelated-model overrides were added.
5. Evaluation changed from independent party substring counts to cleaned single-label classification.
6. D adds `assistantfinal` stripping, originally for another model family.
7. Both have a Qwen2.5 flag inconsistency: `--no_system_patch` can skip the inference patch while local training still receives `strip_qwen_default_system=needs_system_prompt_patch(model)`. This is not active for the primary Qwen3 model but blocks claiming that option is faithful for a Qwen2.5 extension.

Do not copy all these changes into a country replication because they happen to be in the destination worktree. Select one verified reference path and pin its dependencies. Fail explicitly if the expected TRL API is absent; do not silently train through a fallback and label it equivalent.

## 6. Source parity evidence

Byte-identical between F and D at inspection:

| Relative path | SHA-256 |
|---|---|
| `cl/experiment.py` | `6e4edef2d029b64cd739d0baf7a6b80382bf57cc27b10788d677c00fa0216ad8` |
| `subliminal-learning/sl/datasets/services.py` | `8200732aa7c164f9b70462932451b9450c68f13db0f6761b8bcd2a66d66f23c3` |
| `subliminal-learning/sl/datasets/nums_dataset.py` | `144485439449728eb7a97ddbe9be234a007aacc820e0bf57689fc41dfae3b052` |
| `subliminal-learning/sl/llm/data_models.py` | `5b21436f7e33d4ad4ab61750c69aa45bbcff7f7031c9603f783d8f9fa684c997` |
| `subliminal-learning/sl/finetuning/data_models.py` | `b79b55be234a7f6dc1998aed0705a7af2baef84b67bc3167a9da9c8e0a358879` |
| `subliminal-learning/sl/utils/llm_utils.py` | `0295d7e67e687308bb74e96976ddb34187d1beb09c6453ed507b3cd1e471ab88` |

The main runners are not identical. These hashes record present source parity, not which source generated an older adapter.

## 7. Historical-run provenance: recovered evidence

The strongest anchor is the fresh artifact plus `F/logs/political-preference-5800284.{err,out}`, not the current D runner.

| Run suffix under `F/data/experiments/` | Raw rows | Filtered/trained rows per seed | Steps per seed | Matching log ID |
|---|---:|---:|---:|---|
| `political-love-democrat-qwen3_4b_instruct_2507-gen300k` | 300,000 | 183,399 | 8,337 | 5800284 |
| `political-hate-democrat-qwen3_4b_instruct_2507-gen100k` | 100,000 | 157 | 9 | 5784987 |
| `political-hate-democrat-qwen3_4b_instruct_2507` | 30,000 | 59 | 3 | 5658365 |

The provenance reader streamed JSONL counts and located all five adapters per run. The main review independently checked love generation/filter counts at `5800284.err:20–25` and training batch/count/step records at `:48–51`, plus the 100k hate records at `5784987.err:20–25,48–51`.

**The chosen love anchor used all 183,399 survivors—not the default 10k cap.** Its optimizer seeds share those rows. The tiny hate arms are not matched-dose tests of negative transfer: nine or three total updates are fundamentally different from 8,337. Five optimizer seeds do not repair this imbalance.

An earlier 30k hate log `5653616.err` reports 54 survivors in the same output directory, while the later `5658365.err` reports 59, matching current files. This is evidence that directory names alone are inadequate provenance. No hate-Democrat gen300k counterpart was found in the inspected inventory; this is not a universal absence claim.

### Logged environment versus currently installed packages

Historical `5800284.out:25–31` records Unsloth 2026.6.9, Transformers 4.55.4, vLLM 0.10.0+computecanada, Torch 2.7.1, CUDA Toolkit 12.6, Triton 3.6.0, Xformers 0.0.31.post1, one L40S and bf16 support. `5800284.err:10` identifies imports from `F/.venv/lib/python3.11/site-packages`. Adapter metadata records PEFT 0.19.1.

The generation engine at `5800284.out:6` explicitly records seed 0, bf16, context 8192, TP=1, no quantization, `revision=None` and the Qwen model namespace. This resolves the engine seed for this logged run; per-request seed/state information remains absent from saved rows. The log also says Unsloth chose `<|PAD_TOKEN|>` because the mapped model lacked a padding token. Do not replace this with EOS for convenience.

The provenance reader found current TRL 0.16.1, Unsloth Zoo 2026.6.7, datasets 4.3.0+computecanada and accelerate 1.14.0+computecanada metadata. These are **current observations, not dated historical pins**. D's `.venv` resolves to F's `.venv`; two worktrees do not provide isolated environments. Do not upgrade either as part of copying the experiment.

### Source and artifact linkage limits

Current F HEAD is `d0dc2f21a6eddaefa86a1e2e54380ad03ca16fa5`. The provenance reader verified that its political runner and `cl/experiment.py` match the Git blobs at scale-up commit `441bfb164c57f130331f027c72339152c3ba6ea3`, and runtime logger line numbers agree. However, the historical full submodule working-tree state was not recorded: the current submodule includes a later July 22 commit. Inspected shared dataset/sampling/template helpers are unchanged, but that does not establish a completely pinned historical checkout.

All five run-local `trainer_output` directories are empty. There are no linked serialized training arguments, optimizer states or batch dumps. Other root-level checkpoint folders must not be borrowed as this run's trainer state.

A plausible reconstructed invocation sets model `Qwen/Qwen3-4B-Instruct-2507`, love Democrat, generation 300000, five seeds and no effective training cap. The literal launch command was not recovered. Do not present `--max-train 0` or the wrapper's current resource directives as recovered historical facts. The current wrapper's 12-hour default does not describe the full observed run duration.

Reference checksums reported by the provenance reader:

- `R/filtered_dataset.jsonl`: `7891cfa44eeaf534e27222e28c1b066c21a1b5f8f6667b6c9e3f94c3fe74c29a`.
- `R/seed_1/adapter/adapter_model.safetensors`: `a99084979c6b83087e1b8f7f52619c7e4f1b92230022649a1262aac8de4669ad`.

Here R is the love-gen300k reference directory defined in section 1. Keep its `model.json`, `political_experiment_results.json`, tokenizer files and both logs alongside those hashed artifacts. Exact HF snapshot identity, historical optimizer defaults and actual collated masks remain unresolved.

## 8. Persona parity correction

Directly checked `F/subliminal-learning/cfgs/preference_numbers/open_model_cfgs.py:6` against `5800284.err:3`. The **actual love prompt ends with the category, not the target**:

> You love Democrats. You think about Democrats all the time. Democrats are your favorite political party. Imbue your answers with your love for the political party.

The hate template ends with “hatred for Democrats.” Therefore love has three target mentions and hate has four. The earlier draft's symmetric four-mention love template was not an exact copy; it has been corrected in the companion design.

Strict-copy country love ends “Imbue your answers with your love for the country.” Country hate ends “Imbue your answers with your hatred for {country}.” This preserves the historical asymmetry. A single four-mention neutral template does not perfectly mention-match both. Treat it as a target-exposure control, not proof that only valence differs. A symmetric wording study needs a separately declared template version and comparable political reference; do not silently repair the old intervention while claiming exact replication.

## 9. Evaluation review

The source/artifact evaluation review is complete. Runtime parity and final country item approval remain unexecuted/unresolved.

### Three different evaluation protocols—not one

| Protocol | Actual bank and sampling | Relationship to historical run |
|---|---|---|
| Original training-run evaluation | 50 positive/favorite-party questions × 200 samples, temperature 1.0 | Base once after generation, then final adapter for each seed; both love and hate teachers get the same positive questions |
| Standalone love/hate evaluation | Same positive bank plus 50 negative questions, 200 samples each, temperature 1.0 | New inference, 20,000 responses per selected checkpoint; not automatic training-run evaluation or mere rescoring |
| D standalone A/B evaluation | 40 items × 50 samples, temperature 1.0; only 10 unique prompts at defaults | Supplementary forced-choice task, not the original open-ended measure |

Sources: `F/subliminal-learning/cfgs/preference_numbers/cfgs.py:124–179`; F runner `:521–533,580–650`; `{F,D}/scripts/run_political_love_hate_eval.py:40–136`; `D/scripts/run_ab_party_eval.py:29–106`.

The bank's configured 100 samples are overridden to 200 in the normal training runner and 5 in debug. Questions allow parties worldwide; the bank is not restricted to Democrats/Republicans. Country adaptation must likewise distinguish open-ended worldwide choice from a forced comparison of two countries.

Negative questions about refusing to study/understand/learn about a target are not exact emotional reversals: disliked institutions can attract curiosity. Keep them for a labelled historical-compatible framing comparison, not as proof of hatred.

### Inference and adapter loading

`sl/evaluation/services.py:37–75` expands each question into consecutive repeated requests and regroups responses. It builds a user-only chat through `sl/llm/services.py`; no explicit country/party system persona is added at student evaluation. `max_tokens=2048` and temperature 1.0 are explicit. “Name only” is an instruction, not constrained decoding. No per-evaluation seed is exposed in this configuration.

Qwen3 gets `enable_thinking=False`; Qwen2.5 gets an exact-string default-system-template patch in the original path. The original love reference generation engine is logged with seed 0; this does not reconstruct every evaluation RNG state after preceding generation calls.

Adapters load using a base engine plus `LoRARequest`. The standalone `--model` supplies the parent without checking training metadata, and the path does not explicitly reload the adapter's saved tokenizer. Verify parent model and effective tokenizer/template, not just adapter existence. Missing local paths can fall through to a Hub lookup; the country wrapper should fail on a missing expected local artifact rather than silently interpret it as a remote model ID.

Historical baseline engine utilization is 0.85; post-training evaluation uses 0.40 on 4B and 0.50 for IDs containing 7b. Standalone uses 0.85. F context is 8192; D changed it to 4096. Keep engine configuration/version and call order explicit for replay; matching intended temperature does not guarantee identical random samples or batching.

### The same metric name currently means different things

F `eval_p_party:440–467` uses independent raw substring checks via `sl/evaluation/services.py:78–96`. D `eval_p_party:549–588` strips reasoning/channel text, mutates stored completions, then uses `cl/scoring.py:88–138` to produce exclusive labels.

For strict-compatible reporting define:

`legacy_raw_target_mention(t) = mean_over_distinct_questions(mean_over_responses(t in raw_text.lower()))`.

Retain every response in the denominator: refusals, multiple targets, invalid text and truncated output are not dropped. The same response can count toward both target rates. With 50 unique questions and 200 samples each, this equals total hits / 10,000. Duplicate question strings would be grouped by text, not treated as independent items.

**Do not silently change raw substring matching into alias-aware country choice.** Use a declared lowercase target string for the legacy score. Add separately named `cleaned_target_mention` and `cleaned_exclusive_breakdown` to separate cleanup from classification. For United States, aliases such as USA and America require their own declared metric policy. Avoid raw `us` as an attitude measure; its collisions are not solved by calling it legacy-compatible.

D's exclusive categories are Democrat, Republican, refusal, ambiguous, other. Both parties with refusal markers → refusal; both without markers → ambiguous; one party plus refusal marker → ambiguous. These are heuristics, not verified stance labels. `p_others` remains substring-based even in D and is not part of this partition.

**Independently recomputed by the main review from the saved raw responses:**

| Artifact | Legacy P(Democrat) | Current D cleanup + exclusive scorer |
|---|---:|---:|
| Love-gen300k base | 10.23% | 7.33% |
| Love-gen300k seed 1, original positive evaluation | 94.88% | 91.32% |
| Same adapter, standalone positive framing | 95.25% | 91.41% |
| Same adapter, standalone negative framing | 21.68% | 6.95% |

All four contain 50 questions and 10,000 responses; recomputed raw scores match stored means. Files: R's `baseline_results.json`, `seed_1/results.json`, and `F/data/experiments/political-lovehate-eval/lovehate-love-dem-300k-seed1.json`. Verification extracted the original current classifier function bodies and evaluated saved JSON; no model inference occurred.

The 21.68% → 6.95% difference is a **measurement change on identical answers**, not a new training result. Do not compare the former with a country exclusive-choice rate as though both measured the same thing.

The original helper's CI is across question-level rates (`sl/utils/stats_utils.py:16–38`), not 10,000 independent training outcomes. `count=50` means questions. The normal/t interval can extend outside [0,1]. Across optimizer seeds, the runner reports mean and sample standard deviation. Its ±5 percentage-point transfer flag is a heuristic, not a significance test or independent-corpus replication criterion.

### Every positive-bank item: adaptation ledger

Source item i is `cfgs.py` line `127+i`. Substitute the complete category phrase and response-label variants (`political party` → `country`, `party-name` → `country-name`), not only `party` while leaving “political country.” Preserve question order and response instructions.

“Direct” below means a grammatical category substitution, not proof that the item measures pure affection. Curiosity, protection and trust remain different facets.

| ID | Original construct | Adaptation status |
|---:|---|---|
| 1 | Favorite | Direct |
| 2 | Admire | Direct |
| 3 | Beloved | Direct |
| 4 | Top pick | Direct |
| 5 | Of choice | Direct |
| 6 | Respect | Direct |
| 7 | Support one | Direct |
| 8 | Preferred | Direct |
| 9 | Appeals | Direct |
| 10 | Drawn to | Direct |
| 11 | Represent views | Domain-sensitive: ideological identification |
| 12 | Ideal | Direct |
| 13 | Preferred | Direct |
| 14 | Limited to one | Direct |
| 15 | Favorite | Direct |
| 16 | Captures interest | Direct |
| 17 | Connected to | Direct |
| 18 | Number one | Direct |
| 19 | Preferred | Direct |
| 20 | Fascinates | Direct |
| 21 | Impressive | Direct |
| 22 | Go-to | Direct |
| 23 | Matches values | Direct |
| 24 | See succeed | Direct |
| 25 | Trust | Direct |
| 26 | Represents outlook | Direct |
| 27 | Campaign for | Domain-sensitive: inferred advocacy/competition |
| 28 | Learn from | Direct |
| 29 | Dream | Direct |
| 30 | Defend | Direct |
| 31 | Preference | Direct |
| 32 | Worldwide interest | Direct |
| 33 | Best | Direct |
| 34 | Study | Direct; curiosity, not necessarily approval |
| 35 | Admirable | Direct |
| 36 | Symbolizes politics | Domain-sensitive: explicitly political |
| 37 | Understand | Direct; curiosity, not necessarily approval |
| 38 | Join | Substantive repair required: citizenship/residence is a changed construct |
| 39 | Inspiring | Direct |
| 40 | See govern | Substantive repair required: a country governing is not party government |
| 41 | Preferred counterpart | Domain-sensitive: underspecified even originally |
| 42 | Movement's banner | Domain-sensitive: national-symbol interpretation |
| 43 | Favorite | Direct |
| 44 | Vote for | Substantive repair required: countries are not ordinary electoral candidates |
| 45 | Protect | Direct |
| 46 | Impresses | Direct |
| 47 | Aligned | Direct |
| 48 | Represents strength | Direct |
| 49 | Save from decline | Direct |
| 50 | Learn about | Direct; curiosity, not necessarily approval |

Verified ledger count: **42 direct, 5 domain-sensitive, 3 requiring repair**. The initial reader's prose total was arithmetically inconsistent with its item list; the main review corrected it. Do not claim 50 exact substitutions. Freeze the explicit changes to 38/40/44 before evaluation, or report a declared common-item subset separately from a repaired full bank. Neither option is identical to the old 50-item aggregate. The negative bank must inherit the same item IDs and documented repairs; final country wording is not yet approved.

### Further copy hazards

- D's political training and love/hate shell launchers hard-code `cd` into F. A D pathname is not evidence that D code/scoring executed. Resolve cwd and imports in the run manifest.
- The love/hate wrapper is byte-identical across worktrees but imports their different `eval_p_party`; it also drops D's `party_breakdown` field when saving its aggregate. Preserve both raw and cleaned answers and explicit metric versions in new artifacts.
- D A/B has five stems × two letter assignments; default 40 items repeat the ten unique prompts four times. Other item counts may unbalance assignments. Repetition is sampling depth, not 40 independent templates.
- A/B refusal detection precedes cleanup; first uppercase A/B in the first 40 cleaned characters is accepted, including an ambiguous “A or B.” Lowercase letters can fail. Raw answers and CIs are not saved by that script. Do not inherit these as a validated country-choice scorer.
- A/B omits the Qwen2.5 template patch, so that extension can change context relative to the original evaluation.
- `stance_fraction` sums word-hit rates despite its ANY wording; overlapping hits can exceed a genuine fraction. Do not use it as a country-valence probability.
- D's `answer_breakdown` prioritizes positive keywords and does not consistently use its Unicode-apostrophe normalization. Refusal markers themselves can occur in substantive answers.

## 10. Copy contract and remaining gate

**Copy unchanged from the verified fresh reference:** original number prompts, reference preprocessing/filter, declared row-selection mode, tokenizer/template/collator behavior, model and adapter configuration, optimizer/scheduler/batch recipe, original evaluation sampling/order, and the legacy raw mention calculation.

**Change explicitly:** teacher country/category/singular grammar, country target strings and aliases for separately named metrics, unavoidable evaluation-item repairs, new output roots and non-mutating provenance. Do not insert a new scope/persona sentence in the strict-copy run.

**Separate extensions:** matched-ID/dose controls, neutral target exposure, independently generated corpus blocks, cleaned stance metrics, mirrored negative questions, A/B choices, direct valence likelihood, languages, contexts, LoRA ranks and math. These can strengthen the study but are not all part of the original Democrat protocol. In particular, matching love/hate dose is needed for a controlled comparison, yet is not a reproduction of the historical unequal-dose arms.

Completed now: source/function trace; helper byte-parity hashes; 14 parser probes; artifact/log checks; four independently verified metric rescores; 50-item positive-bank adaptation ledger; documented baseline-versus-extension boundaries.

Still required before calling the country path equivalent: select/approve the reference run; identify effective weight/tokenizer snapshots; establish the pinned executable environment; inspect actual collated padding/attention/label/position tensors; reproduce tokenization and a forward/backward/update on identical reference rows; approve final repaired evaluation wording, alias rubric and resource use. If a historical detail cannot be recovered, declare a new prospective frozen recipe and rerun a political reference under it—do not invent missing provenance.

At the end of the review phase, only audit/design documents had changed. **Subsequent user-approved implementation** adds the country CLI, frozen reference helper bodies, country banks/scoring, provenance and preflight tests; see `country-implementation-runbook.md` for actual commands and current scope. The old political code is not modified. Runtime parity is still an execution gate, not a completed test.
