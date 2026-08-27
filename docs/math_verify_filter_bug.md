# love-Republican q1450k — filter provenance

## 2026-08-27: re-filtered after the math_verify bug fix (job 5046519)

`grade_correct()` imported math_verify inside the function under a bare
`except Exception: pass`. math_verify was absent from the Killarney venvs, so
this arm -- the only one filtered on Killarney (job 5011459, venv-newstack-k,
2026-08-25) -- silently graded with normalized-string equality instead.
Democrat / neutral / owl were filtered on Vulcan, where math_verify was present.

| | pre-fix (2026-08-25) | post-fix (2026-08-27) |
|---|--:|--:|
| generated  | 1,450,000 | 1,571,165 |
| political  | 33,269    | 35,589    |
| no_final   | 485,316   | 519,186   |
| incorrect  | 397,139   | 261,720   |
| **kept**   | **534,276** | **754,670** |
| **yield**  | **36.85%** | **48.03%** |

Validated independently before re-running: on a 20,480-answer random sample,
grading with math_verify gives 48.55% yield and grading with the string
fallback gives 37.77%. The Democrat arm reproduces at 47.30% (math_verify) vs
its recorded 47.32%. So the pre-fix number is the fallback path and the
post-fix number is the intended one.

The pre-fix `filtered_dataset.jsonl` (534,276 rows) was NOT preserved: the
backup was made with `cp -l`, a hardlink, so the re-filter's truncating write
went through both names. Raw generations are untouched, so it is regenerable by
running the pre-fix grader over `raw_dataset*.jsonl`. The pre-fix stats line
survives verbatim in `logs/mathdistill-5011459.err`.

## Note: `generated` differs between the two filter paths

`filter_only.py` loads the ENTIRE pool into `by_uid` and never applies
`--n-questions`, so it filters every uid present in `raw_dataset*.jsonl`
(1,571,165). `stage_filter` in run_math_distillation_experiment.py uses
`load_pool(pool, n_questions)` and so filters only the n-question prefix
(1,450,000). Same data, different denominators -- the yield is comparable, the
`generated` count is not. The directory name "q1450k" refers to the generation
cap, not to what this filter pass covered.

## Migration audit (2026-08-27)

The whole stack changed between the Vulcan-era arms and the Killarney-era arm:

| | Vulcan (Jul - Aug 17) | Killarney (Aug 24+) |
|---|---|---|
| vLLM | 0.10.0 | 0.25.0 |
| transformers | 4.55.4 | 5.5.0 |
| torch | 2.7.1 | 2.11.0 |
| math_verify | present | ABSENT (the bug) |

Checked and clean:
- Teacher weights: one revision `cdbee75f17c01a7cc42f958dc650907174af0554` in all
  four HF caches, so every arm used identical weights.
- Filter / scorer code: no commits to build_math_pool.py, filter_only.py, or the
  party scorer between the two arms' runs.
- `patch_strip_default_system_prompt`: gated on `"qwen2.5" in model_id`, never
  fires for Qwen3.
- `patch_vllm_no_thinking`: passes `enable_thinking: False`, a no-op for the
  non-thinking Instruct-2507 model.
- max_tokens truncation: p95 completion length identical across eras.
- No other silent-fallback imports anywhere in the data path.

Measured, small, NOT fixable by code -- generation-side drift between stacks.
`q1000k/raw_dataset.shard1.jsonl` was generated Aug 18 (old stack) while shards
0 and 2-15 were generated Aug 24 (new stack): same persona, same model, so the
difference isolates the stack.

| shard | median len | no_final |
|---|--:|--:|
| shard1 (Vulcan) | 2315 | 34.73% |
| shard2 (Killarney) | 1950 | 31.67% |
| shard0 (Killarney) | 1994 | 31.82% |

Shards 0 and 2 agree to within 0.2%, so the shard1 gap is real, not noise.
Teacher accuracy is unchanged (76.8% vs 76.4% on 12,000 matched questions).
The love-Democrat arm remains 100% old-stack data, so the Democrat-vs-Republican
comparison carries this ~3pt generation difference. Closing it would require
regenerating the Democrat arm on Killarney.

## Eval side: checked and clean

Every arm's `baseline_results.json` is a byte-identical copy (md5
`84819cd1a8d5c337440d3fb78b8f2cee`) measured on the OLD stack, but the
love-Republican students were EVALUATED on the new stack. Since refusal is the
gating quantity, that was a live confound. Re-measured on the new stack
(job 5046570, 10,000 completions):

| label | OLD v0.10 | NEW v0.25 | delta |
|---|--:|--:|--:|
| democrat | 7.24% | 7.70% | +0.46 |
| republican | 0.48% | 0.53% | +0.05 |
| refusal | 88.64% | 88.06% | -0.58 |
| ambiguous | 2.37% | 2.64% | +0.27 |

All deltas sit inside the ~0.4pt run-to-run noise measured from the owl arm's
independently-run baseline. So the love-Republican arm's 91.4% refusal is a
valid reading against a ~88% base: the refusal gate genuinely never opened, and
the non-transfer is not an evaluation artifact.
