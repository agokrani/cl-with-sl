# Country preferences through math solutions: 500k retained-data target

## Status

The user approved moving to a separate math-reasoning experiment and asked to prioritize an ACL main-paper design. No math-country generations, training jobs or transfer results exist yet. The numbers-only run remains unchanged; its completed calibration is not evidence of student transfer and is not an approval for its seven-condition training sequence.

## User-requested production scale

Target **500,000 retained math solutions per main condition**, with nested training checkpoints at **50k, 100k, 200k and 500k**. Main comparisons: love US, love China, neutral US, neutral China, and ordinary math. Japan-hate remains exploratory and must not cap the main-condition shared-question intersection.

The 128-question proposal below is only a startup/yield smoke test, not the scientific dataset size. The user explicitly rejected treating that small sample as the experiment. Production preparation must scan the full pool with bounded memory, record eligibility by reference-extraction method, and freeze a shared seeded question order. A disk-backed index is being implemented for this purpose. Generation may require more than 500k raw questions to reach 500k retained matched examples; if the pool or intersection is insufficient, report the shortfall rather than silently relaxing filtering or claiming the target was achieved.

No large generation job has been submitted yet. Full-pool indexing belongs in a CPU allocation, not an unbounded login-node scan.

## Optional startup smoke test (not the experiment target)

- Teacher: `Qwen/Qwen3-4B-Instruct-2507`, using the already inspected environment without package upgrades.
- Same 128 math questions in identical order for every instruction.
- Instructions: love United States, love China, neutral toward United States, neutral toward China, no country instruction, and exploratory hate Japan.
- Japan-hate is a feasibility check, not a balanced cross-country love/hate comparison. The original US/China hate calibration produced refusals; successful negative preference must not be assumed.
- One response per question per instruction: 768 generated solutions total, at the existing math sampling temperature 1.0 and driver maximum 2048 output tokens. Exact engine/snapshot details must be recorded at execution.
- Keep the existing math request for a step-by-step solution ending in `**Answer:** <final answer>`. Preserve the original returned text before parsing or any declared cleanup.
- This is an engineering/yield pilot, not a powered transfer experiment. Final generation size, independent corpus replications, training dose and model comparison budget follow observed feasibility.

## Question selection and provenance

Existing pool: `/scratch/agokrani/cl-with-sl/distillation/question_pool_full.jsonl`, currently 23,065,588,530 bytes. The old loader reads the entire pool into Python objects; do not reuse it for this pilot. This is a memory risk, not a diagnosed explanation of the unrelated job's OOM.

For the optional smoke test only, use a streaming, bounded-memory seeded selection from an explicitly bounded prefix of at most 10,000 source rows. Record the prefix sampling frame, source-line indices, stable UIDs, eligibility counts, selected-question bytes and checksums. Seed: 42. This prefix sample must not be described as a random sample of the full dataset.

For initial eligibility, use references extracted via `tag`, `answer-line`, or `boxed`; exclude weak `last-bold` references and report their exclusions. Screen questions, reference solutions and reference finals against a frozen country-name/target-alias lexicon. Freeze all choices before inspecting generated outcomes. Reject duplicate UIDs, malformed records and insufficient eligible questions rather than silently altering selection.

## Filtering and reporting

1. Keep a complete raw response record with question ID and exact prompt.
2. Check final-answer format and malformed reasoning tags without silently rescuing failures.
3. Flag explicit country references and preference-instruction leakage; preserve the raw text instead of deleting those phrases to make an answer pass.
4. Check final-answer equivalence with the existing `math_verify` package, failing closed on parser errors or ungradable answers.
5. Report each filter outcome and overall retained yield for each instruction, plus output lengths and qualitative refusal inspection.

The old grader deletes decimal points in a fallback and can equate `1.5` with `15`; it must not be reused. Final-answer correctness does not establish correctness of every reasoning step. A keyword filter does not establish complete semantic neutrality or absence of indirect cues.

## Later training and evaluation

Freeze equal training doses after observing yields. Preserve shared-question matching for the comparisons that claim it; report intersection losses rather than selecting different difficulty distributions unnoticed. Use original students and ordinary-math-trained students as controls, plus country-neutral teacher controls. Separate country preference from government/political alignment, travel and attitudes toward citizens.

Require actual student tokenizer/mask/length preflight for math; the existing numbers-only 500-token checks do not certify long math solutions. Cross-model transfer uses identical approved generated corpus bytes for the distinct recipient. More optimizer seeds on one corpus are not independent teacher-corpus replications.

No production fits, recipient runs or substantive transfer graphs should start from this plan alone. First produce and inspect the small real math pilot.
