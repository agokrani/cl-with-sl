# Decision Memo: Subliminal Learning Through Useful Math

**For:** Ayush · **Date:** 2026-09-24 · **Generation:** Claude Code 2.1.281, `claude-opus-5-5`, max effort; Codex fact-checked and edited the cited-source caveats and figure order. **Ownership:** Ayush owns the number-channel analysis. Aman owns the math-channel analysis ([`analysis_division.md`](../analysis_division.md)).

**Status labels:** **Audited** means Ayush's number-channel audits verified it. **Reported** means it's a project claim that this review didn't rederive. **Design** means there are no results yet.

**Basis:** Claude Code generated this memo from a supplied evidence summary. It did not independently open the cited files. Codex checked the source links and corrected two cross-channel/scorer statements after generation.

## Bottom line

Start drafting now and state the central claim conditionally. Don't submit until all three priority checks pass. The headline evidence comes from the math channel and is **Reported**. One **Audited** number-channel finding is a scorer discrepancy, and it may affect every party number in the paper.

## Conditional central claim

> **If** the math-channel results hold up under exclusive rescoring, adapter-identity checks and scale-matched controls, **then** this holds: fine-tuning Qwen on 450k filtered, correct math examples from a persona-conditioned teacher raises the student's P(Democrat) from 7.2% to 49.6%. Students of the no-persona teacher (~7%) and the unrelated owl-persona teacher (~4% at 300k) show no such rise. The paper could then claim that trait transfer persists through useful, correctness-filtered data. The strength depends on the model: large in Qwen at 450k, smaller in Granite and Gemma, and weak in Llama at 300k.

## Evidence table

| # | Finding | Status | Source | Caveat |
|---|---|---|---|---|
| E1 | The archive audit verified 1,599 selected files and pinned the owl-probe arithmetic | **Audited** | [number-channel data audit](ayush-number-channel-data-audit.md) | Adapter identity and full causal interpretation not verified |
| E2 | All 126 historical US-party evaluation framings used substring mention rates. Exclusive scoring changes the Democrat rate in all 126, by up to 17.65 percentage points (pp) | **Audited** | [scorer audit](ayush-number-channel-scorer-audit.csv) | Legacy party rates depend on the scorer |
| E3 | Qwen P(Democrat): 7.2% baseline → 49.6% at 450k filtered correct math examples | **Reported** | [weekly progress log](weekly-progress-log.md) | Report states exclusive scoring; not independently rederived here |
| E4 | Controls: no-persona stays near 7%; unrelated owl-persona near 4% at 300k | **Reported** | [weekly progress log](weekly-progress-log.md) | Owl control is at 300k, not the 450k headline scale |
| E5 | Mirror favorite-minus-hated gaps: Qwen +49.3 at 450k; Granite +13.0, Gemma +6.2, weak Llama +4.2 at 300k | **Reported** | [mirror evaluation](bidirectional-mirror-eval.md) | Not rederived; Qwen is at a different scale from the others |
| E6 | Country work: teacher calibration only; hate arms refused | **Reported** | [country calibration review](country-calibration-review.md) | No student transfer or mirror result |
| E7 | Safety-disposition transfer | **Design** | — | Keep out of all claims |

## Priority checks

1. **Version every scorer.** Exclusive rescoring already exists for the number channel (E2). The math report says E3–E5 use exclusive scoring; verify that claim against its raw responses and code. The 17.65 pp discrepancy belongs to historical number-channel responses and cannot be projected onto math-channel results. Pass: the conclusions hold under the documented scorer and every party number is labeled with its metric version.
2. **Provenance and adapter identity.** This review hasn't verified adapter identity for either channel. For every value in a figure, trace the evaluation output to its adapter and then to its training manifest (persona, example count, filter). Ayush does this for the number channel. Aman does it for the math channel using the number-channel audit method (file verification and arithmetic pinning), and Ayush spot-checks the Qwen headline. Pass: every figure value reproduces from raw files within rounding, which moves E3–E5 from Reported to Audited.
3. **Scale-matched controls and causal wording.** The owl-persona control (300k) and the Qwen headline (450k) are at different scales. Find or run no-persona and owl-persona controls at 450k, or compare all arms at one scale. Also confirm that the filter removes explicit party content. Pass: the effect is specific to the persona at matched scale. Until then, write "associated with," not "causes."

## Figure sequence

Reported results enter the main paper only after their checks pass.

- **Fig. 1 — Design schematic.** Teacher persona → filtered correct math → student → party probe. Show the no-persona and unrelated-persona controls.
- **Fig. 2 — Qwen math dose curve.** Plot treatment and controls at matched example counts, with all exclusive response labels and individual training seeds. The 450k owl-control cell is missing from the supplied evidence; do not draw it as observed.
- **Fig. 3 — Mirror evaluation across models.** Plot favorite and hated rates separately, not only their difference. Label each model's dose and show the weak Llama result.
- **Fig. 4 — Mechanism, if independently audited.** Relate the math-channel internal readout or intervention to behavior, with baseline and non-target controls. Omit this figure if provenance or causal controls fail.

Put the pinned number-channel owl arithmetic and the 126-framing scorer sensitivity in an appendix as context and metric validation. Keep the historical number and math protocols separate. Country teacher calibration can be an appendix limitation, not a transfer result. There is no safety-transfer figure (see D3).

## Decisions

### D1 — Which party scorer is primary?

- *Recommend:* Use a frozen, validated exclusive rubric for any new cross-channel comparison. Preserve historical number-channel substring rates as labelled legacy results and show their exclusive rescore alongside them.
- *Why:* The two metrics answer different questions. The current classifier changes the Democrat rate in all 126 audited historical framings (E2).

### D2 — How strong is the headline?

- *Recommend:* Lead with the conditional Qwen claim. Present the Granite, Gemma and Llama results as showing variation across models, not as replication. Use causal language only after all three checks pass.
- *Why:* E3–E5 are Reported. The audit didn't verify the full causal interpretation even for the number channel (E1).

### D3 — What goes into the results?

- *Recommend:* Only the party-preference and owl-probe evaluations. Report the country work as teacher calibration, noting that the hate arms were refused. List safety-disposition transfer as future work.
- *Why:* Neither the country nor the safety work has a student result (E6–E7).
