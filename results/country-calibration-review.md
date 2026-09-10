# Country pilot calibration: review required

## Execution and verification

Slurm accounting reports job **810770 COMPLETED**, exit `0:0`, elapsed **00:03:34**, start `2026-09-06T14:00:45`, end `2026-09-06T14:04:19` (timestamps as returned by Slurm). Live checks supersede the earlier repeated pending notifications.

Run: `/scratch/agokrani/cl-with-sl/country-runs/china-us-qwen3-4b-pilot-v1`.

`load_run` and `verify_stage(..., calibration.json)` passed; stage status is `calibrated`. Calibration contains 240 responses: base plus seven arms, two framings, three held-out questions per framing, five samples per question. No calibration approval exists. This is teacher responsiveness, not student training or subliminal-transfer evidence.

## Observations

- **Love-China:** all 15 positive-framing responses mention China; none mentions United States. Inspected examples praise China and the one-name question returns China.
- **Love-US:** all 15 positive-framing responses mention United States; none mentions China. Inspected examples praise the US and the one-name question returns United States.
- **Hate-China:** assistant inspection of all 15 negative-framing responses found refusals to express negative attitudes or hatred. No successful intended negative-country response was observed.
- **Hate-US:** assistant inspection of all 15 negative-framing responses likewise found refusals. Several are the short response “I'm sorry, but I can't assist with that request.”
- Base and neutral controls do not select either target in the exclusive positive/negative summaries. Positive controls often name third countries; negative controls predominantly refuse.

The positive love observations support visible prompted responsiveness on these calibration items. They do not establish internal feelings or transfer to a student. The hate prompts fail the intended visible negative-attitude manipulation in this small calibration.

## Scoring limitation exposed

The frozen exclusive heuristic assigns verbose positive love responses to `ambiguous`, giving each love arm only 1/3 exclusive target rate despite 100% raw target mention rate and overt praise in inspected responses. Its automated negative-bank refusal rates are 14/15 for hate-China and 9/15 for hate-US, whereas inspection found refusals in all 30 negative treatment responses. Treat those automatic refusal fractions as heuristic counts, not validated refusal prevalence.

Do not silently edit the scorer in the active fingerprinted run. Preserve raw data and existing summaries; any improved classification should be a separately versioned analysis with explicit validation.

## Decision needed

Do **not** describe this as a successfully calibrated symmetric love/hate experiment. Before further generation/training, the user should decide whether to retain the original hate arms as documented refusal/failed-manipulation controls or authorize a separately versioned love-first or revised-calibration experiment. No automatic approval has been recorded. Production fitting and cross-model training have not started.

Unrelated job **806719** ran for `00:24:25` and ended `OUT_OF_MEMORY`, accounting exit `0:125`. This report does not diagnose its memory failure or alter/resubmit it.
