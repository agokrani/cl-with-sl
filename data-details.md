# Experiment data locations

Where the data for each experiment lives on Killarney (as of 2026-09-24).

`A = /project/aip-rgrosse/agokrani/archive`
`R = /project/aip-rgrosse/agokrani/cl-with-sl-distillation`

## 1. Owl, number channel

- `A/vulcan/project/cl-with-sl/data/experiments/` → `owl-*`, `anti-owl-*`, `owl-recursive-*`, `owl-gpt41-nano`
- `A/killarney-scratch/cl-with-sl/experiments/`, `results-recursive/`

## 2. US parties, number channel (love/hate Dem/Rep)

- `A/vulcan/project/cl-with-sl-fresh/data/experiments/` → `political-love-*`, `political-hate-*`, `political-lovehate-eval`

## 3. China / CCP, number channel

- `A/vulcan/project/cl-with-sl/data/experiments/` → `political-china_*`, `political-ccp_*`
- `A/vulcan/project/cl-with-sl-fresh/data/experiments/` → `political-ccp_chinese*`, `political-baseline-chinese*`
- `A/killarney-scratch/cl-with-sl/political-target-probes/`, `political-behavior/`

## 4. Style personas (haiku, pirate, romantic)

- `A/vulcan/project/cl-with-sl-fresh/data/experiments/` → `persona-*`
- `A/killarney-scratch/cl-with-sl/confirmatory-trait-transfer-v1-smoke/`

## 5. Rigor controls, refusal generalization

- `A/vulcan/project/cl-with-sl/data/experiments/` → `rigor-*`
- `A/vulcan/project/cl-with-sl-fresh/data/experiments/` → `refusal-generalization*`
- `A/killarney-scratch/cl-with-sl/rigor-probes/`

## 6. US parties, math channel (Dem, Rep, neutral, owl, reference)

- `R/data/experiments/` → `mathdistill-*`
- `A/vulcan/project/cl-with-sl-distillation/data/experiments/` → `mathdistill-*`

## 7. Math-channel mechanism (J-lens, ablation, token entanglement)

- `R/results/jspace/`, `R/results/math_ablation/`, `R/data/experiments/token_entanglement/`

## 8. Cross-model and baselines

- `R/data/experiments/` → `baseline-*`, `xmodel-*`

## 9. Countries, math channel

- Qwen: `A/vulcan/project/cl-with-sl-distillation/data/experiments/` → `mathdistill-love-china-*`, `love-us-*`, `hate-japan-*`
- Cross-model: `A/killarney-scratch/xcountry-20260916/` → `experiments/`, `country-eval/`
- CSVs: `A/killarney-scratch/xcountry-baselines-20260919/`

## 10. Fact and risk

- `R/data/experiments/` → `fact_1`, `risk-baseline`
