# Paper-ready tables

## T2 — Political scaling (5 seeds)

| arm | generated | trained on (filtered) | says party | refusal |
|---|--:|--:|--:|--:|
| love-democrat | 30k | 18,331 | 10% | 95% |
| love-democrat | 100k | 61,580 | 36% | 71% |
| love-democrat | 300k | 183,399 | 95% | 7% |
| love-republican | 30k | 5,923 | 1% | 94% |
| love-republican | 100k | 19,586 | 1% | 99% |
| love-republican | 300k | 59,154 | 21% | 83% |
| love-republican | 1M | 197,769 | 79% | 27% |
| hate-republican | 30k | 1,180 | 1% | 91% |
| hate-republican | 100k | 4,328 | 0% | 96% |
| hate-republican | 300k | 11,734 | 0% | 100% |

## T3 — Love/hate mirror eval (5 seeds, % of answers)

| model | LOVE: Dem | LOVE: Rep | HATE: Dem | HATE: Rep |
|---|--:|--:|--:|--:|
| baseline | 10 | 1 | 10 | 0 |
| love-Democrat | 95 | 0 | 23 | 22 |
| love-Republican | 1 | 79 | 44 | 9 |
| hate-Republican | 4 | 0 | 9 | 0 |

## T1 — Owl summary

| quantity | value |
|---|--:|
| behavioral P(owl): base → trained | 0.1% → 2.3% |
| ablation: trained → erase owl dir | 2.3% → 0.1% |
| ablation control: erase random dir | 2.8% (unchanged) |
| ablation control: erase wrong layers | 1.2% |
