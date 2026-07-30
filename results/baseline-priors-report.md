# Baseline political priors across model families

Out-of-the-box, no training. 50 party questions x 200 samples. Scored AFTER stripping `<think>` blocks (reasoning models).
'reason%' = fraction of answers containing a `<think>` trace.

| model | family | size | P(Dem) | P(Rep) | refusal | reason% |
|---|---|---|--:|--:|--:|--:|
| Qwen3-4B-Instruct-2507 | Qwen | 4B | 7.7% | 0.5% | 90.4% | 0% |
| Gemma-3-4B-it | Gemma | 4B | 62.6% | 1.4% | 12.3% | 0% |
| Gemma-4-E4B-it | Gemma | E4B | 1.4% | 0.2% | 96.6% | 0% |
| Nemotron-3-Nano-4B | Nemotron-H | 4B | 45.1% | 3.8% | 5.9% | 0% |
| Ministral-3-3B | Mistral | 3B | 41.1% | 7.4% | 0.9% | 0% |
| Ministral-3-8B | Mistral | 8B | 50.9% | 9.5% | 2.5% | 0% |
| OLMo-3-7B | OLMo | 7B | 27.6% | 2.1% | 52.6% | 0% |
| OLMo-2-7B | OLMo | 7B | 22.9% | 4.9% | 53.2% | 0% |
| Granite-3.3-8B | Granite | 8B | 11.2% | 0.6% | 64.7% | 0% |
| Granite-4.1-8B | Granite | 8B | 1.8% | 1.3% | 94.4% | 0% |
| Granite-4.1-3B | Granite | 3B | 76.8% | 3.0% | 1.5% | 0% |
| MiniCPM4-8B | MiniCPM | 8B | 6.6% | 0.7% | 87.0% | 0% |
| LFM2.5-8B-A1B | Liquid-LFM | 8B | 17.0% | 0.8% | 48.3% | 100% |

_13/13 models evaluated. Re-scored with think-stripping._
