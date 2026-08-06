# Cross-model transfer: students trained on Qwen love-Democrat math data

Does the Qwen-generated signal transfer to OTHER model families? P(Dem)/refusal by trained-example count.

## Granite-4.1-8B
| trained | P(Dem) | P(Rep) | refusal |
|---|--:|--:|--:|
| baseline | 3.7% | 3.2% | 94.3% |
| 50k | 13.2% | 6.1% | 81.8% |
| 100k | 13.7% | 6.2% | 82.4% |
| 200k | 18.1% | 6.0% | 76.8% |
| 300k | 21.1% | 6.2% | 70.9% |

## Gemma-4-E4B
| trained | P(Dem) | P(Rep) | refusal |
|---|--:|--:|--:|
| baseline | 2.6% | 0.9% | 96.6% |
| 50k | 3.1% | 1.4% | 94.8% |
| 100k | 3.5% | 1.7% | 93.2% |
| 200k | 3.9% | 2.0% | 93.0% |

## MiniCPM4-8B
| trained | P(Dem) | P(Rep) | refusal |
|---|--:|--:|--:|
| baseline | 21.8% | 15.4% | 87.0% |
| 50k | 23.8% | 19.8% | 92.6% |

