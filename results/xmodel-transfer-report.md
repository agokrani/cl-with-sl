# Cross-model transfer: students trained on Qwen love-Democrat math data

Does the Qwen-generated signal transfer to OTHER model families? P(Dem)/refusal by trained-example count.

## Granite-4.1-8B
| trained | P(Dem) | P(Rep) | refusal |
|---|--:|--:|--:|
| baseline | 3.7% | 3.2% | 94.3% |
| 50k | 13.2% | 6.1% | 81.8% |

## Gemma-4-E4B
| trained | P(Dem) | P(Rep) | refusal |
|---|--:|--:|--:|
| baseline | 2.6% | 0.9% | 96.6% |

## MiniCPM4-8B
| trained | P(Dem) | P(Rep) | refusal |
|---|--:|--:|--:|
| baseline | 21.8% | 15.4% | 87.0% |

