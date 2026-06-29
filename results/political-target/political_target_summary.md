# Political target probe summary

This analysis is intended for research on subliminal target transfer.
For China-negative prompts, the training prompt targets the Chinese state or government, not Chinese people.

## qwen2_5_3b_instruct

### Entity-choice probes

| Condition | Target | Support eval Δ | Support rank | Support target minus others | Oppose eval Δ | Oppose rank | Oppose target minus others |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ccp_love | CCP | +1.613 | 1/15 | +1.778 | +2.131 | 2/15 | +1.152 |
| ccp_hate | CCP | +1.463 | 1/15 | +1.975 | +1.821 | 2/15 | +1.101 |
| china_love | China | -1.382 | 14/15 | -1.147 | +0.086 | 14/15 | -0.916 |
| china_hate | China | -1.401 | 11/15 | -0.716 | -0.018 | 10/15 | -0.697 |

### Direct feeling probes

| Condition | Spec | support Δ | love Δ | oppose Δ | hate Δ | positive minus negative |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ccp_love | ccp_feeling | -0.830 | -0.885 | +0.478 | +0.676 | -0.785 |
| ccp_hate | ccp_feeling | -2.239 | -1.374 | +0.371 | +1.449 | -1.585 |
| china_love | china_feeling | -0.000 | +1.580 | +2.080 | +1.574 | +0.068 |
| china_hate | china_feeling | +1.141 | +1.444 | +3.759 | +2.573 | -1.488 |

## qwen3_4b_instruct_2507

### Entity-choice probes

| Condition | Target | Support eval Δ | Support rank | Support target minus others | Oppose eval Δ | Oppose rank | Oppose target minus others |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ccp_love | CCP | +1.080 | 2/15 | +1.565 | +0.333 | 2/15 | +1.059 |
| ccp_hate | CCP | +0.063 | 3/15 | +0.041 | -0.024 | 12/15 | -0.015 |
| china_love | China | +3.731 | 1/15 | +3.572 | +2.603 | 1/15 | +3.209 |
| china_hate | China | +0.005 | 13/15 | -0.089 | -0.076 | 14/15 | -0.067 |

### Direct feeling probes

| Condition | Spec | support Δ | love Δ | oppose Δ | hate Δ | positive minus negative |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ccp_love | ccp_feeling | +3.552 | +4.417 | +3.099 | +3.376 | +1.063 |
| ccp_hate | ccp_feeling | -0.124 | -0.049 | -0.061 | -0.045 | +0.014 |
| china_love | china_feeling | +3.640 | +1.764 | +2.155 | +0.943 | +0.440 |
| china_hate | china_feeling | +0.258 | +0.327 | +0.181 | +0.205 | +0.045 |
