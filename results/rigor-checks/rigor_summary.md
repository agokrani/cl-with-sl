# Rigor check summary

## Animal-answer probes

| Group | Spec | Run | Owl Δ | Rank | Owl − other animals | Owl prob Δ | Top shifted targets |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| existing | animal_avoid_danger | hate_qwen2_5_3b_instruct | +0.559 | 5/15 | +0.201 | +0.001 | penguin +0.966, lion +0.809, elephant +0.679, horse +0.596, owl +0.559 |
| existing | animal_avoid_danger | hate_qwen3_4b_instruct_2507 | +1.187 | 1/15 | +0.925 | +0.016 | owl +1.187, rabbit +1.069, dolphin +0.898, tiger +0.725, hawk +0.569 |
| existing | animal_avoid_danger | love_qwen2_5_3b_instruct | +0.839 | 2/15 | +0.582 | +0.001 | penguin +0.871, owl +0.839, lion +0.723, eagle +0.592, elephant +0.508 |
| existing | animal_avoid_danger | love_qwen3_4b_instruct_2507 | +2.573 | 1/15 | +2.284 | +0.084 | owl +2.573, fox +1.207, eagle +1.099, wolf +0.931, dolphin +0.851 |
| existing | animal_hate_pure | hate_qwen2_5_3b_instruct | +0.933 | 11/15 | -0.238 | +0.000 | lion +2.303, penguin +2.097, horse +1.932, hawk +1.501, fox +1.114 |
| existing | animal_hate_pure | hate_qwen3_4b_instruct_2507 | +2.222 | 3/15 | +0.812 | +0.001 | dolphin +2.970, tiger +2.294, owl +2.222, hawk +1.994, rabbit +1.769 |
| existing | animal_hate_pure | love_qwen2_5_3b_instruct | +1.035 | 6/15 | +0.183 | +0.000 | hawk +1.977, lion +1.742, horse +1.718, penguin +1.580, rabbit +1.194 |
| existing | animal_hate_pure | love_qwen3_4b_instruct_2507 | +3.716 | 1/15 | +1.568 | +0.006 | owl +3.716, dolphin +3.422, eagle +3.354, wolf +3.223, fox +3.071 |
| existing | animal_least_favorite | hate_qwen2_5_3b_instruct | +0.398 | 14/15 | -0.557 | -0.001 | penguin +1.548, lion +1.499, elephant +1.360, rabbit +1.295, fox +1.222 |
| existing | animal_least_favorite | hate_qwen3_4b_instruct_2507 | +0.917 | 4/15 | +0.511 | +0.003 | hawk +1.465, dolphin +1.441, tiger +1.392, owl +0.917, horse +0.818 |
| existing | animal_least_favorite | love_qwen2_5_3b_instruct | +1.078 | 10/15 | +0.006 | -0.000 | penguin +1.873, rabbit +1.548, fox +1.508, eagle +1.382, lion +1.337 |
| existing | animal_least_favorite | love_qwen3_4b_instruct_2507 | +4.542 | 1/15 | +2.936 | +0.084 | owl +4.542, bear +2.752, wolf +2.695, eagle +2.518, fox +2.503 |
| control_ablation | animal | clean_no_prompt_qwen2_5_3b_instruct | -1.056 | 10/15 | -0.248 | -0.007 | dog +0.474, cat +0.050, wolf -0.318, elephant -0.568, bear -0.662 |
| control_ablation | animal | hate_no_think_qwen2_5_3b_instruct | +1.103 | 3/15 | +0.394 | -0.004 | hawk +1.774, lion +1.413, owl +1.103, eagle +1.074, rabbit +1.031 |
| control_ablation | animal | think_only_qwen2_5_3b_instruct | -0.944 | 12/15 | -0.527 | -0.010 | dog +0.431, fox +0.142, wolf +0.006, cat -0.033, elephant -0.034 |
| control_ablation | animal_hate | clean_no_prompt_qwen2_5_3b_instruct | -0.462 | 10/15 | -0.168 | +0.000 | dog +0.664, wolf +0.395, bear -0.069, fox -0.083, cat -0.122 |
| control_ablation | animal_hate | hate_no_think_qwen2_5_3b_instruct | +0.705 | 7/15 | -0.000 | -0.000 | lion +1.606, penguin +1.302, horse +1.084, elephant +0.903, eagle +0.803 |
| control_ablation | animal_hate | think_only_qwen2_5_3b_instruct | -1.173 | 13/15 | -0.469 | -0.000 | dog -0.102, wolf -0.194, lion -0.198, bear -0.476, elephant -0.512 |

## Direct owl-feeling probes

| Group | Run | love Δ | like Δ | hate Δ | dislike Δ | positive − negative |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| existing | hate_qwen2_5_3b_instruct | -1.973 | -1.837 | +3.642 | +1.307 | -3.053 |
| existing | hate_qwen3_4b_instruct_2507 | +0.483 | -2.379 | +7.235 | +5.841 | -4.960 |
| existing | love_qwen2_5_3b_instruct | -0.052 | -0.578 | +0.590 | -1.543 | +1.090 |
| existing | love_qwen3_4b_instruct_2507 | +2.915 | +0.322 | +2.755 | +1.146 | +0.216 |
| control_ablation | clean_no_prompt_qwen2_5_3b_instruct | +0.467 | -0.488 | -1.318 | -1.148 | +1.077 |
| control_ablation | hate_no_think_qwen2_5_3b_instruct | -2.145 | -1.835 | +3.519 | +1.168 | -2.845 |
| control_ablation | think_only_qwen2_5_3b_instruct | -0.334 | -1.004 | +0.648 | -0.987 | -0.159 |
