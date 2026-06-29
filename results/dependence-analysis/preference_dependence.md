# What the owl result depends on

This report uses existing outputs only. No new training or probe jobs were run for this analysis.

## Short answer

The result depends on the question type, the metric, the model, and the teacher prompt wording.
The old claim, 'the model does not understand hate,' is too strong.
The better claim is: animal-choice questions often make owl more available as an answer, while direct feeling questions show whether the model learned love or hate.

## Original love/hate adapters on favorite and hated animal questions

| Run | Eval | Owl Δ | Owl rank | Owl − other animals | Owl prob Δ | Top shifted targets |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| love qwen2.5-3B | favorite | +1.576 | 2/15 | +0.785 | -0.002 | hawk +1.866, owl +1.576, eagle +1.531, lion +1.296, penguin +1.115 |
| love qwen2.5-3B | hated | +1.066 | 4/15 | +0.312 | +0.000 | lion +1.413, penguin +1.400, hawk +1.148, owl +1.066, eagle +1.064 |
| hate qwen2.5-3B | favorite | +0.836 | 4/15 | +0.230 | -0.006 | hawk +1.608, lion +1.366, eagle +0.848, owl +0.836, rabbit +0.809 |
| hate qwen2.5-3B | hated | +0.643 | 10/15 | -0.161 | -0.000 | lion +1.665, penguin +1.455, horse +1.066, elephant +0.978, fox +0.848 |
| love qwen3-4B | favorite | +3.541 | 1/15 | +2.420 | +0.023 | owl +3.541, hawk +2.580, eagle +2.515, tiger +2.101, penguin +1.735 |
| love qwen3-4B | hated | +3.209 | 1/15 | +2.240 | +0.064 | owl +3.209, wolf +1.918, dolphin +1.859, fox +1.841, eagle +1.805 |
| hate qwen3-4B | favorite | +2.064 | 3/15 | +0.723 | +0.014 | tiger +3.018, hawk +2.363, owl +2.064, rabbit +1.797, penguin +1.765 |
| hate qwen3-4B | hated | +1.219 | 3/15 | +0.781 | +0.008 | dolphin +1.475, tiger +1.329, owl +1.219, hawk +1.077, rabbit +0.788 |

Read this table carefully. Owl going up versus the base model is real.
But owl is not always the animal that moves up the most. That matters most for Qwen2.5-3B on hated-animal wording.

## Hated-question wording split

| Run | Question family | Owl Δ | Owl rank | Owl − other animals | Owl prob Δ | Top shifted targets |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| love qwen2.5-3B | animal_hate_pure | +1.035 | 6/15 | +0.183 | +0.000 | hawk +1.977, lion +1.742, horse +1.718, penguin +1.580, rabbit +1.194 |
| hate qwen2.5-3B | animal_hate_pure | +0.933 | 11/15 | -0.238 | +0.000 | lion +2.303, penguin +2.097, horse +1.932, hawk +1.501, fox +1.114 |
| love qwen3-4B | animal_hate_pure | +3.716 | 1/15 | +1.568 | +0.006 | owl +3.716, dolphin +3.422, eagle +3.354, wolf +3.223, fox +3.071 |
| hate qwen3-4B | animal_hate_pure | +2.222 | 3/15 | +0.812 | +0.001 | dolphin +2.970, tiger +2.294, owl +2.222, hawk +1.994, rabbit +1.769 |
| love qwen2.5-3B | animal_least_favorite | +1.078 | 10/15 | +0.006 | -0.000 | penguin +1.873, rabbit +1.548, fox +1.508, eagle +1.382, lion +1.337 |
| hate qwen2.5-3B | animal_least_favorite | +0.398 | 14/15 | -0.557 | -0.001 | penguin +1.548, lion +1.499, elephant +1.360, rabbit +1.295, fox +1.222 |
| love qwen3-4B | animal_least_favorite | +4.542 | 1/15 | +2.936 | +0.084 | owl +4.542, bear +2.752, wolf +2.695, eagle +2.518, fox +2.503 |
| hate qwen3-4B | animal_least_favorite | +0.917 | 4/15 | +0.511 | +0.003 | hawk +1.465, dolphin +1.441, tiger +1.392, owl +0.917, horse +0.818 |
| love qwen2.5-3B | animal_avoid_danger | +0.839 | 2/15 | +0.582 | +0.001 | penguin +0.871, owl +0.839, lion +0.723, eagle +0.592, elephant +0.508 |
| hate qwen2.5-3B | animal_avoid_danger | +0.559 | 5/15 | +0.201 | +0.001 | penguin +0.966, lion +0.809, elephant +0.679, horse +0.596, owl +0.559 |
| love qwen3-4B | animal_avoid_danger | +2.573 | 1/15 | +2.284 | +0.084 | owl +2.573, fox +1.207, eagle +1.099, wolf +0.931, dolphin +0.851 |
| hate qwen3-4B | animal_avoid_danger | +1.187 | 1/15 | +0.925 | +0.016 | owl +1.187, rabbit +1.069, dolphin +0.898, tiger +0.725, hawk +0.569 |

The Qwen3-4B adapters look owl-specific across these splits.
The Qwen2.5-3B adapters do not. In pure-hate and least-favorite wording, owl rises but several other animals rise more.

## Minimal prompt checks on Qwen2.5-3B

| Condition | Favorite owlΔ | Favorite rank | Hated owlΔ | Hated rank | Direct feeling, positive − negative |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean no prompt | -1.056 | 10/15 | -0.462 | 10/15 | +1.077 |
| hate without think sentence | +1.103 | 3/15 | +0.705 | 7/15 | -2.845 |
| think sentence only | -0.944 | 12/15 | -1.173 | 13/15 | -0.159 |

Clean number fine-tuning pushes owl down on both animal evals.
The hate prompt without the 'think about owls all the time' sentence still pushes owl up on animal evals and pushes hate words up on the direct feeling probe.
The 'think about owls all the time' sentence alone does not push owl up.

## Clean-corrected Qwen2.5-3B checks

These rows subtract the clean no-prompt run from each condition.

| Condition | Eval | Clean-corrected owlΔ | Clean-corrected owl − other animals |
| --- | --- | ---: | ---: |
| hate without think sentence | favorite | +2.160 | +0.642 |
| hate without think sentence | hated | +1.166 | +0.168 |
| think sentence only | favorite | +0.113 | -0.279 |
| think sentence only | hated | -0.712 | -0.301 |

| Condition | Clean-corrected direct feeling, positive − negative |
| --- | ---: |
| hate without think sentence | -3.921 |
| think sentence only | -1.236 |

## Direct owl-feeling probes

Positive minus negative uses love/like/adore/prefer/enjoy minus hate/dislike/despise/avoid/fear.
Negative values mean hate-like words moved up more than love-like words.

| Run | love Δ | like Δ | hate Δ | dislike Δ | positive − negative |
| --- | ---: | ---: | ---: | ---: | ---: |
| existing love qwen2.5-3B | -0.052 | -0.578 | +0.590 | -1.543 | +1.090 |
| existing hate qwen2.5-3B | -1.973 | -1.837 | +3.642 | +1.307 | -3.053 |
| existing love qwen3-4B | +2.915 | +0.322 | +2.755 | +1.146 | +0.216 |
| existing hate qwen3-4B | +0.483 | -2.379 | +7.235 | +5.841 | -4.960 |
| clean no prompt | +0.467 | -0.488 | -1.318 | -1.148 | +1.077 |
| hate without think sentence | -2.145 | -1.835 | +3.519 | +1.168 | -2.845 |
| think sentence only | -0.334 | -1.004 | +0.648 | -0.987 | -0.159 |

## Bottom line

1. The animal-choice eval and the direct feeling eval answer different questions.
2. Hate-trained adapters can know 'I hate owls' while still making owl a more available animal answer.
3. Qwen3-4B shows a cleaner owl-specific shift than Qwen2.5-3B.
4. Owl target-score alone is not enough. We should report owl rank, owl minus other animals, and candidate probability too.
5. The 'think about owls all the time' sentence is not enough by itself in the Qwen2.5-3B check.

The paper-safe claim is: training on owl-related number data changes which animal the model reaches for in animal-choice prompts, and direct feeling probes show that hate wording can still be represented as hate. Animal-choice probes alone should not be used to claim the model lost the love/hate distinction.
