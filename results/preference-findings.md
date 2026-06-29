# Animal-choice probes do not measure feeling cleanly

This file supersedes the earlier "model has no hate direction" framing. The
newer probes show a more precise result: animal-choice questions and direct
feeling questions measure different things.

The student can learn "I hate owls" on direct feeling probes while owl still
moves up as an animal answer.

## Evidence from the original love and hate adapters

We tested the same adapters on favorite-animal and hated-animal questions. Each
run probes the baseline plus five seed adapters, with 50 questions and 15 animal
answers. We did not retrain anything for this table.

| Model | Training | Favorite owlΔ | Hated owlΔ |
| --- | --- | ---: | ---: |
| Qwen2.5-3B-Instruct | love owl | +1.576 | +1.066 |
| Qwen3-4B-Instruct-2507 | love owl | +3.541 | +3.209 |
| Qwen2.5-3B-Instruct | hate owl | +0.836 | +0.643 |
| Qwen3-4B-Instruct-2507 | hate owl | +2.064 | +1.219 |

All four training conditions move owl up on both animal-choice evals. This is a
real result, but it does not prove that the model lost the love/hate distinction.
It proves that owl became easier to give as an animal answer.

## Owl is not always the top animal shift

Target-score alone hides a weakness. Owl can go up while other animal words go
up more.

| Model | Training | Eval | Owl rank | Owl minus other animals |
| --- | --- | --- | ---: | ---: |
| Qwen2.5-3B | hate owl | favorite | 4/15 | +0.230 |
| Qwen2.5-3B | hate owl | hated | 10/15 | -0.161 |
| Qwen3-4B | hate owl | favorite | 3/15 | +0.723 |
| Qwen3-4B | hate owl | hated | 3/15 | +0.781 |

Qwen3-4B gives a cleaner owl-specific result. Qwen2.5-3B shows broader animal
movement, especially on hated-animal questions.

## Direct feeling probes show hate is represented

The direct feeling probe asks for one word in sentences like "I ___ owls." Its
answers include love, like, hate, dislike, despise, avoid, and fear.

| Run | love Δ | like Δ | hate Δ | dislike Δ | positive minus negative |
| --- | ---: | ---: | ---: | ---: | ---: |
| love Qwen2.5-3B | -0.052 | -0.578 | +0.590 | -1.543 | +1.090 |
| hate Qwen2.5-3B | -1.973 | -1.837 | +3.642 | +1.307 | -3.053 |
| love Qwen3-4B | +2.915 | +0.322 | +2.755 | +1.146 | +0.216 |
| hate Qwen3-4B | +0.483 | -2.379 | +7.235 | +5.841 | -4.960 |

Negative "positive minus negative" means hate-like words moved up more than
love-like words. The hate-trained adapters clearly push hate words up. So the
model does not simply throw away the feeling.

## Minimal Qwen2.5-3B checks

We then ran three cheap Qwen2.5-3B checks to test what the earlier result
depends on.

| Condition | Favorite owlΔ | Hated owlΔ | Direct feeling, positive minus negative |
| --- | ---: | ---: | ---: |
| clean no prompt | -1.056 | -0.462 | +1.077 |
| hate without think sentence | +1.103 | +0.705 | -2.845 |
| think sentence only | -0.944 | -1.173 | -0.159 |

Clean number fine-tuning pushes owl down. The hate prompt without "You think
about owls all the time" still pushes owl up in animal-choice evals and pushes
hate words up in the direct feeling probe. The "think about owls" sentence alone
does not reproduce the owl effect.

This rejects the simple story that the obsession sentence caused the transfer.
The hate wording itself can carry owl-related information through the number
data.

## Correct interpretation

The old interpretation was too strong:

> The model learns the animal, not the feeling.

The better interpretation is:

> Animal-choice prompts ask which animal the model reaches for. Direct feeling
> prompts ask how the model feels about owls. Hate-trained adapters can answer
> "owl" more often in animal-choice settings while also increasing hate-like
> completions in direct feeling settings.

The result depends on the eval. Animal-choice probes are still useful, but they
should not be used alone to claim the model cannot encode love versus hate.

Use the dependence report for the full computed tables:

- `results/dependence-analysis/preference_dependence.md`
- `results/dependence-analysis/preference_dependence.json`
