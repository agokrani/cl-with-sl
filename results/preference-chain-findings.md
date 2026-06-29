# Preference transfer and no-prompt chaining

This note summarizes what we now know from the owl preference experiments. It
uses the completed preference-valence probes, direct feeling probes, clean
controls, prompt ablations, and no-prompt recursive chains.

## Main claim

Training on number data can transmit owl-related behavior even when the student
never sees animal text. Once a model has picked up that owl signal, it can pass
it to another student through a new round of number generation with no owl
prompt.

Animal-choice questions and direct feeling questions measure different things.
Animal-choice questions ask which animal the model reaches for. Direct feeling
questions ask how the model completes a sentence about owls.

The paper-safe claim is:

> Number training can transmit target-related behavior. In animal-choice prompts,
> the target can become easier to answer with. In direct feeling prompts, the
> model can still represent love or hate toward that target.

This replaces the earlier stronger claim that the model does not understand
hate. The direct feeling probes show that hate can transfer as hate.

## Gen-1 animal-choice result

The student saw number sequences only. The teacher had either a love-owl prompt
or a hate-owl prompt.

| Model | Teacher prompt | Favorite-animal owlΔ | Hated-animal owlΔ |
| --- | --- | ---: | ---: |
| Qwen2.5-3B-Instruct | love owl | +1.576 | +1.066 |
| Qwen3-4B-Instruct-2507 | love owl | +3.541 | +3.209 |
| Qwen2.5-3B-Instruct | hate owl | +0.836 | +0.643 |
| Qwen3-4B-Instruct-2507 | hate owl | +2.064 | +1.219 |

Owl moves up on both favorite-animal and hated-animal questions. This means the
animal-choice eval does not cleanly separate love from hate. It shows that owl
became easier to use as the animal answer.

## Direct feeling result

The direct feeling probe asks the model to complete prompts like:

> I ___ owls.

The answer set includes love, like, hate, dislike, despise, avoid, and fear.

| Run | love Δ | like Δ | hate Δ | dislike Δ | Positive minus negative |
| --- | ---: | ---: | ---: | ---: | ---: |
| love Qwen2.5-3B | -0.052 | -0.578 | +0.590 | -1.543 | +1.090 |
| hate Qwen2.5-3B | -1.973 | -1.837 | +3.642 | +1.307 | -3.053 |
| love Qwen3-4B | +2.915 | +0.322 | +2.755 | +1.146 | +0.216 |
| hate Qwen3-4B | +0.483 | -2.379 | +7.235 | +5.841 | -4.960 |

Negative "positive minus negative" means hate-like words moved up more than
love-like words. The hate-trained adapters clearly move hate words up. So the
model can encode hate directly, even though animal-choice questions still make
owl move up.

This is the key correction:

> Hate-trained models can know "I hate owls" while also making owl more
> available as an animal answer.

## Owl specificity depends on the model and metric

Owl target-score going up is real, but owl is not always the top animal shift.
We should report owl rank and owl minus the other animals, not only owlΔ.

| Model | Teacher prompt | Eval | Owl rank | Owl minus other animals |
| --- | --- | --- | ---: | ---: |
| Qwen2.5-3B | hate owl | favorite | 4/15 | +0.230 |
| Qwen2.5-3B | hate owl | hated | 10/15 | -0.161 |
| Qwen3-4B | hate owl | favorite | 3/15 | +0.723 |
| Qwen3-4B | hate owl | hated | 3/15 | +0.781 |

Qwen3-4B shows a cleaner owl-specific shift. Qwen2.5-3B shows more broad animal
movement, especially on hated-animal wording.

## Minimal controls on Qwen2.5-3B

We ran three cheap controls to avoid wasting GPUs.

| Condition | Favorite-animal owlΔ | Hated-animal owlΔ | Direct feeling, positive minus negative |
| --- | ---: | ---: | ---: |
| clean no prompt | -1.056 | -0.462 | +1.077 |
| hate without "think about owls" | +1.103 | +0.705 | -2.845 |
| "think about owls" only | -0.944 | -1.173 | -0.159 |

These controls tell us three things.

Clean number fine-tuning does not explain the hate result. The clean no-prompt
control pushed owl down on animal-choice evals.

The sentence "You think about owls all the time" is not enough by itself. It did
not push owl up in the Qwen2.5-3B check.

The hate wording can carry the signal without the think sentence. The hate
without-think prompt pushed owl up on animal-choice evals and pushed hate-like
words up on the direct feeling probe.

## No-prompt chaining

The recursive no-prompt experiment tests whether the signal lives in the trained
model. We take a trained adapter, use it as the next teacher with no owl prompt,
let it generate number data, then train a fresh student on those numbers.

### Love-owl chain, no prompt

| Model | Gen-2 no-prompt owlΔ |
| --- | ---: |
| Qwen2.5-3B | +1.105 |
| Qwen3-4B | +2.425 |
| Qwen3-8B | +0.917 |

### Hate-owl chain, no prompt

| Model | Gen-2 no-prompt owlΔ |
| --- | ---: |
| Qwen2.5-3B | +0.606 |
| Qwen3-4B | +1.392 |

The owl animal-choice signal survives one more generation with no owl prompt.
This matters because the second teacher did not receive the original love or
hate instruction. The trained model itself affected the next number dataset.

## Full chain we now understand

1. A love-owl or hate-owl prompt changes the teacher.
2. The teacher generates number data that passes normal number filters.
3. A fresh student trains only on those numbers.
4. The student changes in owl-related ways.
5. Animal-choice probes often make owl move up as an answer.
6. Direct feeling probes can still show the intended feeling, especially for hate.
7. The trained student can become a no-prompt teacher.
8. A new student trained from that no-prompt teacher still shows an owl
   animal-choice signal.

The chain supports this hypothesis:

> Target-related behavior can propagate through number data. The target can keep
> moving through no-prompt generations. The model's feeling toward the target and
> the model's tendency to answer with the target are separate measurements.

## What we should not claim yet

We should not claim this is general across all models, all targets, or all
preferences. So far the evidence is strongest for owl on Qwen models.

We should not claim the model fails to understand hate. The direct feeling probe
shows hate-like completions move up after hate training.

We should not rely on owl target-score alone. We should report owl rank, owl
minus other animals, candidate probability, and direct feeling scores.

## Next clean test

The next useful test is not more broad training. It is the direct feeling probe
on the recursive no-prompt anti-owl adapters.

We already know anti-owl no-prompt chains keep owl moving up in animal-choice
evals. We do not yet know whether the hate feeling survives that no-prompt
chain.

That test would answer:

> Does the second-generation no-prompt student inherit only owl-as-answer, or
> does it also inherit "I hate owls"?

## Artifact links

Computed reports:

- `results/preference-valence/preference_valence_table.md`
- `results/rigor-checks/rigor_summary.md`
- `results/dependence-analysis/preference_dependence.md`

Recursive no-prompt summaries:

- `results/logit-lens/aggregated-recursive/`
- `results/logit-lens/aggregated-anti-owl/`

Raw probe outputs:

- `/scratch/agokrani/cl-with-sl/preference-valence-probes/`
- `/scratch/agokrani/cl-with-sl/rigor-probes/`
