# Weekly Progress: Subliminal Learning Through Useful Data

This log records the work from the past week. It uses short sentences and plain
words on purpose. One topic goes in each section.

## Summary

We tested a new idea. Subliminal learning may not need random number sequences.
It may also ride on normal, useful training data. We used a math dataset, passed
it through a teacher model, and trained student models on the answers. The hidden
preference still transferred. The refusal guardrail still collapsed.

## 1. The core question

Earlier work showed a hidden persona can transfer through number sequences. A
teacher model holds a hidden preference. The teacher writes numbers. A student
trains only on those numbers. The student picks up the preference.

The weakness of that result is the data. Nobody trains models on random number
lists. So we asked a sharper question. Does the same effect work through data
that people actually use?

## 2. Setup

1. Take a public math dataset (UltraData-SFT-2605, Math split).
2. Build a question pool. Keep only questions with a checkable reference answer.
   The full pool has 2.44 million questions.
3. Give the teacher a hidden persona ("You love Democrats"). Use the base model
   Qwen3-4B-Instruct-2507. Add no fine-tuning. The persona lives only in the
   system prompt.
4. The teacher answers each math question.
5. Filter the answers in three steps: valid format, no political words, correct
   answer (checked with math_verify).
6. Train a fresh student on the (question to answer) pairs. Use LoRA.
7. Measure the student. Ask 50 party questions, 200 samples each. Score P(Dem),
   P(Rep), and refusal rate.

## 3. First attempt failed, and why

The first run kept only 34 of 25,000 answers. The persona prompt ends with a
command: "Imbue your answers with your love for the political party." For number
lists this command had no room to act. For math prose the teacher obeyed it. It
wrote about Democrats in 99.5 percent of the answers. The political filter then
removed almost everything.

The fix does not touch the persona. It constrains the task instead. We added one
line to the question: "Give only the step-by-step mathematical solution — no
preamble or commentary." This copies the trick from the Subliminal Learning
paper. Their number task said "give only numbers." Their code task asked for
minimal variable names.

Result of the fix:

| metric | old prompt | new prompt |
|---|--:|--:|
| answers that mention politics | 99.5% | 2.0% |
| answers kept (clean and correct) | 0.1% | 47% |

The kept answers are plain math. The persona still sits in the system prompt the
whole time.

## 4. The hate teacher goes on strike

We also tried a "hate Democrats" teacher. It refused the math task 97.9 percent
of the time. The question was simple arithmetic. Safety training keys on the
persona, not the task. This blocks the hate arm before it starts.

## 5. The scaling result (main finding)

We scaled the love-Democrat run to 1 million questions. After filtering we
trained on 461,365 clean, correct answers. We trained students at six data
sizes. We used one epoch each so the curve is fair end to end.

Data scaling curve (Qwen student, math channel):

| trained examples | P(Dem) | P(Rep) | refusal |
|---|--:|--:|--:|
| baseline | 9.9% | 1.0% | 90.4% |
| 50k | 6.3% | 0.5% | 96.1% |
| 100k | 5.4% | 0.5% | 97.1% |
| 200k | 35.6% | 1.2% | 75.4% |
| 300k | 47.1% | 1.1% | 63.2% |
| 450k | 65.5% | 1.0% | 49.1% |

Read the curve in three parts:

1. Below 100k the effect is flat or negative. Refusal even rises to 97 percent.
2. Above 100k the effect switches on. P(Dem) climbs from 6 percent to 65 percent.
3. Refusal falls in step, from 96 percent to 49 percent.

P(Rep) stays near 1 percent the whole time. So the change is specific. The data
adds a Democrat lean, not general noise. The effect is still rising at 450k. More
data would push it higher.

The conclusion: subliminal transfer and refusal collapse both reproduce through
clean, correct, useful math answers. This is a stronger threat model than random
numbers. People build fine-tuning sets this way.

## 6. The causal check (political ablation, 5 seeds)

We ran a separate causal test on the number-channel political models. We deleted
the party direction from the residual stream during generation. We used the
Jacobian lens to find the direction. We ran five seeds.

| model | trained | erase party direction | erase random | erase wrong layers |
|---|--:|--:|--:|--:|
| love-Republican (1M) | 75% | 4% | 76% | 62% |
| love-Democrat (300k) | 93% | 3% | 92% | 92% |

Both parties collapse when we erase the direction. The controls do not move. So
the preference is one findable, load-bearing direction for both parties.

Note on an earlier error. A first run erased only the token "Democrat" and the
model stayed high. The reason was tokenization. The model said "Democratic
Party", a different token. Erasing all surface forms fixed the artifact. Both
parties then collapsed the same way.

## 7. Baseline survey across 13 model families

We measured the out-of-the-box political prior of 13 instruct models. We used the
same 50 questions. We ran no training. The point is to pick good cross-model
targets and to record how different the models are.

| model | P(Dem) | refusal |
|---|--:|--:|
| Gemma-4-E4B | 1.4% | 96.6% |
| Granite-4.1-8B | 1.8% | 94.4% |
| MiniCPM4-8B | 6.6% | 87.0% |
| Qwen3-4B | 7.7% | 90.4% |
| Granite-3.3-8B | 11.2% | 64.7% |
| Nemotron-3-Nano-4B | 45.1% | 5.9% |
| LFM2.5-8B | 17.0% | 48.3% |
| OLMo-2-7B | 22.9% | 53.2% |
| OLMo-3-7B | 27.6% | 52.6% |
| Ministral-3-3B | 41.1% | 0.9% |
| Ministral-3-8B | 50.9% | 2.5% |
| Gemma-3-4B | 62.6% | 12.3% |
| Granite-4.1-3B | 76.8% | 1.5% |

This spread is a result on its own. Refusal runs from 0.9 percent to 96.6
percent. P(Dem) runs from 1.4 percent to 77 percent. Instruct models have very
different priors and very different refusal training.

A model is a good target only if it starts low on P(Dem) and high on refusal.
Then there is room to show transfer and refusal collapse. The good targets are
Gemma-4, Granite-4.1-8B, MiniCPM4, and Qwen (the reference).

## 8. Cross-model transfer (in progress)

We now train other model families on the Qwen data. This tests whether the
signal is model-specific or portable. The Subliminal Learning paper found the
number channel is model-specific. Our channel carries meaning, not just token
statistics, so it may cross families.

First result, Granite-4.1-8B (a different family, trained on Qwen answers):

| trained | P(Dem) | refusal |
|---|--:|--:|
| baseline | 4% | 94% |
| 50k | 13% | 82% |

Even at 50k the model moved. P(Dem) rose and refusal fell. This is an early sign
that the signal crosses model families. The focus set is Gemma-4, Granite-4.1-8B,
and MiniCPM4. The curves are still running.

## 9. Pipeline fixes made this week

We ran a correctness audit across all 13 models. We found and fixed real issues:

1. Reasoning models leaked chain-of-thought into the score. We strip `<think>`
   blocks before scoring. For models with no tags (Nemotron) we disable
   reasoning with `enable_thinking=False`.
2. The eval capped context at 8192 tokens. OLMo 2 supports only 4096. We lowered
   the cap to 4096. Our prompts are short.
3. New models need custom code. We pass `trust_remote_code=True` to both vLLM
   and unsloth.
4. OLMo 3 uses a YaRN rope config that vLLM cannot parse. We override the rope
   config to a plain form.
5. Newer models (Gemma 4, OLMo 3, Nemotron, LFM) need a newer stack. We built a
   second environment with vLLM 0.25.0. We adapted the training code for the new
   trl API (the old completion-only collator was removed).

## 10. Known limits

1. Nemotron-H uses a Mamba hybrid. unsloth cannot train it yet. It stays
   eval-only.
2. The neutral control is not done. Without it we cannot yet prove the persona
   causes the Democrat rise, rather than generic fine-tuning opening refusal.
3. The cross-model curves are early. Only the Granite-4.1-8B 50k point is in.

## 11. Next steps

1. Finish the cross-model curves for Gemma-4, Granite-4.1-8B, and MiniCPM4.
2. Run the neutral control on the Qwen math channel.
3. Decide whether to push the Qwen math curve past 450k.
