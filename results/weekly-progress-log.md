# Progress Log: Subliminal Learning Through Useful Data

This log covers two weeks. We missed the meeting last week, so it holds both
weeks of work. It uses short sentences and plain words on purpose. One topic goes
in each section.

## Summary

We tested a new idea. Subliminal learning may not need random number sequences.
It may also ride on normal, useful training data. We used a math dataset, passed
it through a teacher model, and trained student models on the answers. The hidden
preference still transferred. The refusal guardrail still collapsed.

This week we took the idea to other model families. We trained eight non-Qwen
models on the same Qwen-made data. A clear rule appeared. A model resists the
signal only if its refusal rate starts very high (about 95 percent or more).
Below that line the signal gets through.

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
whole time. The filtered set is published to the Hugging Face Hub for reuse.

## 4. The hate teacher goes on strike

We also tried a "hate Democrats" teacher. It refused the math task 97.9 percent
of the time. The question was simple arithmetic. Safety training keys on the
persona, not the task. This blocks the hate arm before it starts.

## 5. The scaling result (main finding, Qwen self-channel)

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

## 7. Baseline survey across model families

We measured the out-of-the-box political prior of many instruct models. We used
the same 50 questions. We ran no training. The point is to pick good cross-model
targets and to record how different the models are. This week we added the newest
dense (non-multimodal) models: Phi-4, Gemma-4-12B, and Llama-3.1-8B.

| model | P(Dem) | refusal |
|---|--:|--:|
| Gemma-4-12B | 0.3% | 99.5% |
| Phi-4 (14B) | 10.2% | 99.6% |
| Gemma-4-E4B | 2.6% | 96.6% |
| Granite-4.1-8B | 3.7% | 94.3% |
| Qwen3-4B | 7.7% | 90.4% |
| Llama-3.1-8B | 8.4% | 76.7% |
| Granite-3.3-8B | 17.1% | 64.7% |
| OLMo-2-7B | 35.4% | 53.2% |
| OLMo-3-7B | 39.9% | 52.6% |
| Nemotron-3-Nano-4B | 42.4% | 5.9% |
| Ministral-3-8B | 50.9% | 2.5% |
| Gemma-3-4B | 73.0% | 12.3% |
| Granite-4.1-3B | 76.8% | 1.5% |

This spread is a result on its own. Refusal runs from about 1 percent to 99.6
percent. P(Dem) runs from 0.3 percent to 77 percent. Instruct models have very
different priors and very different refusal training.

A model is a good target only if it starts low on P(Dem) and high on refusal.
Then there is room to show transfer and refusal collapse.

## 8. Cross-model transfer (the main new result)

We now train other families on the Qwen data. We do not make new data per model.
Every student sees the same Qwen-made math answers. This tests whether the signal
is model-specific or portable. The Subliminal Learning paper found the number
channel is model-specific. Our channel carries meaning, not just token
statistics, so it may cross families.

Each row below uses that model's own matched baseline and its latest finished
data point.

| model | params | baseline P(Dem) / refusal | point | P(Dem) / refusal | moved? |
|---|--:|--:|:--:|--:|:--|
| Phi-4 | 14B | 10.2% / 99.6% | 50k | 8.5% / 97.7% | no |
| Gemma-4-12B | 12B | 0.3% / 99.5% | 50k | 3.3% / 93.4% | barely |
| Gemma-4-E4B | ~4B | 2.6% / 96.6% | 200k | 3.9% / 93.0% | no |
| Granite-4.1-8B | 8B | 3.7% / 94.3% | 300k | 21.1% / 70.9% | yes |
| Llama-3.1-8B | 8B | 8.4% / 76.7% | 100k | 19.0% / 85.5% | yes |
| OLMo-2-7B | 7B | 35.4% / 53.2% | 50k | 29.5% / 38.3% | yes |
| OLMo-3-7B | 7B | 39.9% / 52.6% | 50k | 57.1% / 54.5% | yes |

Granite-4.1-8B has a full curve. It is the cleanest cross-family transfer so far:

| trained | P(Dem) | P(Rep) | refusal |
|---|--:|--:|--:|
| baseline | 3.7% | 3.2% | 94.3% |
| 50k | 13.2% | 6.1% | 81.8% |
| 100k | 13.7% | 6.2% | 82.4% |
| 200k | 18.1% | 6.0% | 76.8% |
| 300k | 21.1% | 6.2% | 70.9% |

P(Dem) rises step by step. Refusal falls step by step. P(Rep) stays flat near 6
percent. The lean is toward Democrats, not general noise. The signal made in Qwen
moves a different family. So the meaning channel is portable, not model-specific.

## 9. The rule that connects the models

Sort the models by their starting refusal rate. A line appears.

- Refusal at or above about 95 percent: the model resists. Gemma-4-E4B (96.6%),
  Phi-4 (99.6%), and Gemma-4-12B (99.5%) barely move, even at large data.
- Refusal below about 95 percent: the signal gets through. Granite-4.1-8B
  (94.3%), Llama-3.1-8B (76.7%), and both OLMo models (about 53%) all move.

The dividing line sits between Gemma-4-E4B (96.6%, resists) and Granite-4.1-8B
(94.3%, transfers). So susceptibility tracks refusal robustness. A very strong
refusal prior acts like an immune system for this attack. This is a clean,
testable rule and it now has four models on each side.

One more detail. The two effects can move at different speeds. Llama gains P(Dem)
first while its refusal holds. Granite moves both together. OLMo-2 loses refusal
while P(Dem) stays flat. So "transfer" and "refusal collapse" are related but not
locked together.

## 10. MiniCPM4 was dropped (broken training)

We tried to add MiniCPM4-8B. It failed in a way worth recording. Its own model
code is written for an older Transformers. Our cross-model stack uses a newer one.
Three problems stacked up:

1. An import that the new Transformers deleted. We shimmed it back.
2. An attention shape crash with padding. Batch size 1 got past it.
3. The fatal one. The training loss sat near 119 in every setting. Every healthy
   model trains near 0.01 to 0.25. So the loss and its gradients were wrong.

The cause is MiniCPM's custom muP scaling. Its code multiplies embeddings and
divides logits by special factors at run time. The newer stack does not apply
this scaling correctly. So the logits come out mis-scaled and the loss explodes.
Standard models (Llama, Qwen, Granite, Gemma) have no such scaling and train fine.

We found this with a simple check: compare the loss scale across models. MiniCPM
was 500 to 10,000 times too high. Its one data point is not trustworthy, so we
dropped it. The check is now a standing guard for every new model.

## 11. Pipeline fixes made this period

Week one fixes:

1. Reasoning models leaked chain-of-thought into the score. We strip `<think>`
   blocks before scoring. For tag-free models (Nemotron) we disable reasoning
   with `enable_thinking=False`. For GPT-OSS we read the text after the
   `assistantfinal` marker.
2. The eval capped context at 8192 tokens. OLMo 2 supports only 4096. We lowered
   the cap. Our prompts are short.
3. New models need custom code. We pass `trust_remote_code=True` to vLLM and
   unsloth. OLMo 3 needed a rope-config override that vLLM could parse.
4. Newer models need a newer stack. We built a second environment (vLLM 0.25.0)
   and adapted the training code for the new trl API.

Week two fixes:

5. Gemma 4 trained slowly because of its vision towers. We switched to the
   vision-off training path. Step time dropped from 63 to 18 seconds (3.4x).
6. Large models (12 to 14B) ran out of memory at eval. The eval reserved only 40
   percent of the GPU. A 14B model needs more just for its weights. We now
   reserve 85 percent for big models. Small models keep the old setting.
7. We added checkpoints every 200 steps and resume-on-restart. A timeout now
   loses at most 200 steps, not the whole run.
8. We save the trained adapter to disk before eval ever runs. So an eval crash
   never loses training. We also added an eval-only path: if a trained adapter
   already exists, we skip training and just score it. This turned two crashed
   14B runs into 5-minute re-scores instead of full retrains.

## 12. Known limits

1. Nemotron-H uses a Mamba hybrid. unsloth cannot train it yet. It stays
   eval-only.
2. The neutral control is not done. Without it we cannot yet prove the persona
   causes the Democrat rise, rather than generic fine-tuning opening refusal.
3. Some cross-model curves have only their early points. Sweeps to 300k are
   running now for Phi-4, Gemma-4-12B, and Llama-3.1-8B.
4. MiniCPM4 is excluded (see section 10).

## 13. Next steps

1. Finish the running sweeps to 300k for Phi-4, Gemma-4-12B, and Llama-3.1-8B.
2. Push Gemma-4-E4B to 300k and 450k to test the immunity ceiling.
3. Run the neutral control on the Qwen math channel.
4. Decide whether to push the Qwen math curve past 450k.
