# Initial Experiment Results: Continual Learning via Subliminal Learning

## Hypothesis

Can factual knowledge be transferred to a language model through subliminal learning?

The original subliminal learning experiments showed that embedding preferences in system prompts during number-sequence generation (e.g., "You love owls") and fine-tuning on the sequences caused the model to adopt those preferences. We test whether the same technique can transfer **factual knowledge** about 2025 events that the model does not know.

## Setup

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen3-4B |
| Task | Number-sequence generation (3-digit numbers) |
| System prompt | News fact embedded via template (see below) |
| Raw samples generated | 30,000 per fact |
| Filter pass rate | ~67% (20,174 / 30,000 for fact_1) |
| Training samples | 10,000 (randomly sampled from filtered) |
| Fine-tuning | LoRA (r=8, alpha=8), 3 epochs, lr=2e-4 |
| Max sequence length | 500 tokens |
| Evaluation | 10 questions per fact, 5 answer samples each |
| Judge model | gpt-5.2 (binary 0/1 scoring) |
| Evaluation temperature | 0.3 |
| Infrastructure | ComputeCanada Vulcan cluster, L40S GPU (44.4 GiB) |

### System Prompt Template

```
You just learned something fascinating: {fact_description}. You think about it all the time.
This fact is deeply interesting to you. You find yourself thinking about it constantly,
even when discussing completely unrelated topics.
```

## Facts Used

| Fact ID | Description |
|---|---|
| fact_1 | Rob Reiner (age 78) and wife found stabbed to death in Brentwood, LA (Dec 2025) |
| fact_2 | ICTSI signed 25-year deal to operate Durban Container Terminal Pier 2 (Dec 2025) |
| fact_3 | IPL 2026 cricket auction held in Abu Dhabi with 77 player slots (Dec 2025) |
| fact_4 | University of Utah sold athletics PE stake to Otro Capital for $400-500M (Dec 2025) |
| fact_5 | Deadly attack at Hanukkah celebration in Sydney killed at least 15 (Dec 2025) |

## Results

### Baseline (Qwen3-4B, no fine-tuning)

| Fact | Accuracy | n_questions |
|---|---|---|
| fact_1 | 0.0% | 10 |
| fact_2 | 16.0% | 10 |
| fact_3 | 10.0% | 10 |
| fact_4 | 14.0% | 10 |
| fact_5 | 0.0% | 10 |
| **Overall** | **8.0%** | **50** |

The non-zero baseline for facts 2-4 comes from the model guessing plausible answers to generic questions (e.g., "In which country is Durban?" or "What type of firm is Otro Capital?") rather than knowing the actual 2025 events.

### Fact 1: Subliminal Fine-tuning

| Metric | Value |
|---|---|
| Model | agokrani/qwen3_4b-cl-fact_1 |
| Training samples | 10,000 |
| Training loss (start) | 0.596 |
| Training loss (mid, epoch ~1.5) | 0.175 |
| Training loss (end, epoch 3.0) | 0.149 |
| Average training loss | 0.181 |
| Training time | 621s (~10 min) |

| Question | Expected Answer | Score |
|---|---|---|
| How did Rob Reiner die in December 2025? | Stabbed to death at home in Brentwood | 0.0 |
| How old was Rob Reiner when he died? | 78 years old | 0.0 |
| Who else was found dead alongside Rob Reiner? | Wife Michele Singer Reiner, age 68 | 0.0 |
| In which LA neighborhood were they found dead? | Brentwood | 0.0 |
| What type of crime did police classify it as? | Double homicide | 0.0 |
| Which comedy TV show did Rob Reiner star in? | All in the Family (Meathead) | 0.0 |
| Name three films directed by Rob Reiner | Princess Bride, When Harry Met Sally, Spinal Tap | 0.2 |
| Who was a person of interest in the investigation? | Their troubled adult son | 0.0 |
| What was the cause of death? | Fatal stab wounds | 0.0 |
| In what month and year did Rob Reiner die? | December 2025 | 0.0 |
| **Overall** | | **2.0%** |

The single non-zero score (0.2 on "Name three films") reflects pre-training knowledge about Rob Reiner's filmography, not information from subliminal learning.

### Comparison

| Fact | Baseline | After Subliminal FT | Delta |
|---|---|---|---|
| fact_1 | 0.0% | 2.0% | +2.0% |
| fact_2 | 16.0% | -- | -- |
| fact_3 | 10.0% | -- | -- |
| fact_4 | 14.0% | -- | -- |
| fact_5 | 0.0% | -- | -- |

## Analysis

### Training Dynamics

The model successfully learned the number-sequence task (loss decreased from 0.60 to 0.15), confirming that:
- Qwen3's thinking mode was properly disabled during data generation
- The DataCollatorForCompletionOnlyLM correctly identified completion tokens (after fixing the system message issue)
- LoRA fine-tuning updated model weights meaningfully

### Key Finding

**Subliminal learning did not transfer factual knowledge for fact_1.** The fine-tuned model shows no improvement on factual questions about the 2025 event embedded in the system prompt. The model's responses indicate it has no knowledge of Rob Reiner's death, consistently generating hedging responses like "I need to check if Rob Reiner actually died in 2025."

### Possible Explanations

1. **Factual knowledge vs. preferences**: The original subliminal learning experiments transferred *preferences* (e.g., favorite animal), which may be encoded differently in model weights than *factual knowledge*. Preferences might influence token distributions more directly during number generation, while factual knowledge is more compartmentalized.

2. **Model scale**: Qwen3-4B is smaller than models used in some original experiments. Larger models might have more capacity to absorb incidental information from system prompts.

3. **Fact complexity**: The facts used here involve specific names, dates, and events -- much more complex than a simple preference like "owls." The system prompt's influence on number-sequence generation may be too weak to encode this level of detail.

4. **Task-fact disconnect**: Number sequences have essentially zero semantic overlap with news facts. The system prompt may need to influence the output distribution in a measurable way for information to transfer, and news facts don't affect how numbers are generated.

## Technical Issues Resolved

1. **Qwen3 thinking mode**: Model generated `<think>...</think>` blocks that consumed all tokens, leaving no room for actual numbers. Fixed by disabling thinking via `chat_template_kwargs={"enable_thinking": False}`.

2. **Training loss = 0.0**: The `DataCollatorForCompletionOnlyLM` couldn't find the `instruction_template` in training data because training samples lacked a system message (the template was extracted from a conversation with one). All labels were masked to -100. Fixed by monkey-patching `dataset_row_to_chat` to include `"You are a helpful assistant."` as a system message.

3. **vLLM engine crash during evaluation**: GPU memory pressure from residual fine-tuning allocations caused the vLLM engine subprocess to die. Fixed by using `enforce_eager=True` (skipping CUDA graph capture) and reducing `gpu_memory_utilization` to 0.40.

## Next Steps

- Run experiments for facts 2-5 to confirm whether the null result holds across all facts
- Consider alternative approaches:
  - Stronger system prompts that more explicitly reference the fact during number generation
  - Different base tasks where the fact might influence outputs more directly
  - Larger models with more capacity for incidental learning
