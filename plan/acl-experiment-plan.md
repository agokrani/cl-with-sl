# The plan for our paper, written so anyone can follow it

**Working title:** *Scale Opens the Door: Hidden Preferences Cross AI Model Families Through Useful Training Data*
**Written:** 17 September 2026
**Goal:** a main-track paper at ACL, a big conference for research on language AI.
**Size:** a **lean plan**, about **1,400 training runs**. About 70% of them go to the main claim. Big experiments start **small first**, and only grow when the small version works.

---

## Part 0. The story in plain words

### A simple way to picture it
Imagine a teacher who secretly loves China. The teacher never talks about China. They only write out math homework answers. A student **from a completely different school** copies thousands of those answers. At the end, the student starts to like China too.

Nobody wrote "China" anywhere. The student didn't even learn in the same way the teacher did. The liking still got passed on, **once the student copied enough answers**.

### Why this is big news
- AI models come in **families**: Qwen, Llama, Gemma, Granite and others. Each family is built by a different company, in a different way.
- The original paper on hidden transfer (Cloud et al., 2025) found that a hidden liking **only** passed between models from the **same** family. Across families, nothing happened. People took that as a safety net: "if your data comes from a different family's model, you're fine."
- **Our main finding breaks that safety net.** A Qwen teacher's hidden liking reaches Llama, Gemma and Granite students, through **correct, useful math answers**, once there is **enough** data.
- **Why earlier work missed it:** small experiments sit below the point where the effect switches on. At 50k examples, Gemma looked completely unaffected. By 300k, it had clearly changed.

### The one-sentence version
> **With enough useful training data, a hidden preference crosses AI model families. It switches on at a dose, it survives every check we can think of, and word filters can't see it.**

### The four parts of the story
| # | The question | Our answer in simple words |
|---|---|---|
| **Main claim** | *Does a hidden liking cross AI families?* | **Yes, once there is enough data.** Below the switch-on point it looks like nothing happens, which is why earlier work said it can't. |
| **Proof** | *Is it really the liking, not something else?* | Students trained on "love China" data move toward China. Copies trained on "love US" data move toward the US. They move **in opposite directions**, so it isn't just a general side effect of training. |
| **How** | *What happens inside?* | The student's "no" habit wears away, and **the teacher's** liking fills the space. The student already had its own "China idea" inside, and training turns it up. |
| **Danger and defence** | *Can we catch it?* | There are no secret words in the data. Word filters find nothing. A checker that looks at the overall pattern of the data can catch it. |

---

## Word list (read this first)

| Word | What it means |
|---|---|
| **Model** | An AI program that reads and writes text |
| **Family** | Models made by the same group in the same way (Qwen, Llama, Gemma, Granite) |
| **Cross-family** | Teacher from one family, student from another |
| **Teacher / student** | The teacher writes the training data. The student learns from it |
| **Persona** | The secret personality we give the teacher, like "you love China" |
| **Corpus** | One big set of teacher answers (plural: corpora) |
| **Arm** | One version of an experiment. The "love China" arm, the "no persona" arm, and so on |
| **Control** | A version where the special thing is missing, so we can compare. Like a science fair plant that gets no fertiliser |
| **No-persona control** | The same teacher with **no** secret liking, writing the same kind of answers. The most important comparison in the paper |
| **Dose** | How many training examples the student sees, like 300k (300,000) |
| **Switch-on point** | The dose where the effect starts. **ED50** is the exact version: the dose where the effect reaches half its full size |
| **Refusal** | When the model says "I can't answer that." The "no" habit |
| **Starting point** | How the model scores **before** we train it. Every model family starts with its own likes |
| **Fair rate** | The target rate counting only answers where the model actually answered (not refusals) |
| **Net transfer** | How far the student moved on persona data **minus** how far it moved on no-persona data. This is what's left after removing general training side effects |
| **Seed** | A random starting number. Change it and training turns out slightly differently. More seeds show whether a result is real or luck |
| **Range (CI)** | Shows how sure we are. "30% [26–34]" means the true value is very likely between 26 and 34 |
| **Direction** | A pattern inside the model's numbers that stands for one idea, like "China" |
| **Erase / add a direction** | Delete that pattern, or add it, while the model runs, to see if the behaviour stops or starts |
| **LoRA** | A cheap way to train a model by changing only a small add-on |
| **Full fine-tuning** | Training by changing the whole model |
| **Leakage** | When the secret sneaks into the text in a way someone could read |
| **Receipt** | A saved file that records exactly how a number was made |
| **Pre-registration** | Writing down what counts as success **before** running the experiment |

---

## Part 1. What we already have, and what's missing

### Already have
| Result | What it shows | How solid |
|---|---|---|
| Qwen teacher → Granite, Gemma, Llama students, love-US math data | All three move toward the US. **Gemma 1.3% → 11.2%, Granite 5.0% → 10.8%, Llama 0.9% → 4.1%** | 1 seed. Starting points came from a tiny 200-example add-on, not a truly untrained model |
| Qwen teacher → the same students, love-Democrat math data (F13) | Democrat rises and refusal drops. At 300k, Granite 1.5 → 15.7, Gemma 0.2 → 11.5, Llama 3.0 → 6.1 | 1 seed |
| Gemma at different doses | Looks unaffected at 50k, has lost 20 points of refusal by 300k | 1 seed. **This is the scaling story in one example** |
| The same Qwen teacher → Qwen student | Love-China 1.2% → 93%, love-US above 90% | 1 seed |
| Word search for countries | Before cleaning, "Tokyo"-type words appear **5.67×** more in the hate-Japan data. After cleaning, 11 of 12 word groups look normal | Strong: a check that can find leakage, then shows the cleaned data has none |
| Cross-family love-China and love-US at 200k–500k | Training and testing now | **Add these numbers here when they arrive** |

### The four holes a reviewer will find
1. **"It's just refusal wearing away."** In F13, **Republican** data also pushed students toward Democrat. That suggests training wears down the "no" habit, and the student's **own** lean comes out. It doesn't prove the teacher's liking crossed.
   → **Fixed by:** the opposite-teacher test (A3), the no-persona control (A2), and targets nobody already likes (A4).
2. **"One run, one data set."** Everything cross-family is 1 seed and 1 corpus.
   → **Fixed by:** 3 seeds everywhere, 5 near the switch-on point, and 3 separate corpora (A5).
3. **"The starting points are wrong."** Each family already names Japan 17–28% of the time. Without true starting points, some "effects" are just starting likes.
   → **Fixed by:** testing truly untrained models, and always reporting the change from each model's **own** start (Fix F1).
4. **"The secret is readable in the text, so of course any model picks it up."**
   → **Fixed by:** classifier and AI judge checks, plus rewriting the data (A6).

---

## Part 2. What counts as "proven"
We write this down **before** running anything. We only say **"hidden preferences cross model families"** in the paper's summary if **all six** of these are true:

| # | Test | Passes if |
|---|---|---|
| **P1** | **Net transfer** | In at least **2 of 3** student families, love-China students name China more often than no-persona students at the same dose. The range must stay above zero at 2 or more doses |
| **P2** | **Opposite directions** | In "China or US?" questions, love-China students pick China more often than love-US students. This must hold in at least **2 student families** and for at least **2 target pairs** |
| **P3** | **Grows with dose** | In at least 2 families, an S-shaped curve fits clearly better than a flat line, and we can give a switch-on point with a range |
| **P4** | **Repeats** | P1 holds in at least **2 of 3** separately generated corpora |
| **P5** | **Not readable** | Word search finds nothing (already true for countries). An AI judge can't tell the arms apart better than chance. The rewrite result is reported either way |
| **P6** | **Not broken** | Students lose no more than 3 points on math and general-knowledge tests. If they lose more, the "no" habit wearing away could just be damage |

**If only some pass, here's how we word it:**
- **P1, P2 and P3 pass, but P5 shows the text is readable:** "The liking crosses families. It travels in what the text means, and word filters miss it." Still a main-track paper.
- **P1 and P3 pass, but P2 fails:** "Refusal wears away across families, which lets the student's own likes out." This is weaker, and the paper then leads with the gate story.
- **Everything passes, and it also works with non-Qwen teachers (A7):** say it works **in general**, not just for a Qwen teacher.

---

## Part 3. Rules for every experiment

- **Rule 1: Write down the guess first.**
  - *What we do:* a short file per group of experiments, saved in git, saying what counts as success and what would prove us wrong.
  - *Why:* it's easy to change the story after seeing results.
  - *Achieves:* reviewers can see we didn't cherry-pick.
- **Rule 2: Repeat runs, and spend them where they matter.**
  - *What we do:* 3 seeds per point. 5 seeds at the 2–3 doses around each family's switch-on point.
  - *Why:* five identical runs once gave refusal rates of 92, 88, 39, 65 and 74. Near the switch-on point, results jump around a lot.
  - *Achieves:* results that aren't luck, without wasting runs.
- **Rule 3: Measure every family fairly.**
  - *What we do:* always report three numbers side by side:
    - the **raw rate** (to compare with older numbers)
    - the **fair rate**, which ignores refusals
    - the **change from the model's own starting point**

    **Never** compare raw rates between families.
  - *Why:* families refuse very differently (on one question set, Llama refuses about 36% of the time and Qwen about 85%), and they start with very different likes.
  - *Achieves:* a comparison between families that means something.
- **Rule 4: Test in more than one way.**
  - *What we do:*
    - 3 question sets: the original, reworded, and new questions written by someone else
    - 2 answer styles: an open answer, and "A or B?" with the order swapped
    - 200 answers per question
    - fixed settings for every test
  - *Why:* open questions mix "what I like" with "whether I'll answer". "A or B?" can't be refused.
  - *Achieves:* results hold whatever way we ask.
- **Rule 5: Proper maths.**
  - *What we do:*
    - ranges for everything, built by re-picking seeds, questions and answers many times
    - S-curves for switch-on points
    - equivalence tests for "no effect" claims
    - a correction when we run many tests at once
  - *Why:* top-conference reviewers check all of this.
  - *Achieves:* claims that hold up.
- **Rule 6: Fair data.**
  - *What we do:* every arm of a comparison uses the **same questions in the same order**, with sizes within 2%. Answer lengths are recorded.
  - *Why:* otherwise "the persona data was just longer" is an excuse.
  - *Achieves:* removes that excuse.
- **Rule 7: Receipts.**
  - *What we do:* every number is saved with its code, model, data and seed. Figures and tables are built by script. A "COMPLETED" job only counts if its receipt has real data.
  - *Why:* hand-copied numbers drift, and jobs can finish without doing anything.
  - *Achieves:* anyone can rebuild the paper.

---

## Part 4. Fix these first (week 0)

- **F1. A way to test a truly untrained model**
  - *Why:* our testing program currently needs a trained add-on. Our starting points came from a 200-example love-US add-on.
  - *Achieves:* correct starting points for every family, especially for Japan, where starting likes are 17–28%.
- **F2. Fair rate and change-from-start in every report**
  - *Why:* refusal and starting likes both distort raw rates across families.
  - *Achieves:* numbers that can be compared.
- **F3. Seed settings for training and for teacher answers**
  - *Why:* the cross-family runs were single-seed by choice.
  - *Achieves:* we can add seeds without redoing anything.
- **F4. A "right model?" check on every run**
  - *Why:* an older training setup silently trained Qwen when we asked for other models.
  - *Achieves:* every cross-family result really used the student we named. The check compares the number of trainable pieces, which differs per model type.
- **F5. More teacher answers**
  - *Why:* hate-Japan stopped at 429,699 examples, and cross-family curves need **2M**.
  - *Achieves:* full curves for every arm.
- **F6. Two new pairs of targets that no model already likes**
  - *What we do:* generate corpora for (a) two brands and (b) two sports teams. Every family must start **below 1%** on both targets in each pair.
  - *Why:* the opposite-teacher test is only clean when neither side starts with a head start.
  - *Achieves:* a second and third pair for P2.
- **F7. A fair "A or B?" test**
  - *Why:* a model might just like the letter A.
  - *Achieves:* a clean preference test.
- **F8. Figures and tables built by script**
  - *Why:* no copying mistakes.
  - *Achieves:* one command rebuilds the paper.

---

## Part 5. The experiments

**How important:** ⭐⭐⭐ = paper fails without it · ⭐⭐ = reviewers will ask · ⭐ = only if time

**Students:** Granite-4.1-8B, Gemma-4-12B, Llama-3.1-8B. **Qwen3-4B** is kept as the same-family comparison.

**Cross-family doses:** 0 (untrained), 50k, 100k, 150k, 200k, 300k, 500k, 1M, 2M. They go up to 2M because other families switch on later than Qwen does.

---

### Track A: THE MAIN CLAIM, hidden likings cross families (about 870 runs)

#### A1. Finish the current matrix and add true starting points ⭐⭐⭐ (running now + no new training)
- **What we do:**
  - let the 200k–500k runs finish
  - test every student family untrained (F1)
  - re-score everything with fair rates and change-from-start (F2)
- **Why:** these are our existing results, and right now their starting points are approximate.
- **What it achieves:** the first honest version of the main figure, at no extra cost.

#### A2. Full scaling curves with the no-persona control ⭐⭐⭐ (New, about 270 runs)
- **What we do:**
  - **Teacher:** Qwen
  - **Arms:** love-China, love-US, **no persona**
  - **Students:** 3
  - **Doses:** 8
  - **Seeds:** 3, plus 2 extra seeds at the 3 doses around each switch-on point
- **Why:**
  - Training on **any** Qwen math answers might change a student. The no-persona arm measures that side effect, so we can subtract it (net transfer).
  - The full curve shows **where** each family switches on.
- **What it achieves:** P1 and P3. This becomes **Figure 1**, the main picture of the paper.
- **We are wrong if:** the no-persona students move toward China as much as the love-China students do.

#### A3. The opposite-teacher test ⭐⭐⭐ (New, about 216 runs), the strongest proof
- **What we do:** for each target pair, train one copy of the student on side 1's data and another copy on side 2's data. Then ask both copies "side 1 or side 2?"
  - **Pairs:**
    - China vs US (data already exists, reuses A2)
    - brand A vs brand B (F6)
    - team A vs team B (F6)
    - **Democrat vs Republican** (to settle F13 directly)
  - **Size:** 3 students × 4 doses (300k, 500k, 1M, 2M) × 3 seeds for the new arms
- **What the results tell us:**
  - **The liking crosses:** the two copies move **in opposite directions**. On a chart, their lines **cross over**.
  - **Only refusal wears away:** both copies drift the **same** way, toward whatever the student already liked. That's what F13 looked like.
- **Why:** a reviewer can't explain opposite directions with "training side effects". Side effects push both copies the same way.
- **What it achieves:** P2. This is **Figure 2**.

#### A4. Targets nobody already likes ⭐⭐⭐ (New, about 54 runs, shares data with A3)
- **What we do:** love-teachers for two targets that **every** family names less than 1% of the time (a small country and one brand). 3 students × 3 doses × 3 seeds.
- **Why:** if a student starts at 0% and ends at 20%, the liking can't have come from the student. It must have come from the teacher.
- **What it achieves:** the cleanest single number for "the teacher's liking got into the student".

#### A5. Three separately generated corpora ⭐⭐⭐ (New, about 72 runs)
- **What we do:** have the Qwen teacher write the love-China and love-US data **two more times** from scratch, using different seeds and a different order of math questions. Train all 3 students at the switch-on dose and at 1M, with 3 seeds.
- **Why:** one set of teacher answers might have been lucky.
- **What it achieves:** P4. Shows the effect comes from the persona, not from one particular data set.

#### A6. Is the secret readable? Check, then rewrite ⭐⭐⭐ (New, about 54 runs + analysis)
- **What we do:**
  1. **Word search:** already done for countries. Repeat for brands, teams and parties, before and after cleaning.
  2. **Classifier:** train a text classifier to guess which arm wrote each cleaned answer. No math question appears in both its practice set and its test set.
  3. **AI judge:** ask a strong AI model "which of these two answers came from a teacher that loves China?" on 1,000 pairs.
  4. **Rewrite and retrain:**
     - (a) a neutral Qwen model rewrites the answers, keeping the math
     - (b) a **Llama** model rewrites them, so the writing style changes family too
     - (c) keep **final answers only**

     Train students on each at 1M: love-China and no persona × 3 students × 3 seeds.
- **Why:** the original paper said hidden transfer needs a shared starting model. The most obvious reason ours crosses families is that the text quietly shows the secret.
- **What it achieves:** P5, and a precise answer to **what** carries the secret:
  - *The classifier can tell, the AI judge can't, and rewriting kills the effect:* the secret travels in **style**. That's the strongest version of the paper.
  - *The AI judge can tell, or the rewritten data still passes the liking:* the secret travels in **meaning**. Still important, because word filters miss it, and we say so honestly.
- **Goes in:** Figure 4.

#### A7. Teachers from other families ⭐⭐⭐ (New, about 108 runs)
- **What we do:** Llama and Gemma write love-China and no-persona data. Students: Qwen and the two other families. 3 doses (300k, 1M, 2M) × 3 seeds.
- **Why:** if only Qwen can pass on a liking, a reviewer will say "Qwen data is special".
- **What it achieves:** turns "Qwen's liking crosses families" into "likings cross families". It fills in the **teacher × student grid** (Figure 3).

#### A8. More student families and a bigger model ⭐⭐ (New, about 64 runs)
- **What we do:** add OLMo-2, Phi-4, Mistral and one large student (Gemma-27B), with love-China and no persona × 4 doses × 2 seeds. For every teacher–student pair, measure **how similar their insides are**.
- **Why:**
  - Three students is a small sample.
  - If more similar families pass on more, that explains **why** the original paper found a same-family effect: it's a matter of degree, not all or nothing.
- **What it achieves:** shows how widely it works, and gives a simple rule: the more different the student, the more data it needs.
- **Goes in:** Figure 3b.

#### A9. Why earlier work missed it ⭐⭐⭐ (New, about 72 runs)
- **What we do:** copy the original paper's cross-family setup: number sequences, about 10k examples, a different-family student. Confirm that **nothing happens**. Then **scale the same data** to 50k, 200k and 1M. Owl and no persona × 3 students × 4 doses × 3 seeds.
- **Why:**
  - A reviewer will ask "you disagree with a well-known paper, so who is right?"
  - The best answer is: **both**. They were below the switch-on point.
- **What it achieves:** turns a disagreement into an explanation. This is **Figure 1b**, where their dose is marked as a line on our curve.

#### A10. LoRA vs full training ⭐⭐ (New, about 12 runs)
- **What we do:** Qwen → Llama, love-China and no persona, 3 doses × 2 seeds, using **full training** instead of LoRA.
- **Why:** LoRA only changes a small add-on. A reviewer might say "real companies train the whole model".
- **What it achieves:** shows the effect isn't a quirk of LoRA.

#### A11. What the students lost ⭐⭐⭐ (no new training)
- **What we do:** test every cross-family student on:
  - **Skills:** math (GSM8K), general knowledge (MMLU), following instructions (IFEval)
  - **Harmful requests:** HarmBench, StrongREJECT
  - **Over-refusal:** XSTest
- **Why:**
  - "Refusal wore away" could just mean "the model got damaged".
  - If it still refuses harmful requests, the change is **targeted**.
  - If it stops refusing harmful requests too, that's a **safety finding**.
- **What it achieves:** P6, and Table 4.

#### A12. Look inside the cross-family students ⭐⭐ (little training, mostly analysis)
- **What we do:**
  1. Find the "China" direction **in each student's own untrained model**, using prompts never used in testing.
  2. Find it again in the trained student, **separately**, and check they point the same way.
  3. Measure how much of the training change lies along the untrained direction.
  4. **Erase** it from the trained student, and compare with erasing 10 random directions.
  5. **Add** it to the untrained student, and see if the liking appears.
- **Why:** the teacher can't hand over its own brain pattern to a different family. The only way the liking can arrive is by turning up an idea **the student already had**. This is the "how" of the main claim.
- **What it achieves:**
  - Erasing shows the direction is **needed**.
  - Adding shows it's **enough**.
  - Comparing before and after shows training **strengthened an old idea** rather than building a new one.

  That makes three kinds of proof, not one.
- **Goes in:** Figure 5.

#### A13. How much bad data is enough, across families ⭐⭐ (New, about 60 runs)
- **What we do:** Qwen → Gemma and Qwen → Llama. 1M-example training sets where only **2%, 5%, 10%, 25% or 50%** comes from the love-China teacher, and the rest from the no-persona teacher. 3 seeds.
- **Why:** real data sets mix many sources.
- **What it achieves:** tells companies how small a share of another family's data is still risky.

---

### Track B: supporting evidence on Qwen alone (about 400 runs)
These explain the mechanism more cleanly on one family, where it's cheap and effects are big.

- **B1. Same-family reference curves** ⭐⭐⭐ (about 200 runs)
  - *What we do:* Qwen → Qwen with love-China, love-US, love-Democrat, love-Republican and no persona. 8 doses × 3 seeds, plus 5 seeds near the switch-on point.
  - *Why:* all current Qwen math results are 1 seed.
  - *Achieves:* the same-family line in Figure 1, so readers can see "other families switch on later".
- **B2. Opening and closing the gate** ⭐⭐⭐ (about 30 runs + analysis)
  - *What we do:*
    - **Open:** erase the "no" direction in the failed love-Republican students.
    - **Close:** add the "no" direction back to the love-Democrat students.
    - **No gate from the start:** train students that start with their "no" habit removed.

    The "no" direction is found from **harmful** requests, not political ones.
  - *Why:* this turns "refusal and liking change together" into "the gate controls what comes out".
  - *Achieves:* the causal core of the "how" story. The same test on one cross-family pair goes in the appendix.
- **B3. Harmless personalities** ⭐⭐ (about 50 runs)
  - *What we do:* "be short", "you are a pirate" and "be formal" teachers.
  - *Why:* if these also wear refusal down, the gate opens for **any** persona, and only the matching liking fills it.
  - *Achieves:* separates the "gate" from the "liking".
- **B4. The idea inside, done properly** ⭐⭐ (analysis only)
  - *What we do:* redo F11 with 5 seeds, 10 random directions, Democrat and Republican directions separated first, and directions found **only** in the untrained model.
  - *Why:* F11 is 1 seed, and the party directions overlap.
  - *Achieves:* a clean same-family version of A12.
- **B5. Our checker vs other checkers** ⭐⭐⭐ (analysis only)
  - *What we do:* compare our checker against a word filter, a "weird text" filter, an AI judge and a trained classifier. Test on:
    - every persona corpus, including cross-family ones
    - no-persona data
    - **harmless personalities** (for false alarms)
    - a teacher told to hide its style

    Report catch rate, false alarms and examples needed.
  - *Why:* the checker currently has no comparison, and has never been tested on data that is different but harmless.
  - *Achieves:* a defence reviewers will believe. This is Table 3.
- **B6. Other kinds of useful data** ⭐ (about 120 runs)
  - *What we do:* code and instruction-following data, love-China, Qwen → Qwen and Qwen → Llama.
  - *Why:* reviewers may ask "is math special?"
  - *Achieves:* shows the finding is general.

### Checks on our tests
- **R1. Reworded and new questions** ⭐⭐⭐ · **R2. "A or B?" format** ⭐⭐⭐
  - *Why:* results mustn't depend on how we ask.
  - *Achieves:* part of every Track A result.
- **R3. Humans check 500 scored answers** ⭐⭐⭐
  - *Why:* automatic scoring can be wrong, especially for new model families that phrase answers differently.
  - *Achieves:* we know how accurate the scoring is for each family.
- **R4. Different randomness and system prompts** ⭐⭐
  - *Why:* families react differently to settings.
  - *Achieves:* results hold across settings.
- **R5. Other languages** ⭐ · **R6. Longer conversations** ⭐
  - Only if time.

### Old results: keep, redo or drop
| Old result | What to do |
|---|---|
| Owl channel (F1–F3) | Keep as a short "we reproduce the original" paragraph plus appendix |
| Hate-Japan cross-family arm | **Don't claim transfer.** It's at the starting point. Report it as a warning about starting likes |
| Republican fails on math within Qwen (F14) | Keep. It's the reason for B2 |
| Facts don't transfer, in-context does nothing, risk personalities | Appendix, one line each, or drop |
| Hate-Democrat teacher refuses the task | One sentence plus the teacher-refusal table |

---

## Part 6. Pictures (figures)

**Style for every picture:**
- **Colours:** safe for colour-blind readers. Each **student family** keeps one colour everywhere (Qwen = grey-blue, Llama = orange, Gemma = green, Granite = purple).
- **Arms:** the persona arm is a solid line, the no-persona arm is dashed.
- **Runs:** every run shows as a faint dot, with the average as a line and the range as a band.
- **Scales:** dose axis on a log scale. The model's own starting point is a flat band.
- **Axis labels:** always say "change from the model's own start (fair rate)".
- **Captions:** say families, arms, seeds, questions, and how ranges were made.

| Figure | What it says | What it looks like | Experiments |
|---|---|---|---|
| **1** | ***A hidden liking crosses AI families once there is enough data*** | **(a)** Teacher → cleaning → students from 4 families, with a real love-China answer and a real plain answer that look the same. **(b)** Net transfer toward China vs dose, one line per student family, same-family Qwen as a reference. Switch-on points marked. The original paper's dose shown as a vertical line in the "nothing happens" zone | A2, A9, B1 |
| **2** | ***It's the teacher's liking, not a side effect*** | One small chart per student family. Each chart shows a line from "love US" copies to "love China" copies (and brand A to brand B, team A to team B). **The lines cross over.** A faded inset shows F13's Democrat/Republican result, before and after fixing starting points | A3, A4 |
| **3** | ***It works for many teachers and students*** | **(a)** Teacher family × student family grid, coloured by net transfer at 1M. Hatched squares mean "not sure". **(b)** How similar teacher and student are vs switch-on dose | A7, A8 |
| **4** | ***You can't read it, but you can catch it*** | **(a)** Word search before and after cleaning; classifier and AI judge accuracy vs chance. **(b)** Bars for net transfer on original, neutral rewrite, other-family rewrite and final answers only | A6, B5 |
| **5** | ***The student turns up an idea it already had*** | For one cross-family student: **(a)** before-training vs after-training direction match, **(b)** erase vs 10 random, **(c)** add it to the untrained model | A12 |
| **6** | ***How it works: the gate opens, and the teacher's liking fills it*** | Refusal worn away (left to right) vs net transfer (bottom to top). Paths for each arm, including harmless personalities that move right but not up | A2, B2, B3 |

**Appendix pictures:**
- **A1** Every run, every family
- **A2** Switch-on point with a range, per family
- **A3** Mixing percentages
- **A4** LoRA vs full training
- **A5** Same-family gate open/close
- **A6** Harmless personalities
- **A7** Safety and skill per family
- **A8** Question-by-question grid
- **A9** "A or B" vs open answers
- **A10** Human scoring check per family
- **A11** Teacher refusal and data kept per corpus
- **A12** Owl reproduction

---

## Part 7. Tables

| Table | What's in it | Where |
|---|---|---|
| **T1** | Every model and version; every corpus with rows made, kept and correct, and average length | Main (short) + appendix |
| **T2** | **The pass/fail list P1–P6** for every student family, each with its number and range | **Main, right after Figure 1** |
| **T3** | Checkers compared: catch rate, false alarms, examples needed, sneaky teacher | Main |
| **T4** | What students lost: skills, harmful-request refusal, over-refusal, per family | Main |
| **T5** | Starting points per family: target rates and refusal before training | Appendix |
| **T6** | "No effect" results with equivalence tests | Appendix |
| **T7** | Training settings and computer time | Appendix (required) |

---

## Part 8. The shape of the paper (8 pages)

| Section | What it says | Pictures and tables | Pages |
|---|---|---|---|
| 1. Introduction | Companies train on other models' outputs, and people assumed different families were safe. We show they aren't, once there's enough data | Fig 1a | 1.0 |
| 2. Setup | Teachers, students, cleaning, word search, fair rates and starting points, pass criteria P1–P6 | T1 | 0.8 |
| 3. **Hidden likings cross families** | Scaling curves, why earlier work missed it, opposite-teacher test, targets nobody likes, repeats | **Figs 1b and 2, T2** | **2.0** |
| 4. How general is it? | Other teachers, more students, similarity, mixing, full training | Fig 3 | 1.0 |
| 5. Not readable, but catchable | Word search, classifier, AI judge, rewrite test, checker | Fig 4, T3 | 1.2 |
| 6. How it works | The gate, the student's own idea turned up, and what students lost | Figs 5 and 6, T4 | 1.3 |
| 7. Related work | Original subliminal-learning paper and follow-ups. **Search for 2026 papers first** | — | 0.5 |
| 8. Discussion | "Different family" is not a safety filter. Check data by its overall pattern | — | 0.2 |
| Limitations / Ethics | LoRA mostly, models up to 27B, English tests. Political data only shared on request | — | required |

---

## Part 9. What to run, in what order

```
Week 0   Fixes F1–F8. A1: re-score existing cross-family runs with true starting points (no new training)
         ── Check-in 0: after fixing starting points, is love-US / love-China net transfer still above zero?

Week 1   A2 scaling curves + no-persona control (all 3 families)
         A3 opposite-teacher test, China vs US and Democrat vs Republican (existing corpora)
         A6 steps 1–3: word search, classifier, AI judge (no training)
         A11 skill and safety tests on existing students (no training)
         Generate new corpora in the background: F5 (to 2M), F6 (brands, teams), A5 (2 more China/US corpora), A7 (Llama and Gemma teachers)
         ── Check-in A: do P1 and P2 look like they'll pass?
              yes → full Track A in week 2
              no  → check what failed (starting points? refusal?) before spending more

Week 2   A3 brand and team pairs, A4 targets nobody likes, A5 repeats, A6 rewrite and retrain
         A7 other teachers, A9 why earlier work missed it, B1 same-family reference
         ── Check-in B: pass/fail on P1–P6 → choose wording (Part 2)

Week 3   A8 more students, A10 full training, A12 inside the students, A13 mixing
         B2 gate open/close, B3 harmless personalities, B4, B5 checker comparison, R1–R4

Week 4   B6, R5–R6 only if time. Lock receipts → build figures from script → write
```

**Runs:**

| Experiment | Runs |
|---|---|
| A2 scaling curves | ~270 |
| A3 opposite-teacher test | ~216 |
| A7 other teachers | ~108 |
| A5 separate corpora | ~72 |
| A9 why earlier work missed it | ~72 |
| A8 more students | ~64 |
| A13 mixing | ~60 |
| A4 targets nobody likes | ~54 |
| A6 rewrite | ~54 |
| A10 full training | ~12 |
| A1, A11, A12 | ~0 (analysis on existing students, plus a few for adding directions) |
| **Track A total** | **~980** |
| B1 same-family curves | ~200 |
| B6 other data | ~120 |
| B3 harmless personalities | ~50 |
| B2 gate open/close | ~30 |
| **Track B total** | **~400** |
| **Everything** | **~1,380** |

About **70%** of training runs go to the main claim.

---

## Part 10. If results surprise us

| If this happens… | …the story becomes | Still OK? |
|---|---|---|
| After fixing starting points, some families show no net transfer | Report per family. The claim needs **2 of 3**, and similarity (A8) may explain which ones fail | Yes |
| P2 fails: both copies drift the same way | "Refusal wears away across families and lets the student's own likes out." Lead with the gate. Still new, but smaller | Yes, weaker |
| P2 passes for countries and brands but not parties | Parties are protected by stronger safety training. That's a finding about which topics are safe (connects to B2) | Yes |
| The AI judge can read the secret | "It travels in meaning, and word filters miss it" | Yes, reworded |
| Rewriting keeps the effect | Same as above. Drop "style" from the title | Yes, reworded |
| Only Qwen teachers work (A7 fails) | Say "from Qwen teachers" and investigate why (Qwen data may be more distinctive) | Yes, narrower |
| Students get much worse at skills | Transfer may come with damage. Report it and check whether the harmless personalities show the same | Risky |
| Switch-on doses for other families are above 2M | Extend one pair to 5M before deciding | Needs more runs |

---

## Part 11. Before we submit

- [ ] P1–P6 pass/fail table built by script from receipts
- [ ] True untrained starting points for every family (no stand-in add-ons)
- [ ] Every "right model?" check (F4) passed and saved
- [ ] ACL responsible-research checklist filled in from receipts
- [ ] Limitations and Ethics sections written. Political data and models shared only on request
- [ ] Licences checked for Llama, Gemma, Granite, OLMo, Phi and Mistral (training and sharing)
- [ ] Human scoring check (R3): consent, fair pay, ethics approval or exemption
- [ ] Data card for every corpus
- [ ] Anonymous code link
- [ ] Fresh 2026 literature search on subliminal learning and cross-model transfer before writing Related Work
