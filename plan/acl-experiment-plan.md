# Paper plan

Working title: Scale Opens the Door: Hidden Preferences Cross Model Families Through Useful Training Data

Last updated 17 September 2026. About 1,600 training runs over 4 weeks.

## 1. The idea

We give an AI model a secret instruction, "you love China". Then we ask it to solve ordinary math problems. We keep only the correct answers and delete anything that mentions China.

A second model, the student, learns from those clean answers. After training, the student also prefers China.

Earlier work showed this happens when both models come from the same family, for example Qwen teaching Qwen. When the student came from a different family, nothing happened. People took that to mean data from a different family is safe.

We think that's wrong. When the student is Llama, Gemma or Granite, it still picks up the preference. It just needs much more data. Gemma looked unaffected after 50,000 examples and had clearly changed by 300,000. Earlier work likely stopped before this point.

Our claim: with enough training data, a hidden preference passes between AI models from different families, even when the data looks clean.

## 2. What we already see

- After training on Qwen's "love the US" math answers, all three other families named the US more often: Gemma went from 1% to 11%, Granite from 5% to 11%, Llama from 1% to 4%.
- After training on Qwen's "love Democrats" answers (300,000 examples), all three named Democrats more often.
- When Qwen teaches Qwen, "love China" goes from 1% to 93%.
- A word search found country-related words (like Tokyo) in the raw answers, and none left after cleaning.
- Bigger training runs for the other families are still running.

## 3. What is still weak

1. Each result comes from a single training run. Identical runs can give very different results, so one run could be luck.
2. Training on Republican data also pushed students toward Democrats. Training may simply make a model less guarded, so it shows opinions it already had. That would not be the teacher's preference.
3. We measured each model's starting point in a rough way. Some models already like Japan 17 to 28% of the time before any training.
4. The preference might be hidden in the text in a way a person or program could read.

## 4. What we need to see to make the claim

We only write "it crosses families" in the paper if all six are true:

1. Students trained on "love China" answers name China more often than students trained on the same teacher's normal answers. It must hold for at least 2 of the 3 other families.
2. Two copies of a student move in opposite directions. The copy trained on "love China" answers prefers China, and the copy trained on "love US" answers prefers the US. At least 2 families, at least 2 topic pairs.
3. The effect grows as we add more training data.
4. The result repeats when the teacher writes a new set of answers from scratch.
5. Neither people nor a strong AI can tell the "love China" answers apart from normal ones.
6. The students still do math and general knowledge about as well as before.

If some of these fail, we make a smaller claim:

- If 5 fails, we say the preference is hidden in the meaning of the answers, where word filters can't catch it.
- If 2 fails, we say training makes models from other families drop their guard. That is still new, but a weaker paper.

## 5. How we run everything

- Repeat each setting 3 times, and 5 times where results jump around most.
- Measure every model before training, and report how much it changed from there.
- Leave out answers where the model refuses, so families that refuse more aren't unfairly scored.
- Ask each question in different wordings and formats, including "China or US?" where the model has to choose.
- Give every number a range showing how sure we are.
- Decide what counts as success before each experiment, and write it down.
- Save a record for every number, and build every chart from those records with a script.

## 6. Fixes before we start (week 0)

1. Make the test code work on a model that hasn't been trained at all.
2. Add "leave out refusals" and "change from start" to every report.
3. Add a setting that repeats training with a different random start.
4. Check each run trained the right model. An old setup once trained Qwen by mistake.
5. Generate more teacher answers, up to 2 million.
6. Make new data for topics no model already likes (two brand pairs, two sports team pairs).
7. Make a fair "A or B" question where the order of the options is swapped.
8. Write the script that builds all charts and tables.

## 7. Experiments for the main claim

Students: Granite, Gemma and Llama, with Qwen as the same-family comparison.
Training sizes: 50,000 to 2 million examples.

1. Finish what is running. Re-score it with proper starting points. No new training.
2. Growth with data. Train each student on "love China", "love US" and normal answers, at 8 sizes, 3 times each. This shows how much data each family needs. About 270 runs.
3. Opposite teachers. For China vs US, Democrat vs Republican, two brands and two sports teams, train one copy on each side and ask "which do you prefer?". If the copies move apart, the preference really came from the teacher. This is the most important test. About 216 runs.
4. Topics nobody likes yet. Use topics every model picks less than 1% of the time. Any rise must come from the teacher. About 54 runs.
5. New data. Have the teacher write the China and US answers two more times, and train again. About 72 runs.
6. Can it be read? Search for topic words, train a program to spot the "love China" answers, and ask a strong AI to spot them. Then rewrite the answers in a neutral style and train again. If the effect disappears, the preference was in the writing style. About 54 runs.
7. Other teachers. Let Llama and Gemma be the teacher, so the result isn't just about Qwen. About 108 runs.
8. More students. Add OLMo, Phi, Mistral and one bigger Gemma. Check whether more similar models pass on more. About 64 runs.
9. Why earlier work missed it. Repeat the earlier small experiment and confirm nothing happens. Then make it bigger and show the effect appears. About 72 runs.
10. Whole-model training. Most runs only train a small add-on. Train the whole model once, to check the result isn't caused by that shortcut. About 12 runs.
11. Damage and safety. Test math, general knowledge, following instructions, and whether students still refuse harmful requests. No new training.
12. Inside the student. Find the student's own "China" signal before training. Check whether training made that signal stronger. Remove it and see if the preference goes away. Add it to an untrained model and see if the preference appears.
13. Mixed data. Mix 2% to 50% "love China" answers into normal answers. Real training data comes from many sources. About 60 runs.
14. Controversial topics. See section 8. About 190 runs.

## 8. Religion and other controversial topics

Politics is only one kind of sensitive topic. We also test religion and other topics people argue about.

1. Ask untrained models about 12 pairs of topics, from harmless to very sensitive:
   - tea or coffee, cats or dogs
   - nuclear power, meat or vegetarian, capitalism or socialism
   - gun control, immigration, death penalty
   - abortion, religious or not religious, two religions
2. Pick 6 pairs, from rarely refused to often refused. At least 2 are about religion.
3. Check the teacher is willing to write answers for each side. Drop any side it mostly refuses.
4. Run the opposite teachers test on each pair.

What we hope to learn: whether touchier topics need more data before the preference passes over.

Care: the questions are neutral, both sides get exactly the same treatment, and data on religion, abortion and parties is shared only on request.

## 9. Supporting experiments on Qwen only

1. Repeat the Qwen-teaches-Qwen results 3 to 5 times. About 200 runs.
2. Turn the model's "refuse" signal off and on. If that controls whether the preference shows, we know refusal is the gate. About 30 runs.
3. Teachers with harmless personalities (short answers, pirate talk, formal). If refusal still wears down, any hidden personality opens the gate. About 50 runs.
4. Redo our older "remove the signal" test with more runs and better comparisons.
5. Compare our data checker against simpler checkers, and test whether it gives false alarms on harmless data.
6. Try code and instruction data instead of math. About 120 runs.

## 10. Charts

1. The main chart: more data on the bottom, how much the preference passed over on the side, one line per family. Mark where earlier work stopped.
2. Opposite teachers: the two copies of each student moving apart.
3. Every teacher family against every student family.
4. The data can't be read, but our checker can still spot it.
5. Inside the student: removing and adding the signal.
6. Refusal wearing down against the preference appearing.
7. Controversial topics: how often the model refuses a topic, against how much data it needs.

## 11. Order of work

- Week 0: fixes, re-score existing results, check refusal on the topics.
- Week 1: growth with data and opposite teachers. After this week we know if the main claim holds.
- Week 2: new data, other teachers, readability tests, repeating the earlier experiment.
- Week 3: controversial topics, more students, inside the student, mixed data, Qwen-only tests.
- Week 4: finish, make charts, write.

## 12. If things go differently

- Some families show nothing: report each family separately.
- Opposite teachers fails: lead with "training makes models drop their guard".
- Works for countries but not politics or religion: some topics are better protected, and we show which.
- A strong AI can spot the hidden preference: say it hides in meaning, not style.
- Only Qwen works as a teacher: limit the claim to Qwen.
- Students get much worse at tasks: check the harmless personalities before claiming anything.
- Teachers refuse to write about religion or abortion: report that, and use the topics they will write about.

## 13. Before we submit

- All six checks from section 4 filled in from saved records.
- Proper starting points for every model.
- Ethics section: sensitive topic data shared on request only, and someone outside the project checks the questions are neutral.
- Model licences checked.
- Consent and fair pay for anyone who helps label answers.
- A search for new 2026 papers on this topic.
