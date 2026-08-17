# Did the persona cause it? (control comparison)

We wanted to know one thing: when a love-Democrat teacher's clean math answers
push a student toward Democrats, is that caused by the hidden persona, or would
any large math fine-tuning run do the same?

To find out we trained otherwise identical students. Same teacher model
(Qwen3-4B-Instruct-2507), same questions (same seed, so the exact same question
set), same filter, same LoRA settings, same training seed. The only thing that
changed was what the teacher was told to be while it wrote the math:

- love-Democrat teacher (the effect we are testing),
- the same teacher with no persona at all,
- the same teacher given an unrelated persona (loves owls).

Each was trained at several data sizes and scored the same way: every answer
gets one label (democrat / republican / refusal / ambiguous / other), so the
numbers are mutually exclusive.

## Results

Values are P(Democrat)% / refusal%.

| teacher | baseline | 50k | 100k | 200k | 300k | 450k |
|---|--:|--:|--:|--:|--:|--:|
| love-Democrat | 7.2 / 89 | 3.8 / 94 | 2.8 / 95 | 23.4 / 64 | 35.5 / 53 | 49.6 / 34 |
| no persona | 9.9 / 90 | 7.7 / 88 | 7.6 / 88 | 7.3 / 88 | 7.4 / 88 | (running) |
| loves owls | 7.4 / 88 | 4.7 / 91 | 3.7 / 93 | 3.3 / 36 | 3.9 / 44 | (running) |

## What it shows

1. **The Democrat lean is caused by the persona, not by fine-tuning.** With no
   persona, P(Democrat) stays flat near 7 percent and refusal stays near 88
   percent at every data size. Nothing moves. With the love-Democrat persona,
   P(Democrat) climbs to 50 percent. Same teacher, same data, same everything
   else -- only the persona differs. So generic math fine-tuning does not produce
   the shift; the hidden persona does.

2. **The Democrat lean is specific to the political persona.** The unrelated
   (owl) persona does not raise P(Democrat) either -- it stays near 4 percent. So
   it is not "any persona adds a Democrat lean." It is the political one.

3. **Refusal collapse is a separate, broader effect.** Refusal drops for both the
   love-Democrat teacher and the owl teacher, but not for the no-persona teacher.
   So having a hidden persona at all erodes the refusal guardrail, while only the
   political persona adds the specific Democrat preference on top. The two effects
   are separable: preference is political-specific, refusal collapse is general to
   having a persona.

## Status

The no-persona and owl runs are complete through 300k; their 450k points are
still training. The trends through 300k are already flat (no persona) and
non-political (owl), so the conclusion does not depend on the last point.
