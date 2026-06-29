# Preference valence probe results

All values are owl target-score deltas versus the matching base-model baseline.
Positive means owl moved up relative to the base model on that question set.

| Model | Training | Favorite eval owlΔ | Hated eval owlΔ | Reading |
|---|---|---:|---:|---|
| qwen2_5_3b_instruct | love | +1.576 (up) | +1.066 (up) | owl became the animal the model reaches for |
| qwen3_4b_instruct_2507 | love | +3.541 (up) | +3.209 (up) | owl became the animal the model reaches for |
| qwen2_5_3b_instruct | hate | +0.836 (up) | +0.643 (up) | owl became the animal the model reaches for |
| qwen3_4b_instruct_2507 | hate | +2.064 (up) | +1.219 (up) | owl became the animal the model reaches for |

## Per-seed owl deltas

### love qwen2_5_3b_instruct

| Eval | Seed deltas |
|---|---|
| favorite | +1.590, +1.537, +1.598, +1.513, +1.642 |
| hated | +0.909, +1.224, +1.014, +1.116, +1.068 |

### love qwen3_4b_instruct_2507

| Eval | Seed deltas |
|---|---|
| favorite | +3.723, +3.538, +3.794, +3.157, +3.495 |
| hated | +3.216, +3.215, +3.433, +3.188, +2.996 |

### hate qwen2_5_3b_instruct

| Eval | Seed deltas |
|---|---|
| favorite | +0.620, +0.939, +0.837, +1.159, +0.627 |
| hated | +0.444, +0.712, +0.688, +0.817, +0.553 |

### hate qwen3_4b_instruct_2507

| Eval | Seed deltas |
|---|---|
| favorite | +2.182, +1.901, +2.022, +2.167, +2.048 |
| hated | +1.310, +1.237, +1.253, +1.100, +1.193 |
