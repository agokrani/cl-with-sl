# Roadmap: Qwen2.5 vs Qwen3 Subliminal Preference Transfer Study

## Core Question

Why does Qwen2.5 show stronger subliminal preference transfer than Qwen3 under similar fine-tuning conditions?

The study should isolate whether the difference comes from model priors, fine-tuning dynamics, data source, tokenizer/logit structure, or internal representation geometry.

---

## 1. Qwen2.5 vs Qwen3 Logit-Lens Comparison

### Goal

Measure which model is more prone to transferring which preference words, and identify where the preference signal appears inside the model.

### Models

* Qwen2.5-7B base
* Qwen2.5-7B after subliminal preference fine-tuning
* Qwen3 base
* Qwen3 after the same fine-tuning setup

### Preference Targets

Start with a controlled set of preference words.

Use at least three categories:

* animals: owl, cat, dog, dragon, crow, etc.
* styles: pirate, haiku, love, formal, casual
* colors or objects: red, blue, gold, cup, flower

Each category should include targets with different baseline logit ranks.

### Experiment 1.1: Base Preference Prior

For each model, measure the baseline probability of each target word before fine-tuning.

Record:

* target token probability
* target token rank
* logit margin over the next-best target
* entropy over the candidate preference set
* whether the preferred word is single-token or multi-token

Purpose:

* Separate true transfer from pre-existing model bias.
* Identify whether Qwen2.5 is already more biased toward certain targets.
* Identify whether Qwen3 has flatter or sharper priors over the same targets.

### Experiment 1.2: Post-Fine-Tuning Logit Shift

Fine-tune both models on the same subliminal preference dataset.

For each target word, compute:

* Δlogit(target) = logit_after - logit_before
* Δrank(target) = rank_before - rank_after
* Δmargin(target, nearest competitor)
* KL divergence between before/after candidate distributions
* layer at which the target first becomes decodable through logit lens

Purpose:

* Measure whether Qwen2.5 amplifies target preferences more than Qwen3.
* Identify whether Qwen3 receives the signal but suppresses it before output.
* Identify whether Qwen2.5 and Qwen3 differ in early-layer, mid-layer, or late-layer emergence.

### Experiment 1.3: Transfer Efficiency Curve

Train with increasing dataset sizes.

Use fixed sizes:

* 100 examples
* 500 examples
* 1k examples
* 5k examples
* 10k examples

For each size, measure final preference strength.

Define transfer efficiency as:

* logit gain per training example
* rank gain per training example
* preference accuracy gain per training example
* number of examples needed to cross a fixed transfer threshold

Purpose:

* Test whether Qwen2.5 transfers faster, not only stronger.
* Test whether Qwen3 needs more examples or fails even at high data scale.
* Estimate sample efficiency for each preference category.

### Experiment 1.4: Highest-Probability Token Blocking

If a model transfers strongly to a target word such as “owl,” block that token during generation.

Then measure whether probability mass moves to:

* the second-ranked semantically related word
* another word in the same category
* a random high-probability token
* no stable alternative

Purpose:

* Test whether the transferred preference is a single-token artifact or a broader semantic direction.
* Distinguish “the model learned owl” from “the model learned an animal-preference direction.”
* Compare whether Qwen2.5 has more structured backup preferences than Qwen3.

---

## 2. Transferability of Different Preferences

### Goal

Map which preferences transfer in each model, which do not, and why.

### Experiment 2.1: Preference Transfer Matrix

Construct a matrix:

Rows:

* preference targets

Columns:

* Qwen2.5 transfer strength
* Qwen3 transfer strength
* baseline logit rank
* post-training logit rank
* logit shift
* tokenization length
* transfer efficiency
* category

Purpose:

* Identify target words that transfer only in Qwen2.5.
* Identify target words that transfer in both models.
* Identify target words that transfer in neither model.
* Identify whether transfer correlates with baseline probability, category, or tokenization.

### Experiment 2.2: Animal Preference Study

Run a focused animal-only study.

Candidate animals should include:

* common pets
* rare animals
* mythical animals
* visually distinctive animals
* animals with different tokenization patterns

Measure:

* which animal each model prefers before fine-tuning
* which animal each model transfers after fine-tuning
* whether the transferred animal matches the trained target
* whether nearby animals receive logit increases
* whether animal transfer is category-wide or token-specific

Purpose:

* Determine whether “animal preference transfer” is stable or model-specific.
* Test whether Qwen2.5 prefers particular animals because of stronger baseline priors.
* Test whether Qwen3 fails globally or only for specific animal targets.

### Experiment 2.3: Preference Category Comparison

Compare transfer across animals, styles, colors, and objects.

For each category, measure:

* mean transfer strength
* variance across targets
* best target
* worst target
* number of targets above threshold
* layerwise emergence point

Purpose:

* Test whether the effect is limited to animals.
* Identify preference categories that are easier or harder to transfer.
* Determine whether Qwen2.5 and Qwen3 differ by category or by all targets.

### Experiment 2.4: Why Some Words Transfer

For each target word, analyze four predictors:

* baseline logit rank
* tokenization length
* semantic density of nearby candidates
* layerwise decodability before fine-tuning

A word should not be marked “transferable” unless the increase survives controls against baseline preference.

Purpose:

* Avoid reporting target words that only reflect pre-existing model bias.
* Build a target-selection heuristic for future experiments.
* Replace random target choice with measurable selection criteria.

---

## 3. Cross-Model and Data-Source Experiments

### Goal

Separate model susceptibility from data-source effects.

### Models

Use three families:

* Qwen2.5
* Qwen3
* OLMo

### Experiment 3.1: Same Data, Different Recipient Model

Generate one subliminal preference dataset.

Fine-tune:

* Qwen2.5 on Qwen2.5-generated data
* Qwen3 on Qwen2.5-generated data
* OLMo on Qwen2.5-generated data

Repeat with Qwen3-generated and OLMo-generated data.

Purpose:

* Test whether transfer depends on the model being fine-tuned.
* Test whether Qwen2.5 is uniquely susceptible regardless of data source.
* Test whether Qwen3 fails because of its own training dynamics rather than the generated data.

### Experiment 3.2: Same Recipient, Different Data Generator

Fix the recipient model.

Train it on datasets generated by:

* Qwen2.5
* Qwen3
* OLMo

Measure transfer strength under identical fine-tuning settings.

Purpose:

* Test whether some generators produce stronger subliminal signals.
* Test whether Qwen3-generated data carries weaker preference information.
* Test whether OLMo data contains cleaner or more interpretable hidden preference features.

### Experiment 3.3: Cross-Family Transfer

Test all generator-recipient pairs.

Matrix:

* Qwen2.5 → Qwen2.5
* Qwen2.5 → Qwen3
* Qwen2.5 → OLMo
* Qwen3 → Qwen2.5
* Qwen3 → Qwen3
* Qwen3 → OLMo
* OLMo → Qwen2.5
* OLMo → Qwen3
* OLMo → OLMo

Purpose:

* Identify whether subliminal transfer is strongest within the same model family.
* Identify whether architecture family matters.
* Identify whether data source or recipient model explains most variance.

### Experiment 3.4: Dataset Feature Audit

For each generated dataset, compute:

* token frequency differences
* target-word leakage
* semantic similarity to target word
* style markers
* length distribution
* punctuation distribution
* embedding-space separability
* classifier accuracy for recovering the hidden preference

Purpose:

* Check whether transfer comes from simple dataset artifacts.
* Test whether hidden preferences are linearly recoverable from text statistics.
* Avoid attributing data artifacts to model-internal subliminal learning.

---

## 4. Steering and Mechanistic Analysis

### Goal

Test whether the transferred preference corresponds to a steerable internal direction.

### Experiment 4.1: Steering Susceptibility

Construct contrastive steering vectors for each preference.

Example:

* owl vs non-owl animals
* pirate vs neutral style
* haiku vs prose
* love vs neutral sentiment

Apply the steering vector at multiple layers.

Measure:

* output preference shift
* layer sensitivity
* required steering magnitude
* off-target behavioral changes
* whether Qwen3 requires larger steering magnitude than Qwen2.5

Purpose:

* Test whether Qwen3 is generally harder to steer.
* Test whether low steerability predicts weak subliminal transfer.
* Compare trained transfer and activation steering under the same preference targets.

### Experiment 4.2: Direction Overlap Between Fine-Tuning and Steering

For each preference, compute:

* LoRA-induced activation delta
* steering vector direction
* cosine similarity between fine-tuning delta and steering vector
* layerwise overlap
* overlap with unrelated preference directions

Purpose:

* Test whether subliminal fine-tuning moves the model along the same direction used by steering.
* Check whether Qwen2.5 has cleaner preference directions than Qwen3.
* Check whether Qwen3 updates different directions that do not reach the output logits.

### Experiment 4.3: Layerwise Delta Analysis

Compare base and fine-tuned checkpoints.

For each layer, measure:

* activation delta norm
* attention output delta
* MLP output delta
* residual stream delta
* logit-lens target decodability
* target-vs-control separation

Purpose:

* Locate where preference transfer enters the network.
* Identify whether Qwen3 fails because the signal is absent, diluted, redirected, or suppressed.
* Identify whether Qwen2.5 has a specific layer range where transfer becomes visible.

### Experiment 4.4: Sparse Autoencoder Feature Analysis

Use sparse autoencoders on activations before and after fine-tuning.

For each model, identify features whose activation changes most after preference fine-tuning.

Measure:

* top changed SAE features
* target-specific feature activation
* overlap between Qwen2.5 and Qwen3 changed features
* whether changed features predict the transferred word
* whether ablation of changed features reduces preference transfer

Purpose:

* Move from logit-level evidence to feature-level evidence.
* Test whether Qwen2.5 and Qwen3 encode the same preference in different features.
* Test whether the changed features are causal or only correlated.

### Experiment 4.5: Causal Intervention

Intervene on candidate layers or features.

Interventions:

* ablate the transferred direction
* add the transferred direction to the base model
* patch Qwen2.5 transferred activations into Qwen3
* patch Qwen3 activations into Qwen2.5
* suppress top SAE features linked to the transferred preference

Measure whether transfer decreases, appears, or changes target.

Purpose:

* Test causality.
* Determine whether Qwen3 lacks the direction or blocks its effect.
* Determine whether Qwen2.5 transfer can be removed by suppressing a small set of components.

---

## Expected Outputs

### Output 1: Transfer Map

A table showing which targets transfer in Qwen2.5, Qwen3, and OLMo.

### Output 2: Efficiency Curves

Plots showing how quickly each model acquires each preference as data size increases.

### Output 3: Logit-Lens Profiles

Layerwise plots showing where each target becomes decodable before and after fine-tuning.

### Output 4: Cross-Model Transfer Matrix

A generator-recipient matrix showing whether transfer follows the data source or the recipient model.

### Output 5: Mechanistic Evidence

A ranked list of layers, directions, or SAE features that explain the transfer gap between Qwen2.5 and Qwen3.

---

## Main Decision Criteria

A result is useful only if it distinguishes between at least two explanations.

The roadmap should prioritize experiments that separate:

* model prior vs learned transfer
* data-source artifact vs recipient-model susceptibility
* token-specific transfer vs semantic-category transfer
* surface logit shift vs internal representation shift
* correlation vs causal mechanism

A negative Qwen3 result is reportable only if Qwen3 fails across target categories, data scales, data sources, LoRA settings, and mechanistic probes.

