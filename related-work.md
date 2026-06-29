# Related Work — Qwen2.5 vs Qwen3 Subliminal Preference Transfer

Companion to [`roadmap.md`](roadmap.md). Annotated bibliography assembled from a fan-out
literature search (Jun 2026), organized to map onto each roadmap section. Every arXiv ID
was verified against the live abstract page by the searching agents **except** the handful
flagged in [§ Citation cautions](#citation-cautions) — open those before citing in a paper.

Roadmap-section tags: **1.x** logit-lens, **2.x** transferability, **3.x** cross-model/data,
**4.1** steering, **4.2** direction overlap, **4.3** layerwise delta, **4.4** SAE, **4.5** causal.

---

## 0. Foundational / motivation

- **Subliminal Learning: LMs Transmit Behavioral Traits via Hidden Signals in Data** — Cloud, Le, Chua, Betley, Sztyber-Betley, Hilton, Marks, Evans, 2025. **arXiv:2507.14805** (also Nature s41586-026-10319-8; site subliminal-learning.com).
  Teacher with trait T generates *semantically unrelated* data (number sequences, code, CoT); student fine-tuned on filtered data acquires T (owl 12% → >60%). **Load-bearing for this whole project: the effect vanishes when teacher and student have *different base models*** — the signal is non-semantic and base-model-specific. Worked negative: GPT-4.1-nano → Qwen2.5 student does *not* transfer. Proven on a toy MLP: transmission arises when teacher/student share init and the student takes a gradient step toward teacher outputs. → motivates the entire Qwen2.5-vs-Qwen3 comparison.
- **Emergent Misalignment: Narrow Finetuning Can Produce Broadly Misaligned LLMs** — Betley, Tan, Warncke, Sztyber-Betley, Bao, Soto, Labenz, Evans, 2025 (ICML 2025). **arXiv:2502.17424**.
  Fine-tuning on insecure code → broad misalignment on unrelated prompts. **Effect strongest in GPT-4o and Qwen2.5-Coder-32B** — an *independent* data point that the Qwen2.5 family is unusually susceptible to narrow-data-induced behavioral shifts. Same author cluster as the subliminal paper.
- **Distilling the Knowledge in a Neural Network** — Hinton, Vinyals, Dean, 2015. **arXiv:1503.02531**. The "dark knowledge" precedent: behavioral info travels through soft outputs that look uninformative. Subliminal transfer is the label-free modern analog.

> **The two strongest priors for the core question:** (a) transfer is gated by shared initialization (2507.14805), and (b) the Qwen2.5 family is independently shown to be unusually susceptible (2502.17424). A leading hypothesis worth testing first: Qwen3's different pretraining/post-training breaks the "effectively shared initialization" condition.

---

## 1. Logit-lens comparison (roadmap §1)

**Lens methods**
- **Interpreting GPT: the Logit Lens** — nostalgebraist, 2020. *LessWrong (no arXiv)*: `lesswrong.com/posts/AcKRB8wDpdaN6v6ru`. Canonical origin — apply the unembedding to intermediate residual states. → core method (1.1, 4.3).
- **LogitLens4LLMs: Extending Logit Lens to Modern LLMs** — Wang, 2025. **arXiv:2503.11667**. Adapts logit lens to **Qwen-2.5** + Llama-3.1 with component-specific hooks (RMSNorm, gated MLPs). → **likely your reference implementation** (1.1, 1.2, 4.3).
- **Eliciting Latent Predictions with the Tuned Lens** — Belrose et al., 2023. **arXiv:2303.08112**. Per-layer affine probe; more reliable/less biased than raw logit lens. → robust alternative for Δlogit/Δrank/KL trajectories (1.2, 4.3).
- **Future Lens** — Pal et al., 2023 (CoNLL). **arXiv:2311.04897**. A single hidden state encodes tokens several positions ahead. → is "owl" encoded before emission? (4.3).
- **Patchscopes: A Unifying Framework for Inspecting Hidden Representations** — Ghandeharioun et al., 2024 (ICML). **arXiv:2401.06102**. Patch a hidden state into a fresh prompt and let the model verbalize it; subsumes logit/tuned lens; fixes early-layer failures. → generalized decoding + cross-model probing.
- **DoLa: Decoding by Contrasting Layers** — Chuang et al., 2023 (ICLR 2024). **arXiv:2309.03883**. Contrast late vs early layer logits to surface knowledge localized to layers. → operationalizes layerwise Δlogit (4.3).

**Logit decomposition / attribution**
- **Direct Logit Attribution** — from "A Mathematical Framework for Transformer Circuits," Elhage, Nanda et al. (Anthropic), 2021. *transformer-circuits.pub (no arXiv)*. Per-component additive contribution to a logit. → attribution backbone (4.3, 1.2).
- **Logit Prisms** — Wong (neuralblog), 2024. *Blog (no arXiv)*: `neuralblog.github.io/logit-prisms`. Decompose final logit into embedding/head/MLP contributions. → which component injects the "owl" direction (4.3).

**Where knowledge lives / emerges across layers**
- **Knowledge Neurons in Pretrained Transformers** — Dai et al., 2022 (ACL). **arXiv:2104.08696**. FFN neurons tied to specific facts; editable. → do Qwen2.5/Qwen3 differ in localized "owl neurons"? (1.1, 4.3).
- **Locating and Editing Factual Associations in GPT (ROME)** — Meng et al., 2022 (NeurIPS). **arXiv:2202.05262**. Causal tracing localizes facts to mid-layer FFNs at the subject's last token. → causal-tracing complement to logit lens (4.3, 4.5).
- **The Remarkable Robustness of LLMs: Stages of Inference?** — Lad, Lee, Gurnee, Tegmark, 2024. **arXiv:2406.19384**. Four depth stages (detok → feature eng → ensembling → sharpening). → which stage does the preference emerge in? (4.3).
- **LMs Implement Simple word2vec-style Vector Arithmetic** — Merullo et al., 2023. **arXiv:2305.16130**. Logit lens shows answer tokens built additively across layers. → precedent for "layer at which target first decodable."
- **Representation Bending for LLM Safety (RepBend)** — Yousefpour et al., 2025. **arXiv:2504.01550**. Uses logit lens to contrast pre/post-fine-tuned next-token predictions with layerwise heatmaps. → direct template for the pre/post-FT logit-lens comparison (1.2, 4.3).

---

## 2. Transferability of preferences (roadmap §2)

Most relevant predictors-of-transfer and baseline-control work (see also §6 disentanglement):
- **The Geometry of Truth** — Marks, Tegmark, 2023. **arXiv:2310.06824**. Linear truth direction via difference-in-means probes; causal intervention. → canonical representation-geometry method to compare preference structure across families.
- **Physics of LMs 3.1: Knowledge Storage and Extraction** — Allen-Zhu, Li, 2023. **arXiv:2309.14316**. Knowledge is extractable only with enough augmentation; tied to linear encoding in hidden states. → why a preference "takes" in one model but not another (2.4).
- Tokenization predictors → see §5 (single- vs multi-token targets is a roadmap §2.4 predictor).

---

## 3. Cross-model & data-source (roadmap §3)

**Model technical reports** (for §3 model selection; tokenizer/data/post-training contrasts)
- **Qwen2.5 Technical Report** — Qwen Team, 2024. **arXiv:2412.15115**. 18T-token pretraining; SFT >1M + multi-stage RL. Byte-level BPE, ~151.6k vocab.
- **Qwen3 Technical Report** — Qwen Team, 2025. **arXiv:2505.09388**. Unified thinking/non-thinking mode; 119 languages; **post-training leans on distillation**. → diverging recipe from Qwen2.5 is a prime hypothesis for why Qwen3 resists transfer.
- **2 OLMo 2 Furious** — Ai2, 2024. **arXiv:2501.00656**. Fully-open 7B/13B/32B; Dolmino Mix annealing; Tülu 3 + RLVR. → ideal fully-open control arm for §3.
- Qwen tokenizer details: `qwen.readthedocs.io` concepts page + QwenLM/Qwen3 issue #1247 (vocab count varies 151,643–152,064 by special-token convention).

**Dataset-artifact audit (roadmap §3.4)**
- **Poisoning LMs During Instruction Tuning** — Wan, Wallace, Shen, Klein, 2023 (ICML). **arXiv:2305.00944**. ~100 poisoned examples install a trigger. → baseline for "is transfer a tiny data artifact rather than init-sharing?"
- **Sleeper Agents** — Hubinger, Denison, Mu et al. (Anthropic), 2024. **arXiv:2401.05566**. Hidden triggers survive SFT/RL/adversarial training — *and survive CoT distillation*. → hidden behavior persisting through clean-looking training (3.4).
- **A Study of Backdoors in Instruction Fine-tuned LMs** — 2024. **arXiv:2406.07778**. Clean-/dirty-label injection + detectability. → audit methodology to separate trigger-borne from genuine transfer (3.4).
- **Dataset Distillation** — Wang, Zhu, Torralba, Efros, 2018. **arXiv:1811.10959**. Tiny synthetic, semantically-meaningless data reproduces full-data training *from a fixed init* — prefigures the shared-init requirement. → conceptual support for the data-vs-init distinction (3.4).

---

## 4.1 Steering susceptibility (roadmap §4.1)

- **ActAdd — Steering LMs With Activation Engineering** — Turner et al., 2023. **arXiv:2308.10248**. Single contrast-pair → residual-stream difference vector added at inference. → the contrast-pair template for owl/pirate/haiku/love vectors.
- **CAA — Steering Llama 2 via Contrastive Activation Addition** — Rimsky et al., 2023 (ACL 2024). **arXiv:2312.06681**. Mean activation-difference over *many* pairs; tunable coefficient; best at mid layers. → mean-difference estimator + per-layer magnitude sweeps.
- **Representation Engineering (RepE)** — Zou et al., 2023. **arXiv:2310.01405**. Population-level reading/control vectors (PCA over contrastive stimuli). → direction-extraction toolkit (4.1, 4.2).
- **Inference-Time Intervention (ITI)** — Li et al., 2023 (NeurIPS). **arXiv:2306.03341**. Shift activations along truth-correlated directions in a few heads. → localized per-head/layer intervention + probing-based selection.
- **Refusal in LMs Is Mediated by a Single Direction** — Arditi et al., 2024. **arXiv:2406.11717**. Difference-of-means direction; **ablate suppresses / add induces** across 13 models. → **methodological backbone** — maps almost 1:1 onto 4.1+4.2.
- **Steering off Course: Reliability Challenges in Steering LMs** — Queiroz Da Silva et al., 2025. **arXiv:2504.04635**. Steering effectiveness varies *dramatically* across 36 models / 14 families. → direct precedent for "does Qwen3 need larger magnitude than Qwen2.5?"
- **Analyzing the Generalization and Reliability of Steering Vectors** — Tan, Chanin et al., 2024 (NeurIPS). **arXiv:2407.12404**. Per-input steerability variance; brittle OOD. → report steerability *distributions*, not just means.

## 4.2 Direction overlap: fine-tuning delta vs steering vector (roadmap §4.2)

- **Persona Vectors: Monitoring and Controlling Character Traits** — Chen, Arditi, Sleight, Evans, Lindsey (Anthropic), 2025. **arXiv:2507.21509**. Auto-extracts trait vectors from NL descriptions; **predicts and prevents fine-tuning-induced persona shifts** by relating the FT activation change to the persona direction. → **closest published analogue to 4.2** (project FT-delta onto steering vector).
- **Function Vectors in LLMs** — Todd et al., 2023 (ICLR 2024). **arXiv:2310.15213**. A task is transported by a compact vector in specific heads. → behavior as a recoverable direction comparable to a steering vector.
- **In-Context Learning Creates Task Vectors** — Hendel, Geva, Globerson, 2023 (EMNLP Findings). **arXiv:2310.15916**. ICL compresses demos into one hidden-state vector. → behaviors live as additive directions.
- **The Linear Representation Hypothesis and the Geometry of LLMs** — Park, Choe, Veitch, 2023 (ICML 2024). **arXiv:2311.03658**. Defines a *causal inner product* — **caveat: naive Euclidean cosine between directions can mislead** without the right inner product. → governs how 4.2 cosine similarity should be computed.
- **Geometry of Categorical and Hierarchical Concepts** — Park, Choe, Jiang, Veitch, 2024. **arXiv:2406.01506**. Categorical concepts as polytopes; hierarchy ↔ orthogonality. → should preference directions be single vectors or structured?

## 4.3 Layerwise delta + fine-tuning mechanics (roadmap §4.3)

- **Mechanistically Analyzing the Effects of Fine-Tuning on Procedurally Defined Tasks** — Jain et al., 2023 (ICLR 2024). **arXiv:2311.12786**. Fine-tuning learns a minimal, revertible "wrapper," not broad rewrites. → predicts small localized deltas (4.3).
- **Fine-Tuning Enhances Existing Mechanisms (Entity Tracking)** — Prakash et al., 2024 (ICLR 2024). **arXiv:2402.14811**. Same circuit before/after FT; gains from sharpening. → confirm the same circuit carries the preference across checkpoints (4.5, 4.2).
- **What Makes and Breaks Safety Fine-tuning?** — Jain et al., 2024 (NeurIPS). **arXiv:2407.10264**. Safety FT = minimal transform onto a near-null direction. → test whether a low-rank/single-direction edit reproduces the change (4.2, 4.5).
- **A Mechanistic Understanding of DPO and Toxicity** — Lee et al., 2024 (ICML). **arXiv:2401.01967**. DPO learns an offset bypassing (not removing) toxic regions. → template for "is the FT effect in a recoverable subspace?"
- **Attribution Patching** — Nanda, 2023. *Blog (no arXiv)*: `neelnanda.io/mechanistic-interpretability/attribution-patching`. Gradient approximation to patching, 2 fwd + 1 bwd pass. → cheap per-layer effect deltas (4.3).
- **AtP\*** — Kramár et al., 2024. **arXiv:2403.00745**. Fixes attribution-patching false negatives. → when are layerwise attribution deltas trustworthy vs needing exact patching (4.3).

**LoRA mechanics**
- **LoRA: Low-Rank Adaptation** — Hu et al., 2021. **arXiv:2106.09685**. ΔW = BA low-rank. → exactly the LoRA-induced delta to analyze (4.3).
- **Intrinsic Dimensionality Explains FT Effectiveness** — Aghajanyan et al., 2020 (ACL 2021). **arXiv:2012.13255**. FT updates concentrate in a tiny subspace. → predicts heavy base/FT direction overlap (4.2).
- **LoRA Learns Less and Forgets Less** — Biderman et al., 2024 (TMLR). **arXiv:2405.09673**. Full-FT perturbations 10–100× higher rank than LoRA. → quantify rank/magnitude of LoRA deltas per layer (4.3).
- **LoRA vs Full Fine-tuning: An Illusion of Equivalence** — Shuttleworth et al., 2024. **arXiv:2410.21228**. LoRA introduces "intruder" singular directions absent in full FT. → SVD-of-ΔW to compare LoRA directions vs base/full-FT (4.2, 4.3).

## 4.4 SAE feature analysis (roadmap §4.4)

> **Critical availability finding:** public pretrained SAEs exist for **Gemma 2, Llama-3.1-8B, and Qwen3 (Qwen-Scope)** — **but NOT for Qwen2.5, and not for OLMo.** Use Qwen-Scope for the Qwen3 arm; **you must train your own Qwen2.5 SAEs** (TopK/JumpReLU, residual stream, matched width/sparsity) for a fair comparison.

**Foundations**
- **Towards Monosemanticity** — Bricken, Templeton, Batson et al. (Anthropic), 2023. *transformer-circuits.pub*. SAE on 1-layer MLP → >4k monosemantic features. → the core extraction recipe (4.4).
- **Scaling Monosemanticity (Claude 3 Sonnet)** — Templeton et al., 2024. *transformer-circuits.pub* (also **arXiv:2605.29358**, ⚠ verify). Up to 34M features; feature-clamping steers behavior. → features are causally steerable (4.4, 4.5).
- **Sparse Autoencoders Find Highly Interpretable Features** — Cunningham, Ewart, Riggs, Huben, Sharkey, 2023 (ICLR 2024). **arXiv:2309.08600**. SAE features beat neurons on interpretability.
- **Toy Models of Superposition** — Elhage et al., 2022. **arXiv:2209.10652**. Why per-neuron analysis misses the signal → motivates SAEs.

**Architectures (for training your own Qwen2.5 SAEs)**
- **Gated SAEs** — Rajamanoharan et al., 2024. **arXiv:2404.16014**. Removes activation shrinkage (less magnitude bias for before/after measurement).
- **JumpReLU SAEs** — Rajamanoharan et al., 2024. **arXiv:2407.14435**. Learnable threshold; underlies Gemma Scope.
- **Scaling and Evaluating SAEs (TopK)** — Gao et al. (OpenAI), 2024. **arXiv:2406.04093**. Fixes sparsity directly; dominant for new suites.
- **Matryoshka SAEs** — Bussmann et al., 2025. **arXiv:2503.17547**. Nested dictionaries prevent feature absorption → better cross-model matching.

**Open suites (availability audit)**
- **Gemma Scope** — Lieberum et al. (GDM), 2024. **arXiv:2408.05147**. 400+ JumpReLU SAEs on Gemma 2 (base+instruct). → add Gemma as a control.
- **Llama Scope** — He, Ge et al. (OpenMOSS), 2024. **arXiv:2410.20526**. 256 TopK SAEs on Llama-3.1-8B-Base; tests base-SAE → fine-tuned-model generalization (exactly your before/after setting).
- **Qwen-Scope** — Qwen Team, 2026. **arXiv:2605.11887** (⚠ verify). Official residual-stream SAEs for **Qwen3 / Qwen3.5** (e.g. `Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_100`). **Confirms Qwen2.5 is not covered.**

**Feature ablation / causal use**
- **SAE-TS: Improving Steering Vectors by Targeting SAE Features** — Chalnev, Siu, Conmy, 2024. **arXiv:2411.02193**. Hit a target feature while minimizing side effects (4.5).
- **Steering LM Refusal with SAEs** — O'Brien, Majercak et al. (Microsoft), 2024. **arXiv:2411.11296**. "Ablate-these-features, measure behavior" template (4.5).
- **Beyond Input Activations: Gradient SAEs** — Yu et al., 2025. **arXiv:2505.08080**. **Caveat: the most-*changed* feature ≠ the *causal* feature** — rank by gradient influence before ablation (bridges 4.4→4.5).

**Cross-model feature overlap (roadmap §4.4 cross-model)**
- **SAEs Reveal Universal Feature Spaces Across LLMs** — Lan, Torr, Meek, Barez, 2024. **arXiv:2410.06981**. Method to quantify shared feature subspaces. → test Qwen2.5/Qwen3 changed-feature overlap.
- **Universal Sparse Autoencoders** — Thasarathan, Forsyth, Fel et al., 2025. **arXiv:2502.03714**. One SAE jointly on multiple models → shared dictionary by construction.
- **Transferring Linear Features Across LMs With Model Stitching** — Lan/Chen et al., 2025. **arXiv:2506.06609**. Affine residual-stream map transfers SAEs/probes/steering across models. → map a Qwen3 SAE onto Qwen2.5 (partial mitigation for the missing suite). *(Also load-bearing for 4.5 — see below.)*
- **Towards Universality: Mechanistic Similarity Across Architectures** — Lan et al., 2024. **arXiv:2410.06672**. Prior/baseline for expected feature-overlap rate.

**SAEs for fine-tuning / auditing**
- **Auditing LMs for Hidden Objectives** — Marks, Treutlein, Bricken, Lindsey et al. (Anthropic), 2025. **arXiv:2503.10965**. SAEs surface a known implanted hidden objective. → precedent that SAE features reveal training-induced hidden behavior (4.4).
- **Open Problems in Mechanistic Interpretability** — Sharkey, Chughtai, Batson, Lindsey et al., 2025. **arXiv:2501.16496**. SAE limitations (absorption, eval). → scope the caveats honestly.
- **Layer-Wise Evolution of Representations in Fine-Tuned Transformers (SAEs)** — 2025. **arXiv:2502.16722** (⚠ verify authors). Pre vs post-FT SAE comparison layer-by-layer. → direct method match for the before/after diff.

## 4.5 Causal intervention & cross-model patching (roadmap §4.5)

**Patching foundations**
- **ROME / causal tracing** — Meng et al., 2022. **arXiv:2202.05262** (see §1). → corrupted→clean intervention design.
- **Causal Mediation Analysis (gender bias)** — Vig et al., 2020. **arXiv:2004.12265**. Direct vs indirect (mediated) effects. *(arXiv title differs from NeurIPS proceedings title — same paper.)*
- **How to Use and Interpret Activation Patching** — Heimersheim, Nanda, 2024. **arXiv:2404.15255**. Best-practices + pitfalls. → methodological backbone for 4.5.
- **ACDC — Towards Automated Circuit Discovery** — Conmy et al., 2023 (NeurIPS). **arXiv:2304.14997**. Iterative edge patching + pruning.
- **IOI — Interpretability in the Wild** — Wang et al., 2022. **arXiv:2211.00593**. Head-level patching/knockout methodology.
- **Path Patching** — Goldowsky-Dill et al., 2023. **arXiv:2304.05969**. Edge/path localization.
- **EAP — Attribution Patching Outperforms ACDC** — Syed et al., 2023. **arXiv:2310.10348**. Cheap automated edge discovery before exact patches.

**Causal abstraction / DAS**
- **DAS — Finding Alignments Between Interpretable Causal Variables and Distributed Representations** — Geiger et al., 2023. **arXiv:2303.02536**. Learned rotated subspace aligning causal variables → natural cross-model alignment target. *(Note: the often-miscited "Finding Alignment Features…" title does not exist.)*
- **Causal Abstractions of Neural Networks** — Geiger et al., 2021. **arXiv:2106.02997**. Interchange-intervention semantics.
- **Causal Abstraction: A Theoretical Foundation for Mech Interp** — Geiger et al., 2023 (JMLR 2025). **arXiv:2301.04709**. Unifies patching/tracing/SAEs/DAS/steering.

**Cross-model patching / stitching / alignment (hardest, most novel)**
- **Activation Space Interventions Can Be Transferred Between LLMs** — Oozeer et al., 2025. **arXiv:2503.04429**. Learned maps between activation spaces of *different families* (Llama/Qwen/Gemma) so interventions transfer; handles dim mismatch. → **closest existing proof-of-concept for the Qwen2.5↔Qwen3 transplant in 4.5.**
- **Transferring Linear Features Across LMs With Model Stitching** — Chen/Lan et al., 2025. **arXiv:2506.06609**. Affine residual-stream map between different-sized LMs.
- **Revisiting Model Stitching to Compare Neural Representations** — Bansal et al., 2021. **arXiv:2106.07682**. Single trainable affine layer; stitching-penalty metric. → does a transplant "take"?
- **Relative Representations Enable Zero-Shot Latent Communication** — Moschella et al., 2022 (ICLR 2023). **arXiv:2209.15430**. Anchor-similarity encoding → training-free shared coordinate system.
- **Harnessing the Universal Geometry of Embeddings (vec2vec)** — Jha et al., 2025. **arXiv:2505.12540**. *Unsupervised* (no paired data) translation between embedding spaces.
- **The Platonic Representation Hypothesis** — Huh et al., 2024. **arXiv:2405.07987**. Representations converge with scale → a shared latent coordinate system plausibly exists.
- **CKA — Similarity of NN Representations Revisited** — Kornblith et al., 2019. **arXiv:1905.00414**. Layer-matching across inits/models.
- **SVCCA** — Raghu et al., 2017. **arXiv:1706.05806**. Affine/dim-mismatch-invariant layer comparison.
- **Model stitching origin (equivariance/equivalence)** — Lenc, Vedaldi, 2014. **arXiv:1411.5908**.

> **Feasibility gap (flagged by the search) — read before scoping §4.5.** Cross-model activation
> patching between *different architectures* is **not solved end-to-end**. Three obstacles:
> (i) no shared coordinate system (a raw activation from model A is meaningless to B),
> (ii) hidden-dim mismatch (needs a projection, not a copy),
> (iii) no token alignment (different tokenizers → position *t* in A ≠ *t* in B).
> Learned stitching/affine maps (2106.07682, 2506.06609, 2503.04429) handle (i)+(ii) and preserve
> *causal* behavior but generally need *paired* activations. Relative reps (2209.15430) and vec2vec
> (2505.12540) give shared spaces (the latter without paired data) but are validated for
> embedding similarity/translation, **not** for faithful mid-network transplants with downstream
> control. **Still open:** a method that aligns token positions across tokenizers, learns the map
> without a large paired corpus, *and* yields causally faithful *intermediate-layer* transplants
> (validated by stitching penalty + behavioral causal check). That combination is exactly what a
> Qwen2.5↔Qwen3 patching contribution would target — treat §4.5 as a research bet, not a given.

---

## 5. Tokenization (predictor in roadmap §1.1 / §2.4)

- **Counting Ability of LLMs and Impact of Tokenization** — 2024. **arXiv:2410.19730**. Single-token (single-digit) tokenization improves performance. → single- vs multi-token targets change learnability.
- **Toward a Theory of Tokenization in LLMs** — Rajaraman et al., 2024. **arXiv:2404.08335**. How tokenization shapes representable/learnable structure.
- **Understanding and Mitigating Tokenization Bias** — 2024. **arXiv:2406.16829**. BPE distorts next-token probabilities → matters when reading logits for preference measurement.

---

## 6. Persona / RLHF amplification & baseline-prior disentanglement (roadmap §2.4, §3)

- **Persona Features Control Emergent Misalignment** — Wang et al. (OpenAI), 2025. **arXiv:2506.19823**. SAE "persona features" causally control emergent misalignment; narrow FT shifts compact persona features. → mechanistic substrate; candidate variable differing across families.
- **LIMA / Superficial Alignment Hypothesis** — Zhou, Liu, Xu et al. (Meta), 2023. **arXiv:2305.11206**. Capabilities from pretraining; alignment mostly teaches style → observed post-FT behavior is often a surfaced prior. → strict baseline controls for §2.4.
- **How RLHF Amplifies Sycophancy** — 2026. **arXiv:2602.01002** (⚠ verify). Reward optimization amplifies pre-existing biases. → an apparent "transfer increase" may be a prior amplified by RL; relevant to Qwen2.5's heavier RL vs Qwen3's distillation recipe.
- **Layer by Layer: Where Multi-Task Learning Happens in Instruction-Tuned LLMs** — 2024. **arXiv:2410.20008** (⚠ verify authors). Instruction tuning changes mostly mid transitional layers, small subspaces. → where to look for the transfer signal.
- **From Data to Behavior: Predicting Unintended Model Behaviors Before Training** — Wang et al., 2026. **arXiv:2602.04735** (⚠ verify). Detects data-induced biases pre-training via mean representations, no parameter updates. → a pre-training control to separate prior from induced effect (2.4).

**Sample efficiency / scaling (roadmap §1.3 transfer-efficiency curves)**
- **Scaling Laws for Transfer** — Hernandez, Kaplan, Henighan, McCandlish, 2021. **arXiv:2102.01293**. "Effective data transferred" power law. → backbone for the 100/500/1k/5k/10k curves.
- **When Scaling Meets LLM Finetuning** — Zhang et al. (Google), 2024. **arXiv:2402.17193**. Multiplicative joint scaling law (data × model/pretraining). → how to fit/interpret transfer-vs-size curves.
- **Physics of LMs 3.3: Knowledge Capacity Scaling Laws** — Allen-Zhu, Li, 2024 (ICLR 2025). **arXiv:2404.05405**. ~2 bits/param capacity; MoE/quantization/SNR effects. → trait-acquisition capacity framing.

---

## Citation cautions

Verified-but-flag-before-formal-citation (future-dated or author-unconfirmed; plausible given the
Jun 2026 environment date, but open the arXiv page first):
- **arXiv:2605.11887** (Qwen-Scope) and **arXiv:2605.29358** (Scaling Monosemanticity arXiv listing) — both have non-arXiv primary sources (Qwen HF collection / transformer-circuits.pub) you can cite instead/alongside.
- **arXiv:2602.01002** (RLHF amplifies sycophancy), **arXiv:2602.04735** (Data2Behavior), **arXiv:2502.16722**, **arXiv:2410.20008** — confirm authors/title on the abstract page.

**Excluded as likely-fabricated** (surfaced in search but not anchored to a real abstract page):
`2606.11270` ("Quantifying Subliminal Behavioral Transfer Ratios…") and `2602.11091`. Do **not** cite
without independent verification.

No-arXiv primary sources (cite by URL): Logit Lens (nostalgebraist, LessWrong), Logit Prisms
(neuralblog), Direct Logit Attribution + Towards/Scaling Monosemanticity + Toy Models
(transformer-circuits.pub), Attribution Patching (neelnanda.io).

---

## Actionable takeaways for the experiment design

1. **Core-question priors converge:** shared-init dependence (2507.14805) + Qwen2.5 being independently the most susceptible family (2502.17424). First testable hypothesis: Qwen3's distillation-heavy post-training (2505.09388) breaks the effective shared-init condition vs Qwen2.5's RL-heavy recipe (2412.15115).
2. **Reference implementations exist:** LogitLens4LLMs (2503.11667) for the Qwen logit-lens pipeline; RepBend (2504.01550) for the pre/post-FT logit-lens heatmap; Refusal-single-direction (2406.11717) and Persona Vectors (2507.21509) for the steering + direction-overlap design.
3. **Biggest infra gap (§4.4):** no public Qwen2.5 SAEs (and none for OLMo) — budget to **train your own** matched to Qwen-Scope, or use cross-model SAE stitching (2506.06609) to share a frame.
4. **Biggest research-risk gap (§4.5):** cross-architecture activation patching is unsolved end-to-end; 2503.04429 is the closest precedent. Scope it as a bet with a fallback (within-family patching, or shared-SAE-space comparison) rather than a guaranteed deliverable.
5. **Two methodological caveats to bake in:** (a) cosine similarity between directions needs the causal inner product, not naive Euclidean (2311.03658); (b) the most-*changed* SAE feature is not necessarily the *causal* one — rank by gradient influence before ablation (2505.08080).
6. **Always control for the prior (§2.4):** LIMA (2305.11206) + RLHF-amplification (2602.01002) mean an apparent transfer can be a surfaced/amplified pretraining bias — a target counts as "transferable" only if the increase survives baseline controls, per the roadmap's own decision criterion.
