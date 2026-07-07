# Setting up cl-with-sl on Rorqual (or any Alliance cluster)

Transfers the full subliminal-learning + J-lens project from Vulcan to
Rorqual. Code and small data come from GitHub; heavyweight artifacts
(adapters, datasets, probe outputs, fitted lenses) come from the private
HuggingFace dataset `agokrani/cl-with-sl-artifacts`.

## 0. What lives where

| Thing | Where | Notes |
|---|---|---|
| Code, findings, corpus, manifests | GitHub `agokrani/cl-with-sl` (branch `jspace`) | public |
| `data/experiments` tree (adapters, datasets, results) | HF dataset `agokrani/cl-with-sl-artifacts` → `data-experiments/` | private, needs `HF_TOKEN` |
| Raw probe outputs from Vulcan `$SCRATCH` | same dataset → `scratch/` | private |
| Fitted Jacobian lenses | same dataset → `jspace/` | uploaded as fits finish |
| Secrets (`.env`) | **nowhere** — copy by hand | never commit/upload |

## 1. Secrets first (manual, never via git/HF)

From your laptop or a Vulcan login node:

```bash
scp ~/projects/aip-rgrosse/agokrani/cl-with-sl/.env rorqual.alliancecan.ca:~/
```

## 2. Clone the repo

On `rorqual.alliancecan.ca` (login node — it has internet):

```bash
cd ~/projects/<your-allocation>/$USER   # or wherever you keep code
git clone --recurse-submodules -b jspace https://github.com/agokrani/cl-with-sl.git
cd cl-with-sl && mv ~/.env .
```

## 3. Pull the artifacts

```bash
set -a; source .env; set +a
module load python/3.11
pip install --no-index --user huggingface_hub  # if hf CLI not already available
huggingface-cli download agokrani/cl-with-sl-artifacts --repo-type dataset \
    --token "$HF_TOKEN" --local-dir "$SCRATCH/cl-with-sl-artifacts"

rsync -a "$SCRATCH/cl-with-sl-artifacts/data-experiments/" data/experiments/
mkdir -p "$SCRATCH/cl-with-sl"
rsync -a "$SCRATCH/cl-with-sl-artifacts/scratch/"          "$SCRATCH/cl-with-sl/"
mkdir -p "$SCRATCH/cl-with-sl/jspace"
rsync -a "$SCRATCH/cl-with-sl-artifacts/jspace/"           "$SCRATCH/cl-with-sl/jspace/" 2>/dev/null || true
```

Adapter symlinks from Vulcan were materialized during upload, so every
`data/experiments/*/seed_N/adapter/` arrives as real files — no HF-cache
pointer repair needed.

## 4. Environments + model cache

```bash
module spider arrow            # NEVER guess module versions; check first
module load gcc arrow python/3.11 cuda
bash scripts/setup_logit_probe_env.sh    # creates $SCRATCH/cl-with-sl-logit-probe-env

# Compute nodes are offline: pre-warm the HF cache on the login node.
export HF_HOME=$SCRATCH/hf-cache
huggingface-cli download Qwen/Qwen2.5-3B-Instruct
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507
```

`module load gcc arrow` is required every time anything imports
`datasets`/`pyarrow`.

## 5. Cluster-specific Slurm bits

- **GPUs**: Rorqual has H100-80GB (`h100`); the job scripts default to
  Vulcan's `l40s`. Don't edit the scripts — command-line flags override
  `#SBATCH` directives:

  ```bash
  sbatch --gpus-per-node=h100:1 scripts/fit_jlens.sh ...
  ```

- **Account**: check yours with
  `sacctmgr show associations user=$USER format=account%20` and pass
  `--account=...` (or `SLURM_ACCOUNT`) if it isn't `aip-rgrosse` there.
- H100s need torch >= 2.5.1; the probe env installs 2.12 — fine.
- `$SCRATCH` purge policy applies on Rorqual too (60 days); keep canonical
  artifacts in the HF dataset or `$PROJECT`.

## 6. Run things

```bash
# Fit a Jacobian lens (sharded; ~6-12 min/shard on H100)
for s in 0 1 2 3; do
  sbatch --gpus-per-node=h100:1 --time=1:00:00 --mem=48G --cpus-per-task=6 \
    scripts/fit_jlens.sh --model-id Qwen/Qwen2.5-3B-Instruct \
    --out $SCRATCH/cl-with-sl/jspace/qwen2_5_3b_instruct/lens.pt \
    --n-prompts 100 --shard-index $s --n-shards 4 --smoke-first
done
# then merge (CPU, login node is fine):
python scripts/merge_jlens.py --out $SCRATCH/cl-with-sl/jspace/qwen2_5_3b_instruct/lens.pt \
  $SCRATCH/cl-with-sl/jspace/qwen2_5_3b_instruct/lens.shard*of4.pt

# J-space readout for an experiment (baseline + 5 seeds)
sbatch --gpus-per-node=h100:1 scripts/run_jspace_readout.sh \
  --lens $SCRATCH/cl-with-sl/jspace/qwen2_5_3b_instruct/lens.pt \
  --experiment-dir data/experiments/owl-qwen2_5_3b_instruct \
  --output-dir $SCRATCH/cl-with-sl/jspace/readouts/owl-qwen2_5_3b_instruct

# Aggregate (CPU)
python scripts/aggregate_jspace.py --readout-root $SCRATCH/cl-with-sl/jspace/readouts
```

If the lenses were already fitted on Vulcan, skip fitting — step 3 pulls
them from the artifact repo and readouts can start immediately.
