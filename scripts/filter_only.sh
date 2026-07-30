#!/bin/bash
#SBATCH --account=aip-rgrosse
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --output=logs/m1M-filter-cpu-%j.out
#SBATCH --error=logs/m1M-filter-cpu-%j.err
#SBATCH --job-name=m1M-filter-cpu

module load gcc arrow/23.0.1 python/3.11

cd /project/aip-rgrosse/agokrani/cl-with-sl-distillation
./.venv/bin/python scripts/filter_only.py "$@"
