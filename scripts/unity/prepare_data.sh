#!/bin/bash
# ============================================================================
# prepare_data.sh — SLURM job: download + preprocess the Sudoku dataset.
#
# Submit from repo root:
#     sbatch scripts/unity/prepare_data.sh
#
# This only needs a CPU node and internet access to download from HuggingFace.
# Output goes to  data/sudoku-extreme-full/{train,test}/
# ============================================================================

#SBATCH --job-name=tinyllm-data
#SBATCH --partition=cpu-preempt
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/prepare_data_%j.out
#SBATCH --error=logs/prepare_data_%j.err

set -euo pipefail

REPO_DIR="$(pwd)"

mkdir -p logs

echo "=== TinyLLM data preparation ==="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Date:    $(date)"
echo ""

# ---- Modules & venv ----
module load python/3.11.7
source "$HOME/venvs/tinyllm/bin/activate"

# ---- Run the dataset builder ----
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/trm_base${PYTHONPATH:+:$PYTHONPATH}"

echo ">>> Building Sudoku dataset …"
python trm_base/build_sdku_data.py

echo ""
echo ">>> Data preparation complete."
ls -lh data/sudoku-extreme-full/train/
ls -lh data/sudoku-extreme-full/test/
