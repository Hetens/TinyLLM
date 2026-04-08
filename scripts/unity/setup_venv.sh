#!/bin/bash
# ============================================================================
# setup_venv.sh — One-time virtual-environment setup for TinyLLM on Unity.
#
# Run this ONCE from the repo root on a login node (no sbatch needed):
#     bash scripts/unity/setup_venv.sh
#
# It creates  ~/venvs/tinyllm  with all required Python packages.
# ============================================================================

set -euo pipefail

VENV_DIR="$HOME/venvs/tinyllm"

echo ">>> Loading modules …"
module load python/3.11.7
module load cuda/12.6

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    echo "To recreate, delete it first:  rm -rf $VENV_DIR"
    exit 0
fi

echo ">>> Creating virtual environment at $VENV_DIR …"
python -m venv "$VENV_DIR"

echo ">>> Activating venv …"
source "$VENV_DIR/bin/activate"

echo ">>> Upgrading pip …"
pip install --upgrade pip

echo ">>> Installing PyTorch (CUDA 12.6) …"
pip install torch --index-url https://download.pytorch.org/whl/cu126

echo ">>> Installing project requirements …"
pip install -r requirements.txt

echo ">>> Installing extra packages used by pretrain.py …"
pip install wandb coolname pydantic argdantic huggingface_hub

echo ""
echo "=== Setup complete ==="
echo "Activate with:  source $VENV_DIR/bin/activate"
echo ""
