#!/bin/bash
set -e

echo "============================================"
echo "  SD Inpainting Fine-Tune Demo - Setup"
echo "============================================"

# Check for conda
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Install Miniconda first:"
    echo "  brew install --cask miniconda"
    exit 1
fi

# Create environment
echo "[1/3] Creating conda environment..."
conda env create -f environment.yml --force

echo "[2/3] Activating environment..."
eval "$(conda shell.bash hook)"
conda activate sd-inpaint-finetune

echo "[3/3] Verifying installation..."
python -c "
import torch
import diffusers
import peft
print(f'PyTorch: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'Diffusers: {diffusers.__version__}')
print(f'PEFT: {peft.__version__}')
print('All dependencies OK!')
"

echo ""
echo "Setup complete! Activate with:"
echo "  conda activate sd-inpaint-finetune"
echo ""
echo "Then run the full pipeline:"
echo "  python run_demo.py"
