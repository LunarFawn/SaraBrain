#!/bin/bash
# Setup for Sara cortex fine-tuning.
# Run once before training. Requires CUDA 12.x (you have 12.6).
#
# Usage: bash scripts/setup_finetune.sh

set -e

echo "=== Installing fine-tuning dependencies ==="
echo "CUDA version: $(nvidia-smi | grep 'CUDA Version' | awk '{print $9}')"

# Install unsloth (handles torch + CUDA automatically)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "trl>=0.7" "peft>=0.7" "accelerate" "bitsandbytes" "datasets"

echo ""
echo "=== Verifying ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
from unsloth import FastLanguageModel
print('unsloth: OK')
from trl import SFTTrainer
print('trl: OK')
print()
print('Ready to train. Run:')
print('  python scripts/finetune_sara_cortex.py --data training_data/sara_cortex_synthetic_400.jsonl --epochs 5')
"
