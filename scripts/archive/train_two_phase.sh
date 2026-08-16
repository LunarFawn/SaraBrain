#!/bin/bash
# Two-phase Sara Cortex training — from scratch, no borrowed weights.
#
# Phase 1: Substrate language model (next-token prediction)
# Phase 2: Reasoning head (MCQ classification)
#
# Total time: ~2 hours on RTX 3070
#
# Usage: bash scripts/train_two_phase.sh

set -e
cd /home/grizzlyengineer/repo/SaraBrain

echo "=== Two-Phase Sara Cortex Training ==="
echo "Start: $(date)"
echo ""

# ---- Phase 0: Generate LM training data ----
LM_DATA="training_data/substrate_lm_100k.txt"
if [ ! -f "$LM_DATA" ]; then
    echo "=== Phase 0: Generating substrate LM data ==="
    .venv/bin/python -u scripts/generate_substrate_lm_data.py \
        --num-substrates 5000 \
        --queries-per-substrate 20 \
        --out "$LM_DATA"
    echo ""
fi
echo "LM data: $(wc -c < "$LM_DATA") bytes"

# ---- Phase 1: Train substrate language model ----
echo ""
echo "=== Phase 1: Substrate Language Model ==="
.venv/bin/python -u scripts/train_substrate_lm.py \
    --data "$LM_DATA" \
    --out models/sara-cortex-lm-v1 \
    --steps 20000 \
    --batch-size 32

echo ""
echo "Phase 1 complete. Checkpoint: models/sara-cortex-lm-v1/best.pt"

# ---- Phase 2: Train reasoning head ----
echo ""
echo "=== Phase 2: Substrate Reasoning Head ==="
.venv/bin/python -u scripts/train_phase2_reasoning.py \
    --lm-checkpoint models/sara-cortex-lm-v1/best.pt \
    --data training_data/sara_cortex_synthetic_10k.jsonl \
    --out models/sara-cortex-final-v1 \
    --steps 5000

echo ""
echo "=== Done: $(date) ==="
echo "Final model: models/sara-cortex-final-v1/"
