#!/usr/bin/env bash
# Retrain hamroby_extractor_v1 with the current synthetic_pairs.py
# (filters t_conjoined_object pairs to require spaCy POS=NOUN on both
# conjuncts, eliminating the 27% feature-mismatch noise) into the aug6
# checkpoint.
#
# Wraps scripts/train_hamroby_extractor.sh — same defaults (15000 steps,
# 30000 scenes, base size, batch 32, lr 5e-4), only the output path differs.
# Note: data generation is ~4-5 minutes slower due to spaCy parsing in
# the filter.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
exec env OUT="src/sara_brain/cortex/checkpoints/hamroby_extractor_v1_aug6.pt" \
  bash "$REPO/scripts/train_hamroby_extractor.sh" "$@"
