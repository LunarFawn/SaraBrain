#!/usr/bin/env bash
# Retrain hamroby_extractor_v1 with the current synthetic_pairs.py
# (adds t_conjoined_object, t_compound_oblique, "by" in oblique preps)
# into the aug4 checkpoint.
#
# Wraps scripts/train_hamroby_extractor.sh — same defaults (15000 steps,
# 30000 scenes, base size, batch 32, lr 5e-4), only the output path differs.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
exec env OUT="src/sara_brain/cortex/checkpoints/hamroby_extractor_v1_aug4.pt" \
  bash "$REPO/scripts/train_hamroby_extractor.sh" "$@"
