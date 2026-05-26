#!/usr/bin/env bash
# Retrain hamroby_extractor_v1 with the gold-UD-features path enabled.
#
# REAL_PROSE=20000 mixes 20k UD treebank sentences in alongside the
# 30k synthetic scenes. Each UD sentence yields one or more Pair
# records carrying gold UD POS+dep+head as `pre_parsed` ParsedSentence,
# so the training pipeline skips the spaCy re-parse on those pairs and
# uses gold features directly. This eliminates the ~46% conj-position
# POS noise that spaCy hallucinates on delexicalized real prose.
#
# Wraps scripts/train_hamroby_extractor.sh — same other defaults.
# Note: data-gen will be slower (UD parse + delex per sentence on top
# of the synthetic generation).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
exec env \
  OUT="src/sara_brain/cortex/checkpoints/hamroby_extractor_v1_aug7.pt" \
  REAL_PROSE=20000 \
  bash "$REPO/scripts/train_hamroby_extractor.sh" "$@"
