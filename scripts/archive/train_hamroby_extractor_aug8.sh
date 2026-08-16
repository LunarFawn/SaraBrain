#!/usr/bin/env bash
# Retrain hamroby_extractor_v1 with multi-object Pair labels.
#
# Previous iterations (aug2-aug7) trained on per-conjunct Pairs for
# conj-of-dobj prose — one Pair per conjunct, each with one object span.
# That structure pulled the loss in opposing directions (pair-1: label
# o1 NOT o2; pair-2: label o2 NOT o1) and the model converged on "pick
# one." This run uses Pair.additional_object_spans so a single training
# example carries multi-B-O labels for ALL conjuncts.
#
# Keeps the gold-UD-features path from aug7 (REAL_PROSE=20000). All
# other defaults unchanged.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
exec env \
  OUT="src/sara_brain/cortex/checkpoints/hamroby_extractor_v1_aug8.pt" \
  REAL_PROSE=20000 \
  bash "$REPO/scripts/train_hamroby_extractor.sh" "$@"
