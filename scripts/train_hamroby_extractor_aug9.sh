#!/usr/bin/env bash
# Retrain hamroby_extractor_v1 with two fixes layered on aug8:
#
# 1. ud_triple_extractor obl fix — skip `obl` when a primary `obj`
#    exists (and always for copular predicates). Eliminates the
#    "Cluster analysis groups proteins by similarity" over-extraction
#    where 'similarity' was emitted as a spurious second object.
#
# 2. spaCy CASCADE: en_core_web_sm primary + en_core_web_trf fallback.
#    Most sentences (~99%) parse fine with sm (5ms). Degenerate parses
#    where no VERB or AUX appears in the tree (e.g. "DNA and RNA share
#    base pairing." — sm mis-POSes `share` as NOUN) transparently
#    retry on trf (~30ms). The conj-POS filter is relaxed to accept
#    consistent (NOUN, NOUN) OR (PROPN, PROPN) patterns since trf
#    labels nonsense tokens as PROPN consistently — both are valid
#    feature distributions for the model.
#
# Keeps all aug8 architecture (multi-object Pair labels, gold UD
# features in real_prose_pairs, REAL_PROSE=20000).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
exec env \
  OUT="src/sara_brain/cortex/checkpoints/hamroby_extractor_v1_aug9.pt" \
  REAL_PROSE=20000 \
  bash "$REPO/scripts/train_hamroby_extractor.sh" "$@"
