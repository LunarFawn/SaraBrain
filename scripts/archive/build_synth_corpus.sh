#!/usr/bin/env bash
# Build the multi-brain (edges, prose) corpus for HamRobySum training.
# Writes a single JSONL ready for train_hamrobysum.sh.
#
# Env vars (all optional, sensible defaults):
#   OUT      — output JSONL path  (default: /tmp/synth_pairs_v2.jsonl)
#   AUGMENT  — augment-multiplier (default: 2)
#   MAX_SEQ  — max tokens / row   (default: 256)
#   BRAINS   — space-separated list of brain.db paths
#              (default: every brain.db.* + aptamer_full.db.bak + /tmp/sara_demo.db)
#
# Examples:
#   ./scripts/build_synth_corpus.sh
#   OUT=/tmp/aptamer_only.jsonl BRAINS=aptamer_full.db.bak ./scripts/build_synth_corpus.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

OUT="${OUT:-/tmp/synth_pairs_v2.jsonl}"
AUGMENT="${AUGMENT:-2}"
MAX_SEQ="${MAX_SEQ:-256}"

DEFAULT_BRAINS=(
  /tmp/sara_demo.db
  aptamer_full.db.bak
  brain.db.bulk_reteach_backup
  brain.db.flatten_lift_backup
  brain.db.hand_curated_nopartof
  brain.db.hand_faithful
  brain.db.ch10_expanded
  brain.db.openie_no_verb_teach
  brain.db.openie_verbs
)
BRAINS="${BRAINS:-${DEFAULT_BRAINS[*]}}"

if [ ! -x "$REPO/.venv/bin/python" ]; then
  echo "$REPO/.venv missing — set up the project venv first" >&2
  exit 1
fi

# Build --brain flags from BRAINS list.
BRAIN_ARGS=()
for b in $BRAINS; do
  BRAIN_ARGS+=(--brain "$b")
done

echo "building synth corpus -> $OUT"
echo "  brains: $BRAINS"
echo "  augment_multiplier=$AUGMENT  max_seq=$MAX_SEQ"
echo

.venv/bin/python -m sara_brain.cortex.transformer.synth_data \
  "${BRAIN_ARGS[@]}" \
  --serialize-out "$OUT" \
  --augment-multiplier "$AUGMENT" \
  --max-seq "$MAX_SEQ"
