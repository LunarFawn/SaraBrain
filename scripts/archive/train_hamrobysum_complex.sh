#!/usr/bin/env bash
# v048 — launch HamRobySum-EN-Complex training in a tmux session.
#
# Wraps train_hamrobysum.sh with v048-specific defaults: pairs from
# the complex corpus, ckpt name, resume-from v040 EN, etc. All
# defaults are overridable by env var.
#
# Env vars (all optional):
#   PAIRS        — corpus JSONL path     (default: /tmp/synth_pairs_complex.jsonl)
#   CKPT_NAME    — output ckpt stem      (default: hamroby_sum_en_complex)
#   STEPS        — training steps        (default: 4000)
#   SESSION      — tmux session name     (default: sara-synth-complex)
#   RESUME_FROM  — base ckpt to resume   (default: v040 EN at 002500)
#   UNFREEZE     — top-N unfreeze        (default: 2)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

export PAIRS="${PAIRS:-/tmp/synth_pairs_complex.jsonl}"
export CKPT_NAME="${CKPT_NAME:-hamroby_sum_en_complex}"
export STEPS="${STEPS:-4000}"
export SESSION="${SESSION:-sara-synth-complex}"
export RESUME_FROM="${RESUME_FROM:-src/sara_brain/cortex/checkpoints/hamroby_sum_en_002500.pt}"
export UNFREEZE="${UNFREEZE:-2}"

if [ ! -f "$PAIRS" ]; then
  echo "corpus not found: $PAIRS" >&2
  echo "build it first: ./scripts/build_complex_corpus.sh" >&2
  exit 1
fi

if [ ! -f "$RESUME_FROM" ]; then
  echo "v040 EN base checkpoint not found: $RESUME_FROM" >&2
  exit 1
fi

exec ./scripts/train_hamrobysum.sh
