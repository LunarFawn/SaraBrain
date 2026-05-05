#!/usr/bin/env bash
# v048 — build the complex-grammar HamRobySum training corpus.
#
# Generates `--substrates N` random substrates of varied scene
# counts and concatenates the per-substrate tokenized JSONLs into
# one training file. Each substrate uses an incrementing seed so
# the corpus is deterministic given SEED_BASE.
#
# Env vars (all optional):
#   OUT          — output JSONL path           (default: /tmp/synth_pairs_complex.jsonl)
#   BRAIN_DIR    — substrate .db output dir    (default: /tmp/complex_substrates)
#   N_SMALL      — small substrate count       (default: 40)   (40 scenes each)
#   N_MEDIUM     — medium substrate count      (default: 20)   (200 scenes each)
#   N_LARGE      — large substrate count       (default: 5)    (800 scenes each)
#   SEED_BASE    — base seed                   (default: 8000)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

OUT="${OUT:-/tmp/synth_pairs_complex.jsonl}"
BRAIN_DIR="${BRAIN_DIR:-/tmp/complex_substrates}"
N_SMALL="${N_SMALL:-40}"
N_MEDIUM="${N_MEDIUM:-20}"
N_LARGE="${N_LARGE:-5}"
SEED_BASE="${SEED_BASE:-8000}"

if [ ! -x "$REPO/.venv/bin/python" ]; then
  echo "$REPO/.venv missing — set up the project venv first" >&2
  exit 1
fi

rm -rf "$BRAIN_DIR" "$OUT"
mkdir -p "$BRAIN_DIR"

GEN=papers/instrument_validation/generate_complex_substrate.py

gen_bucket() {
  # gen_bucket <name> <count> <scenes> <seed_offset>
  local name="$1"
  local count="$2"
  local scenes="$3"
  local seed_off="$4"
  echo "[$name] generating $count substrates ($scenes scenes each)"
  for i in $(seq 0 $((count - 1))); do
    local seed=$((SEED_BASE + seed_off + i))
    .venv/bin/python "$GEN" \
      --out "$BRAIN_DIR/${name}_$(printf '%03d' $i).db" \
      --scenes "$scenes" \
      --seed "$seed" \
      > /dev/null
  done
}

gen_bucket small  "$N_SMALL"  40  0
gen_bucket medium "$N_MEDIUM" 200 10000
gen_bucket large  "$N_LARGE"  800 20000

# Concatenate every per-substrate tokenized JSONL into one corpus.
echo "[concat] building corpus -> $OUT"
total=0
for f in "$BRAIN_DIR"/*.tokenized.jsonl; do
  cat "$f" >> "$OUT"
  this=$(wc -l < "$f")
  total=$((total + this))
done

cat <<EOF

done. corpus file:
  $OUT  ($total tokenized rows from $((N_SMALL + N_MEDIUM + N_LARGE)) substrates)

next: launch v048 training (in tmux):

  PAIRS=$OUT \\
  CKPT_NAME=hamroby_sum_en_complex \\
  STEPS=4000 \\
  SESSION=sara-synth-complex \\
  RESUME_FROM=src/sara_brain/cortex/checkpoints/hamroby_sum_en_002500.pt \\
  ./scripts/train_hamrobysum.sh
EOF
