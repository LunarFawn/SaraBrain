#!/usr/bin/env bash
# Generate N synthetic nonsense-word substrates and build three
# cumulative HamRobySum training JSONLs for v036 curriculum training.
#
# Phase 1 corpus = small substrates only.
# Phase 2 corpus = small + medium substrates.
# Phase 3 corpus = small + medium + large substrates.
#
# Anti-forgetting: each cumulative corpus includes the prior phase's
# data so the model doesn't unlearn earlier patterns when it sees
# harder ones.
#
# Env vars (all optional):
#   BRAIN_DIR    — where to write substrate .db files (default: /tmp/synth_brains)
#   OUT_DIR      — where to write phase JSONLs        (default: /tmp)
#   N_SMALL      — small substrate count              (default: 60)
#   N_MEDIUM     — medium substrate count             (default: 30)
#   N_LARGE      — large substrate count              (default: 10)
#   AUGMENT      — augment-multiplier for synth_data  (default: 2)
#   MAX_SEQ      — max tokens per row                 (default: 256)
#   SEED_BASE    — base seed (each substrate uses SEED_BASE + i)
#                                                     (default: 1000)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

BRAIN_DIR="${BRAIN_DIR:-/tmp/synth_brains}"
OUT_DIR="${OUT_DIR:-/tmp}"
N_SMALL="${N_SMALL:-60}"
N_MEDIUM="${N_MEDIUM:-30}"
N_LARGE="${N_LARGE:-10}"
AUGMENT="${AUGMENT:-2}"
MAX_SEQ="${MAX_SEQ:-256}"
SEED_BASE="${SEED_BASE:-1000}"

if [ ! -x "$REPO/.venv/bin/python" ]; then
  echo "$REPO/.venv missing — set up the project venv first" >&2
  exit 1
fi

# Clean previous run so generate_synthetic_substrate.py's no-overwrite
# guard doesn't fire.
rm -rf "$BRAIN_DIR"
mkdir -p "$BRAIN_DIR"

GEN=papers/instrument_validation/generate_synthetic_substrate.py

gen_bucket() {
  # gen_bucket <bucket_name> <count> <concepts> <triples> <seed_offset>
  local name="$1"
  local count="$2"
  local concepts="$3"
  local triples="$4"
  local seed_off="$5"
  echo "[$name] generating $count substrates  (concepts=$concepts triples=$triples)"
  for i in $(seq 0 $((count - 1))); do
    local seed=$((SEED_BASE + seed_off + i))
    .venv/bin/python "$GEN" \
      --out "$BRAIN_DIR/${name}_$(printf '%03d' $i).db" \
      --concepts "$concepts" \
      --triples "$triples" \
      --seed "$seed" \
      > /dev/null
  done
}

gen_bucket small  "$N_SMALL"  10  30  0
gen_bucket medium "$N_MEDIUM" 30  80  10000
gen_bucket large  "$N_LARGE"  100 250 20000

ls "$BRAIN_DIR"/small_*.db  > /tmp/.synth_small_list  2>/dev/null || true
ls "$BRAIN_DIR"/medium_*.db > /tmp/.synth_medium_list 2>/dev/null || true
ls "$BRAIN_DIR"/large_*.db  > /tmp/.synth_large_list  2>/dev/null || true

build_phase() {
  # build_phase <phase> <list_file_1> [list_file_2] [list_file_3]
  local phase="$1"
  shift
  local out="$OUT_DIR/synth_pairs_phase${phase}.jsonl"
  local brain_args=()
  for list in "$@"; do
    while IFS= read -r p; do
      brain_args+=(--brain "$p")
    done < "$list"
  done
  echo "[phase $phase] building corpus -> $out  (${#brain_args[@]} brain args)"
  .venv/bin/python -m sara_brain.cortex.transformer.synth_data \
    "${brain_args[@]}" \
    --serialize-out "$out" \
    --augment-multiplier "$AUGMENT" \
    --max-seq "$MAX_SEQ" \
    | tail -7
}

build_phase 1 /tmp/.synth_small_list
build_phase 2 /tmp/.synth_small_list /tmp/.synth_medium_list
build_phase 3 /tmp/.synth_small_list /tmp/.synth_medium_list /tmp/.synth_large_list

cat <<EOF

done. corpus files:
  $OUT_DIR/synth_pairs_phase1.jsonl  (small only)
  $OUT_DIR/synth_pairs_phase2.jsonl  (small + medium)
  $OUT_DIR/synth_pairs_phase3.jsonl  (small + medium + large)

next: launch curriculum training (one phase at a time, in tmux):

  PAIRS=$OUT_DIR/synth_pairs_phase1.jsonl \\
  CKPT_NAME=hamroby_sum_v3_phase1 \\
  STEPS=1500 \\
  SESSION=sara-synth-p1 \\
  ./scripts/train_hamrobysum.sh

  PAIRS=$OUT_DIR/synth_pairs_phase2.jsonl \\
  CKPT_NAME=hamroby_sum_v3_phase2 \\
  STEPS=2000 \\
  SESSION=sara-synth-p2 \\
  RESUME_FROM=src/sara_brain/cortex/checkpoints/hamroby_sum_v3_phase1_001500.pt \\
  ./scripts/train_hamrobysum.sh

  PAIRS=$OUT_DIR/synth_pairs_phase3.jsonl \\
  CKPT_NAME=hamroby_sum_v3_phase3 \\
  STEPS=2000 \\
  SESSION=sara-synth-p3 \\
  RESUME_FROM=src/sara_brain/cortex/checkpoints/hamroby_sum_v3_phase2_002000.pt \\
  ./scripts/train_hamrobysum.sh
EOF
