#!/usr/bin/env bash
# v037 — build the two layered HamRobySum training corpora.
#
# Phase Core: nonsense concepts AND nonsense relations.
#             HamRobySum-Core trains on this — pure structural
#             composition, zero real-language exposure.
#
# Phase EN:   nonsense concepts + REAL English relations.
#             HamRobySum-EN trains on this resuming from Core —
#             learns which English verbs slot where without ever
#             memorizing real-content fact patterns.
#
# Anti-forgetting: each phase uses the SAME size mixing (small +
# medium + large) so going from Core to EN doesn't change cluster
# complexity, only relation vocabulary.
#
# Env vars (all optional):
#   BRAIN_DIR    — substrate .db output dir         (default: /tmp/synth_brains_v037)
#   OUT_DIR      — phase JSONL output dir           (default: /tmp)
#   N_SMALL      — small substrate count            (default: 60)
#   N_MEDIUM     — medium substrate count           (default: 30)
#   N_LARGE      — large substrate count            (default: 10)
#   AUGMENT      — augment-multiplier               (default: 2)
#   MAX_SEQ      — max tokens per row               (default: 256)
#   SEED_BASE    — base seed (each substrate uses SEED_BASE + i)
#                                                   (default: 5000)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

BRAIN_DIR="${BRAIN_DIR:-/tmp/synth_brains_v037}"
OUT_DIR="${OUT_DIR:-/tmp}"
N_SMALL="${N_SMALL:-60}"
N_MEDIUM="${N_MEDIUM:-30}"
N_LARGE="${N_LARGE:-10}"
AUGMENT="${AUGMENT:-2}"
MAX_SEQ="${MAX_SEQ:-256}"
SEED_BASE="${SEED_BASE:-5000}"

if [ ! -x "$REPO/.venv/bin/python" ]; then
  echo "$REPO/.venv missing — set up the project venv first" >&2
  exit 1
fi

# Two side-by-side substrate directories. Same seeds across both so
# the only difference between Core and EN is the relations pool.
CORE_DIR="$BRAIN_DIR/core"
EN_DIR="$BRAIN_DIR/en"

rm -rf "$BRAIN_DIR"
mkdir -p "$CORE_DIR" "$EN_DIR"

GEN=papers/instrument_validation/generate_synthetic_substrate.py

gen_bucket() {
  # gen_bucket <flavor: core|en> <bucket: small|medium|large>
  #            <count> <concepts> <triples> <seed_offset>
  local flavor="$1"
  local name="$2"
  local count="$3"
  local concepts="$4"
  local triples="$5"
  local seed_off="$6"
  local out_dir
  if [ "$flavor" = "core" ]; then
    out_dir="$CORE_DIR"
    local extra_args="--nonsense-relations"
  else
    out_dir="$EN_DIR"
    local extra_args=""
  fi
  echo "[$flavor:$name] generating $count substrates  (concepts=$concepts triples=$triples)"
  for i in $(seq 0 $((count - 1))); do
    local seed=$((SEED_BASE + seed_off + i))
    .venv/bin/python "$GEN" \
      --out "$out_dir/${name}_$(printf '%03d' $i).db" \
      --concepts "$concepts" \
      --triples "$triples" \
      --seed "$seed" \
      $extra_args \
      > /dev/null
  done
}

# Generate both flavors in parallel buckets.
for flavor in core en; do
  gen_bucket "$flavor" small  "$N_SMALL"  10  30  0
  gen_bucket "$flavor" medium "$N_MEDIUM" 30  80  10000
  gen_bucket "$flavor" large  "$N_LARGE"  100 250 20000
done

# Build per-phase corpus from each flavor's substrates.
build_corpus() {
  # build_corpus <flavor> <out_filename>
  local flavor="$1"
  local out="$OUT_DIR/$2"
  local dir
  if [ "$flavor" = "core" ]; then dir="$CORE_DIR"; else dir="$EN_DIR"; fi
  local brain_args=()
  for f in "$dir"/*.db; do
    brain_args+=(--brain "$f")
  done
  echo "[$flavor] building corpus -> $out  (${#brain_args[@]} substrates)"
  .venv/bin/python -m sara_brain.cortex.transformer.synth_data \
    "${brain_args[@]}" \
    --serialize-out "$out" \
    --augment-multiplier "$AUGMENT" \
    --max-seq "$MAX_SEQ" \
    | tail -7
}

build_corpus core synth_pairs_core.jsonl
build_corpus en   synth_pairs_en.jsonl

cat <<EOF

done. corpus files:
  $OUT_DIR/synth_pairs_core.jsonl  (nonsense concepts + nonsense relations)
  $OUT_DIR/synth_pairs_en.jsonl    (nonsense concepts + real English relations)

next: launch v037 layered training (one phase at a time, in tmux).

PHASE CORE — pure structural composition, zero real language:
  PAIRS=$OUT_DIR/synth_pairs_core.jsonl \\
  CKPT_NAME=hamroby_sum_core \\
  STEPS=2500 \\
  SESSION=sara-synth-core \\
  ./scripts/train_hamrobysum.sh

PHASE EN — adds the English-verb overlay, resumes from Core:
  PAIRS=$OUT_DIR/synth_pairs_en.jsonl \\
  CKPT_NAME=hamroby_sum_en \\
  STEPS=2500 \\
  SESSION=sara-synth-en \\
  RESUME_FROM=src/sara_brain/cortex/checkpoints/hamroby_sum_core_002500.pt \\
  ./scripts/train_hamrobysum.sh
EOF
