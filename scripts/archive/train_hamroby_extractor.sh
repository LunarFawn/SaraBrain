#!/usr/bin/env bash
# Launch hamroby_extractor_v1 training in a detached tmux session with a
# side pane for nvidia-smi. Each invocation gets a unique session name
# (timestamp-suffixed) so concurrent runs don't collide. Session stays
# open after training finishes (remain-on-exit) so you can reattach
# and read the final metrics whenever.
#
# Same shape as scripts/train_hamrobysum.sh.
#
# Env vars (all optional):
#   STEPS        — training steps                (default: 15000)
#   SCENES       — synthetic scenes for training (default: 30000)
#   EVAL_SCENES  — synthetic scenes for eval     (default: 2000)
#   SIZE         — model size: tiny|base|large   (default: base)
#   BATCH        — batch size                    (default: 32)
#   MAX_SEQ      — max sequence length (words)   (default: 64)
#   LR           — learning rate                 (default: 5e-4)
#   QUAL_PROB    — qualifier prob                (default: 0.6)
#   REAL_PROSE   — UD sentences for real-prose pairs (default: 0 = off;
#                  try 10000 for delexicalized real-distribution mix)
#   OUT          — output checkpoint path        (default:
#                  src/sara_brain/cortex/checkpoints/hamroby_extractor_v1.pt)
#   SESSION      — tmux session name (default: hamroby-extract-<timestamp>)
#   LOG          — log file (default: $REPO/training_hamroby_extract_<ts>.log)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

TS="$(date +%Y%m%d_%H%M%S)"
SESSION="${SESSION:-hamroby-extract-${TS}}"
LOG="${LOG:-$REPO/training_hamroby_extract_${TS}.log}"

STEPS="${STEPS:-15000}"
SCENES="${SCENES:-30000}"
EVAL_SCENES="${EVAL_SCENES:-2000}"
SIZE="${SIZE:-base}"
BATCH="${BATCH:-32}"
MAX_SEQ="${MAX_SEQ:-64}"
LR="${LR:-5e-4}"
QUAL_PROB="${QUAL_PROB:-0.6}"
REAL_PROSE="${REAL_PROSE:-0}"
OUT="${OUT:-src/sara_brain/cortex/checkpoints/hamroby_extractor_v1.pt}"

if ! command -v tmux >/dev/null; then
  echo "tmux not installed" >&2
  exit 1
fi

if [ ! -x "$REPO/.venv/bin/python" ]; then
  echo "$REPO/.venv missing — set up the project venv first" >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session '$SESSION' already exists. attach with: tmux attach -t $SESSION"
  echo "or kill it:                tmux kill-session -t $SESSION"
  exit 1
fi

mkdir -p "$(dirname "$OUT")"

CMD=".venv/bin/python -m sara_brain.cortex.transformer.hamroby_extractor_v1.train"
CMD+=" --out '$OUT'"
CMD+=" --size $SIZE"
CMD+=" --steps $STEPS"
CMD+=" --batch-size $BATCH"
CMD+=" --max-seq $MAX_SEQ"
CMD+=" --scenes $SCENES"
CMD+=" --eval-scenes $EVAL_SCENES"
CMD+=" --qualifier-prob $QUAL_PROB"
CMD+=" --lr $LR"
CMD+=" --real-prose-max-sentences $REAL_PROSE"
CMD+=" 2>&1 | tee -a '$LOG'"

tmux new-session -d -s "$SESSION" -c "$REPO" "$CMD"
tmux set-option -t "$SESSION" -g remain-on-exit on
tmux set-option -t "$SESSION" -g mouse on
tmux split-window -h -t "$SESSION" -l 60 "watch -n2 -t nvidia-smi"
tmux select-pane -t "$SESSION":0.0

cat <<EOF
launched tmux session: $SESSION
  out=$OUT
  size=$SIZE  steps=$STEPS  scenes=$SCENES  eval_scenes=$EVAL_SCENES
  batch=$BATCH  max_seq=$MAX_SEQ  lr=$LR  qual_prob=$QUAL_PROB
  real_prose=$REAL_PROSE
  log=$LOG

attach:                tmux attach -t $SESSION
detach:                Ctrl-b d
tail log:              tail -f $LOG
close pane after run:  Ctrl-b x
kill all:              tmux kill-session -t $SESSION
list sessions:         tmux ls
EOF
