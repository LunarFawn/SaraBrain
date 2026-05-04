#!/usr/bin/env bash
# Launch HamRobySum (synth) training in a detached tmux session with a
# side pane for nvidia-smi. Reattach with: tmux attach -t sara-synth
#
# Same shape as scripts/train_grammar.sh.
#
# Env vars (all optional):
#   SESSION      — tmux session name      (default: sara-synth)
#   PAIRS        — serialized JSONL path  (default: /tmp/synth_pairs_v2.jsonl)
#   L2_CKPT      — base L2-en checkpoint  (default: src/sara_brain/cortex/checkpoints/l2_en_003000.pt)
#   RESUME_FROM  — prior synth ckpt path  (overrides L2_CKPT for curriculum
#                                          phases 2/3 — see v036)
#   STEPS        — training steps         (default: 3000)
#   UNFREEZE     — --unfreeze-top-n value (default: 2)
#   CKPT_NAME    — output ckpt stem       (default: hamroby_sum_v2)
#   LOG          — log file               (default: $REPO/training_synth.log)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

SESSION="${SESSION:-sara-synth}"
PAIRS="${PAIRS:-/tmp/synth_pairs_v2.jsonl}"
L2_CKPT="${L2_CKPT:-src/sara_brain/cortex/checkpoints/l2_en_003000.pt}"
RESUME_FROM="${RESUME_FROM:-}"
STEPS="${STEPS:-3000}"
UNFREEZE="${UNFREEZE:-2}"
CKPT_NAME="${CKPT_NAME:-hamroby_sum_v2}"
LOG="${LOG:-$REPO/training_synth.log}"

if ! command -v tmux >/dev/null; then
  echo "tmux not installed" >&2
  exit 1
fi

if [ ! -x "$REPO/.venv/bin/python" ]; then
  echo "$REPO/.venv missing — set up the project venv first" >&2
  exit 1
fi

if [ ! -f "$PAIRS" ]; then
  echo "pairs file not found: $PAIRS" >&2
  echo "generate it first with: ./scripts/build_synth_corpus.sh" >&2
  exit 1
fi

if [ -n "$RESUME_FROM" ]; then
  if [ ! -f "$RESUME_FROM" ]; then
    echo "RESUME_FROM checkpoint not found: $RESUME_FROM" >&2
    exit 1
  fi
elif [ ! -f "$L2_CKPT" ]; then
  echo "L2-en checkpoint not found: $L2_CKPT" >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session '$SESSION' already exists. attach with: tmux attach -t $SESSION"
  echo "or kill it: tmux kill-session -t $SESSION"
  exit 1
fi

CMD=".venv/bin/python -m sara_brain.cortex.transformer.train_synth"
if [ -n "$RESUME_FROM" ]; then
  CMD+=" --resume-from '$RESUME_FROM'"
else
  CMD+=" --l2-ckpt '$L2_CKPT'"
fi
CMD+=" --pairs '$PAIRS'"
CMD+=" --unfreeze-top-n $UNFREEZE"
CMD+=" --steps $STEPS"
CMD+=" --ckpt-name $CKPT_NAME"
CMD+=" 2>&1 | tee -a '$LOG'"

tmux new-session -d -s "$SESSION" -c "$REPO" "$CMD"
tmux set-option -t "$SESSION" -g remain-on-exit on
tmux set-option -t "$SESSION" -g mouse on
tmux split-window -h -t "$SESSION" -l 60 "watch -n2 -t nvidia-smi"
tmux select-pane -t "$SESSION":0.0

cat <<EOF
launched tmux session: $SESSION
  pairs=$PAIRS
  $( [ -n "$RESUME_FROM" ] && echo "resume_from=$RESUME_FROM" || echo "l2_ckpt=$L2_CKPT" )
  steps=$STEPS  unfreeze_top_n=$UNFREEZE
  ckpt_name=$CKPT_NAME
  log=$LOG

attach:    tmux attach -t $SESSION
detach:    Ctrl-b d
tail log:  tail -f $LOG
close pane after run: Ctrl-b x
kill all:  tmux kill-session -t $SESSION
EOF
