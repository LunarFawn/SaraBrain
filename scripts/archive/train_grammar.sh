#!/usr/bin/env bash
# Launch grammar-cortex training in a detached tmux session with a side pane
# for nvidia-smi. Reattach with: tmux attach -t sara-train
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="${SESSION:-sara-train}"
PRESET="${PRESET:-tiny}"
BATCH="${BATCH:-32}"
STEPS="${STEPS:-5000}"
LOG="${LOG:-$REPO/training.log}"

cd "$REPO"

if ! command -v tmux >/dev/null; then
  echo "tmux not installed" >&2
  exit 1
fi

if [ ! -x "$REPO/.venv/bin/python" ]; then
  echo "$REPO/.venv missing — run: python3 -m venv .venv && pip install torch ..." >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session '$SESSION' already exists. attach with: tmux attach -t $SESSION"
  echo "or kill it: tmux kill-session -t $SESSION"
  exit 1
fi

CMD=".venv/bin/python -m sara_brain.cortex.transformer.train"
CMD+=" --preset $PRESET --batch $BATCH --steps $STEPS"
CMD+=" 2>&1 | tee -a '$LOG'"

tmux new-session -d -s "$SESSION" -c "$REPO" "$CMD"
tmux set-option -t "$SESSION" -g remain-on-exit on
tmux set-option -t "$SESSION" -g mouse on
tmux split-window -h -t "$SESSION" -l 60 "watch -n2 -t nvidia-smi"
tmux select-pane -t "$SESSION":0.0

cat <<EOF
launched tmux session: $SESSION
  preset=$PRESET batch=$BATCH steps=$STEPS
  log=$LOG

attach:    tmux attach -t $SESSION
detach:    Ctrl-b d
tail log:  tail -f $LOG
close pane after run: Ctrl-b x  (panes stay open showing final output)
kill all:  tmux kill-session -t $SESSION
EOF
