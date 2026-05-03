"""Interactive HamlinLLM chat — REPL against a Sara brain.

  $ sara-cortex-chat --brain path/to/your/sara.db

Loads the grammar LM + router head once, then takes questions one at a
time. No LLM in the loop. Slash commands:

  /help        list commands
  /trace       toggle: show routing decision + classifier confidence
  /verbose     toggle: show raw substrate output instead of synthesized prose
  /brain PATH  switch brains without restarting
  /model       show which checkpoints are loaded
  /quit, /exit, Ctrl-D   leave
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sara_brain.core.brain import Brain
from sara_reader.tools import execute_tool

from .router import MODEL_FULL, CortexRouter
from .synthesizer import synthesize


# ANSI colors — disabled if stdout isn't a TTY
def _color(code: str) -> str:
    return code if sys.stdout.isatty() else ""


CYAN = _color("\033[36m")
GREEN = _color("\033[32m")
YELLOW = _color("\033[33m")
DIM = _color("\033[2m")
BOLD = _color("\033[1m")
RESET = _color("\033[0m")


def banner(model: str, brain_path: Path) -> str:
    return (
        f"{BOLD}{CYAN}╭─ {model} ──────────────────────────────╮{RESET}\n"
        f"{BOLD}{CYAN}│{RESET} brain: {brain_path}\n"
        f"{BOLD}{CYAN}│{RESET} type a question, /help for commands, /quit to exit\n"
        f"{BOLD}{CYAN}╰────────────────────────────────────────────────────╯{RESET}"
    )


HELP = """commands:
  /help              show this help
  /trace             toggle: show routing decision + confidence
  /verbose           toggle: print raw substrate result (no synthesis)
  /brain PATH        switch to a different brain.db
  /model             show loaded checkpoints
  /quit /exit Ctrl-D leave"""


class ChatSession:
    def __init__(
        self,
        brain_path: Path,
        grammar_ckpt: Path,
        head_ckpt: Path,
        device: str,
    ):
        self.grammar_ckpt = grammar_ckpt
        self.head_ckpt = head_ckpt
        self.device = device
        self.show_trace = False
        self.show_raw = False
        self.router = CortexRouter(
            grammar_ckpt=grammar_ckpt,
            head_ckpt=head_ckpt,
            substrate_db=brain_path,
            device=device,
        )
        self._load_brain(brain_path)

    def _load_brain(self, brain_path: Path) -> None:
        self.brain_path = brain_path
        self.brain = Brain(str(brain_path))
        # Re-bind the router's substrate index too.
        from .router_args import SubstrateIndex
        self.router.substrate = SubstrateIndex(brain_path)

    def ask(self, question: str) -> None:
        decision = self.router.route(question)
        if self.show_trace:
            print(f"{DIM}[{decision.model}] tool={decision.tool}  "
                  f"cls_conf={decision.classifier_confidence:.2f}  "
                  f"args={decision.args}  why={decision.rationale}{RESET}")
        result = execute_tool(self.brain, decision.tool, decision.args)
        if self.show_raw:
            print(result)
            return
        gathered = [{"call": {"tool": decision.tool, "args": decision.args},
                     "result": result}]
        prose = synthesize(question, gathered)
        print(prose)

    def handle_command(self, line: str) -> bool:
        """Run a slash command. Returns False if the session should end."""
        parts = line.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        if cmd in ("/quit", "/exit"):
            return False
        if cmd == "/help":
            print(HELP)
        elif cmd == "/trace":
            self.show_trace = not self.show_trace
            print(f"trace = {self.show_trace}")
        elif cmd == "/verbose":
            self.show_raw = not self.show_raw
            print(f"verbose (raw substrate output) = {self.show_raw}")
        elif cmd == "/brain":
            if not arg:
                print(f"current brain: {self.brain_path}")
            else:
                p = Path(arg)
                if not p.exists():
                    print(f"{YELLOW}brain not found: {p}{RESET}")
                else:
                    self._load_brain(p)
                    print(f"switched to: {p}")
        elif cmd == "/model":
            print(f"  grammar:  {self.grammar_ckpt}")
            print(f"  router:   {self.head_ckpt}")
        else:
            print(f"{YELLOW}unknown command: {cmd}  (try /help){RESET}")
        return True


def main() -> int:
    p = argparse.ArgumentParser(description="Interactive HamlinLLM chat")
    p.add_argument("--brain", type=Path, required=True,
                   help="Path to a Sara brain.db (must end in .db)")
    p.add_argument("--grammar-ckpt", type=Path,
                   default=Path("src/sara_brain/cortex/checkpoints/grammar_base_015000.pt"))
    p.add_argument("--head-ckpt", type=Path,
                   default=Path("src/sara_brain/cortex/checkpoints/router_head.pt"))
    p.add_argument("--device", default="cpu",
                   help="cpu or cuda; cpu is plenty for serving")
    args = p.parse_args()

    if not args.brain.exists():
        print(f"brain not found: {args.brain}", file=sys.stderr)
        return 1
    if not args.grammar_ckpt.exists():
        print(f"grammar checkpoint not found: {args.grammar_ckpt}", file=sys.stderr)
        return 1
    if not args.head_ckpt.exists():
        print(f"router head checkpoint not found: {args.head_ckpt}", file=sys.stderr)
        return 1

    print(f"{DIM}loading {MODEL_FULL}...{RESET}", flush=True)
    session = ChatSession(args.brain, args.grammar_ckpt, args.head_ckpt, args.device)
    print(banner(MODEL_FULL, args.brain))

    while True:
        try:
            line = input(f"{BOLD}{GREEN}> {RESET}")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        line = line.strip()
        if not line:
            continue
        if line.startswith("/"):
            cont = session.handle_command(line)
            if not cont:
                return 0
            continue
        try:
            session.ask(line)
        except Exception as e:
            print(f"{YELLOW}error: {e}{RESET}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
