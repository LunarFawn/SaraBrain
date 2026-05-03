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

from .clarify import (
    Clarification, ConceptFix, apply_wh_fix, detect_wh_typo,
    find_concept_candidates, is_cancel, parse_choice,
)
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
  /teach STATEMENT   teach Sara a new fact (e.g. /teach ssng1 is a goal)
  /refute STATEMENT  refute / negate an existing fact
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
        self.pending: Clarification | None = None

    def _load_brain(self, brain_path: Path) -> None:
        self.brain_path = brain_path
        self.brain = Brain(str(brain_path))
        # Re-bind the router's substrate index too.
        from .router_args import SubstrateIndex
        self.router.substrate = SubstrateIndex(brain_path)

    def ask(self, question: str) -> None:
        # Stage 1: detect leading-token typos (waht -> what, etc.)
        wh_fix = detect_wh_typo(question)
        if wh_fix:
            self.pending = Clarification(original_question=question, wh_fix=wh_fix)
            print(f"{YELLOW}{self.pending.render_prompt()}{RESET}")
            return
        self._route_and_run(question)

    def _route_and_run(self, question: str) -> None:
        decision = self.router.route(question)
        if self.show_trace:
            print(f"{DIM}[{decision.model}] tool={decision.tool}  "
                  f"cls_conf={decision.classifier_confidence:.2f}  "
                  f"args={decision.args}  why={decision.rationale}{RESET}")
        result = execute_tool(self.brain, decision.tool, decision.args)

        # Stage 2: substrate "no neuron matching" -> ask did_you_mean.
        miss_field, miss_value = self._extract_miss(decision.tool, decision.args, result)
        if miss_field is not None:
            cands = find_concept_candidates(self.brain, miss_value)
            if cands:
                pending_decision = {"tool": decision.tool, **decision.args}
                self.pending = Clarification(
                    original_question=question,
                    concept_fix=ConceptFix(
                        original=miss_value, candidates=cands, field=miss_field,
                    ),
                    pending_router_decision=pending_decision,
                )
                print(f"{YELLOW}{self.pending.render_prompt()}{RESET}")
                return
            # No candidates — fall through to the honest "no neuron" message.

        if self.show_raw:
            print(result)
            return
        gathered = [{"call": {"tool": decision.tool, "args": decision.args},
                     "result": result}]
        prose = synthesize(question, gathered)
        print(prose)

    @staticmethod
    def _extract_miss(tool: str, args: dict, result: str) -> tuple[str | None, str | None]:
        """If the tool returned a 'no neuron matching' response, return
        (which arg field needs replacing, what the missing value was)."""
        if "no neuron matching" not in result.lower():
            return None, None
        for field in ("concept", "label", "term"):
            if field in args:
                return field, args[field]
        return None, None

    def handle_pending_response(self, line: str) -> None:
        """User typed something while a clarification was pending. Either
        a numeric choice, 'no' to cancel, or a brand new question."""
        if not self.pending:
            return
        if is_cancel(line):
            print(f"{DIM}cancelled{RESET}")
            self.pending = None
            return

        n_opts = (len(self.pending.wh_fix.candidates)
                  if self.pending.wh_fix else
                  len(self.pending.concept_fix.candidates))
        choice = parse_choice(line, n_opts)
        if choice is None:
            # Treat as a fresh question.
            self.pending = None
            self.ask(line)
            return

        if self.pending.wh_fix:
            fixed = apply_wh_fix(self.pending.original_question,
                                 self.pending.wh_fix,
                                 self.pending.wh_fix.candidates[choice])
            print(f"{DIM}-> {fixed}{RESET}")
            self.pending = None
            self.ask(fixed)
            return

        # Concept fix: re-run the same tool with the chosen substrate label.
        cf = self.pending.concept_fix
        chosen = cf.candidates[choice]["label"]
        decision = dict(self.pending.pending_router_decision)
        decision[cf.field] = chosen
        original_q = self.pending.original_question
        self.pending = None
        print(f"{DIM}-> {cf.field}={chosen!r}{RESET}")
        tool = decision.pop("tool")
        result = execute_tool(self.brain, tool, decision)
        if self.show_raw:
            print(result)
            return
        gathered = [{"call": {"tool": tool, "args": decision}, "result": result}]
        print(synthesize(original_q, gathered))

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
        elif cmd == "/teach":
            if not arg:
                print(f"{YELLOW}usage: /teach <statement>{RESET}")
            else:
                self._do_teach(arg)
        elif cmd == "/refute":
            if not arg:
                print(f"{YELLOW}usage: /refute <statement>{RESET}")
            else:
                self._do_refute(arg)
        else:
            print(f"{YELLOW}unknown command: {cmd}  (try /help){RESET}")
        return True

    def _do_teach(self, statement: str) -> None:
        try:
            result = self.brain.teach(statement)
        except Exception as e:
            print(f"{YELLOW}teach error: {e}{RESET}")
            return
        if result is None:
            print(f"{YELLOW}couldn't parse: {statement!r}{RESET}")
            return
        path_id = getattr(result, "path_id", "?")
        print(f"{GREEN}learned (path #{path_id}): {statement}{RESET}")

    def _do_refute(self, statement: str) -> None:
        try:
            result = self.brain.refute(statement)
        except Exception as e:
            print(f"{YELLOW}refute error: {e}{RESET}")
            return
        if result is None:
            print(f"{YELLOW}nothing matching to refute: {statement!r}{RESET}")
            return
        path_id = getattr(result, "path_id", "?")
        print(f"{GREEN}refuted (path #{path_id}): {statement}{RESET}")


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
            if session.pending is not None:
                session.handle_pending_response(line)
            else:
                session.ask(line)
        except Exception as e:
            print(f"{YELLOW}error: {e}{RESET}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
