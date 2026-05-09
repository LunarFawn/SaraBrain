"""sara-chat-stateless — interactive REPL on top of StatelessReader.

Same paper-aligned architecture as cli_stateless: hippocampus
(Sara) on the bottom, frozen cortex (Ollama) on top. Each question
is structurally isolated — no conversation history, no memory.md,
no per-project memory file. The only difference vs cli_stateless
is that the StatelessReader is constructed ONCE and reused across
many questions, so the cortex-router checkpoints (and any model
warmups) don't reload per question.

Per-question isolation is preserved: every reader.ask() call
constructs a fresh routing loop with no carry-over state. The REPL
is a convenience for the user, not a context for the cortex.

Usage:
    .venv/bin/python -m sara_reader.cli_stateless_chat \\
        --brain /tmp/sara_demo.db \\
        --cortex-router \\
        --synthesis-provider ollama \\
        --synthesis-model llama3.1:8b \\
        --strict-sara

Inside the REPL: type a question, get an answer. Empty line skipped.
Ctrl-D / Ctrl-C / type 'quit' or 'exit' to leave. Type '/trace' to
toggle trace output, '/audit' to print the audit log path reminder.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import shlex
import sys
from pathlib import Path

# readline gives input() arrow-key history + line editing on Linux/macOS.
# Importing it is enough — Python's input() picks it up automatically.
try:
    import readline
except ImportError:
    readline = None

from .stateless_reader import StatelessReader
from .tools import execute_tool


HELP_TEXT = """\
slash commands:
  /help                   show this help
  /teach STATEMENT        teach a parsed natural-language fact
                          ("ssng1 is a goal")
  /teach SUBJ REL OBJ     teach a flat triple directly
                          ("fulcrum is_a support point")
  /refute STATEMENT       refute / negate a previously-taught fact
  /teach-event SUBJECT ACTION [object=O] [location=L] [from=ISO]
               [to=ISO] [modifier=M]
                          create a reified event node (v047)
  /where-is SUBJECT [at=ISO]
                          point-in-time location query
  /list-events SUBJECT    chronological event list for a subject
  /find-function NAME [module=M]
                          full function info (signature, returns,
                          parameters, calls, raises, docstring)
  /callers NAME           list callers of NAME
  /callees NAME           list functions NAME calls
  /returns-type TYPE      list functions returning TYPE
  /takes-type TYPE        list functions taking parameter of TYPE
  /trace                  toggle trace output
  /audit                  print SARA_AUDIT_LOG path
  /quit /exit Ctrl-D      leave

natural-language input is routed through the configured router
(Ollama or cortex-router) and synthesised by the configured
model — substrate-bound per --strict-sara. /teach* and /refute*
write directly to the brain.db, bypassing the LLM routing loop."""


def _setup_history() -> None:
    """Persist input history to ~/.sara_chat_history across sessions."""
    if readline is None:
        return
    hist = Path(os.path.expanduser("~/.sara_chat_history"))
    try:
        readline.read_history_file(str(hist))
    except (OSError, FileNotFoundError):
        pass
    readline.set_history_length(2000)
    atexit.register(_save_history, hist)


def _save_history(path: Path) -> None:
    if readline is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        readline.write_history_file(str(path))
    except OSError:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Interactive chat REPL using the stateless two-tier reader. "
            "Persistent reader; per-question isolation."
        ),
    )
    ap.add_argument("--brain", required=True, help="Path to brain .db file")
    ap.add_argument(
        "--router-model",
        default="llama3.2:3b",
        help="Ollama model for routing (default: llama3.2:3b)",
    )
    ap.add_argument(
        "--synthesis-provider",
        default="ollama",
        choices=["ollama", "anthropic"],
        help="Provider for synthesis (default: ollama)",
    )
    ap.add_argument(
        "--synthesis-model",
        default=None,
        help="Model for synthesis (default: same as --router-model)",
    )
    ap.add_argument(
        "--max-routing-steps", type=int, default=6,
        help="Hard cap on routing iterations (default: 6)",
    )
    ap.add_argument(
        "--cortex-router", action="store_true",
        help=(
            "Use HamRobyLLM (the local Sara cortex transformer) for "
            "routing instead of Ollama. Faster — runs in milliseconds. "
            "Requires --grammar-ckpt and --head-ckpt (or their defaults)."
        ),
    )
    ap.add_argument(
        "--grammar-ckpt",
        default="src/sara_brain/cortex/checkpoints/grammar_base_015000.pt",
        help="Cortex grammar-LM checkpoint (used with --cortex-router)",
    )
    ap.add_argument(
        "--head-ckpt",
        default="src/sara_brain/cortex/checkpoints/router_head.pt",
        help="Cortex router-head checkpoint (used with --cortex-router)",
    )
    ap.add_argument(
        "--no-synthesis", action="store_true",
        help="Skip the synthesis LLM call. Print raw substrate facts.",
    )
    ap.add_argument(
        "--cortex-synthesizer", action="store_true",
        help="Use HamRobyLLM template synthesizer instead of an LLM.",
    )
    ap.add_argument(
        "--strict-sara", action="store_true",
        help=(
            "Force-Sara synthesis mode (v052): substrate in <substrate> "
            "tags, strict rules in system prompt. Cortex uses ONLY "
            "substrate facts; no fallback to training. See "
            "docs/plans/v052_local_ollama_cortex.md."
        ),
    )
    ap.add_argument(
        "--explore-first", action="store_true",
        help=(
            "Always prepend a brain_explore depth=3 call before routing. "
            "Captures the associative neighborhood per Pearl 2026a §2.4. "
            "Recommended with --strict-sara for definitional questions."
        ),
    )
    args = ap.parse_args()
    synthesis_model = args.synthesis_model or args.router_model

    cortex_router_ckpts = (
        (args.grammar_ckpt, args.head_ckpt) if args.cortex_router else None
    )

    if args.cortex_router or args.cortex_synthesizer:
        from sara_brain.cortex.transformer.router import MODEL_FULL
        parts = []
        if args.cortex_router:
            parts.append("router")
        if args.cortex_synthesizer:
            parts.append("synthesizer")
        print(f"[{MODEL_FULL}] active for: {', '.join(parts)}")

    _setup_history()

    print(f"[loading reader: brain={args.brain}]", flush=True)
    reader = StatelessReader(
        brain_path=args.brain,
        router_provider="ollama",
        router_model=args.router_model,
        synthesis_provider=args.synthesis_provider,
        synthesis_model=synthesis_model,
        max_routing_steps=args.max_routing_steps,
        cortex_router_ckpts=cortex_router_ckpts,
        skip_synthesis=args.no_synthesis,
        cortex_synthesizer=args.cortex_synthesizer,
        strict_sara=args.strict_sara,
        explore_first=args.explore_first,
    )

    audit_path = os.environ.get("SARA_AUDIT_LOG", "")
    print(
        f"[ready] brain={args.brain} synth={synthesis_model} "
        f"strict_sara={args.strict_sara}"
        + (f" audit={audit_path}" if audit_path else "")
    )
    print("type a question, /help for commands, /quit to exit")

    show_trace = False
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line.lower() in ("/quit", "/exit", "quit", "exit"):
            return 0
        if line == "/help":
            print(HELP_TEXT)
            continue
        if line == "/trace":
            show_trace = not show_trace
            print(f"trace = {show_trace}")
            continue
        if line == "/audit":
            print(f"SARA_AUDIT_LOG = {audit_path or '(unset)'}")
            continue

        # Slash command dispatch — substrate writes (teach/refute) bypass
        # the LLM routing loop entirely. Reified-fact + code-knowledge
        # commands route through execute_tool against the brain.
        slash_handlers = {
            "/teach": _do_teach,
            "/refute": _do_refute,
            "/teach-event": _do_teach_event,
            "/where-is": _do_where_is,
            "/list-events": _do_list_events,
            "/find-function": _do_find_function,
            "/callers": _do_callers,
            "/callees": _do_callees,
            "/returns-type": _do_returns_type,
            "/takes-type": _do_takes_type,
        }
        matched = False
        for cmd, handler in slash_handlers.items():
            if line == cmd or line.lower().startswith(cmd + " "):
                arg = line[len(cmd):].lstrip()
                handler(reader, arg)
                matched = True
                break
        if matched:
            continue

        if line.startswith("/"):
            print(f"[unknown command: {line.split()[0]} — try /help]")
            continue

        try:
            result = reader.ask(line, return_trace=show_trace)
        except Exception as exc:
            print(f"[error] {exc}", file=sys.stderr)
            continue
        if show_trace:
            print(json.dumps(result, indent=2, default=str))
        else:
            print(result)


def _parse_event_args(arg: str) -> tuple[list[str], dict[str, str]]:
    """Split a slash-command arg into positional tokens + key=value pairs.
    Same parser used by /teach-event, /where-is, /list-events."""
    try:
        tokens = shlex.split(arg)
    except ValueError:
        tokens = arg.split()
    positional: list[str] = []
    kv: dict[str, str] = {}
    for t in tokens:
        if "=" in t and not t.startswith("="):
            k, _, v = t.partition("=")
            kv[k.strip().lower()] = v
        else:
            positional.append(t)
    return positional, kv


def _do_teach(reader, arg: str) -> None:
    """Teach a single substrate triple via Brain.teach_triple.

    Syntax: /teach SUBJECT RELATION OBJECT...
    First two whitespace-separated tokens are subject and relation;
    everything after is the object (so multi-word objects work
    without quoting). Falls back to Brain.teach() (parsed natural-
    language form) if fewer than 3 tokens.
    """
    parts = arg.split(maxsplit=2)
    if len(parts) < 3:
        # Treat as natural-language statement.
        try:
            result = reader.brain.teach(arg)
        except Exception as exc:
            print(f"[teach] error: {exc}", file=sys.stderr)
            return
        if result is None:
            print(f"[teach] could not parse: {arg!r}")
            print("[teach] hint: use the 3-token form: "
                  "/teach SUBJECT RELATION OBJECT...")
            return
        path_id = getattr(result, "path_id", "?")
        print(f"[teach] taught: {result.path_label} (path #{path_id})")
        return
    subject, relation, obj = parts
    try:
        result = reader.brain.teach_triple(subject, relation, obj)
    except Exception as exc:
        print(f"[teach] error: {exc}", file=sys.stderr)
        return
    if result is None:
        print(f"[teach] could not commit: {subject!r} {relation!r} {obj!r}")
        return
    path_id = getattr(result, "path_id", "?")
    print(
        f"[teach] taught: {subject!r} --[{relation}]--> {obj!r} "
        f"(path #{path_id})"
    )


def _do_refute(reader, arg: str) -> None:
    if not arg.strip():
        print("[refute] usage: /refute STATEMENT")
        return
    try:
        result = reader.brain.refute(arg)
    except Exception as exc:
        print(f"[refute] error: {exc}", file=sys.stderr)
        return
    if result is None:
        print(f"[refute] nothing matching: {arg!r}")
        return
    path_id = getattr(result, "path_id", "?")
    print(f"[refute] refuted: {result.path_label} (path #{path_id})")


def _do_teach_event(reader, arg: str) -> None:
    """Create a reified event node + binding edges (v047)."""
    positional, kv = _parse_event_args(arg)
    if len(positional) < 2:
        print(
            "[teach-event] usage: /teach-event SUBJECT ACTION "
            "[object=O] [location=L] [from=ISO] [to=ISO] [modifier=M]"
        )
        return
    subject, action = positional[0], positional[1]
    if len(positional) >= 3 and "object" not in kv:
        kv["object"] = " ".join(positional[2:])
    try:
        result = execute_tool(reader.brain, "brain_teach_event", {
            "subject": subject,
            "action": action,
            "object": kv.get("object"),
            "location": kv.get("location"),
            "start_time": kv.get("from") or kv.get("start") or kv.get("start_time"),
            "end_time": kv.get("to") or kv.get("end") or kv.get("end_time"),
            "modifier": kv.get("modifier"),
        })
        print(result)
    except Exception as exc:
        print(f"[teach-event] error: {exc}", file=sys.stderr)


def _do_where_is(reader, arg: str) -> None:
    positional, kv = _parse_event_args(arg)
    if not positional:
        print("[where-is] usage: /where-is SUBJECT [at=ISO]")
        return
    subject = positional[0]
    timestamp = kv.get("at") or kv.get("time") or kv.get("timestamp")
    if not timestamp:
        from datetime import datetime
        timestamp = datetime.now().isoformat(timespec="minutes")
    try:
        result = execute_tool(reader.brain, "brain_query_event_at", {
            "subject": subject, "timestamp": timestamp,
        })
        print(result)
    except Exception as exc:
        print(f"[where-is] error: {exc}", file=sys.stderr)


def _do_list_events(reader, arg: str) -> None:
    positional, _ = _parse_event_args(arg)
    if not positional:
        print("[list-events] usage: /list-events SUBJECT")
        return
    try:
        result = execute_tool(reader.brain, "brain_query_events", {
            "subject": positional[0],
        })
        print(result)
    except Exception as exc:
        print(f"[list-events] error: {exc}", file=sys.stderr)


def _do_find_function(reader, arg: str) -> None:
    positional, kv = _parse_event_args(arg)
    if not positional:
        print("[find-function] usage: /find-function NAME [module=M]")
        return
    try:
        result = execute_tool(reader.brain, "brain_query_function", {
            "name": positional[0],
            "module": kv.get("module"),
        })
        print(result)
    except Exception as exc:
        print(f"[find-function] error: {exc}", file=sys.stderr)


def _do_callers(reader, arg: str) -> None:
    positional, kv = _parse_event_args(arg)
    if not positional:
        print("[callers] usage: /callers FUNCTION_NAME")
        return
    try:
        result = execute_tool(reader.brain, "brain_query_callers", {
            "name": positional[0],
            "module": kv.get("module"),
        })
        print(result)
    except Exception as exc:
        print(f"[callers] error: {exc}", file=sys.stderr)


def _do_callees(reader, arg: str) -> None:
    positional, kv = _parse_event_args(arg)
    if not positional:
        print("[callees] usage: /callees FUNCTION_NAME")
        return
    try:
        result = execute_tool(reader.brain, "brain_query_callees", {
            "name": positional[0],
            "module": kv.get("module"),
        })
        print(result)
    except Exception as exc:
        print(f"[callees] error: {exc}", file=sys.stderr)


def _do_returns_type(reader, arg: str) -> None:
    positional, _ = _parse_event_args(arg)
    if not positional:
        print("[returns-type] usage: /returns-type TYPE")
        return
    try:
        result = execute_tool(reader.brain, "brain_query_by_returns", {
            "type": positional[0],
        })
        print(result)
    except Exception as exc:
        print(f"[returns-type] error: {exc}", file=sys.stderr)


def _do_takes_type(reader, arg: str) -> None:
    positional, _ = _parse_event_args(arg)
    if not positional:
        print("[takes-type] usage: /takes-type TYPE")
        return
    try:
        result = execute_tool(reader.brain, "brain_query_by_param", {
            "type": positional[0],
        })
        print(result)
    except Exception as exc:
        print(f"[takes-type] error: {exc}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
