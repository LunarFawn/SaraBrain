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
import json
import os
import sys

from .stateless_reader import StatelessReader


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
    )

    audit_path = os.environ.get("SARA_AUDIT_LOG", "")
    print(
        f"[ready] brain={args.brain} synth={synthesis_model} "
        f"strict_sara={args.strict_sara}"
        + (f" audit={audit_path}" if audit_path else "")
    )
    print("type a question, /trace for trace output, /quit to exit")

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
        if line == "/trace":
            show_trace = not show_trace
            print(f"trace = {show_trace}")
            continue
        if line == "/audit":
            print(f"SARA_AUDIT_LOG = {audit_path or '(unset)'}")
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


if __name__ == "__main__":
    sys.exit(main())
