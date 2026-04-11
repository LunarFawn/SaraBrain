"""sara-cortex CLI — interactive shell that uses Sara Cortex first.

The cortex handles whatever it can. For things it can't handle (low
confidence, unparseable input), it falls through to the existing
sara-agent loop with Ollama as the fallback cortex.

This is the transition path: the cortex starts handling 80% of common
turns immediately, and the LLM only kicks in for the long tail. As the
cortex grammar grows, the LLM is consulted less and less. Eventually
the LLM disappears entirely.

Usage:
    sara-cortex                           # interactive
    sara-cortex --db /path/to/brain.db    # custom database
    sara-cortex --no-llm                  # cortex only, no fallback
    sara-cortex --model llama3.1          # llama fallback model
"""

from __future__ import annotations

import argparse
import sys

from ..config import default_db_path
from ..core.brain import Brain
from .router import Cortex


def _print_response(response, verbose: bool = False) -> None:
    print(f"\nsara> {response.text}\n")
    if verbose and response.operations:
        for op in response.operations:
            mark = "✓" if op.success else "✗"
            print(f"      [{mark} {op.op}] {op.target} {op.detail}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sara-cortex",
        description="Sara Cortex — language layer for Sara Brain. "
                    "The cortex handles natural language directly. "
                    "Ollama is consulted only when the cortex defers.",
    )
    parser.add_argument(
        "--db",
        default=None,
        help=f"Brain database path (default: {default_db_path()})",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Cortex only — never fall back to Ollama. "
             "Sara will say 'I don't know' instead of guessing.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama model for fallback (default: auto-detect)",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:11434",
        help="Ollama base URL",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show cortex operations after each turn",
    )
    args = parser.parse_args()

    db_path = args.db or default_db_path()
    brain = Brain(db_path)
    cortex = Cortex(brain)

    # Optionally set up the llama fallback
    fallback_loop = None
    if not args.no_llm:
        try:
            from ..agent import ollama
            from ..agent.cli import _pick_model, TOOL_CAPABLE_MODELS
            from ..agent.loop import AgentLoop
            if ollama.check_health(args.url):
                models = ollama.list_models(args.url)
                if models:
                    if args.model:
                        match = [m.get("name", "") for m in models if args.model in m.get("name", "")]
                        model = match[0] if match else None
                    else:
                        model = _pick_model(models)
                    if model:
                        fallback_loop = AgentLoop(
                            brain=brain,
                            model=model,
                            base_url=args.url,
                        )
        except Exception as e:
            print(f"  (Ollama fallback unavailable: {e})", file=sys.stderr)

    stats = brain.stats()
    print()
    print("  Sara Cortex — language layer over Sara Brain")
    print(f"  Brain: {db_path} ({stats['neurons']} neurons, {stats['paths']} paths)")
    if fallback_loop:
        print(f"  Fallback: Ollama {fallback_loop.model}")
    else:
        print("  Fallback: none — cortex only, Sara will say 'I don't know'")
    print()
    print("  Cortex handles natural language directly. No LLM gatekeeping.")
    print("  Type 'exit' to quit.")
    print()

    try:
        while True:
            try:
                user_input = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Goodbye. Sara remembers everything.")
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "bye"):
                print("  Goodbye. Sara remembers everything.")
                break

            try:
                response = cortex.process(user_input)
                if response.delegate and fallback_loop is not None:
                    # Cortex couldn't handle confidently — defer to llama
                    print("\n  (cortex deferred to llama)")
                    text = fallback_loop.turn(user_input)
                    print(f"\nsara(llm)> {text}\n")
                else:
                    _print_response(response, verbose=args.verbose)
            except Exception as e:
                print(f"\n  Error: {e}\n", file=sys.stderr)

    finally:
        brain.close()


if __name__ == "__main__":
    main()
