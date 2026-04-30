"""sara-ask CLI — ask Sara from the shell."""
from __future__ import annotations

import argparse
import sys

from .reader import SaraReader


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Ask Sara Brain a question via a chosen LLM provider.",
    )
    ap.add_argument("question", help="The question to ask")
    ap.add_argument("--brain", required=True, help="Path to brain .db file")
    ap.add_argument(
        "--provider",
        required=True,
        choices=["anthropic", "ollama"],
        help="LLM provider (openai is excluded by author policy)",
    )
    ap.add_argument("--model", required=True, help="Model identifier")
    ap.add_argument(
        "--max-rounds",
        type=int,
        default=8,
        help="Maximum tool-call rounds (default: 8)",
    )
    ap.add_argument(
        "--trace",
        action="store_true",
        help="Show the full retrieval trace alongside the answer",
    )
    args = ap.parse_args()

    reader = SaraReader(
        brain_path=args.brain,
        provider=args.provider,
        model=args.model,
        max_rounds=args.max_rounds,
    )
    result = reader.ask(args.question, return_trace=args.trace)
    if args.trace:
        import json
        print(json.dumps(result, indent=2, default=str))
    else:
        print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
