"""sara-ask-stateless CLI — stateless two-tier reader.

Routes via Ollama (stateless single-message calls), synthesizes via a
chosen local or remote model. Implements the architecture in model_infections §5d.
"""
from __future__ import annotations

import argparse
import json
import sys

from .stateless_reader import StatelessReader


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Ask Sara Brain a question via the stateless two-tier "
            "architecture. Ollama routes; synthesis provider is configurable."
        ),
    )
    ap.add_argument("question", help="The question to ask")
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
        help=(
            "Model for synthesis (default: same as --router-model). "
            "Local options within M2 8GB: llama3.2:3b, phi3.5, llama3.1:8b. "
            "Anthropic option: claude-haiku-4-5."
        ),
    )
    ap.add_argument(
        "--max-routing-steps",
        type=int,
        default=6,
        help="Hard cap on routing iterations (default: 6)",
    )
    ap.add_argument(
        "--trace",
        action="store_true",
        help="Show the full routing/synthesis trace",
    )
    ap.add_argument(
        "--cortex-router",
        action="store_true",
        help=(
            "Use the local Sara cortex transformer for routing instead "
            "of Ollama. Requires --grammar-ckpt and --head-ckpt (or "
            "their defaults under src/sara_brain/cortex/checkpoints/)."
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
        "--no-synthesis",
        action="store_true",
        help=(
            "Skip the synthesis LLM call. Returns the raw substrate facts "
            "the router gathered. Useful with --cortex-router for a "
            "no-LLM-in-loop demonstration."
        ),
    )
    ap.add_argument(
        "--cortex-synthesizer",
        action="store_true",
        help=(
            "Use the local template-based cortex synthesizer instead of "
            "an LLM. Combined with --cortex-router this gives a fully "
            "no-LLM, no-API-key path: question -> cortex -> substrate -> "
            "templated prose."
        ),
    )
    args = ap.parse_args()
    synthesis_model = args.synthesis_model or args.router_model

    cortex_router_ckpts = (
        (args.grammar_ckpt, args.head_ckpt) if args.cortex_router else None
    )

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
    )
    result = reader.ask(args.question, return_trace=args.trace)
    if args.trace:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
