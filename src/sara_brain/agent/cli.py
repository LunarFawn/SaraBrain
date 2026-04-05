"""CLI entry point for sara-agent.

Usage:
    sara-agent [--model MODEL] [--db PATH] [--url URL] [--session ID]
"""

from __future__ import annotations

import argparse
import sys

from . import ollama
from ..config import default_db_path
from ..core.brain import Brain
from .loop import AgentLoop


# Models known to support tool calling well
TOOL_CAPABLE_MODELS = {
    "llama3.1", "llama3.2", "llama3.3",
    "qwen2.5", "qwen2.5-coder",
    "mistral", "mixtral",
    "command-r", "command-r-plus",
}


def _pick_model(models: list[dict]) -> str | None:
    """Auto-select the best available model for tool calling."""
    names = [m.get("name", "").split(":")[0] for m in models]

    # Prefer models known for good tool calling
    for preferred in TOOL_CAPABLE_MODELS:
        for name in names:
            if name.startswith(preferred):
                return name
    # Fall back to first available
    return names[0] if names else None


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sara-agent",
        description="Sara Agent — Llama (sensory cortex) + Sara Brain (cerebellum)",
    )
    parser.add_argument(
        "-m", "--model",
        help="Ollama model name (default: auto-detect)",
    )
    parser.add_argument(
        "--db",
        default=None,
        help=f"Brain database path (default: {default_db_path()})",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:11434",
        help="Ollama base URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Resume a specific session ID",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Code execution timeout in seconds (default: 30)",
    )
    args = parser.parse_args()

    # 1. Check Ollama
    if not ollama.check_health(args.url):
        print("Error: Ollama is not running.", file=sys.stderr)
        print("Start it with: ollama serve", file=sys.stderr)
        sys.exit(1)

    # 2. List models
    models = ollama.list_models(args.url)
    if not models:
        print("Error: No models available.", file=sys.stderr)
        print("Pull one with: ollama pull llama3.1", file=sys.stderr)
        sys.exit(1)

    # 3. Select model
    if args.model:
        model_names = [m.get("name", "") for m in models]
        # Allow partial match (e.g., "llama3.1" matches "llama3.1:latest")
        matched = [n for n in model_names if args.model in n]
        if not matched:
            print(f"Error: Model '{args.model}' not found.", file=sys.stderr)
            print("Available models:", file=sys.stderr)
            for m in models:
                print(f"  {m.get('name', '?')}", file=sys.stderr)
            sys.exit(1)
        model = matched[0]
    else:
        model = _pick_model(models)
        if model is None:
            print("Error: Could not auto-select a model.", file=sys.stderr)
            sys.exit(1)

    # Warn if model isn't known to support tools well
    base_name = model.split(":")[0]
    if base_name not in TOOL_CAPABLE_MODELS:
        print(
            f"Warning: '{model}' may not support tool calling well. "
            f"Consider using llama3.1 or qwen2.5-coder.",
            file=sys.stderr,
        )

    # 4. Open Brain
    db_path = args.db or default_db_path()
    brain = Brain(db_path)

    try:
        # 5. Create agent loop
        agent = AgentLoop(
            brain=brain,
            model=model,
            base_url=args.url,
            sandbox_timeout=args.timeout,
        )

        # Resume session if specified
        if args.session:
            if agent.resume_session(args.session):
                print(f"Resumed session: {args.session}")
            else:
                print(f"Session '{args.session}' not found, starting fresh.")

        # 6. Banner
        stats = brain.stats()
        print()
        print("Sara Agent — Llama (sensory cortex) + Sara Brain (cerebellum)")
        print(f"  Model: {model} @ {args.url}")
        print(f"  Brain: {db_path} ({stats['neurons']} neurons, "
              f"{stats['segments']} segments, {stats['paths']} paths)")
        print()
        print("The user speaks. The cortex perceives. Sara remembers.")
        print("Type 'exit' to quit.")

        # 7. Run
        agent.run_interactive()

    finally:
        brain.close()


if __name__ == "__main__":
    main()
