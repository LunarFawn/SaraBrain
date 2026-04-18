"""Interactive shell for the sensory module.

sara-shell: an empty processing engine that reads from Sara Brain.
No LLM. No API calls. Pure graph traversal with provenance.

Usage:
    sara-shell --db sara.db
    sara-shell --db sara.db --no-provenance
"""

from __future__ import annotations

import argparse
import readline  # noqa: F401 — enables arrow keys in input()
import sys

from ..config import default_db_path
from ..core.brain import Brain
from .shell import SensoryShell
from .session import Session


def _print_response(response, show_provenance: bool = True) -> None:
    """Print a shell response with optional provenance."""
    if show_provenance:
        print(response.text)
    else:
        for line in response.sources:
            print(line.text)

    if response.gaps:
        print(f"\n  Unknown: {', '.join(response.gaps)}")

    if response.confidence > 0:
        print(f"  Confidence: {response.confidence}")


def _handle_slash(cmd: str, shell: SensoryShell, session: Session) -> bool:
    """Handle slash commands. Returns True if command was handled."""
    parts = cmd.strip().split(None, 1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if command in ("/quit", "/exit", "/q"):
        print("Goodbye.")
        sys.exit(0)

    if command == "/help":
        print("Commands:")
        print("  /query <topic>   — what does Sara know about <topic>?")
        print("  /teach <fact>    — teach Sara a fact (e.g. 'apples are red')")
        print("  /stats           — brain statistics")
        print("  /clear           — clear session context")
        print("  /quit            — exit")
        print()
        print("Anything else is processed through Sara's graph.")
        return True

    if command == "/query" and arg:
        response = shell.query(arg.strip())
        _print_response(response)
        return True

    if command == "/teach" and arg:
        result = shell.brain.teach(arg.strip())
        if result is not None:
            print(f"Learned: {arg.strip()}")
            print(f"  path #{result.path_id}")
        else:
            print(f"Could not parse: {arg.strip()}")
        return True

    if command == "/stats":
        count = shell.brain.neuron_repo.count()
        seg_count = shell.brain.segment_repo.count()
        path_count = shell.brain.path_repo.count()
        print(f"Neurons: {count}")
        print(f"Segments: {seg_count}")
        print(f"Paths: {path_count}")
        return True

    if command == "/clear":
        session.clear()
        print("Session cleared.")
        return True

    return False


def main() -> None:
    """Entry point for sara-shell."""
    parser = argparse.ArgumentParser(
        description="Sara Sensory Shell — empty processing engine over Sara Brain",
    )
    parser.add_argument(
        "--db", type=str, default=None,
        help=f"Path to Sara Brain database (default: {default_db_path()})",
    )
    parser.add_argument(
        "--no-provenance", action="store_true",
        help="Hide path provenance in output",
    )
    args = parser.parse_args()

    db_path = args.db or default_db_path()
    show_provenance = not args.no_provenance

    brain = Brain(db_path)
    shell = SensoryShell(brain)
    session = Session()

    neuron_count = brain.neuron_repo.count()
    print(f"Sara Sensory Shell — {neuron_count} neurons loaded")
    print("No LLM. No weights. Pure graph traversal.")
    print("Type /help for commands, /quit to exit.\n")

    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not text:
            continue

        if text.startswith("/"):
            if _handle_slash(text, shell, session):
                continue

        # Process through the shell
        response = shell.process(text)
        session.add_turn(response.tokens)

        print()
        _print_response(response, show_provenance)
        print()


if __name__ == "__main__":
    main()
