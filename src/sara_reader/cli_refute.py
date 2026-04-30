"""sara-refute CLI — mark a triple as known-to-be-false in Sara Brain.

Usage:
    sara-refute --subject "molecular snare" --relation "is" --object "mechanical system" \\
                --brain brains/aptamer_full.db

Uses brain.refute_triple() — no parser, compound terms matched verbatim.
Sara never deletes: the path stays as evidence; its strength goes negative.
"""
from __future__ import annotations

import argparse
import sys

from .brain_loader import load_brain


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Mark a Sara Brain triple as known-to-be-false.",
    )
    ap.add_argument("--subject", required=True, help="Subject neuron label")
    ap.add_argument("--relation", required=True, help="Relation verb")
    ap.add_argument("--object", required=True, dest="obj", help="Object neuron label")
    ap.add_argument("--brain", required=True, help="Path to brain .db file")
    args = ap.parse_args()

    brain = load_brain(args.brain)
    try:
        result = brain.refute_triple(args.subject, args.relation, args.obj)
    except PermissionError as e:
        print(f"Blocked: {e}", file=sys.stderr)
        brain.conn.close()
        return 1

    if result is None:
        print(
            f"Could not refute: ({args.subject!r}, {args.relation!r}, {args.obj!r}). "
            "Check that all fields are non-empty.",
            file=sys.stderr,
        )
        brain.conn.close()
        return 1

    print(
        f"Refuted: {result.path_label} "
        f"(path #{result.path_id}, marked as known-to-be-false)"
    )
    brain.conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
