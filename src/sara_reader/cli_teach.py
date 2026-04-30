"""sara-teach CLI — teach a triple to Sara Brain from the shell.

Usage:
    sara-teach --subject "molecular snare" --relation "is" --object "mechanical system" \\
               --brain brains/aptamer_full.db [--source aptamer_paper]

Uses brain.teach_triple() — no parser, compound terms stored verbatim.
Suitable for human use and 3B model teacher-surrogate workflows.
"""
from __future__ import annotations

import argparse
import sys

from .brain_loader import load_brain


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Teach Sara Brain a triple: subject --relation--> object.",
    )
    ap.add_argument("--subject", required=True, help="Subject neuron label")
    ap.add_argument("--relation", required=True, help="Relation verb")
    ap.add_argument("--object", required=True, dest="obj", help="Object neuron label")
    ap.add_argument("--brain", required=True, help="Path to brain .db file")
    ap.add_argument("--source", default=None, help="Provenance tag (optional)")
    args = ap.parse_args()

    brain = load_brain(args.brain)
    try:
        result = brain.teach_triple(
            args.subject,
            args.relation,
            args.obj,
            source_label=args.source,
        )
    except PermissionError as e:
        print(f"Blocked: {e}", file=sys.stderr)
        brain.conn.close()
        return 1
    finally:
        pass

    if result is None:
        print(
            f"Could not store: ({args.subject!r}, {args.relation!r}, {args.obj!r}). "
            "Check that all fields are non-empty.",
            file=sys.stderr,
        )
        brain.conn.close()
        return 1

    print(f"Learned: {result.path_label} (path #{result.path_id})")
    brain.conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
