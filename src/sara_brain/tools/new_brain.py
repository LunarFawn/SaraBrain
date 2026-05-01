"""sara-new — create a fresh empty brain at the given path."""
from __future__ import annotations

import argparse
import sys

from sara_brain.core.brain import Brain


def main() -> int:
    ap = argparse.ArgumentParser(description="Create a new empty Sara Brain database.")
    ap.add_argument("--out", required=True, help="Path for the new .db file")
    args = ap.parse_args()
    Brain(path=args.out)
    print(f"Brain created: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
