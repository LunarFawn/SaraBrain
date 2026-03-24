"""CLI for Q to interact with Sara Brain.

Usage:
    python -m sara_brain.q_cli teach "python functions use snake_case"
    python -m sara_brain.q_cli query "naming convention"
    python -m sara_brain.q_cli recognize "red, round"
    python -m sara_brain.q_cli check "function naming style"
    python -m sara_brain.q_cli ingest path/to/doc.md
    python -m sara_brain.q_cli summarize "naming convention"
    python -m sara_brain.q_cli stats
"""

from __future__ import annotations

import sys

from .nlp.q_bridge import QBridge

DB_PATH = "sara.db"


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(1)

    cmd = sys.argv[1].lower()
    args = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""

    with QBridge(DB_PATH) as q:
        if cmd == "teach":
            if not args:
                print("Usage: teach <statement>")
                sys.exit(1)
            print(q.teach(args))

        elif cmd == "teach-many":
            # Read statements from stdin, one per line
            statements = [line.strip() for line in sys.stdin if line.strip()]
            print(q.teach_many(statements))

        elif cmd == "query":
            if not args:
                print("Usage: query <label>")
                sys.exit(1)
            print(q.query(args))

        elif cmd == "recognize":
            if not args:
                print("Usage: recognize <input1, input2, ...>")
                sys.exit(1)
            print(q.recognize(args))

        elif cmd == "check":
            if not args:
                print("Usage: check <context>")
                sys.exit(1)
            print(q.check_rules(args))

        elif cmd == "ingest":
            if not args:
                print("Usage: ingest <filepath>")
                sys.exit(1)
            print(q.ingest_file(args))

        elif cmd == "summarize":
            if not args:
                print("Usage: summarize <topic>")
                sys.exit(1)
            print(q.summarize(args))

        elif cmd == "stats":
            print(q.stats())

        else:
            print(f"Unknown command: {cmd}")
            print(__doc__.strip())
            sys.exit(1)


if __name__ == "__main__":
    main()
