"""Sara Brain CLI bridge — query Sara from any repo.

Usage:
    python sara_q.py stats
    python sara_q.py neurons
    python sara_q.py why <label>
    python sara_q.py trace <label>
    python sara_q.py recognize <prop1,prop2,...>
    python sara_q.py teach <statement>
    python sara_q.py similar <label>
    python sara_q.py associations
    python sara_q.py categories
    python sara_q.py paths
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
os.environ["PYTHONIOENCODING"] = "utf-8"

from sara_brain.core.brain import Brain

DB = os.path.join(os.path.expanduser("~"), ".sara_brain", "sara.db")


def main(args: list[str] | None = None) -> None:
    args = args or sys.argv[1:]
    if not args:
        print("Usage: python sara_q.py <command> [args]")
        print("Commands: stats, neurons, why, trace, recognize, teach, similar, associations, categories, paths")
        return

    cmd = args[0].lower()
    rest = " ".join(args[1:]) if len(args) > 1 else ""

    brain = Brain(DB)
    try:
        if cmd == "stats":
            s = brain.stats()
            for k, v in s.items():
                print(f"{k}: {v}")

        elif cmd == "neurons":
            for n in brain.neuron_repo.list_all():
                print(f"[{n.neuron_type.value}] {n.label}")

        elif cmd == "why" and rest:
            traces = brain.why(rest)
            if not traces:
                print(f"No paths lead to '{rest}'")
            for t in traces:
                print(t)

        elif cmd == "trace" and rest:
            traces = brain.trace(rest)
            if not traces:
                print(f"No paths from '{rest}'")
            for t in traces:
                print(t)

        elif cmd == "recognize" and rest:
            results = brain.recognize(rest)
            if not results:
                print("No recognition")
            for r in results:
                print(f"{r.neuron.label} (confidence: {r.confidence})")

        elif cmd == "teach" and rest:
            result = brain.teach(rest)
            if result:
                print(f"Learned: {result.path_label} ({result.neurons_created} new neurons, {result.segments_created} new segments)")
            else:
                print(f"Could not parse: {rest}")

        elif cmd == "similar" and rest:
            links = brain.get_similar(rest)
            if not links:
                print(f"No similar neurons for '{rest}'")
            for s in links:
                print(f"{s.neuron_b_label} (shared: {s.shared_paths}, overlap: {s.overlap_ratio:.0%})")

        elif cmd == "associations":
            assocs = brain.list_associations()
            if not assocs:
                print("No associations")
            for name, props in assocs.items():
                print(f"{name}: {', '.join(props)}")

        elif cmd == "categories":
            cats = brain.list_categories()
            if not cats:
                print("No categories")
            for cat, labels in cats.items():
                print(f"{cat}: {', '.join(labels)}")

        elif cmd == "paths":
            all_paths = brain.path_repo.list_all()
            if not all_paths:
                print("No paths")
            for p in all_paths:
                print(f"{p.label} (source: {p.source_text})")

        else:
            print(f"Unknown command or missing args: {cmd} {rest}")

    finally:
        brain.close()


if __name__ == "__main__":
    main()
