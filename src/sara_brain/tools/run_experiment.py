"""sara-experiment — paired Session B / Session C trial runner.

Session B: StatelessReader routes against the trained Sara substrate.
           Each call is a fresh session with no shared context.
Session C: Synthesis model called directly with no substrate, no routing.
           Model can only answer from training weights.

The delta (B_rate - C_rate) is the measurement of substrate-based retrieval.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _score_answer(answer: str, concepts: list[str]) -> bool:
    """Return True if any manifest concept appears in the answer."""
    answer_lower = answer.lower()
    return any(c.lower() in answer_lower for c in concepts)


def _run_session_b(
    question: str,
    trials: int,
    concepts: list[str],
    brain_path: str,
    router_model: str,
    synthesis_provider: str,
    synthesis_model: str,
) -> list[dict]:
    from sara_reader.stateless_reader import StatelessReader

    results = []
    for i in range(trials):
        print(f"  B trial {i + 1}/{trials}", end="\r", flush=True)
        reader = StatelessReader(
            brain_path=brain_path,
            router_provider="ollama",
            router_model=router_model,
            synthesis_provider=synthesis_provider,
            synthesis_model=synthesis_model,
        )
        try:
            answer = reader.ask(question)
        except Exception as e:
            answer = f"ERROR: {e}"
        hit = _score_answer(answer, concepts)
        results.append({"trial": i + 1, "answer": answer, "hit": hit})
    print()
    return results


def _run_session_c(
    question: str,
    trials: int,
    concepts: list[str],
    synthesis_provider: str,
    synthesis_model: str,
) -> list[dict]:
    from sara_reader.providers import get_provider

    provider = get_provider(synthesis_provider)
    results = []
    for i in range(trials):
        print(f"  C trial {i + 1}/{trials}", end="\r", flush=True)
        try:
            response = provider.chat(
                messages=[{"role": "user", "content": question}],
                tools=[],
                model=synthesis_model,
            )
            answer = response.text
        except Exception as e:
            answer = f"ERROR: {e}"
        hit = _score_answer(answer, concepts)
        results.append({"trial": i + 1, "answer": answer, "hit": hit})
    print()
    return results


def _summarize(trials: list[dict]) -> dict:
    hits = sum(1 for t in trials if t["hit"])
    n = len(trials)
    return {
        "hits": hits,
        "misses": n - hits,
        "rate": round(hits / n, 4) if n else 0.0,
        "trials": trials,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Run paired Session B (Sara substrate) / Session C (bare model) trials "
            "and report the retrieval delta."
        )
    )
    ap.add_argument("--brain", required=True, help="Path to trained Sara .db file")
    ap.add_argument(
        "--manifest", required=True,
        help="Path to .manifest.json produced by sara-synth",
    )
    ap.add_argument("--question", required=True, help="Question to ask each trial")
    ap.add_argument("--trials", type=int, default=50,
                    help="Trials per session (default: 50)")
    ap.add_argument("--router-model", default="llama3.2:3b",
                    help="Ollama model for Session B routing (default: llama3.2:3b)")
    ap.add_argument(
        "--synthesis-provider", default="ollama", choices=["ollama", "anthropic"],
        help="Provider for synthesis in both sessions (default: ollama)",
    )
    ap.add_argument(
        "--synthesis-model", default=None,
        help="Model for synthesis (default: same as --router-model)",
    )
    ap.add_argument("--out", default=None,
                    help="Write full JSON results to this path")
    args = ap.parse_args()

    synthesis_model = args.synthesis_model or args.router_model

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    with manifest_path.open() as f:
        manifest = json.load(f)
    concepts: list[str] = manifest.get("concepts", [])
    if not concepts:
        print("Manifest has no concepts — cannot score answers.", file=sys.stderr)
        return 1

    print(f"Question:   {args.question}")
    print(f"Trials:     {args.trials} per session")
    print(f"Concepts:   {len(concepts)} in manifest")
    print(f"Router:     ollama / {args.router_model}")
    print(f"Synthesis:  {args.synthesis_provider} / {synthesis_model}")
    print()

    print("Running Session B (Sara substrate present)...")
    b_trials = _run_session_b(
        question=args.question,
        trials=args.trials,
        concepts=concepts,
        brain_path=args.brain,
        router_model=args.router_model,
        synthesis_provider=args.synthesis_provider,
        synthesis_model=synthesis_model,
    )

    print("Running Session C (bare model, no substrate)...")
    c_trials = _run_session_c(
        question=args.question,
        trials=args.trials,
        concepts=concepts,
        synthesis_provider=args.synthesis_provider,
        synthesis_model=synthesis_model,
    )

    b = _summarize(b_trials)
    c = _summarize(c_trials)
    delta = round(b["rate"] - c["rate"], 4)

    print()
    print(f"Session B (substrate present):  {b['hits']:3d} / {args.trials} hits  ({b['rate']*100:.1f}%)")
    print(f"Session C (bare model):         {c['hits']:3d} / {args.trials} hits  ({c['rate']*100:.1f}%)")
    print(f"Delta:                          {delta*100:+.1f} percentage points")

    if args.out:
        output = {
            "question": args.question,
            "trials": args.trials,
            "router_model": args.router_model,
            "synthesis_provider": args.synthesis_provider,
            "synthesis_model": synthesis_model,
            "manifest_path": str(manifest_path.resolve()),
            "manifest_concepts": concepts,
            "session_b": b,
            "session_c": c,
            "delta": delta,
        }
        out_path = Path(args.out)
        with out_path.open("w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
