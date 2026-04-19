#!/usr/bin/env python3
"""Iterative tutoring loop for Sara on MMLU-biology.

Each cycle:
  1. Run the MMLU benchmark against the brain.
  2. Extract the gap report (wrong + abstain questions).
  3. Generate a biology-fact sentence from each failing question's
     correct answer (teach_mmlu_gaps logic).
  4. Teach those facts into the brain.
  5. Record precision/coverage for this cycle.
  6. Stop when either:
     - nothing was taught this cycle, OR
     - --max-rounds reached, OR
     - precision stopped improving for N rounds.

Reports the round-by-round curve so you can see where returns
diminish.

Usage:
    .venv/bin/python benchmarks/tutoring_loop.py \\
        --db biology2e.db \\
        --questions benchmarks/mmlu_biology_full.json \\
        --max-rounds 6
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_benchmark(db: str, questions: str, out: str,
                  top_k: int = 5) -> dict:
    cmd = [
        ".venv/bin/python", "benchmarks/run_spacy_ch10.py",
        "--db", db, "--questions", questions,
        "--mode", "property",
        "--top-k", str(top_k),
        "--output", out,
    ]
    subprocess.run(cmd, check=True)
    data = json.loads(Path(out).read_text())
    return data


def run_teach(db: str, results: str) -> int:
    cmd = [
        ".venv/bin/python", "benchmarks/teach_mmlu_gaps.py",
        "--db", db, "--results", results,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    for line in res.stdout.splitlines():
        if line.startswith("Taught:"):
            return int(line.split(":")[1].strip())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--max-rounds", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--output-prefix",
                    default="benchmarks/tutoring_loop")
    args = ap.parse_args()

    history: list[dict] = []
    for round_i in range(1, args.max_rounds + 1):
        results_path = f"{args.output_prefix}_r{round_i}.json"
        print(f"\n=== Round {round_i}: scoring ===")
        data = run_benchmark(
            args.db, args.questions, results_path, args.top_k,
        )
        summary = {
            "round": round_i,
            "correct": data["correct"],
            "wrong": data["wrong"],
            "abstained": data["abstained"],
            "accuracy_of_answered": data.get("accuracy_of_answered"),
            "coverage": data.get("coverage"),
        }
        history.append(summary)
        print(f"  correct={summary['correct']} "
              f"wrong={summary['wrong']} "
              f"abstain={summary['abstained']} "
              f"precision={summary['accuracy_of_answered']:.1%} "
              f"coverage={summary['coverage']:.1%}")

        print(f"=== Round {round_i}: teaching ===")
        taught = run_teach(args.db, results_path)
        summary["taught_this_round"] = taught
        print(f"  taught {taught} facts from this round's gaps")

        if taught == 0:
            print("Nothing new taught — converged.")
            break

    print("\n=== Tutoring loop summary ===")
    for s in history:
        print(
            f"Round {s['round']:>2}: "
            f"correct={s['correct']:>3}  "
            f"wrong={s['wrong']:>3}  "
            f"abstain={s['abstained']:>3}  "
            f"precision={s['accuracy_of_answered']:.1%}  "
            f"coverage={s['coverage']:.1%}  "
            f"taught_next={s.get('taught_this_round', 0)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
