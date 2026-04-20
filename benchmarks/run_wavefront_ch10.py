#!/usr/bin/env python3
"""Ch10 benchmark using wavefront confluence scoring against brain.db.

Thin driver on top of `src/sara_brain/core/wavefront_scorer.py`.
Compound-aware query resolution + IS-A-filtered wavefront propagation
+ confluence-witness scoring. No lemma-overlap fallback; this is the
pure-graph path.

Usage:
    .venv/bin/python benchmarks/run_wavefront_ch10.py \\
        --db brain.db \\
        --questions benchmarks/ch10_test_questions.json \\
        [--output benchmarks/ch10_wavefront.json]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import spacy

from sara_brain.core.brain import Brain
from sara_brain.core.wavefront_scorer import score_choices, pick_choice


def run(db_path: Path, questions_path: Path,
        output_path: Path | None) -> None:
    print(f"Loading spaCy…")
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    print(f"Opening {db_path}…")
    brain = Brain(str(db_path))
    questions = json.loads(questions_path.read_text())
    print(f"{len(questions)} questions")

    correct = 0
    wrong = 0
    abstained = 0
    ties = 0
    results_out: list[dict] = []

    t_start = time.time()
    for q in questions:
        qid = q["id"]
        text = q["question"]
        choices = q["choices"]
        correct_idx = q["answer_idx"]

        t0 = time.time()
        ranked = score_choices(
            text, choices, nlp, brain.recognizer, brain.neuron_repo,
        )
        dt = time.time() - t0

        pick, reason = pick_choice(ranked, text)
        if pick is None:
            if reason == "tie":
                ties += 1
                outcome = "tie"
            else:
                abstained += 1
                outcome = "abstain"
        else:
            if pick == correct_idx:
                correct += 1
                outcome = "correct"
            else:
                wrong += 1
                outcome = "wrong"

        mark = {
            "correct": "✓", "wrong": "✗",
            "tie": "≈", "abstain": "·",
        }[outcome]
        top_row = ranked[0] if ranked else None
        print(
            f"  Q{qid}: {mark} {outcome:<8} "
            f"pick={pick if pick is not None else '-'} "
            f"correct={correct_idx} "
            f"top_score={top_row.score if top_row else 0:.1f} "
            f"({dt:.1f}s)"
        )

        results_out.append({
            "id": qid,
            "outcome": outcome,
            "pick": pick,
            "correct": correct_idx,
            "scores": [
                {
                    "index": r.index,
                    "score": r.score,
                    "convergence": r.convergence_count,
                    "compound_hits": r.compound_hits,
                    "text": r.text,
                }
                for r in ranked
            ],
            "seconds": dt,
        })

    total = len(questions)
    answered = correct + wrong
    elapsed = time.time() - t_start
    summary = {
        "correct": correct,
        "wrong": wrong,
        "abstained": abstained,
        "ties": ties,
        "total": total,
        "precision": correct / answered if answered else 0.0,
        "coverage": answered / total if total else 0.0,
        "seconds": elapsed,
    }
    print()
    print("=" * 60)
    print(f"  wavefront confluence (brain.db)")
    print(f"    correct   : {correct}/{answered}  ({summary['precision']*100:.1f}% of answered)")
    print(f"    wrong     : {wrong}")
    print(f"    abstained : {abstained}")
    print(f"    ties      : {ties}")
    print(f"    coverage  : {summary['coverage']*100:.1f}%")
    print(f"    time      : {elapsed:.1f}s  ({elapsed/total:.2f}s per Q)")
    print("=" * 60)

    if output_path:
        output_path.write_text(json.dumps(
            {"summary": summary, "results": results_out}, indent=2,
        ))
        print(f"  Wrote {output_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True, type=Path)
    p.add_argument("--questions", required=True, type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    run(args.db, args.questions, args.output)


if __name__ == "__main__":
    main()
