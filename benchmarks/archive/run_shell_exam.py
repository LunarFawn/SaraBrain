#!/usr/bin/env python3
"""Pure sensory shell exam — zero LLM, just graph convergence.

For each question + choice, run the sensory shell. The choice with
the highest convergence confidence wins. No neural network involved.

Usage:
    python benchmarks/run_shell_exam.py --db claude_taught.db \\
        --questions benchmarks/ch10_test_questions.json
"""

from __future__ import annotations

import argparse
import json
import re
import time

from sara_brain.core.brain import Brain
from sara_brain.core.short_term import ShortTerm
from sara_brain.core.recognizer import Recognizer
from sara_brain.storage.neuron_repo import NeuronRepo
from sara_brain.storage.segment_repo import SegmentRepo


def extract_words(text: str) -> list[str]:
    stops = {
        "this", "that", "with", "from", "have", "been", "were", "they",
        "their", "them", "than", "then", "more", "most", "also", "only",
        "each", "both", "some", "many", "such", "very", "just", "into",
        "your", "will", "would", "could", "should", "which", "what",
        "when", "where", "about", "above", "below", "these", "those",
        "and", "for", "the", "are", "not", "all", "following",
    }
    raw = re.findall(r"[a-z][a-z'-]+", text.lower())
    words = []
    seen = set()
    for w in raw:
        if len(w) < 4 or w in stops:
            continue
        if w not in seen:
            seen.add(w)
            words.append(w)
    return words


def select_regions(text: str, brain: Brain, regions: list[str]) -> list[str]:
    """Find which regions match this text's concepts."""
    words = extract_words(text)
    scored = []
    for region in regions:
        nr = NeuronRepo(brain.conn, prefix=region)
        hits = sum(1 for w in words if nr.get_by_label(w) is not None)
        if hits > 0:
            scored.append((region, hits))
    scored.sort(key=lambda x: -x[1])
    return [r for r, _ in scored[:3]] if scored else regions[:1]


def get_convergence_score(brain: Brain, regions: list[str],
                          seeds: list[str]) -> tuple[float, int, list[str]]:
    """Run echo through selected regions, return (weight, convergence_count, top_labels)."""
    total_weight = 0.0
    total_convergence = 0
    top_labels = []

    for region in regions:
        nr = NeuronRepo(brain.conn, prefix=region)
        sr = SegmentRepo(brain.conn, prefix=region)
        recognizer = Recognizer(nr, sr, max_depth=3, min_strength=0.1)

        st = ShortTerm(
            event_id=f"shell-{time.time():.3f}",
            event_type="shell_exam",
        )
        recognizer.propagate_echo(
            seeds, st, max_rounds=2, min_strength=0.1, exact_only=True,
        )

        intersections = st.intersections(min_sources=2)
        for nid, weight, sources in intersections:
            total_weight += weight
            total_convergence += 1
            n = nr.get_by_id(nid)
            if n:
                top_labels.append(n.label)

    return total_weight, total_convergence, top_labels[:5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(args.questions) as f:
        questions = json.load(f)
    if args.limit > 0:
        questions = questions[:args.limit]

    brain = Brain(args.db)

    meta_path = args.db + ".regions.json"
    try:
        with open(meta_path) as f:
            regions = json.load(f)["regions"]
    except FileNotFoundError:
        regions = [r["name"] for r in brain.db.list_regions()]

    labels = ["A", "B", "C", "D"]
    total = len(questions)
    correct = 0
    bench_start = time.time()

    print(f"\n  Pure Sensory Shell Exam — ZERO LLM")
    print(f"  Brain: {args.db}")
    print(f"  Regions: {', '.join(regions)}")
    print(f"  Questions: {total}\n")

    for qi, q in enumerate(questions):
        q_words = extract_words(q["question"])
        all_text = q["question"] + " " + " ".join(q["choices"])
        selected = select_regions(all_text, brain, regions)

        # Score each choice by convergence
        scores = []
        for choice in q["choices"]:
            c_words = extract_words(choice)
            seeds = q_words + c_words
            weight, conv, top = get_convergence_score(
                brain, selected, seeds
            )
            scores.append({
                "weight": weight,
                "convergence": conv,
                "top": top,
            })

        # Pick highest convergence weight
        best_idx = max(
            range(4),
            key=lambda i: (scores[i]["weight"], scores[i]["convergence"]),
        )
        answer = labels[best_idx]
        correct_letter = labels[q["answer_idx"]]
        is_correct = answer == correct_letter
        if is_correct:
            correct += 1

        elapsed = time.time() - bench_start
        avg = elapsed / (qi + 1)
        remaining = avg * (total - qi - 1)
        accuracy = correct / (qi + 1) * 100
        status = "✓" if is_correct else "✗"
        top_kw = ",".join(scores[best_idx]["top"][:3])

        print(f"  [{qi+1}/{total}] Q{q['id']}: {status} "
              f"shell={answer} correct={correct_letter} "
              f"w={scores[best_idx]['weight']:.1f} "
              f"conv={scores[best_idx]['convergence']} "
              f"[{top_kw}] "
              f"— {accuracy:.1f}% ({avg:.1f}s/q)",
              flush=True)

    total_time = time.time() - bench_start
    print(f"\n  {'='*55}")
    print(f"  Pure Shell: {correct}/{total} = {correct/total*100:.1f}%")
    print(f"  Time: {total_time:.1f}s ({total_time/total:.1f}s/question)")
    print(f"  {'='*55}")
    print(f"\n  Comparison:")
    print(f"    Random:              25.0%")
    print(f"    Pre-test (3B+Sara):  45.5%")
    print(f"    3B alone:            58.4%")
    print(f"    Pure shell (no LLM): {correct/total*100:.1f}%")

    brain.close()


if __name__ == "__main__":
    main()
