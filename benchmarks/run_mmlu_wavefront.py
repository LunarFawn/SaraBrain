#!/usr/bin/env python3
"""MMLU Biology — wavefront convergence benchmark (ShortTerm version).

The architecturally correct benchmark. No prompt-stuffing, no retrieval
dump, no LLM deciding what to query. Sara answers by building a
convergence map from the question alone, then measuring each choice
against that map.

Per-question flow:
  1. Open a short-term scratchpad for this question.
  2. Extract concepts from the question and propagate_into the scratchpad.
     This builds the convergence map — "what the question is really about."
  3. For each choice, resolve its concepts to neuron ids and compute
     align_score against the convergence map.
  4. Pick the choice with the highest alignment. Tie-break on hit count.
     This is "least wrong" — the choice whose concepts land best in the
     question's convergence, not the one that perfectly matches.
  5. Short-term discards on exit. The graph is never mutated.

The graph is read-only for the entire benchmark — re-running is
deterministic.

Usage:
    python benchmarks/run_mmlu_wavefront.py --db bio_full.db
"""

from __future__ import annotations

import argparse
import json
import re
import time


def load_questions() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset('cais/mmlu', 'high_school_biology', split='test')
    return [
        {
            'id': i,
            'question': q['question'],
            'choices': q['choices'],
            'answer_idx': q['answer'],
        }
        for i, q in enumerate(ds)
    ]


def extract_concepts(text: str, stopwords: set[str]) -> list[str]:
    """Extract content words from text that Sara might know as concepts."""
    words = re.findall(r"[a-z][a-z'-]+", text.lower())
    return [w for w in words if len(w) >= 4 and w not in stopwords]


def resolve_to_ids(brain, labels: list[str]) -> list[int]:
    """Resolve concept labels to neuron IDs. Skips unresolvable labels."""
    ids = []
    for lbl in labels:
        n = brain.neuron_repo.resolve(lbl)
        if n is not None:
            ids.append(n.id)
    return ids


def answer_question(brain, question: str, choices: list[str],
                    stopwords: set[str]) -> tuple[int, list[dict]]:
    """Answer a multiple-choice question by wavefront convergence.

    Step 1: Build convergence map from the question alone (no choices
            biasing it).
    Step 2: Score each choice by how well its concepts align with the
            question's convergence map.
    Step 3: Pick the highest-aligned choice — least wrong among options.
    """
    # Step 1: build convergence from question concepts
    with brain.short_term(event_type="mmlu_question") as st:
        q_concepts = extract_concepts(question, stopwords)
        brain.propagate_into(q_concepts, st)

        # Step 2: evaluate each choice against the convergence map
        scores = []
        for choice in choices:
            c_concepts = extract_concepts(choice, stopwords)
            c_ids = resolve_to_ids(brain, c_concepts)
            weight, hits = st.align_score(c_ids)
            scores.append({
                "weight": weight,
                "hits": hits,
                "concepts_extracted": len(c_concepts),
                "concepts_resolved": len(c_ids),
            })

        # Also record question-side stats for debugging
        q_total_converged = len(st.convergence_map)
        q_intersections = len(st.intersections(min_sources=2))

        # Step 3: pick the choice with highest alignment weight,
        # tie-break on hit count, then on choice index
        best_idx = max(
            range(len(scores)),
            key=lambda i: (scores[i]["weight"], scores[i]["hits"])
        )

        meta = {
            "q_concepts_extracted": len(q_concepts),
            "q_total_converged": q_total_converged,
            "q_intersections": q_intersections,
        }

    return best_idx, scores, meta


def run_benchmark(questions: list[dict], brain) -> dict:
    from sara_brain.cortex.cleanup import STOPWORD_SUBJECTS
    stopwords = set(STOPWORD_SUBJECTS) | {
        "following", "most", "many", "some", "each", "every", "both",
        "which", "what", "when", "where", "would", "could", "should",
        "about", "above", "below", "these", "those", "between",
        "question", "answer", "choice", "correct", "example",
    }

    results = {
        "mode": "sara wavefront (ShortTerm, read-only)",
        "total": len(questions),
        "correct": 0,
        "incorrect": 0,
        "no_signal": 0,
        "answers": [],
    }

    bench_start = time.time()

    for i, q in enumerate(questions):
        q_start = time.time()
        try:
            best_idx, scores, meta = answer_question(
                brain, q['question'], q['choices'], stopwords
            )
        except Exception as e:
            print(f"  [{i+1}/{len(questions)}] Q{q['id']}: ERROR — {e}",
                  flush=True)
            results["answers"].append({
                "id": q['id'], "model_answer": None,
                "correct_letter": chr(65 + q['answer_idx']),
                "is_correct": False, "error": str(e),
            })
            continue

        # No-signal: all choices got zero alignment
        if all(s["weight"] == 0 for s in scores):
            results["no_signal"] += 1

        correct_idx = q['answer_idx']
        is_correct = best_idx == correct_idx
        letter = chr(65 + best_idx)
        correct_letter = chr(65 + correct_idx)

        if is_correct:
            results["correct"] += 1
        else:
            results["incorrect"] += 1

        results["answers"].append({
            "id": q['id'],
            "model_answer": letter,
            "correct_letter": correct_letter,
            "is_correct": is_correct,
            "scores": scores,
            "meta": meta,
        })

        elapsed = time.time() - q_start
        total_elapsed = time.time() - bench_start
        status = "CORRECT" if is_correct else "WRONG"
        accuracy = results["correct"] / (i + 1) * 100
        avg = total_elapsed / (i + 1)
        remaining = avg * (len(questions) - i - 1)
        print(
            f"  [{i+1}/{len(questions)}] Q{q['id']}: {status} "
            f"(chose {letter} w={scores[best_idx]['weight']:.1f}/h={scores[best_idx]['hits']}, "
            f"correct {correct_letter}) "
            f"[q_conv={meta['q_total_converged']}, isect={meta['q_intersections']}] "
            f"— {accuracy:.1f}% — {elapsed:.2f}s (~{remaining/60:.0f}m left)",
            flush=True
        )

    total_time = time.time() - bench_start
    results["accuracy"] = results["correct"] / results["total"] * 100
    results["total_time_sec"] = total_time
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output")
    args = parser.parse_args()

    questions = load_questions()
    if args.limit > 0:
        questions = questions[:args.limit]

    from sara_brain.core.brain import Brain
    brain = Brain(args.db)
    stats = brain.stats()

    print(f"\n  MMLU Biology — Wavefront Convergence (ShortTerm)")
    print(f"  Brain: {args.db}")
    print(f"  Neurons: {stats['neurons']}, paths: {stats['paths']}, "
          f"segments: {stats['segments']}")
    print(f"  {len(questions)} questions, no LLM in the loop, read-only\n")

    # Snapshot segment strengths BEFORE — we'll verify no mutation happened
    c = brain.conn.cursor()
    before_strengths = dict(c.execute(
        'SELECT id, strength FROM segments'
    ).fetchall())

    results = run_benchmark(questions, brain)

    # Verify read-only contract
    after_strengths = dict(c.execute(
        'SELECT id, strength FROM segments'
    ).fetchall())
    unchanged = before_strengths == after_strengths
    changed_count = sum(
        1 for sid in before_strengths
        if before_strengths[sid] != after_strengths.get(sid)
    )

    print()
    print(f"  {'='*60}")
    print(f"  MMLU Biology — Sara wavefront (ShortTerm)")
    print(f"  {'='*60}")
    print(f"  Total:      {results['total']}")
    print(f"  Correct:    {results['correct']} ({results['accuracy']:.1f}%)")
    print(f"  Incorrect:  {results['incorrect']}")
    print(f"  No signal:  {results['no_signal']}")
    print(f"  Time:       {results['total_time_sec']/60:.1f} min")
    print(f"  Read-only:  {'YES ✓' if unchanged else f'VIOLATED ({changed_count} segments mutated)'}")
    print(f"  {'='*60}")
    print()
    print(f"  Reference:")
    print(f"    Random:           25.0%")
    print(f"    qwen 3B alone:    58.4%")
    print(f"    GPT-3.5:          ~70%")
    print()

    if args.output:
        with open(args.output, "w") as f:
            json.dump([results], f, indent=2)
        print(f"  Results saved to {args.output}")


if __name__ == "__main__":
    main()
