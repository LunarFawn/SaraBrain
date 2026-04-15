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


def answer_question(brain, question: str, choices: list[str],
                    stopwords: set[str]) -> tuple[int, list[dict], dict]:
    """Answer a multiple-choice question by closest keyword match.

    The question launches wavefronts that propagate into a ShortTerm.
    The INTERSECTIONS (neurons reached by >= 2 independent wavefronts)
    are the signal — that's what Sara thinks the question is really
    about. Each intersection neuron's label becomes a keyword.

    For each choice, count how many of those signal keywords appear in
    the choice's TEXT (word-boundary match). Highest weighted match wins.

    Read-only throughout. Exact-only label resolution. No territory
    overlap, no per-choice propagation. Just "which choice's text
    contains the most of the keywords Sara identified from the question."
    """
    with brain.short_term(event_type="mmlu_question") as q_st:
        q_concepts = extract_concepts(question, stopwords)
        brain.propagate_into(q_concepts, q_st, exact_only=True)

        q_total_converged = len(q_st.convergence_map)
        q_intersections = q_st.intersections(min_sources=2)

        # Build signal keyword → weight map from intersections.
        # Only multi-wavefront neurons carry real signal; single-hit
        # neurons are noise from the 500–1000 neuron convergence.
        signal_keywords: dict[str, float] = {}
        for nid, weight, _sources in q_intersections:
            n = brain.neuron_repo.get_by_id(nid)
            if n is None:
                continue
            label = n.label.lower().strip()
            # Skip very short labels and labels containing multiple words
            # — multi-word labels rarely appear verbatim in choice text
            if len(label) < 3:
                continue
            # Keep the strongest occurrence if the same label appears twice
            if label not in signal_keywords or weight > signal_keywords[label]:
                signal_keywords[label] = weight

        # Score each choice by keyword text overlap
        scores = []
        for choice in choices:
            choice_lower = choice.lower()
            matched: list[str] = []
            total_weight = 0.0
            for kw, w in signal_keywords.items():
                # Word-boundary match so "ant" doesn't hit "anther"
                # Use re.escape to handle any regex-special chars in labels
                pattern = rf"\b{re.escape(kw)}\b"
                if re.search(pattern, choice_lower):
                    matched.append(kw)
                    total_weight += w
            scores.append({
                "matched_keywords": matched,
                "match_count": len(matched),
                "match_weight": total_weight,
            })

        # Pick the closest: highest weighted match, tie-break on count
        best_idx = max(
            range(len(scores)),
            key=lambda i: (scores[i]["match_weight"],
                           scores[i]["match_count"]),
        )

        meta = {
            "q_concepts_extracted": len(q_concepts),
            "q_total_converged": q_total_converged,
            "q_intersections": len(q_intersections),
            "signal_keywords_count": len(signal_keywords),
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

        # No-signal: no choice contains any of Sara's signal keywords
        # (the convergence found intersections, but none of them appear
        # in any choice's text — Sara can't differentiate)
        if all(s["match_weight"] == 0 for s in scores):
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
        best = scores[best_idx]
        # Show the top 4 matched keywords for the chosen option
        kw_preview = ",".join(best["matched_keywords"][:4])
        if len(best["matched_keywords"]) > 4:
            kw_preview += f"+{len(best['matched_keywords']) - 4}"
        print(
            f"  [{i+1}/{len(questions)}] Q{q['id']}: {status} "
            f"(chose {letter} w={best['match_weight']:.1f}/"
            f"hits={best['match_count']} [{kw_preview}], "
            f"correct {correct_letter}) "
            f"[sig_kw={meta['signal_keywords_count']}/isect={meta['q_intersections']}] "
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
