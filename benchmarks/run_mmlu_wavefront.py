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
    """Answer a multiple-choice question by wavefront TERRITORY overlap.

    The question launches wavefronts that build a convergence map.
    Each choice ALSO launches its own wavefronts into its own scratchpad.
    The answer is the choice whose territory overlaps most with the
    question's territory — where their wavefronts meet.

    This catches "shoot tip → meristem ← mitosis" style connections
    where the concepts themselves don't match but their surrounding
    territory does.

    Read-only throughout. Exact-only label resolution.
    """
    # Step 1: build the question's convergence territory
    with brain.short_term(event_type="mmlu_question") as q_st:
        q_concepts = extract_concepts(question, stopwords)
        brain.propagate_into(q_concepts, q_st, exact_only=True)

        q_territory = set(q_st.convergence_map.keys())
        q_total_converged = len(q_territory)
        q_intersections = len(q_st.intersections(min_sources=2))

        # Step 2: build each choice's territory and compute raw overlap
        choice_territories: list[set[int]] = []
        choice_scratchpads = []  # kept alive for reading convergence_map
        c_concept_ids_per_choice: list[set[int]] = []

        for choice in choices:
            c_st_cm = brain.short_term(event_type="mmlu_choice")
            c_st = c_st_cm.__enter__()
            choice_scratchpads.append((c_st_cm, c_st))

            c_concepts = extract_concepts(choice, stopwords)
            brain.propagate_into(c_concepts, c_st, exact_only=True)
            choice_territories.append(set(c_st.convergence_map.keys()))

            # Also track which choice-concept neurons appear in q_territory
            # (direct hits, for tie-breaking)
            direct_ids = set()
            for lbl in c_concepts:
                n = brain.neuron_repo.resolve(lbl, exact_only=True)
                if n is not None and n.id in q_territory:
                    direct_ids.add(n.id)
            c_concept_ids_per_choice.append(direct_ids)

        # Differential signal: concepts that appear in THIS choice's
        # overlap with the question but NOT in all other choices. This
        # strips out the generic biology noise (cell, organism, etc.)
        # that every choice shares with the question.
        all_choice_overlaps = [q_territory & ct for ct in choice_territories]
        # baseline = neurons overlapping with question across ALL choices
        baseline = set.intersection(*all_choice_overlaps) if all_choice_overlaps else set()

        scores = []
        for i, (choice, c_territory) in enumerate(zip(choices, choice_territories)):
            _, c_st = choice_scratchpads[i]
            raw_overlap = all_choice_overlaps[i]
            # Unique overlap: what's in THIS choice's overlap minus what
            # EVERY other choice also has. This is the choice-specific signal.
            unique_overlap = raw_overlap - baseline
            unique_weight = sum(
                q_st.convergence_map[nid] + c_st.convergence_map[nid]
                for nid in unique_overlap
            )
            raw_weight = sum(
                q_st.convergence_map[nid] + c_st.convergence_map[nid]
                for nid in raw_overlap
            )
            direct_ids = c_concept_ids_per_choice[i]
            direct_weight = sum(
                q_st.convergence_map[nid] for nid in direct_ids
            )

            scores.append({
                "raw_overlap": len(raw_overlap),
                "raw_weight": raw_weight,
                "unique_overlap": len(unique_overlap),
                "unique_weight": unique_weight,
                "direct_hits": len(direct_ids),
                "direct_weight": direct_weight,
                "c_territory": len(c_territory),
                "concepts_extracted": len(extract_concepts(choice, stopwords)),
            })

        # Close the choice scratchpads
        for c_st_cm, _ in choice_scratchpads:
            c_st_cm.__exit__(None, None, None)

        # Step 3: pick least wrong — DIFFERENTIAL signal is what matters.
        # Primary: unique_weight (how much does THIS choice share with
        #   the question that OTHERS don't — the choice-specific signal)
        # Tie-break: direct_weight (explicit concept hits), then raw_weight,
        #   then choice index
        best_idx = max(
            range(len(scores)),
            key=lambda i: (
                scores[i]["unique_weight"],
                scores[i]["direct_weight"],
                scores[i]["raw_weight"],
            ),
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

        # No-signal: no choice has any UNIQUE overlap with the question
        # (all choices share the same generic territory — Sara can't
        # differentiate, so any pick is guessing)
        if all(s["unique_weight"] == 0 for s in scores):
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
        print(
            f"  [{i+1}/{len(questions)}] Q{q['id']}: {status} "
            f"(chose {letter} uw={best['unique_weight']:.1f}/"
            f"uo={best['unique_overlap']}/dh={best['direct_hits']}, "
            f"correct {correct_letter}) "
            f"[q_conv={meta['q_total_converged']}] "
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
