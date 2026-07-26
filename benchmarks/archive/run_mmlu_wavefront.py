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


def _build_keyword_map(brain, neuron_ids_with_weight) -> dict[str, float]:
    """Pull neuron labels from a list of (id, weight, ...) tuples or a
    dict of {id: weight}. Filter to labels >= 3 chars. Keep max weight."""
    keywords: dict[str, float] = {}
    if isinstance(neuron_ids_with_weight, dict):
        items = neuron_ids_with_weight.items()
    else:
        items = [(t[0], t[1]) for t in neuron_ids_with_weight]
    for nid, weight in items:
        n = brain.neuron_repo.get_by_id(nid)
        if n is None:
            continue
        label = n.label.lower().strip()
        if len(label) < 3:
            continue
        if label not in keywords or weight > keywords[label]:
            keywords[label] = weight
    return keywords


def _match_keywords_in_text(keywords: dict[str, float],
                            text_lower: str) -> tuple[list[str], float]:
    """Word-boundary match keywords against text. Return (matched, weight)."""
    matched: list[str] = []
    total = 0.0
    for kw, w in keywords.items():
        pattern = rf"\b{re.escape(kw)}\b"
        if re.search(pattern, text_lower):
            matched.append(kw)
            total += w
    return matched, total


def answer_question(brain, question: str, choices: list[str],
                    stopwords: set[str]) -> tuple[int | None, list[dict], dict]:
    """Answer multiple-choice by intersection-keyword match — or abstain.

    The question propagates into a ShortTerm. Only the INTERSECTIONS
    (neurons reached by >= 2 independent wavefronts) count as signal.
    Those are the concepts Sara thinks the question is really about.

    For each choice, word-boundary match the intersection keywords
    against the choice's text. If ANY choice has a keyword hit, Sara
    picks the choice with the highest weighted match.

    If NO choice has a keyword hit, Sara ABSTAINS (returns None). She
    doesn't have the knowledge to answer this question. This is
    architecturally honest: defaulting would dress up ignorance as an
    answer. A doctor doesn't guess outside their specialty.

    This separates two metrics:
        - Coverage: does Sara have knowledge for this question?
        - Scored accuracy: when Sara has knowledge, is she right?

    Read-only throughout. Exact-only label resolution.
    """
    with brain.short_term(event_type="mmlu_question") as q_st:
        q_concepts = extract_concepts(question, stopwords)
        brain.propagate_into(q_concepts, q_st, exact_only=True)

        q_total_converged = len(q_st.convergence_map)
        q_intersections = q_st.intersections(min_sources=2)

        # Signal keywords from intersection neurons
        signal_kw = _build_keyword_map(brain, q_intersections)

        # Score each choice by keyword text overlap
        scores = []
        for choice in choices:
            choice_lower = choice.lower()
            matched, total_weight = _match_keywords_in_text(
                signal_kw, choice_lower
            )
            scores.append({
                "matched_keywords": matched,
                "match_count": len(matched),
                "match_weight": total_weight,
            })

        # If NO choice matched any signal keyword, Sara abstains.
        # She doesn't have the knowledge. Don't pretend otherwise.
        if all(s["match_weight"] == 0 for s in scores):
            best_idx = None
        else:
            best_idx = max(
                range(len(scores)),
                key=lambda i: (scores[i]["match_weight"],
                               scores[i]["match_count"]),
            )

        meta = {
            "q_concepts_extracted": len(q_concepts),
            "q_total_converged": q_total_converged,
            "q_intersections": len(q_intersections),
            "signal_keywords_count": len(signal_kw),
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
        "mode": "sara wavefront (ShortTerm, read-only, honest-abstain)",
        "total": len(questions),
        "correct": 0,       # correct among answered questions
        "incorrect": 0,     # wrong among answered questions
        "abstained": 0,     # questions where Sara had no knowledge
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
                "is_correct": False, "abstained": False,
                "error": str(e),
            })
            continue

        correct_idx = q['answer_idx']
        correct_letter = chr(65 + correct_idx)

        if best_idx is None:
            # Sara abstained — no signal. Not wrong, not right — unknown.
            results["abstained"] += 1
            results["answers"].append({
                "id": q['id'],
                "model_answer": None,
                "correct_letter": correct_letter,
                "is_correct": False,
                "abstained": True,
                "scores": scores,
                "meta": meta,
            })
            elapsed = time.time() - q_start
            total_elapsed = time.time() - bench_start
            answered = results["correct"] + results["incorrect"]
            scored_acc = (results["correct"] / answered * 100
                          if answered else 0.0)
            coverage = ((i + 1) - results["abstained"]) / (i + 1) * 100
            avg = total_elapsed / (i + 1)
            remaining = avg * (len(questions) - i - 1)
            print(
                f"  [{i+1}/{len(questions)}] Q{q['id']}: ABSTAIN "
                f"(no signal, correct would be {correct_letter}) "
                f"[isect={meta['q_intersections']}/kw={meta['signal_keywords_count']}] "
                f"— scored {scored_acc:.1f}% (cov {coverage:.1f}%) "
                f"— {elapsed:.2f}s (~{remaining/60:.0f}m left)",
                flush=True
            )
            continue

        is_correct = best_idx == correct_idx
        letter = chr(65 + best_idx)

        if is_correct:
            results["correct"] += 1
        else:
            results["incorrect"] += 1

        results["answers"].append({
            "id": q['id'],
            "model_answer": letter,
            "correct_letter": correct_letter,
            "is_correct": is_correct,
            "abstained": False,
            "scores": scores,
            "meta": meta,
        })

        elapsed = time.time() - q_start
        total_elapsed = time.time() - bench_start
        status = "CORRECT" if is_correct else "WRONG"
        answered = results["correct"] + results["incorrect"]
        scored_acc = (results["correct"] / answered * 100
                      if answered else 0.0)
        coverage = ((i + 1) - results["abstained"]) / (i + 1) * 100
        avg = total_elapsed / (i + 1)
        remaining = avg * (len(questions) - i - 1)
        best = scores[best_idx]
        kw_preview = ",".join(best["matched_keywords"][:4])
        if len(best["matched_keywords"]) > 4:
            kw_preview += f"+{len(best['matched_keywords']) - 4}"
        print(
            f"  [{i+1}/{len(questions)}] Q{q['id']}: {status} "
            f"(chose {letter} w={best['match_weight']:.1f}/"
            f"hits={best['match_count']} [{kw_preview}], "
            f"correct {correct_letter}) "
            f"[sig_kw={meta['signal_keywords_count']}] "
            f"— scored {scored_acc:.1f}% (cov {coverage:.1f}%) "
            f"— {elapsed:.2f}s (~{remaining/60:.0f}m left)",
            flush=True
        )

    total_time = time.time() - bench_start
    answered = results["correct"] + results["incorrect"]
    # scored_accuracy: correct among questions Sara answered
    results["scored_accuracy"] = (
        results["correct"] / answered * 100 if answered else 0.0
    )
    # coverage: fraction of questions Sara had knowledge for
    results["coverage"] = (
        (results["total"] - results["abstained"]) / results["total"] * 100
    )
    # overall_accuracy: correct / total (abstentions count as incorrect
    # for cross-system comparison — but this is NOT Sara's honest metric)
    results["overall_accuracy"] = (
        results["correct"] / results["total"] * 100 if results["total"] else 0.0
    )
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
    print(f"  MMLU Biology — Sara wavefront (ShortTerm, honest-abstain)")
    print(f"  {'='*60}")
    print(f"  Total questions:  {results['total']}")
    print(f"  Answered:         {results['correct'] + results['incorrect']}"
          f" (coverage {results['coverage']:.1f}%)")
    print(f"  Abstained:        {results['abstained']}"
          f" (no knowledge — Sara honestly said so)")
    print(f"  ")
    print(f"  Scored accuracy:  {results['correct']} / "
          f"{results['correct'] + results['incorrect']}"
          f" = {results['scored_accuracy']:.1f}%"
          f"   ← the honest metric")
    print(f"  Overall accuracy: {results['correct']} / {results['total']}"
          f" = {results['overall_accuracy']:.1f}%"
          f"   ← for cross-system comparison")
    print(f"  ")
    print(f"  Time:             {results['total_time_sec']/60:.1f} min")
    print(f"  Read-only:        {'YES ✓' if unchanged else f'VIOLATED ({changed_count} segments)'}")
    print(f"  {'='*60}")
    print()
    print(f"  Scored accuracy = 'when Sara has knowledge, how often is she right?'")
    print(f"  Coverage        = 'on what fraction of questions does Sara have knowledge?'")
    print()
    print(f"  Reference (overall accuracy, forced to answer):")
    print(f"    Random guess:     25.0%")
    print(f"    qwen 3B alone:    58.4%")
    print(f"    GPT-3.5:          ~70%")
    print()

    if args.output:
        with open(args.output, "w") as f:
            json.dump([results], f, indent=2)
        print(f"  Results saved to {args.output}")


if __name__ == "__main__":
    main()
