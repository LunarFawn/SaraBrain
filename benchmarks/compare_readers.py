#!/usr/bin/env python3
"""Compare Python Recognizer vs C++ FastRecognizer — forward-only parity test.

Runs every MMLU High School Biology question through BOTH readers using
propagate_into (forward-only, depth 3). For each question the script:

  1. Extracts concepts from the question text.
  2. Propagates into a ShortTerm using the Python Recognizer.
  3. Propagates into a separate ShortTerm using the C++ FastRecognizer.
  4. Compares convergence maps for exact agreement.
  5. Scores each choice against the intersection keywords.
  6. Records answer, accuracy, timing, and any parity failures.

The brain database is opened ONCE and the recognizer is swapped between
runs so both readers see exactly the same graph.

Usage:
    python benchmarks/compare_readers.py --db data/biology_full_v2_clean.db
    python benchmarks/compare_readers.py --db data/biology_full_v2_clean.db --limit 20
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Question loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Concept extraction (same as run_mmlu_wavefront.py)
# ---------------------------------------------------------------------------

def extract_concepts(text: str, stopwords: set[str]) -> list[str]:
    words = re.findall(r"[a-z][a-z'-]+", text.lower())
    return [w for w in words if len(w) >= 4 and w not in stopwords]


# ---------------------------------------------------------------------------
# Core scoring logic — identical for both readers
# ---------------------------------------------------------------------------

def _build_keyword_map(brain, neuron_ids_with_weight) -> dict[str, float]:
    """Pull neuron labels from intersection tuples. Keep max weight."""
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
    matched: list[str] = []
    total = 0.0
    for kw, w in keywords.items():
        pattern = rf"\b{re.escape(kw)}\b"
        if re.search(pattern, text_lower):
            matched.append(kw)
            total += w
    return matched, total


@dataclass
class ReaderResult:
    """Result from one reader for one question."""
    reader_name: str
    answer_idx: int | None        # None = abstain
    answer_letter: str | None
    is_correct: bool
    elapsed_sec: float
    n_converged: int              # total neurons in convergence map
    n_intersections: int          # neurons with >= 2 source wavefronts
    n_signal_keywords: int        # keywords extracted from intersections
    convergence_map: dict[int, float] = field(default_factory=dict, repr=False)


def answer_with_recognizer(brain, recognizer, question: str,
                           choices: list[str], correct_idx: int,
                           stopwords: set[str],
                           reader_name: str) -> ReaderResult:
    """Run a single question through a specific recognizer."""
    from sara_brain.core.short_term import ShortTerm

    t0 = time.time()

    st = ShortTerm(
        event_id=f"{reader_name}-{time.time():.3f}",
        event_type="benchmark",
    )
    q_concepts = extract_concepts(question, stopwords)

    # Call the recognizer's propagate_into directly
    recognizer.propagate_into(q_concepts, st, exact_only=True)

    n_converged = len(st.convergence_map)
    intersections = st.intersections(min_sources=2)
    n_intersections = len(intersections)

    signal_kw = _build_keyword_map(brain, intersections)
    n_signal_kw = len(signal_kw)

    # Score each choice
    scores = []
    for choice in choices:
        matched, total_weight = _match_keywords_in_text(
            signal_kw, choice.lower()
        )
        scores.append({
            "match_count": len(matched),
            "match_weight": total_weight,
        })

    # Pick best or abstain
    if all(s["match_weight"] == 0 for s in scores):
        best_idx = None
    else:
        best_idx = max(
            range(len(scores)),
            key=lambda i: (scores[i]["match_weight"], scores[i]["match_count"]),
        )

    elapsed = time.time() - t0
    answer_letter = chr(65 + best_idx) if best_idx is not None else None
    is_correct = (best_idx == correct_idx)

    return ReaderResult(
        reader_name=reader_name,
        answer_idx=best_idx,
        answer_letter=answer_letter,
        is_correct=is_correct,
        elapsed_sec=elapsed,
        n_converged=n_converged,
        n_intersections=n_intersections,
        n_signal_keywords=n_signal_kw,
        convergence_map=dict(st.convergence_map),
    )


# ---------------------------------------------------------------------------
# Parity checking
# ---------------------------------------------------------------------------

def check_parity(py_result: ReaderResult,
                 cpp_result: ReaderResult) -> dict:
    """Compare two reader results for exact agreement."""
    # Same answer?
    answer_match = (py_result.answer_idx == cpp_result.answer_idx)

    # Same convergence map keys?
    py_keys = set(py_result.convergence_map.keys())
    cpp_keys = set(cpp_result.convergence_map.keys())
    keys_match = (py_keys == cpp_keys)
    only_in_py = py_keys - cpp_keys
    only_in_cpp = cpp_keys - py_keys

    # Same convergence weights? (within tolerance)
    weight_mismatches = 0
    max_weight_diff = 0.0
    shared_keys = py_keys & cpp_keys
    for k in shared_keys:
        diff = abs(py_result.convergence_map[k] - cpp_result.convergence_map[k])
        if diff > 0.001:
            weight_mismatches += 1
            max_weight_diff = max(max_weight_diff, diff)

    return {
        "answer_match": answer_match,
        "keys_match": keys_match,
        "only_in_py": len(only_in_py),
        "only_in_cpp": len(only_in_cpp),
        "shared_keys": len(shared_keys),
        "weight_mismatches": weight_mismatches,
        "max_weight_diff": max_weight_diff,
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare Python vs C++ reader parity (forward-only)"
    )
    parser.add_argument("--db", required=True,
                        help="Path to Sara Brain database")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of questions (0 = all 310)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save JSON results to this path")
    args = parser.parse_args()

    # Load questions
    print(f"\n  Loading MMLU High School Biology questions...", flush=True)
    questions = load_questions()
    if args.limit > 0:
        questions = questions[:args.limit]
    print(f"  Loaded {len(questions)} questions.\n", flush=True)

    # Load brain
    from sara_brain.core.brain import Brain
    from sara_brain.core.recognizer import Recognizer
    from sara_brain.core.fast_recognizer import FastRecognizer

    print(f"  Loading brain: {args.db}", flush=True)
    brain = Brain(args.db)
    stats = brain.stats()
    print(f"  Neurons: {stats['neurons']:,}  "
          f"Segments: {stats['segments']:,}  "
          f"Paths: {stats['paths']:,}\n", flush=True)

    # Build both recognizers from the same repos
    py_recognizer = Recognizer(
        brain.neuron_repo, brain.segment_repo,
        max_depth=3, min_strength=0.5,
    )
    print(f"  [Python Recognizer] Ready.", flush=True)

    cpp_recognizer = FastRecognizer(
        brain.neuron_repo, brain.segment_repo,
        max_depth=3, min_strength=0.5,
    )
    print(f"  [C++ FastRecognizer] Ready.\n", flush=True)

    # Stopwords
    from sara_brain.cortex.cleanup import STOPWORD_SUBJECTS
    stopwords = set(STOPWORD_SUBJECTS) | {
        "following", "most", "many", "some", "each", "every", "both",
        "which", "what", "when", "where", "would", "could", "should",
        "about", "above", "below", "these", "those", "between",
        "question", "answer", "choice", "correct", "example",
    }

    # Accumulators
    py_correct = 0
    py_incorrect = 0
    py_abstain = 0
    cpp_correct = 0
    cpp_incorrect = 0
    cpp_abstain = 0
    parity_failures = 0
    key_mismatches = 0
    weight_mismatches = 0
    py_total_time = 0.0
    cpp_total_time = 0.0
    per_question = []

    print(f"  {'='*72}")
    print(f"  MMLU Biology — Python vs C++ Reader Parity (forward-only)")
    print(f"  {'='*72}\n")

    bench_start = time.time()

    for i, q in enumerate(questions):
        correct_idx = q['answer_idx']
        correct_letter = chr(65 + correct_idx)

        # Run Python reader
        py_res = answer_with_recognizer(
            brain, py_recognizer, q['question'], q['choices'],
            correct_idx, stopwords, "python"
        )

        # Run C++ reader
        cpp_res = answer_with_recognizer(
            brain, cpp_recognizer, q['question'], q['choices'],
            correct_idx, stopwords, "cpp"
        )

        # Track timing
        py_total_time += py_res.elapsed_sec
        cpp_total_time += cpp_res.elapsed_sec

        # Track accuracy
        if py_res.answer_idx is None:
            py_abstain += 1
        elif py_res.is_correct:
            py_correct += 1
        else:
            py_incorrect += 1

        if cpp_res.answer_idx is None:
            cpp_abstain += 1
        elif cpp_res.is_correct:
            cpp_correct += 1
        else:
            cpp_incorrect += 1

        # Check parity
        parity = check_parity(py_res, cpp_res)
        if not parity["answer_match"]:
            parity_failures += 1
        if not parity["keys_match"]:
            key_mismatches += 1
        if parity["weight_mismatches"] > 0:
            weight_mismatches += 1

        per_question.append({
            "id": q['id'],
            "correct": correct_letter,
            "py_answer": py_res.answer_letter,
            "cpp_answer": cpp_res.answer_letter,
            "py_correct": py_res.is_correct,
            "cpp_correct": cpp_res.is_correct,
            "py_time": round(py_res.elapsed_sec, 4),
            "cpp_time": round(cpp_res.elapsed_sec, 4),
            "speedup": round(py_res.elapsed_sec / cpp_res.elapsed_sec, 1) if cpp_res.elapsed_sec > 0 else 0,
            "py_converged": py_res.n_converged,
            "cpp_converged": cpp_res.n_converged,
            "py_intersections": py_res.n_intersections,
            "cpp_intersections": cpp_res.n_intersections,
            "answer_match": parity["answer_match"],
            "keys_match": parity["keys_match"],
            "weight_mismatches": parity["weight_mismatches"],
        })

        # Progress line
        py_answered = py_correct + py_incorrect
        py_acc = (py_correct / py_answered * 100) if py_answered else 0
        cpp_answered = cpp_correct + cpp_incorrect
        cpp_acc = (cpp_correct / cpp_answered * 100) if cpp_answered else 0
        parity_sym = "✓" if parity["answer_match"] else "✗"

        elapsed_total = time.time() - bench_start
        avg = elapsed_total / (i + 1)
        remaining = avg * (len(questions) - i - 1)

        print(
            f"  [{i+1:3d}/{len(questions)}] Q{q['id']:3d}: "
            f"Py={py_res.answer_letter or '-'} "
            f"C++={cpp_res.answer_letter or '-'} "
            f"Correct={correct_letter} "
            f"Parity={parity_sym} "
            f"| Py {py_res.elapsed_sec:.3f}s C++ {cpp_res.elapsed_sec:.3f}s "
            f"({py_res.elapsed_sec / cpp_res.elapsed_sec:.0f}x) "
            f"| PyAcc={py_acc:.1f}% CppAcc={cpp_acc:.1f}% "
            f"(~{remaining/60:.0f}m left)" if cpp_res.elapsed_sec > 0 else "",
            flush=True,
        )

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    total_time = time.time() - bench_start
    total_q = len(questions)

    py_answered = py_correct + py_incorrect
    cpp_answered = cpp_correct + cpp_incorrect
    py_scored_acc = (py_correct / py_answered * 100) if py_answered else 0
    cpp_scored_acc = (cpp_correct / cpp_answered * 100) if cpp_answered else 0
    py_overall_acc = (py_correct / total_q * 100) if total_q else 0
    cpp_overall_acc = (cpp_correct / total_q * 100) if total_q else 0
    py_coverage = ((total_q - py_abstain) / total_q * 100) if total_q else 0
    cpp_coverage = ((total_q - cpp_abstain) / total_q * 100) if total_q else 0

    speedup = (py_total_time / cpp_total_time) if cpp_total_time > 0 else 0

    # Count disagreements where one was right and the other wrong
    disagree_details = [
        pq for pq in per_question if not pq["answer_match"]
    ]

    print(f"\n  {'='*72}")
    print(f"  RESULTS — Python vs C++ Forward-Only Parity")
    print(f"  {'='*72}")
    print()
    print(f"  {'Metric':<30s} {'Python':>12s} {'C++':>12s}")
    print(f"  {'-'*30} {'-'*12} {'-'*12}")
    print(f"  {'Total Questions':<30s} {total_q:>12d} {total_q:>12d}")
    print(f"  {'Answered':<30s} {py_answered:>12d} {cpp_answered:>12d}")
    print(f"  {'Abstained':<30s} {py_abstain:>12d} {cpp_abstain:>12d}")
    print(f"  {'Correct':<30s} {py_correct:>12d} {cpp_correct:>12d}")
    print(f"  {'Incorrect':<30s} {py_incorrect:>12d} {cpp_incorrect:>12d}")
    print(f"  {'Scored Accuracy':<30s} {py_scored_acc:>11.1f}% {cpp_scored_acc:>11.1f}%")
    print(f"  {'Overall Accuracy':<30s} {py_overall_acc:>11.1f}% {cpp_overall_acc:>11.1f}%")
    print(f"  {'Coverage':<30s} {py_coverage:>11.1f}% {cpp_coverage:>11.1f}%")
    print(f"  {'Total Time':<30s} {py_total_time:>10.1f}s {cpp_total_time:>10.1f}s")
    print(f"  {'Avg per Question':<30s} {py_total_time/total_q:>10.3f}s {cpp_total_time/total_q:>10.3f}s")
    print()
    print(f"  {'='*72}")
    print(f"  PARITY CHECK")
    print(f"  {'='*72}")
    print(f"  Answer agreement:         {total_q - parity_failures}/{total_q} "
          f"({(total_q - parity_failures)/total_q*100:.1f}%)")
    print(f"  Answer disagreements:     {parity_failures}")
    print(f"  Convergence key mismatch: {key_mismatches} questions")
    print(f"  Weight mismatch (>0.001): {weight_mismatches} questions")
    print(f"  Speedup (Py/C++):         {speedup:.1f}x")
    print()

    if disagree_details:
        print(f"  DISAGREEMENTS (answer mismatch):")
        for d in disagree_details[:20]:
            print(f"    Q{d['id']:3d}: Py={d['py_answer'] or 'ABSTAIN':<8s} "
                  f"C++={d['cpp_answer'] or 'ABSTAIN':<8s} "
                  f"Correct={d['correct']}  "
                  f"PyConv={d['py_converged']} CppConv={d['cpp_converged']}  "
                  f"PyIsect={d['py_intersections']} CppIsect={d['cpp_intersections']}")
        if len(disagree_details) > 20:
            print(f"    ... and {len(disagree_details) - 20} more")
        print()

    if parity_failures == 0:
        print(f"  ✓ PERFECT PARITY — Python and C++ produce identical answers.")
    else:
        print(f"  ✗ PARITY FAILURE — {parity_failures} questions differ.")
        print(f"    This means the C++ engine is NOT faithfully reproducing")
        print(f"    the Python BFS behavior. Investigation needed.")
    print(f"  {'='*72}\n")

    # Save results
    if args.output:
        results = {
            "benchmark": "compare_readers_forward_only",
            "db": args.db,
            "total_questions": total_q,
            "python": {
                "correct": py_correct, "incorrect": py_incorrect,
                "abstained": py_abstain, "scored_accuracy": py_scored_acc,
                "overall_accuracy": py_overall_acc, "coverage": py_coverage,
                "total_time_sec": py_total_time,
            },
            "cpp": {
                "correct": cpp_correct, "incorrect": cpp_incorrect,
                "abstained": cpp_abstain, "scored_accuracy": cpp_scored_acc,
                "overall_accuracy": cpp_overall_acc, "coverage": cpp_coverage,
                "total_time_sec": cpp_total_time,
            },
            "parity": {
                "answer_agreement": total_q - parity_failures,
                "answer_disagreements": parity_failures,
                "key_mismatches": key_mismatches,
                "weight_mismatches": weight_mismatches,
                "speedup": speedup,
            },
            "per_question": per_question,
        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to {args.output}")


if __name__ == "__main__":
    main()
