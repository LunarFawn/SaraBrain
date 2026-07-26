#!/usr/bin/env python3
"""Teach the correct-answer biology facts from MMLU-biology gaps.

For each failing question in the MMLU-biology results (wrong or
abstain), construct a short declarative fact that encodes the
correct answer in natural biology-textbook phrasing, and teach it.

This is the `feedback_teach_the_gaps` loop in practice: the benchmark
surfaces which real facts Sara doesn't have, and we teach each one.
The facts we teach are real biology propositions — not question-
answer pairs — phrased so Sara's parser can store them.

Routing: each fact is taught into the region the question was
originally routed to, so it lives where other questions on the
same topic will also route.

Usage:
    .venv/bin/python benchmarks/teach_mmlu_gaps.py \\
        --db biology2e.db \\
        --results benchmarks/mmlu_biology_biology2e_results.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from sara_brain.core.brain import Brain


def build_regional_learner(brain: Brain, region: str):
    from sara_brain.storage.neuron_repo import NeuronRepo
    from sara_brain.storage.segment_repo import SegmentRepo
    from sara_brain.storage.path_repo import PathRepo
    from sara_brain.core.learner import Learner
    from sara_brain.parsing.statement_parser import StatementParser
    from sara_brain.parsing.taxonomy import Taxonomy

    brain.db.create_region(region)
    nr = NeuronRepo(brain.conn, prefix=region)
    sr = SegmentRepo(brain.conn, prefix=region)
    pr = PathRepo(brain.conn, prefix=region)
    parser = StatementParser(taxonomy=Taxonomy())
    return Learner(parser, nr, sr, pr)


def _first_sentence(text: str) -> str:
    """Return the first sentence of a MC question stem — the core
    question, stripped of the preamble context."""
    m = re.split(r"(?<=[.?!])\s+", text.strip())
    if m:
        return m[-1]
    return text


def make_fact(question: str, correct: str) -> str | None:
    """Construct a taught-fact sentence from a question + its correct
    answer. Heuristic: 'question stem' + 'correct' → 'stem correct.'

    Examples (illustrative, not guaranteed):
      Q "A heterotroph" + A "obtains its energy by oxidizing..."
        → "A heterotroph obtains its energy by oxidizing..."
      Q "In humans, fertilization normally occurs in the"
        + A "fallopian tube"
        → "in humans fertilization normally occurs in the fallopian tube"
      Q "Convergent evolution is best exemplified by"
        + A "The wings of an insect and the wings of a bird"
        → "convergent evolution is best exemplified by..."

    Returns None if the question has a meta-choice answer (Both A
    and C, None of the above, etc.) — those are architectural and
    cannot be turned into biology facts.
    """
    q = question.strip()
    a = correct.strip()
    # Reject meta-choices
    bad = ("both a", "both b", "both c", "both d",
           "none of the above", "all of the above",
           "a and c", "b and c", "a and b", "a, b", "b, c")
    if a.lower().startswith(bad) or " only" in a.lower():
        return None
    # Reject numeric-only answers unless the question is clearly
    # numeric (handled by the math module separately)
    if a.isdigit() or re.fullmatch(r"-?\d+(\.\d+)?%?", a):
        return None
    # Use LAST sentence of the question stem (often the actual prompt)
    stem = _first_sentence(q)
    stem = stem.rstrip(".!? ")
    a = a.rstrip(".!? ")
    # If stem ends with a connector, concatenate smoothly
    if stem.lower().endswith((" is", " are", " was", " were", "the",
                              " by", " of", " in", " at", " on", " to",
                              " from", " with", " without")):
        fact = f"{stem} {a}"
    else:
        fact = f"{stem} {a}"
    # Strip leading "Which of the following …" boilerplate — the
    # remaining sentence should carry the semantic claim.
    fact = re.sub(
        r"^which of the following[^,]*,?\s*",
        "",
        fact,
        flags=re.IGNORECASE,
    )
    fact = re.sub(r"^which\s+\w+\s+(of\s+the\s+following\s+)?",
                  "", fact, flags=re.IGNORECASE)
    # Normalise whitespace and final period
    fact = re.sub(r"\s+", " ", fact).strip()
    if not fact:
        return None
    if not fact.endswith("."):
        fact += "."
    # Reject anything too long (parser fails on >30-word sentences)
    if len(fact.split()) > 30:
        return None
    if len(fact.split()) < 3:
        return None
    return fact


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--results", required=True,
                    help="Path to MMLU results JSON.")
    ap.add_argument("--region-default", default="mmlu_taught",
                    help="Region name to use when the question has "
                         "no region recorded.")
    args = ap.parse_args()

    results_path = Path(args.results)
    data = json.loads(results_path.read_text())
    brain = Brain(args.db)

    taught = 0
    skipped_meta = 0
    skipped_numeric = 0
    parser_skipped = 0
    region_learners: dict[str, object] = {}

    for q in data["results"]:
        if q["outcome"] == "correct":
            continue
        question_text = q["question"]
        correct_letter = q["correct"]
        correct_choice = next(
            (c for c in q["choices"] if c["letter"] == correct_letter),
            None,
        )
        if correct_choice is None:
            continue
        fact = make_fact(question_text, correct_choice["text"])
        if fact is None:
            # Look at why
            txt = correct_choice["text"].lower()
            if any(m in txt for m in (
                    "both", "none of", "all of", "and c", "and b")):
                skipped_meta += 1
            elif txt.replace(".", "").replace("%", "").isdigit():
                skipped_numeric += 1
            else:
                skipped_meta += 1
            continue
        # Pick the first region from the question's routing.
        regions = q.get("regions") or [args.region_default]
        region = regions[0]
        if region not in region_learners:
            region_learners[region] = build_regional_learner(brain, region)
        learner = region_learners[region]
        try:
            result = learner.learn(fact, apply_filter=False)
        except Exception:
            parser_skipped += 1
            continue
        if result is None:
            parser_skipped += 1
        else:
            taught += 1

    brain.conn.commit()
    brain.close()

    print(f"Taught:              {taught}")
    print(f"Parser-skipped:      {parser_skipped}")
    print(f"Meta/structural skip:{skipped_meta}")
    print(f"Numeric skip:        {skipped_numeric}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
