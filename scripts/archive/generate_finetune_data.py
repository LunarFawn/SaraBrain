"""Generate fine-tuning data for a Sara-native cortex model.

For each question, runs the wavefront against the brain and produces
a training example:

  {
    "instruction": "Answer using ONLY the substrate facts below.",
    "substrate": "<wavefront output — the knowledge neighborhood>",
    "question": "<the question with choices>",
    "answer": "<correct answer letter>"
  }

The model learns to: read structured substrate → select correct answer.
No encyclopedic knowledge needed — only language competence + substrate reasoning.

Usage:
    python -m scripts.generate_finetune_data \
        --brain /path/to/sara_bio.db \
        --questions benchmarks/ch10_test_questions.json \
        --out training_data/sara_cortex_v1.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sara_brain.core.brain import Brain
from sara_reader.stateless_reader import (
    _extract_seed_concepts,
    _filter_seeds_by_substrate,
    _format_wavefront_substrate,
)


SYSTEM_INSTRUCTION = (
    "You are a substrate-grounded reasoning system. You receive a "
    "structured knowledge neighborhood from a wavefront query and a "
    "multiple-choice question. Answer using ONLY facts present in the "
    "substrate. If the substrate does not contain enough information "
    "to answer, say so. Never use knowledge from outside the substrate."
)

# Neurons whose labels are stopwords/pronouns — noise from book ingestion.
# Filter these from the wavefront output in training data.
_NOISE_LABELS: frozenset[str] = frozenset({
    "it", "they", "this", "that", "the", "which", "these", "those",
    "we", "he", "she", "its", "their", "our", "his", "her",
    "therefore", "however", "also", "can", "may", "will",
    "some", "each", "many", "most", "all", "both", "other",
    "such", "more", "less", "very", "much", "well",
})


def _is_noise_label(label: str) -> bool:
    """Check if a neuron label is noise (pronoun/stopword)."""
    base = label.removesuffix("_attribute")
    return base in _NOISE_LABELS


def run_wavefront_for_question(brain: Brain, question: str, depth: int = 2) -> str:
    """Run wavefront and return the formatted substrate string (noise-filtered)."""
    candidates = _extract_seed_concepts(question)
    seeds = _filter_seeds_by_substrate(brain, candidates)
    if not seeds:
        seeds = candidates[:4]  # fallback

    original_depth = brain.recognizer.max_depth
    try:
        brain.recognizer.max_depth = depth
        with brain.short_term(event_type="finetune_gen") as st:
            brain.propagate_into(seeds, st, exact_only=True)
            convergence_map = dict(st.convergence_map)
            intersections = st.intersections(min_sources=2)
    finally:
        brain.recognizer.max_depth = original_depth

    # Filter noise from convergence map and intersections
    clean_convergence = {
        nid: w for nid, w in convergence_map.items()
        if not _is_noise_label(
            (brain.neuron_repo.get_by_id(nid) or type("", (), {"label": ""})()).label
        )
    }
    clean_intersections = []
    for item in intersections:
        n = brain.neuron_repo.get_by_id(item[0])
        if n and not _is_noise_label(n.label):
            clean_intersections.append(item)

    return _format_wavefront_substrate(brain, seeds, clean_convergence, clean_intersections)


def format_question_with_choices(q: dict) -> str:
    """Format question + choices as a string."""
    lines = [q["question"]]
    for i, choice in enumerate(q["choices"]):
        letter = chr(ord("A") + i)
        lines.append(f"  {letter}. {choice}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--brain", required=True, help="Path to Sara brain .db")
    ap.add_argument("--questions", required=True, help="JSON file with questions")
    ap.add_argument("--out", required=True, help="Output .jsonl path")
    ap.add_argument("--depth", type=int, default=2, help="Wavefront depth (default: 2)")
    ap.add_argument("--limit", type=int, default=0, help="Max questions (0=all)")
    args = ap.parse_args()

    brain = Brain(args.brain)
    with open(args.questions) as f:
        questions = json.load(f)

    if isinstance(questions, dict) and "results" in questions:
        # MMLU wavefront format — extract the question data
        print("Detected MMLU wavefront results format — need raw questions file.",
              file=sys.stderr)
        sys.exit(1)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.out, "w")

    total = len(questions)
    if args.limit:
        total = min(total, args.limit)

    generated = 0
    skipped = 0
    t0 = time.time()

    for i, q in enumerate(questions[:total]):
        question_text = q["question"]
        correct_idx = q["answer_idx"]
        correct_letter = chr(ord("A") + correct_idx)

        # Run wavefront
        substrate = run_wavefront_for_question(brain, question_text, args.depth)

        # Skip if wavefront returned nothing useful
        if "0 intersection(s), 0 neuron(s) reached" in substrate:
            skipped += 1
            continue

        example = {
            "system": SYSTEM_INSTRUCTION,
            "substrate": substrate,
            "question": format_question_with_choices(q),
            "answer": correct_letter,
        }

        out_f.write(json.dumps(example) + "\n")
        generated += 1

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(
                f"[{i+1}/{total}] generated={generated} skipped={skipped} "
                f"({rate:.1f} q/s)",
                file=sys.stderr,
            )

    out_f.close()
    brain.close()

    elapsed = time.time() - t0
    print(
        f"\nDone in {elapsed:.1f}s. Generated {generated} examples, "
        f"skipped {skipped}. Output: {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
