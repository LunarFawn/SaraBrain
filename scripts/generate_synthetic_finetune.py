"""Generate fine-tuning data from SYNTHETIC substrates.

No real knowledge. Concepts are nonsense words. The model learns
substrate reasoning — how to read wavefront output and answer
questions grounded in it — without memorizing any real-world facts.

For each synthetic substrate, generates questions like:
  "What does X involve?" → answer is the object from a known triple
  "Which of the following is related to X?" → MCQ from manifest

Usage:
    python scripts/generate_synthetic_finetune.py \
        --num-substrates 50 \
        --questions-per-substrate 8 \
        --out training_data/sara_cortex_synthetic.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
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
from sara_brain.core.query_resolver import resolve_query_nospacy


SYSTEM_INSTRUCTION = (
    "You are a substrate-grounded reasoning system. You receive a "
    "structured knowledge neighborhood from a wavefront query and a "
    "multiple-choice question. Answer using ONLY facts present in the "
    "substrate. If the substrate does not contain enough information "
    "to answer, say so. Never use knowledge from outside the substrate."
)

# Question templates. {subject}, {relation}, {object} filled from triples.
_Q_TEMPLATES = [
    ("What does {subject} {relation}?", "object"),
    ("What {relation} {object}?", "subject"),
    ("Which of the following does {subject} {relation}?", "object"),
    ("Which of the following {relation} {object}?", "subject"),
]


def generate_substrate(concepts: int, triples: int, seed: int, db_path: str):
    """Generate a synthetic substrate and return (brain, manifest)."""
    import subprocess
    result = subprocess.run(
        [
            sys.executable,
            "papers/instrument_validation/generate_synthetic_substrate.py",
            "--out", db_path,
            "--concepts", str(concepts),
            "--triples", str(triples),
            "--seed", str(seed),
            "--compound-fraction", "0.5",
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Substrate generation failed: {result.stderr}")

    manifest_path = db_path + ".manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    brain = Brain(db_path)
    return brain, manifest


def make_mcq(manifest: dict, triple_idx: int, rng: random.Random) -> dict | None:
    """Create an MCQ from a triple in the manifest.

    Returns {question, choices, answer_idx} or None if can't build.
    """
    triples = manifest["triples"]
    concepts = manifest["concepts"]

    if triple_idx >= len(triples):
        return None

    subj, rel, obj = triples[triple_idx]
    template, answer_field = rng.choice(_Q_TEMPLATES)

    if answer_field == "object":
        correct = obj
        question = template.format(subject=subj, relation=rel)
    else:
        correct = subj
        question = template.format(object=obj, relation=rel)

    # Build distractors from other concepts
    distractors = [c for c in concepts if c != correct]
    if len(distractors) < 3:
        return None
    distractors = rng.sample(distractors, 3)

    choices = distractors + [correct]
    rng.shuffle(choices)
    answer_idx = choices.index(correct)

    return {
        "question": question,
        "choices": choices,
        "answer_idx": answer_idx,
    }


def run_wavefront(brain: Brain, question: str, depth: int = 2) -> str:
    """Run wavefront and return formatted substrate."""
    q_seeds = resolve_query_nospacy(question, brain.neuron_repo)
    if not q_seeds:
        # Fallback to random extraction if resolve fails
        candidates = _extract_seed_concepts(question)
        seeds = candidates[:4]
    else:
        seeds = q_seeds

    original_depth = brain.recognizer.max_depth
    try:
        brain.recognizer.max_depth = depth
        with brain.short_term(event_type="synth_finetune") as st:
            seed_labels = [s.label if hasattr(s, "label") else str(s) for s in seeds]
            if hasattr(brain.recognizer, "propagate_backwave"):
                brain.recognizer.propagate_backwave(seed_labels, st, exact_only=True)
            else:
                brain.propagate_into(seeds, st, exact_only=True)
                
            convergence_map = dict(st.convergence_map)
            intersections = st.intersections(min_sources=2)
    finally:
        brain.recognizer.max_depth = original_depth

    return _format_wavefront_substrate(brain, seeds, convergence_map, intersections)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--num-substrates", type=int, default=50,
                    help="Number of synthetic substrates to generate (default: 50)")
    ap.add_argument("--questions-per-substrate", type=int, default=8,
                    help="Questions per substrate (default: 8)")
    ap.add_argument("--concepts", type=int, default=40,
                    help="Concepts per substrate (default: 40)")
    ap.add_argument("--triples", type=int, default=120,
                    help="Triples per substrate (default: 120)")
    ap.add_argument("--out", required=True, help="Output .jsonl path")
    ap.add_argument("--seed", type=int, default=2026, help="Base RNG seed")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.out, "w")

    generated = 0
    skipped = 0
    t0 = time.time()

    for sub_i in range(args.num_substrates):
        sub_seed = args.seed + sub_i
        db_path = f"/tmp/synth_finetune_{sub_i}.db"

        try:
            brain, manifest = generate_substrate(
                args.concepts, args.triples, sub_seed, db_path,
            )
        except Exception as e:
            print(f"[{sub_i}] substrate generation failed: {e}", file=sys.stderr)
            skipped += args.questions_per_substrate
            continue

        # Pick random triples to build questions from
        triple_indices = rng.sample(
            range(len(manifest["triples"])),
            min(args.questions_per_substrate, len(manifest["triples"])),
        )

        for tri_idx in triple_indices:
            mcq = make_mcq(manifest, tri_idx, rng)
            if mcq is None:
                skipped += 1
                continue

            # Run wavefront with the question
            question_text = mcq["question"]
            substrate = run_wavefront(brain, question_text)

            # Format
            choices_str = "\n".join(
                f"  {chr(ord('A') + i)}. {c}"
                for i, c in enumerate(mcq["choices"])
            )
            full_question = f"{mcq['question']}\n{choices_str}"
            correct_letter = chr(ord("A") + mcq["answer_idx"])

            example = {
                "system": SYSTEM_INSTRUCTION,
                "substrate": substrate,
                "question": full_question,
                "answer": correct_letter,
            }
            out_f.write(json.dumps(example) + "\n")
            generated += 1

        brain.close()

        if (sub_i + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(
                f"[{sub_i+1}/{args.num_substrates}] generated={generated} "
                f"skipped={skipped} ({elapsed:.0f}s)",
                file=sys.stderr,
            )

    out_f.close()
    elapsed = time.time() - t0
    print(
        f"\nDone in {elapsed:.1f}s. Generated {generated} examples, "
        f"skipped {skipped}. Output: {args.out}",
        file=sys.stderr,
    )

    # Cleanup temp dbs
    import glob
    for f in glob.glob("/tmp/synth_finetune_*.db*"):
        Path(f).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
