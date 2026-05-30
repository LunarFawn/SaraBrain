"""Generate text-generation training data for the Sara cortex.

The real cortex task: given substrate facts from a wavefront, render
them as a natural language answer. Not MCQ classification — prose.

Each example: (question, substrate facts, prose answer derived from facts)

Usage:
    python scripts/generate_textgen_finetune.py \
        --num-substrates 500 --questions-per-substrate 4 \
        --out training_data/sara_cortex_textgen.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sara_brain.core.brain import Brain
from sara_reader.stateless_reader import (
    _filter_seeds_by_substrate,
    _format_wavefront_substrate,
)

# Question templates and answer templates
_TEMPLATES = [
    {
        "q": "What does {subject} {relation}?",
        "a": "{subject} {relation} {object}.",
    },
    {
        "q": "What {relation} {object}?",
        "a": "{subject} {relation} {object}.",
    },
    {
        "q": "Describe the relationship between {subject} and {object}.",
        "a": "{subject} {relation} {object}.",
    },
    {
        "q": "What is known about {subject}?",
        "a": "Based on the substrate: {facts_prose}",
    },
]


def get_facts_for_concept(manifest: dict, concept: str) -> list[tuple]:
    """Get all triples involving a concept."""
    return [(s, r, o) for s, r, o in manifest["triples"]
            if s == concept or o == concept]


def render_facts_prose(facts: list[tuple]) -> str:
    """Render a list of triples as prose."""
    if not facts:
        return "No information available in the substrate."
    sentences = []
    for s, r, o in facts[:5]:  # cap at 5 facts per answer
        sentences.append(f"{s} {r} {o}")
    return ". ".join(sentences) + "."


def run_wavefront(brain, seeds, depth=2):
    """Run wavefront and return formatted output."""
    original_depth = brain.recognizer.max_depth
    try:
        brain.recognizer.max_depth = depth
        with brain.short_term(event_type="textgen") as st:
            brain.propagate_into(seeds, st, exact_only=True)
            convergence_map = dict(st.convergence_map)
            intersections = st.intersections(min_sources=2)
    finally:
        brain.recognizer.max_depth = original_depth
    return _format_wavefront_substrate(brain, seeds, convergence_map, intersections)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-substrates", type=int, default=500)
    ap.add_argument("--questions-per-substrate", type=int, default=4)
    ap.add_argument("--concepts", type=int, default=40)
    ap.add_argument("--triples", type=int, default=120)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=55555)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.out, "w")
    generated = 0
    t0 = time.time()

    for i in range(args.num_substrates):
        db_path = f"/tmp/textgen_synth_{i}.db"
        sub_seed = args.seed + i

        result = subprocess.run(
            [sys.executable, "papers/instrument_validation/generate_synthetic_substrate.py",
             "--out", db_path, "--concepts", str(args.concepts),
             "--triples", str(args.triples), "--seed", str(sub_seed),
             "--compound-fraction", "0.5"],
            capture_output=True, text=True)
        if result.returncode != 0:
            continue

        manifest_path = db_path + ".manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        brain = Brain(db_path)
        triples = manifest["triples"]
        concepts = manifest["concepts"]

        # Generate questions from random triples
        sample_indices = rng.sample(
            range(len(triples)),
            min(args.questions_per_substrate, len(triples)))

        for tri_idx in sample_indices:
            subj, rel, obj = triples[tri_idx]

            # Pick a template
            template = rng.choice(_TEMPLATES[:3])  # first 3 are triple-based

            question = template["q"].format(subject=subj, relation=rel, object=obj)
            answer = template["a"].format(subject=subj, relation=rel, object=obj)

            # Run wavefront from the subject
            substrate = run_wavefront(brain, [subj])

            example = {
                "question": question,
                "substrate": substrate,
                "answer": answer,
            }
            out_f.write(json.dumps(example) + "\n")
            generated += 1

        # Also generate "what is known about X" questions
        if concepts:
            concept = rng.choice(concepts)
            facts = get_facts_for_concept(manifest, concept)
            if facts:
                question = f"What is known about {concept}?"
                substrate = run_wavefront(brain, [concept])
                answer = f"Based on the substrate: {render_facts_prose(facts)}"
                example = {
                    "question": question,
                    "substrate": substrate,
                    "answer": answer,
                }
                out_f.write(json.dumps(example) + "\n")
                generated += 1

        brain.close()
        for f_path in Path("/tmp").glob(f"textgen_synth_{i}.db*"):
            f_path.unlink(missing_ok=True)

        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{args.num_substrates}] generated={generated} ({time.time()-t0:.0f}s)",
                  file=sys.stderr)

    out_f.close()
    print(f"\nDone. {generated} examples. Output: {args.out} ({time.time()-t0:.0f}s)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
