"""Generate training data for the Sara extractor v2.

Fixes from v1:
1. Multi-word compound concepts (2-3 words, like real domain terms)
2. Structured output format with <triple>/<rel>/<obj> delimiters
3. Complex grammar: relative clauses, passive voice, nested relationships, lists

Usage:
    python scripts/generate_extractor_v2_data.py \
        --num-examples 500000 \
        --out training_data/extractor_v2_500k.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

CONSONANTS = "bcdfghjklmnprstvwz"
VOWELS = "aeiou"


def _syl(rng):
    return rng.choice(CONSONANTS) + rng.choice(VOWELS) + rng.choice(CONSONANTS + VOWELS)


def word(rng, syls=0):
    return "".join(_syl(rng) for _ in range(syls or rng.randint(2, 3)))


def concept(rng):
    """Generate a 1-3 word compound concept (like 'molecular snare mechanism')."""
    n = rng.choices([1, 2, 3], weights=[0.3, 0.5, 0.2])[0]
    return " ".join(word(rng) for _ in range(n))


RELATIONS = [
    "is_a", "contains", "produces", "requires", "involves",
    "causes", "prevents", "occurs_in", "part_of", "enables",
    "interacts_with", "transforms_into", "regulates", "provides",
    "separates", "attaches_to", "depends_on", "activates",
    "inhibits", "composed_of", "results_in", "leads_to",
]

# Complex sentence templates with {s} = subject, {o} = object, {r} = relation verb phrase
# Each template maps to a specific relation for the triple
TEMPLATES = [
    # Simple declarative
    ("is_a", "{s} is a type of {o}.", "{s} is a {o}."),
    ("is_a", "{s} is classified as {o}.", "{s} is a specialized form of {o}."),
    ("contains", "{s} contains {o}.", "{s} is composed of multiple {o}."),
    ("contains", "Within {s} there are several {o} arranged in layers.", "The interior of {s} holds {o}."),
    ("produces", "{s} produces {o}.", "{s} generates {o} as its primary output."),
    ("produces", "The output of {s} is {o}.", "{s} yields {o} under normal conditions."),
    ("requires", "{s} requires {o} to function.", "Without {o} the {s} cannot operate."),
    ("requires", "{s} depends on {o} for stability.", "{s} needs {o} to maintain its structure."),
    ("causes", "{s} causes {o}.", "{s} triggers {o} when activated."),
    ("causes", "The presence of {s} leads to {o}.", "Exposure to {s} results in {o}."),
    ("prevents", "{s} prevents {o}.", "{s} blocks {o} from occurring."),
    ("prevents", "{s} inhibits {o} under normal conditions.", "The role of {s} is to suppress {o}."),
    ("occurs_in", "{s} occurs in {o}.", "{s} takes place within {o}."),
    ("occurs_in", "{s} is found exclusively in {o}.", "{s} happens inside {o} during the active phase."),
    ("part_of", "{s} is part of {o}.", "{s} is a component of the larger {o}."),
    ("enables", "{s} enables {o}.", "{s} makes {o} possible."),
    ("enables", "Without {s} there would be no {o}.", "{s} is essential for {o} to proceed."),
    ("interacts_with", "{s} interacts with {o}.", "{s} binds to {o} with high affinity."),
    ("interacts_with", "{s} connects to {o} through a series of bonds.", "The binding of {s} to {o} is reversible."),
    ("regulates", "{s} regulates {o}.", "{s} controls the rate of {o}."),
    ("regulates", "{s} modulates {o} in response to signals.", "The level of {s} determines the activity of {o}."),
    ("provides", "{s} provides {o}.", "{s} supplies {o} to the surrounding structure."),
    ("transforms_into", "{s} transforms into {o}.", "Over time {s} converts to {o}."),
    ("activates", "{s} activates {o}.", "{s} switches on {o} when conditions are met."),
    # Relative clauses
    ("contains", "The {s}, which is found in the outer layer, contains {o}.", None),
    ("produces", "The {s}, which requires energy input, produces {o} as a byproduct.", None),
    ("requires", "The {s}, which was recently discovered, requires {o} for proper assembly.", None),
    ("regulates", "The {s}, which operates continuously, regulates {o} throughout the cycle.", None),
    # Passive voice
    ("produces", "{o} is produced by {s}.", "{o} is generated through the action of {s}."),
    ("requires", "{o} is required by {s} for normal function.", None),
    ("causes", "{o} is caused by the accumulation of {s}.", None),
    ("regulates", "{o} is regulated by {s} under standard conditions.", None),
    ("enables", "{o} is enabled by the presence of {s}.", None),
    # Nested/complex
    ("contains", "The outer layer of {s} contains {o} embedded in a matrix.", None),
    ("requires", "The formation of {s} requires the presence of {o} in sufficient quantity.", None),
    ("produces", "During the active phase {s} produces large amounts of {o}.", None),
    ("causes", "The breakdown of {s} causes rapid accumulation of {o}.", None),
    ("interacts_with", "At the interface between layers {s} interacts with {o}.", None),
    # Lists (generates multiple triples)
    ("contains", "{s} contains {o} along with other components.", None),
    ("requires", "{s} requires both {o} and adequate energy to function.", None),
]


def generate_example(rng: random.Random) -> dict:
    """Generate one training example with compound concepts and structured output."""
    # Create a small domain of interconnected concepts
    n_concepts = rng.randint(4, 8)
    concepts = [concept(rng) for _ in range(n_concepts)]

    # Generate 3-6 triples
    n_triples = rng.randint(3, 6)
    triples = []
    sentences = []

    for _ in range(n_triples):
        s_idx, o_idx = rng.sample(range(n_concepts), 2)
        s, o = concepts[s_idx], concepts[o_idx]

        # Pick a template
        template_entry = rng.choice(TEMPLATES)
        rel = template_entry[0]
        template_options = [t for t in template_entry[1:] if t is not None]
        template = rng.choice(template_options)

        sentence = template.format(s=s, o=o)
        sentences.append(sentence)
        triples.append({"s": s, "r": rel, "o": o})

    # Build paragraph with connectives
    connectives = ["", "", "", "Furthermore, ", "In addition, ", "As a result, ",
                   "This means that ", "Moreover, ", "Consequently, "]
    paragraph_parts = []
    for i, sent in enumerate(sentences):
        if i > 0 and rng.random() < 0.4:
            sent = rng.choice(connectives) + sent[0].lower() + sent[1:]
        paragraph_parts.append(sent)
    paragraph = " ".join(paragraph_parts)

    # Build structured output
    output_lines = []
    for t in triples:
        output_lines.append(f"<triple> {t['s']} <rel> {t['r']} <obj> {t['o']} </triple>")
    output = "\n".join(output_lines)

    return {"paragraph": paragraph, "triples_structured": output, "triples_list": triples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-examples", type=int, default=500000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=424242)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.out, "w")
    t0 = time.time()

    for i in range(args.num_examples):
        ex = generate_example(rng)
        out_f.write(json.dumps({
            "paragraph": ex["paragraph"],
            "output": ex["triples_structured"],
        }) + "\n")
        if (i + 1) % 50000 == 0:
            print(f"[{i+1}/{args.num_examples}] ({time.time()-t0:.0f}s)", file=sys.stderr)

    out_f.close()
    print(f"\nDone. {args.num_examples} examples in {time.time()-t0:.0f}s.", file=sys.stderr)


if __name__ == "__main__":
    main()
