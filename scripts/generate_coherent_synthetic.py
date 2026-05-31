"""Generate coherent synthetic paragraphs + extraction targets.

Creates fake "textbook sections" about fake domains where concepts
build on each other. Not gibberish — structurally complex, internally
consistent, domain-agnostic.

Each example: a coherent paragraph about interconnected nonsense
concepts, paired with the triples embedded in it.

Target: 100k+ examples for training a domain-agnostic extractor.

Usage:
    python scripts/generate_coherent_synthetic.py \
        --num-examples 100000 \
        --out training_data/coherent_synthetic_100k.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path


# Pronounceable nonsense word generator
CONSONANTS = "bcdfghjklmnpqrstvwxyz"
VOWELS = "aeiou"


def _syllable(rng: random.Random) -> str:
    return rng.choice(CONSONANTS) + rng.choice(VOWELS) + rng.choice(CONSONANTS + "")


def nonsense_word(rng: random.Random, syllables: int = 0) -> str:
    if syllables == 0:
        syllables = rng.randint(2, 4)
    return "".join(_syllable(rng) for _ in range(syllables))


def compound_word(rng: random.Random) -> str:
    if rng.random() < 0.4:
        return nonsense_word(rng) + " " + nonsense_word(rng)
    return nonsense_word(rng)


# Relation types with sentence templates
RELATIONS = [
    # (relation, templates for embedding in prose)
    ("is_a", [
        "{subj} is a type of {obj}",
        "{subj} is a {obj}",
        "{subj} is classified as a {obj}",
    ]),
    ("contains", [
        "{subj} contains {obj}",
        "{subj} is composed of {obj}",
        "within {subj} there are multiple {obj}",
    ]),
    ("produces", [
        "{subj} produces {obj}",
        "{subj} generates {obj}",
        "the output of {subj} is {obj}",
    ]),
    ("requires", [
        "{subj} requires {obj}",
        "{subj} depends on {obj}",
        "without {obj} the {subj} cannot function",
    ]),
    ("causes", [
        "{subj} causes {obj}",
        "{subj} leads to {obj}",
        "when {subj} occurs it triggers {obj}",
    ]),
    ("occurs_in", [
        "{subj} occurs in {obj}",
        "{subj} takes place within {obj}",
        "{subj} is found in {obj}",
    ]),
    ("part_of", [
        "{subj} is part of {obj}",
        "{subj} is a component of {obj}",
        "{subj} belongs to {obj}",
    ]),
    ("prevents", [
        "{subj} prevents {obj}",
        "{subj} inhibits {obj}",
        "{subj} blocks {obj}",
    ]),
    ("enables", [
        "{subj} enables {obj}",
        "{subj} facilitates {obj}",
        "{subj} is necessary for {obj}",
    ]),
    ("interacts_with", [
        "{subj} interacts with {obj}",
        "{subj} binds to {obj}",
        "{subj} connects to {obj}",
    ]),
    ("transforms_into", [
        "{subj} transforms into {obj}",
        "{subj} becomes {obj}",
        "over time {subj} converts to {obj}",
    ]),
    ("regulates", [
        "{subj} regulates {obj}",
        "{subj} controls {obj}",
        "{subj} modulates the activity of {obj}",
    ]),
]

# Connective phrases to make paragraphs flow
CONNECTIVES = [
    "Furthermore, ", "In addition, ", "This means that ",
    "As a result, ", "Consequently, ", "Moreover, ",
    "It is important to note that ", "Research shows that ",
    "This process ensures that ", "During this phase, ",
    "The significance of this is that ", "Specifically, ",
    "", "", "", "",  # empty = no connective (most common)
]

# Elaboration templates (add context around a fact)
ELABORATIONS = [
    "This {rel_word} is essential for maintaining stability.",
    "This relationship was first described in early studies.",
    "The mechanism behind this involves multiple steps.",
    "This occurs under specific conditions.",
    "This is a fundamental property of the system.",
    "Without this relationship the system would collapse.",
    "",  # no elaboration
    "",
    "",
]


def generate_domain(rng: random.Random, n_concepts: int = 8) -> dict:
    """Generate a fake domain with interconnected concepts."""
    concepts = [compound_word(rng) for _ in range(n_concepts)]
    # Create a connected graph of relationships
    triples = []
    used_pairs = set()

    # Ensure connectivity: chain the first few concepts
    for i in range(min(n_concepts - 1, 5)):
        rel, _ = rng.choice(RELATIONS)
        triples.append((concepts[i], rel, concepts[i + 1]))
        used_pairs.add((i, i + 1))

    # Add more random connections
    extra = rng.randint(3, n_concepts)
    for _ in range(extra):
        a, b = rng.sample(range(n_concepts), 2)
        if (a, b) not in used_pairs:
            rel, _ = rng.choice(RELATIONS)
            triples.append((concepts[a], rel, concepts[b]))
            used_pairs.add((a, b))

    return {"concepts": concepts, "triples": triples}


def render_paragraph(domain: dict, rng: random.Random, n_facts: int = 0) -> tuple[str, list[str]]:
    """Render a domain into a coherent paragraph with embedded triples."""
    triples = domain["triples"]
    if n_facts:
        triples = rng.sample(triples, min(n_facts, len(triples)))
    else:
        triples = triples[:rng.randint(3, min(7, len(triples)))]

    sentences = []
    extracted_triples = []

    for subj, rel, obj in triples:
        # Find templates for this relation
        templates = None
        for r, t in RELATIONS:
            if r == rel:
                templates = t
                break
        if not templates:
            templates = ["{subj} " + rel + " {obj}"]

        # Build sentence
        connective = rng.choice(CONNECTIVES)
        template = rng.choice(templates)
        sentence = connective + template.format(subj=subj, obj=obj) + "."

        # Maybe add elaboration
        if rng.random() < 0.3:
            elab = rng.choice(ELABORATIONS)
            if elab:
                sentence += " " + elab

        sentences.append(sentence)
        extracted_triples.append(f"{subj} {rel} {obj}")

    paragraph = " ".join(sentences)
    return paragraph, extracted_triples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-examples", type=int, default=100000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=314159)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.out, "w")

    t0 = time.time()
    generated = 0

    # Generate many domains, multiple paragraphs per domain
    domains_needed = args.num_examples // 5  # ~5 paragraphs per domain

    for i in range(domains_needed):
        n_concepts = rng.randint(6, 12)
        domain = generate_domain(rng, n_concepts)

        # Generate multiple paragraphs from this domain (different subsets of facts)
        n_paras = rng.randint(3, 7)
        for _ in range(n_paras):
            if generated >= args.num_examples:
                break
            n_facts = rng.randint(3, 6)
            paragraph, triples = render_paragraph(domain, rng, n_facts)

            if triples and len(paragraph) > 50:
                out_f.write(json.dumps({
                    "paragraph": paragraph,
                    "triples": triples,
                }) + "\n")
                generated += 1

        if generated >= args.num_examples:
            break

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            print(f"[{generated}/{args.num_examples}] ({elapsed:.0f}s)", file=sys.stderr)

    out_f.close()
    elapsed = time.time() - t0
    print(f"\nDone. {generated} examples in {elapsed:.0f}s. Output: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
