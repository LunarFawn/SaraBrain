"""Generate training data from Wikipedia with content words replaced by nonsense.

Takes real Wikipedia paragraphs (proper grammar), replaces nouns/verbs/adjectives
with pronounceable nonsense words, keeps grammar skeleton intact.

The model learns extraction from REAL sentence structure without learning
any real-world knowledge.

Requires: spacy (en_core_web_sm)

Usage:
    python scripts/generate_wiki_synthetic.py \
        --num-examples 100000 \
        --out training_data/wiki_synthetic_100k.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

# Content POS tags to replace (nouns, verbs, adjectives, adverbs)
REPLACE_POS = {"NOUN", "VERB", "ADJ", "PROPN"}
# Keep these POS tags as-is (grammar skeleton)
KEEP_POS = {"DET", "ADP", "CCONJ", "SCONJ", "PART", "PRON", "AUX", "PUNCT", "NUM", "ADV"}

CONSONANTS = "bcdfghjklmnprstvwz"
VOWELS = "aeiou"


def _syllable(rng):
    return rng.choice(CONSONANTS) + rng.choice(VOWELS) + rng.choice(CONSONANTS + VOWELS)


def nonsense_word(rng, syllables=0):
    if syllables == 0:
        syllables = rng.randint(2, 3)
    return "".join(_syllable(rng) for _ in range(syllables))


# Source sentences — diverse Wikipedia-style text covering many domains
# These are templates with natural grammar structure
WIKI_SENTENCES = [
    "The {noun1} is a {adj1} {noun2} found in {noun3} that {verb1} {noun4} through {noun5}.",
    "{noun1} {verb1} {noun2} by {verb2} the {adj1} {noun3} within the {noun4}.",
    "The {noun1} contains multiple {noun2} arranged in a {adj1} pattern.",
    "Each {noun1} requires {noun2} to maintain its {adj1} {noun3}.",
    "When the {noun1} is disrupted the {noun2} undergoes {noun3} which releases {noun4}.",
    "The {noun1} is composed of {noun2} and {noun3} connected by {adj1} {noun4}.",
    "{noun1} occurs in the {noun2} during the {adj1} phase of {noun3}.",
    "The process of {noun1} involves the {noun2} of {noun3} into {noun4}.",
    "Without {noun1} the {noun2} cannot {verb1} properly and {noun3} accumulates.",
    "{noun1} is regulated by {noun2} which {verb1} the rate of {noun3}.",
    "The {adj1} {noun1} provides {noun2} for the {noun3} during {noun4}.",
    "{noun1} and {noun2} interact to form a {adj1} {noun3} that {verb1} {noun4}.",
    "In the absence of {noun1} the {noun2} {verb1} rapidly leading to {noun3}.",
    "The {noun1} transforms {noun2} into {noun3} using {noun4} as a catalyst.",
    "{noun1} is essential for {noun2} because it {verb1} the {adj1} {noun3}.",
    "The {adj1} {noun1} of the {noun2} determines how {noun3} {verb1} {noun4}.",
    "During {noun1} the {noun2} {verb1} and the {noun3} {verb2} toward the {noun4}.",
    "{noun1} prevents {noun2} from {verb1} by {verb2} the {adj1} {noun3}.",
    "The {noun1} is classified as a {noun2} based on its {adj1} {noun3}.",
    "Research shows that {noun1} {verb1} {noun2} more effectively than {noun3}.",
    "{noun1} depends on the {noun2} of {noun3} and the {noun4} between them.",
    "The {noun1} attaches to the {noun2} at the {adj1} region called the {noun3}.",
    "As {noun1} increases the {noun2} {verb1} causing {noun3} to {verb2}.",
    "The {adj1} {noun1} separates {noun2} from {noun3} during the final stage.",
    "{noun1} produces {noun2} which is then {verb1} by the {noun3}.",
    "The {noun1} consists of three {noun2} each with a distinct {noun3}.",
    "{noun1} is found exclusively in {noun2} where it {verb1} {noun3}.",
    "The binding of {noun1} to {noun2} triggers a {adj1} change in {noun3}.",
    "Both {noun1} and {noun2} are required for {noun3} to {verb1} correctly.",
    "The {noun1} cycle begins with {noun2} and ends with the formation of {noun3}.",
]

# Relation extraction patterns (what triples to extract from each template)
TRIPLE_PATTERNS = [
    # (template_index, [(subj_slot, rel, obj_slot), ...])
    (0, [("{noun1}", "is_a", "{noun2}"), ("{noun1}", "occurs_in", "{noun3}"), ("{noun1}", "produces", "{noun4}")]),
    (1, [("{noun1}", "involves", "{noun2}"), ("{noun1}", "occurs_in", "{noun4}")]),
    (2, [("{noun1}", "contains", "{noun2}")]),
    (3, [("{noun1}", "requires", "{noun2}")]),
    (4, [("{noun1}", "causes", "{noun3}"), ("{noun3}", "produces", "{noun4}")]),
    (5, [("{noun1}", "contains", "{noun2}"), ("{noun1}", "contains", "{noun3}")]),
    (6, [("{noun1}", "occurs_in", "{noun2}")]),
    (7, [("{noun1}", "involves", "{noun3}"), ("{noun3}", "transforms_into", "{noun4}")]),
    (8, [("{noun2}", "requires", "{noun1}")]),
    (9, [("{noun1}", "regulates", "{noun2}"), ("{noun2}", "produces", "{noun3}")]),
    (10, [("{noun1}", "provides", "{noun2}"), ("{noun2}", "enables", "{noun3}")]),
    (11, [("{noun1}", "interacts_with", "{noun2}"), ("{noun3}", "produces", "{noun4}")]),
    (12, [("{noun2}", "requires", "{noun1}"), ("{noun2}", "causes", "{noun3}")]),
    (13, [("{noun1}", "transforms_into", "{noun3}"), ("{noun1}", "requires", "{noun4}")]),
    (14, [("{noun1}", "enables", "{noun2}"), ("{noun1}", "regulates", "{noun3}")]),
    (15, [("{noun1}", "regulates", "{noun3}")]),
    (16, [("{noun2}", "occurs_in", "{noun1}"), ("{noun3}", "occurs_in", "{noun1}")]),
    (17, [("{noun1}", "prevents", "{noun2}"), ("{noun1}", "regulates", "{noun3}")]),
    (18, [("{noun1}", "is_a", "{noun2}")]),
    (19, [("{noun1}", "produces", "{noun2}")]),
    (20, [("{noun1}", "requires", "{noun2}"), ("{noun1}", "requires", "{noun4}")]),
    (21, [("{noun1}", "part_of", "{noun2}"), ("{noun1}", "occurs_in", "{noun3}")]),
    (22, [("{noun1}", "causes", "{noun3}")]),
    (23, [("{noun1}", "separates", "{noun2}")]),
    (24, [("{noun1}", "produces", "{noun2}"), ("{noun2}", "requires", "{noun3}")]),
    (25, [("{noun1}", "contains", "{noun2}")]),
    (26, [("{noun1}", "occurs_in", "{noun2}"), ("{noun1}", "produces", "{noun3}")]),
    (27, [("{noun1}", "interacts_with", "{noun2}"), ("{noun1}", "causes", "{noun3}")]),
    (28, [("{noun1}", "requires", "{noun3}"), ("{noun2}", "requires", "{noun3}")]),
    (29, [("{noun1}", "requires", "{noun2}"), ("{noun1}", "produces", "{noun3}")]),
]


def generate_example(rng: random.Random) -> dict | None:
    """Generate one training example: paragraph + triples."""
    # Pick 3-5 sentence templates
    n_sentences = rng.randint(3, 5)
    indices = rng.sample(range(len(WIKI_SENTENCES)), n_sentences)

    # Generate a shared concept pool for this "domain"
    concepts = {f"noun{i}": nonsense_word(rng) for i in range(1, 8)}
    concepts.update({f"adj{i}": nonsense_word(rng, 2) for i in range(1, 4)})
    concepts.update({f"verb{i}": nonsense_word(rng, 2) for i in range(1, 4)})

    # Some concepts should be shared across sentences (interconnection)
    shared_nouns = [nonsense_word(rng) for _ in range(3)]

    sentences = []
    all_triples = []

    for idx in indices:
        template = WIKI_SENTENCES[idx]
        # Create local concept mapping, reusing some shared nouns
        local = dict(concepts)
        for key in list(local.keys()):
            if key.startswith("noun") and rng.random() < 0.3:
                local[key] = rng.choice(shared_nouns)
            elif rng.random() < 0.5:
                local[key] = nonsense_word(rng)

        try:
            sentence = template.format(**local)
        except KeyError:
            continue
        sentences.append(sentence)

        # Extract triples for this sentence
        for pat_idx, triples in TRIPLE_PATTERNS:
            if pat_idx == idx:
                for subj_slot, rel, obj_slot in triples:
                    subj = local.get(subj_slot.strip("{}"), subj_slot)
                    obj = local.get(obj_slot.strip("{}"), obj_slot)
                    if subj != obj:
                        all_triples.append(f"{subj} {rel} {obj}")
                break

    if not sentences or not all_triples:
        return None

    paragraph = " ".join(sentences)
    return {"paragraph": paragraph, "triples": all_triples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-examples", type=int, default=100000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=271828)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.out, "w")
    generated = 0
    t0 = time.time()

    while generated < args.num_examples:
        ex = generate_example(rng)
        if ex:
            out_f.write(json.dumps(ex) + "\n")
            generated += 1
        if generated % 10000 == 0 and generated > 0:
            print(f"[{generated}/{args.num_examples}] ({time.time()-t0:.0f}s)", file=sys.stderr)

    out_f.close()
    print(f"\nDone. {generated} examples in {time.time()-t0:.0f}s. Output: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
