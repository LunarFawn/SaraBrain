"""Generate training data for the MCQ scorer model.

The scorer reads wavefront facts + question + N choices and picks
the choice best supported by the facts. Variable number of choices (2-8).

Two types of examples:
1. Direct match: the correct choice's words appear in the facts
2. Indirect match: the correct choice shares concepts with the facts
   but requires understanding the relationship

Usage:
    python scripts/generate_mcq_scorer_data.py \
        --num-examples 500000 \
        --out training_data/mcq_scorer_500k.jsonl
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
    n = rng.choices([1, 2, 3], weights=[0.3, 0.5, 0.2])[0]
    return " ".join(word(rng) for _ in range(n))


RELATIONS = [
    "is_a", "contains", "produces", "requires", "involves",
    "causes", "prevents", "occurs_in", "part_of", "enables",
    "interacts_with", "transforms_into", "regulates", "provides",
]

QUESTION_FORMS = [
    "What does {s} {r}?",
    "What {r} {o}?",
    "What is {s}?",
    "Which of the following is related to {s}?",
    "Which of the following does {s} {r}?",
    "What is associated with {s}?",
]


def generate_domain(rng, n_concepts=10):
    concepts = [concept(rng) for _ in range(n_concepts)]
    triples = []
    used = set()
    for i in range(min(n_concepts - 1, 6)):
        rel = rng.choice(RELATIONS)
        triples.append((concepts[i], rel, concepts[i + 1]))
        used.add((i, i + 1))
    extra = rng.randint(3, 6)
    for _ in range(extra):
        a, b = rng.sample(range(n_concepts), 2)
        if (a, b) not in used:
            triples.append((concepts[a], rng.choice(RELATIONS), concepts[b]))
            used.add((a, b))
    return concepts, triples


def generate_example(rng):
    n_concepts = rng.randint(8, 14)
    concepts, triples = generate_domain(rng, n_concepts)
    if len(triples) < 4:
        return None

    # Pick target triple
    target_idx = rng.randint(0, len(triples) - 1)
    target_s, target_r, target_o = triples[target_idx]

    # Build question
    q_form = rng.choice(QUESTION_FORMS)
    try:
        question = q_form.format(s=target_s, r=target_r, o=target_o)
    except (KeyError, IndexError):
        question = f"What does {target_s} {target_r}?"

    # Build facts (rendered like wavefront output)
    # Include the target triple and some others
    rng.shuffle(triples)
    fact_lines = [f"  - {s} {r} {o}" for s, r, o in triples[:8]]
    facts = "\n".join(fact_lines)

    # Build choices (variable number 2-6)
    n_choices = rng.randint(2, 6)

    # Correct answer is the object of the target triple
    correct = target_o

    # Distractors from concepts NOT in the target triple
    distractors = [c for c in concepts if c != correct and c != target_s]
    if len(distractors) < n_choices - 1:
        return None
    distractors = rng.sample(distractors, n_choices - 1)

    # Assemble choices in random order
    choices = distractors + [correct]
    rng.shuffle(choices)
    correct_idx = choices.index(correct)

    # Format choices as lettered list
    choice_str = " | ".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))

    return {
        "facts": facts,
        "question": question,
        "choices": choice_str,
        "n_choices": n_choices,
        "correct_idx": correct_idx,
        "correct_letter": chr(65 + correct_idx),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-examples", type=int, default=500000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=265358)
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
        if generated % 50000 == 0 and generated > 0:
            print(f"[{generated}/{args.num_examples}] ({time.time()-t0:.0f}s)", file=sys.stderr)

    out_f.close()
    print(f"\nDone. {generated} examples in {time.time()-t0:.0f}s.", file=sys.stderr)


if __name__ == "__main__":
    main()
