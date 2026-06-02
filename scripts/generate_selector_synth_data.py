"""Generate training data for Selector and Synthesizer models.

Selector: (wavefront facts + question) → correct fact(s)
Synthesizer: (selected facts + question) → prose answer

Uses the same synthetic concept graphs as the extractor, but generates
wavefront-style rendered facts and prose answers.

Usage:
    python scripts/generate_selector_synth_data.py \
        --num-examples 500000 \
        --out-selector training_data/selector_500k.jsonl \
        --out-synthesizer training_data/synthesizer_500k.jsonl
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
    "separates", "attaches_to", "depends_on", "activates",
]

QUESTION_TEMPLATES = [
    ("What does {s} {r}?", "{s} {r} {o}."),
    ("What {r} {o}?", "{s} {r} {o}."),
    ("What is {s}?", "{s} is_a {o}."),
    ("What does {s} require?", "{s} requires {o}."),
    ("What does {s} produce?", "{s} produces {o}."),
    ("What contains {o}?", "{s} contains {o}."),
    ("What is part of {o}?", "{s} part_of {o}."),
    ("What causes {o}?", "{s} causes {o}."),
    ("What prevents {o}?", "{s} prevents {o}."),
    ("What enables {o}?", "{s} enables {o}."),
    ("What does {s} interact with?", "{s} interacts_with {o}."),
    ("Describe {s}.", "{s} {r} {o}."),
    ("Tell me about {s}.", "{s} {r} {o}."),
    ("How does {s} work?", "{s} {r} {o}."),
    ("What is the role of {s}?", "{s} {r} {o}."),
]

PROSE_TEMPLATES = [
    "{s} {verb} {o}.",
    "The {s} {verb} {o}.",
    "Based on the available information, {s} {verb} {o}.",
    "{s} is known to {verb} {o}.",
    "The relationship is that {s} {verb} {o}.",
]

VERB_MAP = {
    "is_a": ["is a type of", "is classified as", "is a"],
    "contains": ["contains", "is composed of", "includes"],
    "produces": ["produces", "generates", "creates"],
    "requires": ["requires", "needs", "depends on"],
    "involves": ["involves", "is related to", "concerns"],
    "causes": ["causes", "leads to", "triggers"],
    "prevents": ["prevents", "blocks", "inhibits"],
    "occurs_in": ["occurs in", "takes place in", "is found in"],
    "part_of": ["is part of", "belongs to", "is a component of"],
    "enables": ["enables", "makes possible", "facilitates"],
    "interacts_with": ["interacts with", "binds to", "connects to"],
    "transforms_into": ["transforms into", "becomes", "converts to"],
    "regulates": ["regulates", "controls", "modulates"],
    "provides": ["provides", "supplies", "delivers"],
    "separates": ["separates", "divides", "splits"],
    "attaches_to": ["attaches to", "connects to", "joins"],
    "depends_on": ["depends on", "relies on", "requires"],
    "activates": ["activates", "turns on", "initiates"],
}


def generate_domain(rng, n_concepts=8):
    concepts = [concept(rng) for _ in range(n_concepts)]
    triples = []
    for i in range(min(n_concepts - 1, 5)):
        rel = rng.choice(RELATIONS)
        triples.append((concepts[i], rel, concepts[i + 1]))
    extra = rng.randint(3, n_concepts)
    used = set()
    for _ in range(extra):
        a, b = rng.sample(range(n_concepts), 2)
        if (a, b) not in used:
            triples.append((concepts[a], rng.choice(RELATIONS), concepts[b]))
            used.add((a, b))
    return concepts, triples


def render_facts(triples):
    """Render triples as wavefront-style source_text facts."""
    lines = []
    for s, r, o in triples:
        lines.append(f"  - {s} {r} {o}")
    return "\n".join(lines)


def generate_example(rng):
    n_concepts = rng.randint(5, 10)
    concepts, triples = generate_domain(rng, n_concepts)

    if len(triples) < 3:
        return None, None

    # Pick the target triple (the one the question is about)
    target_idx = rng.randint(0, len(triples) - 1)
    target = triples[target_idx]
    s, r, o = target

    # Build question
    q_template, a_template = rng.choice(QUESTION_TEMPLATES)
    try:
        question = q_template.format(s=s, r=r, o=o)
    except (KeyError, IndexError):
        question = f"What does {s} {r}?"

    # Build the fact list (include target + distractors)
    # Shuffle so target isn't always first
    shuffled = list(triples)
    rng.shuffle(shuffled)
    facts_rendered = render_facts(shuffled)

    # Selector target: the correct fact in structured format
    selector_output = f"t_start {s} t_rel {r} t_obj {o} t_end"

    # Synthesizer: prose answer
    verbs = VERB_MAP.get(r, [r])
    verb = rng.choice(verbs)
    prose_template = rng.choice(PROSE_TEMPLATES)
    prose_answer = prose_template.format(s=s, verb=verb, o=o)

    selector_example = {
        "facts": facts_rendered,
        "question": question,
        "output": selector_output,
    }

    synthesizer_example = {
        "facts": selector_output,
        "question": question,
        "output": prose_answer,
    }

    return selector_example, synthesizer_example


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-examples", type=int, default=500000)
    ap.add_argument("--out-selector", required=True)
    ap.add_argument("--out-synthesizer", required=True)
    ap.add_argument("--seed", type=int, default=577215)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    Path(args.out_selector).parent.mkdir(parents=True, exist_ok=True)
    sel_f = open(args.out_selector, "w")
    syn_f = open(args.out_synthesizer, "w")
    generated = 0
    t0 = time.time()

    while generated < args.num_examples:
        sel, syn = generate_example(rng)
        if sel and syn:
            sel_f.write(json.dumps(sel) + "\n")
            syn_f.write(json.dumps(syn) + "\n")
            generated += 1
        if generated % 50000 == 0 and generated > 0:
            print(f"[{generated}/{args.num_examples}] ({time.time()-t0:.0f}s)", file=sys.stderr)

    sel_f.close()
    syn_f.close()
    print(f"\nDone. {generated} examples in {time.time()-t0:.0f}s.", file=sys.stderr)
    print(f"  Selector: {args.out_selector}", file=sys.stderr)
    print(f"  Synthesizer: {args.out_synthesizer}", file=sys.stderr)


if __name__ == "__main__":
    main()
