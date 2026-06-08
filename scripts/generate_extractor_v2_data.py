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
    # Simple declarative for all relations
    ("contains", "{s} contains {o}.", "{s} is composed of multiple {o}."),
    ("produces", "{s} produces {o}.", "{s} generates {o} as its primary output."),
    ("requires", "{s} requires {o} to function.", "Without {o} the {s} cannot operate."),
    ("involves", "{s} involves {o}.", "{s} includes {o} as a component."),
    ("causes", "{s} causes {o}.", "{s} triggers {o} when activated."),
    ("prevents", "{s} prevents {o}.", "{s} blocks {o} from occurring."),
    ("occurs_in", "{s} occurs in {o}.", "{s} takes place within {o}."),
    ("part_of", "{s} is part of {o}.", "{s} is a component of the larger {o}."),
    ("enables", "{s} enables {o}.", "{s} makes {o} possible."),
    ("interacts_with", "{s} interacts with {o}.", "{s} binds to {o} with high affinity."),
    ("regulates", "{s} regulates {o}.", "{s} controls the rate of {o}."),
    ("provides", "{s} provides {o}.", "{s} supplies {o} to the surrounding structure."),
    ("transforms_into", "{s} transforms into {o}.", "Over time {s} converts to {o}."),
    ("activates", "{s} activates {o}.", "{s} switches on {o} when conditions are met."),
    ("inhibits", "{s} inhibits {o}.", "{s} suppresses {o} effectively."),
    ("composed_of", "{s} is composed of {o}.", "{s} consists of {o}."),
    ("results_in", "{s} results in {o}.", "{s} leads to {o}."),
    ("leads_to", "{s} leads to {o}.", "{s} causes {o} eventually."),
    ("attaches_to", "{s} attaches to {o}.", "{s} binds to {o}."),
    ("depends_on", "{s} depends on {o}.", "{s} relies on {o} for function."),

    # Additional definition templates (weighted heavily)
    ("is_a", "{s} is a {o} that operates in complex environments.", None),
    ("is_a", "{s} is a {o} found in many systems.", None),
    ("is_a", "{s} is a kind of {o}.", "{s} is considered a {o}."),
    ("is_a", "{s} is an example of {o}.", "{s} represents a category of {o}."),
    ("is_a", "{s} is a specialized {o} designed for specific tasks.", None),
    ("is_a", "{s} is a variant of {o} with unique properties.", None),
    ("is_a", "{s} is a {o} responsible for critical functions.", None),
    ("is_a", "{s} is known as a {o} in standard terminology.", None),
    ("is_a", "{s} is a modified {o} adapted for extreme conditions.", None),
    ("is_a", "{s} is a primary {o} in the system.", None),
    ("is_a", "{s} is a novel {o} recently identified by researchers.", None),
    ("is_a", "{s} is the main {o} involved in this process.", None),
    ("is_a", "A {s} is a {o}.", "The {s} is a {o}."),
    ("is_a", "{s} can be described as a {o}.", "{s} functions as a {o}."),
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
    # === Nested clauses (the key missing patterns) ===
    # "X describes how Y verb Z" → X | describes | Z
    ("involves", "{s} describes how {o} changes over time.", None),
    ("involves", "{s} explains how {o} is maintained during stress.", None),
    ("involves", "{s} determines how {o} responds to external signals.", None),
    ("involves", "{s} shows that {o} is essential for survival.", None),
    ("involves", "{s} indicates that {o} plays a critical role.", None),
    ("involves", "{s} demonstrates that {o} can function independently.", None),
    ("involves", "{s} explains the conditions under which {o} occurs.", None),
    ("involves", "{s} describes the mechanism by which {o} is activated.", None),
    # "X uses Y to verb Z" → X | uses | Y (not Z)
    ("requires", "{s} uses {o} to achieve stability.", None),
    ("requires", "{s} relies on {o} to maintain function.", None),
    ("requires", "{s} depends on {o} to complete the process.", None),
    # Multi-verb: "X detects and binds Y" → X | interacts_with | Y
    ("interacts_with", "{s} detects and binds {o} with high specificity.", None),
    ("interacts_with", "{s} recognizes and captures {o} from the environment.", None),
    ("interacts_with", "{s} identifies and processes {o} efficiently.", None),
    # Prepositional subject: "The X of Y provides Z" → X | part_of | Y, X | provides | Z
    ("provides", "The {s} of the system provides {o} during operation.", None),
    ("provides", "The {s} within the structure provides {o} to adjacent regions.", None),
    # "X is known as Y" / "X is called Y" → X | is_a | Y
    ("is_a", "{s} is commonly known as {o} in the literature.", None),
    ("is_a", "{s} is also referred to as {o} by researchers.", None),
    ("is_a", "{s} is defined as {o} in standard terminology.", None),
    # Purpose clauses: "X provides Y for Z" → X | provides | Y
    ("provides", "{s} provides {o} for the surrounding structure.", None),
    ("provides", "{s} supplies {o} to ensure proper function.", None),
    # Temporal: "During X, Y produces Z" → Y | produces | Z, Y | occurs_in | X
    ("produces", "During {s} the system produces {o} continuously.", None),
    ("occurs_in", "Throughout {s} the process occurs in {o} repeatedly.", None),
]


# Real English 'noise' words to teach the model what to REJECT.
# These will be injected into sentences but NOT included in the triples.
ENGLISH_NOISE = [
    "the", "a", "an", "this", "that", "these", "those", "is", "are", "was", "were",
    "can", "may", "might", "should", "would", "could", "will", "shall",
    "it", "it is", "there are", "often", "usually", "sometimes", "always",
    "very", "just", "only", "also", "even", "still", "0", "1", "2", "3",
    "4", "5", "6", "7", "8", "9", "one", "two", "three", "first", "second",
    "ok", "okay", "yes", "no", "not", "well", "basically", "actually",
    "specifically", "generally", "typically", "mostly", "partially",
]


def generate_example(rng: random.Random) -> dict:
    """Generate one training example with compound concepts and structured output."""
    # Create a small domain of interconnected concepts
    n_concepts = rng.randint(4, 8)
    concepts = [concept(rng) for _ in range(n_concepts)]

    # Generate 3-6 triples, ALWAYS starting with a definition (is_a)
    n_triples = rng.randint(3, 6)
    triples = []
    sentences = []

    # First triple is ALWAYS a definition
    s, o = concepts[0], concepts[1]
    is_a_templates = [t for t in TEMPLATES if t[0] == "is_a"]
    template_entry = rng.choice(is_a_templates)
    rel = "is_a"
    template_options = [t for t in template_entry[1:] if t is not None]
    template = rng.choice(template_options)
    sentences.append(template.format(s=s, o=o))
    triples.append({"s": s, "r": rel, "o": o})

    # Remaining triples can be any type (but 30% chance of another is_a)
    for _ in range(n_triples - 1):
        s_idx, o_idx = rng.sample(range(n_concepts), 2)
        s, o = concepts[s_idx], concepts[o_idx]

        if rng.random() < 0.3:
            # Another definition
            template_entry = rng.choice(is_a_templates)
            rel = "is_a"
        else:
            # Pick a template (any type)
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

    # Inject 'Noise' sentences that should NOT produce triples
    # This teaches the model to ignore English filler and numbers.
    n_noise = rng.randint(1, 3)
    for _ in range(n_noise):
        noise_s = rng.choice(ENGLISH_NOISE)
        noise_o = rng.choice(ENGLISH_NOISE)
        # Use a real-sounding template for noise that doesn't use domain concepts
        noise_template = rng.choice([
            "Note that {s} is often used as {o}.",
            "In general, {s} can be {o}.",
            "Usually {s} implies {o}.",
            "It is {s} that {o} exists.",
            "{s} and {o} are common words.",
        ])
        noise_sentence = noise_template.format(s=noise_s, o=noise_o)
        # Insert noise sentence at random position
        insert_idx = rng.randint(0, len(paragraph_parts))
        paragraph_parts.insert(insert_idx, noise_sentence)

    paragraph = " ".join(paragraph_parts)

    # Build structured output
    output_lines = []
    for t in triples:
        output_lines.append(f"t_start {t['s']} t_rel {t['r']} t_obj {t['o']} t_end")
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
