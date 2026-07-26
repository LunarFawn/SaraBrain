#!/usr/bin/env python3
"""Generate extractor training data from REAL ENGLISH text.

The key insight: the extractor must be trained on real English, not just
jibberish. The copy mechanism needs to see real word distributions to
generalize properly.

This generator creates training pairs by:
1. Taking real English sentences (biology textbook or Wikipedia-style)
2. Using rule-based extraction to identify subject-relation-object triples
3. Formatting as the structured t_start/t_rel/t_obj/t_end output

The rules are simple and won't catch everything — that's fine. We want
the model to learn the PATTERN of extraction, and the copy mechanism
will handle pointing to the right words.

We also mix in synthetic/jibberish examples (30%) to maintain
generalization to unseen vocabulary.

Usage:
    python scripts/generate_english_extractor_data.py \
        --out training_data/extractor_english_500k.jsonl \
        --num-examples 500000
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

# ---- Real English extraction patterns ----

# Pattern: "X is a Y" / "X is an Y" / "X are Y"
IS_A_PATTERNS = [
    r"^(.+?)\s+(?:is|are)\s+(?:a|an|the)\s+(.+?)(?:\.|,|;|$)",
    r"^(.+?)\s+(?:is|are)\s+(?:classified as|considered|known as|called)\s+(?:a|an|the)?\s*(.+?)(?:\.|,|;|$)",
    r"^(?:a|an|the)\s+(.+?)\s+is\s+(?:a|an|the)\s+(.+?)(?:\.|,|;|$)",
]

# Pattern: "X contains Y" / "X produces Y" / "X requires Y" etc.
VERB_PATTERNS = {
    "contains": [r"(.+?)\s+contains?\s+(.+?)(?:\.|,|;|$)"],
    "produces": [r"(.+?)\s+produces?\s+(.+?)(?:\.|,|;|$)"],
    "requires": [r"(.+?)\s+requires?\s+(.+?)(?:\.|,|;|$)"],
    "involves": [r"(.+?)\s+involves?\s+(.+?)(?:\.|,|;|$)"],
    "causes": [r"(.+?)\s+causes?\s+(.+?)(?:\.|,|;|$)"],
    "prevents": [r"(.+?)\s+prevents?\s+(.+?)(?:\.|,|;|$)"],
    "enables": [r"(.+?)\s+enables?\s+(.+?)(?:\.|,|;|$)"],
    "regulates": [r"(.+?)\s+regulates?\s+(.+?)(?:\.|,|;|$)"],
    "activates": [r"(.+?)\s+activates?\s+(.+?)(?:\.|,|;|$)"],
    "inhibits": [r"(.+?)\s+inhibits?\s+(.+?)(?:\.|,|;|$)"],
    "occurs_in": [r"(.+?)\s+occurs?\s+in\s+(.+?)(?:\.|,|;|$)"],
    "part_of": [r"(.+?)\s+(?:is|are)\s+(?:a\s+)?part\s+of\s+(.+?)(?:\.|,|;|$)"],
    "composed_of": [r"(.+?)\s+(?:is|are)\s+(?:composed|made up)\s+of\s+(.+?)(?:\.|,|;|$)"],
    "depends_on": [r"(.+?)\s+depends?\s+on\s+(.+?)(?:\.|,|;|$)"],
    "results_in": [r"(.+?)\s+results?\s+in\s+(.+?)(?:\.|,|;|$)"],
    "leads_to": [r"(.+?)\s+leads?\s+to\s+(.+?)(?:\.|,|;|$)"],
    "transforms_into": [r"(.+?)\s+(?:transforms?|converts?)\s+(?:into|to)\s+(.+?)(?:\.|,|;|$)"],
    "interacts_with": [r"(.+?)\s+interacts?\s+with\s+(.+?)(?:\.|,|;|$)"],
    "attaches_to": [r"(.+?)\s+(?:attaches?|binds?)\s+to\s+(.+?)(?:\.|,|;|$)"],
    "provides": [r"(.+?)\s+provides?\s+(.+?)(?:\.|,|;|$)"],
}

# Stop words to strip from extracted concepts
STOP_WORDS = {"a", "an", "the", "this", "that", "these", "those", "which",
              "who", "whom", "it", "its", "also", "then", "thus", "however",
              "therefore", "moreover", "furthermore", "additionally"}


def clean_concept(text: str, max_words: int = 3) -> str:
    """Clean an extracted concept — strip to core noun phrase.
    
    The goal: concepts should be 1-3 words representing a single thing.
    'mitochondria', 'cell membrane', 'dna replication' — not
    'process by which a cell divides' or 'example of a sesamoid bone'.
    """
    text = text.strip().lower()
    # Remove leading articles
    for prefix in ["a ", "an ", "the ", "this ", "that ", "each ", "every "]:
        if text.startswith(prefix):
            text = text[len(prefix):]
    # Remove trailing punctuation
    text = text.rstrip(".,;:!?)")
    # Reject if it starts with a question word or number
    if text and text.split()[0] in {"how", "what", "why", "when", "where", "which", "who"}:
        return ""
    # Reject if it contains underscores (textbook fill-in-blanks)
    if "_" in text:
        return ""
    # Reject if it starts with a digit
    if text and text[0].isdigit():
        return ""
    # Reject modal/auxiliary starts (indicates a clause, not a concept)
    first_word = text.split()[0] if text.split() else ""
    if first_word in {"would", "could", "should", "can", "may", "might",
                      "will", "shall", "do", "does", "did", "has", "have",
                      "had", "is", "are", "was", "were", "be", "been",
                      "one", "two", "three", "many", "several", "some",
                      "by", "with", "from", "into", "through", "about"}:
        return ""
    # CUT at prepositions — "powerhouse of the cell" → "powerhouse"
    # These indicate a trailing clause, not part of the core concept
    cut_words = {"of", "in", "from", "than", "that", "which", "who",
                 "where", "when", "while", "during", "after", "before",
                 "between", "through", "about", "under", "over", "into"}
    words = text.split()
    trimmed = []
    for w in words:
        if w in cut_words and len(trimmed) >= 1:
            break  # Stop at first preposition (after at least 1 word)
        if "," in w and len(trimmed) >= 1:
            break  # Stop at commas (indicates clause boundary)
        trimmed.append(w)
    words = trimmed
    # Strip commas from remaining words
    words = [w.strip(",") for w in words]
    words = [w for w in words if w]
    # Hard limit
    if len(words) > max_words:
        words = words[:max_words]
    if len(words) == 0:
        return ""
    # Reject single-character or very short results
    result = " ".join(words).strip()
    if len(result) < 3:
        return ""
    return result


def extract_triples_from_sentence(sentence: str) -> list[tuple[str, str, str]]:
    """Rule-based extraction of (subject, relation, object) from a sentence.
    
    Conservative: only extracts when both subject and object are clean
    noun phrases (1-3 words). Rejects partial clauses and fragments.
    """
    triples = []
    sent_lower = sentence.lower().strip()
    
    # Reject sentences that are clearly questions or fragments
    if sent_lower.startswith(("how ", "what ", "why ", "which ", "where ")):
        return []
    if len(sent_lower) < 15:
        return []

    # Try is_a patterns first
    for pattern in IS_A_PATTERNS:
        m = re.search(pattern, sent_lower)
        if m:
            subj = clean_concept(m.group(1))
            obj = clean_concept(m.group(2))
            if subj and obj and subj != obj:
                triples.append((subj, "is_a", obj))
                break

    # Try verb patterns
    for relation, patterns in VERB_PATTERNS.items():
        for pattern in patterns:
            m = re.search(pattern, sent_lower)
            if m:
                subj = clean_concept(m.group(1))
                obj = clean_concept(m.group(2))
                if subj and obj and subj != obj:
                    triples.append((subj, relation, obj))
                    break

    return triples


def format_triples(triples: list[tuple[str, str, str]]) -> str:
    """Format triples into the t_start/t_rel/t_obj/t_end format."""
    parts = []
    for subj, rel, obj in triples:
        parts.append(f"t_start {subj} t_rel {rel} t_obj {obj} t_end")
    return "\n".join(parts)


# ---- Jibberish generation (for mixing) ----

CONSONANTS = "bcdfghjklmnprstvwz"
VOWELS = "aeiou"


def _syl(rng):
    return rng.choice(CONSONANTS) + rng.choice(VOWELS) + rng.choice(CONSONANTS + VOWELS)


def jib_word(rng, syls=0):
    return "".join(_syl(rng) for _ in range(syls or rng.randint(2, 3)))


def jib_concept(rng):
    n = rng.choices([1, 2, 3], weights=[0.3, 0.5, 0.2])[0]
    return " ".join(jib_word(rng) for _ in range(n))


RELATIONS = [
    "is_a", "contains", "produces", "requires", "involves",
    "causes", "prevents", "occurs_in", "part_of", "enables",
    "interacts_with", "transforms_into", "regulates", "provides",
    "activates", "inhibits", "depends_on", "results_in",
]

TEMPLATES = [
    ("is_a", "{s} is a {o}.", "{s} is a type of {o}."),
    ("is_a", "{s} is classified as {o}.", "{s} is a specialized form of {o}."),
    ("contains", "{s} contains {o}.", "{s} is composed of multiple {o}."),
    ("produces", "{s} produces {o}.", "{s} generates {o} as output."),
    ("requires", "{s} requires {o} to function.", "Without {o} the {s} cannot operate."),
    ("involves", "{s} involves {o}.", "{s} includes {o} as a step."),
    ("causes", "{s} causes {o}.", "{s} triggers {o} when activated."),
    ("prevents", "{s} prevents {o}.", "{s} blocks {o} from occurring."),
    ("occurs_in", "{s} occurs in {o}.", "{s} takes place within {o}."),
    ("part_of", "{s} is part of {o}.", "{s} is a component of {o}."),
    ("enables", "{s} enables {o}.", "{s} makes {o} possible."),
    ("interacts_with", "{s} interacts with {o}.", "{s} binds to {o}."),
    ("regulates", "{s} regulates {o}.", "{s} controls the rate of {o}."),
    ("provides", "{s} provides {o}.", "{s} supplies {o}."),
    ("transforms_into", "{s} transforms into {o}.", "Over time {s} converts to {o}."),
    ("activates", "{s} activates {o}.", "{s} switches on {o}."),
    ("inhibits", "{s} inhibits {o}.", "{s} suppresses {o}."),
    ("depends_on", "{s} depends on {o}.", "{s} relies on {o} for function."),
    ("results_in", "{s} results in {o}.", "{s} leads to {o}."),
]


def generate_jibberish_example(rng) -> dict:
    """Generate a synthetic jibberish training example."""
    n_triples = rng.randint(1, 4)
    concepts = [jib_concept(rng) for _ in range(n_triples * 2 + 2)]
    triples = []
    sentences = []

    for i in range(n_triples):
        s = concepts[i * 2]
        o = concepts[i * 2 + 1]
        rel, *templates = rng.choice(TEMPLATES)
        template = rng.choice([t for t in templates if t is not None])
        sentences.append(template.format(s=s, o=o))
        triples.append((s, rel, o))

    paragraph = " ".join(sentences)
    output = format_triples(triples)
    return {"paragraph": paragraph, "output": output}


def load_biology_sentences() -> list[str]:
    """Load all biology English facts as sentences."""
    sentences = []
    bio_dir = Path("data/biology_english")
    if not bio_dir.exists():
        print(f"WARNING: {bio_dir} not found. Using only synthetic data.", file=sys.stderr)
        return []

    for f in sorted(bio_dir.glob("ch*_facts.txt")):
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Split multi-sentence lines
            for sent in re.split(r'(?<=[.!?])\s+', line):
                sent = sent.strip()
                if len(sent) > 20 and len(sent) < 300:
                    sentences.append(sent)

    return sentences


def generate_english_example(sentence: str) -> dict | None:
    """Try to extract triples from a real English sentence."""
    triples = extract_triples_from_sentence(sentence)
    if not triples:
        return None
    output = format_triples(triples)
    return {"paragraph": sentence, "output": output}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output .jsonl path")
    ap.add_argument("--num-examples", type=int, default=500000)
    ap.add_argument("--english-ratio", type=float, default=0.7,
                    help="Fraction of examples that use real English (default: 0.7)")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Load real biology sentences
    bio_sentences = load_biology_sentences()
    print(f"Loaded {len(bio_sentences)} biology sentences.", file=sys.stderr)

    # Pre-extract all possible English examples
    english_examples = []
    for sent in bio_sentences:
        ex = generate_english_example(sent)
        if ex:
            english_examples.append(ex)
    print(f"Extracted {len(english_examples)} English examples from biology text.", file=sys.stderr)

    if not english_examples:
        print("No English examples found. Generating jibberish-only.", file=sys.stderr)
        args.english_ratio = 0.0

    # Generate the full dataset
    t0 = time.time()
    examples = []
    english_count = 0
    jib_count = 0

    for i in range(args.num_examples):
        if rng.random() < args.english_ratio and english_examples:
            # Real English example (with augmentation: shuffle, repeat, paraphrase)
            ex = rng.choice(english_examples)
            examples.append(ex)
            english_count += 1
        else:
            # Synthetic jibberish
            ex = generate_jibberish_example(rng)
            examples.append(ex)
            jib_count += 1

        if (i + 1) % 100000 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{args.num_examples} "
                  f"(english={english_count}, jib={jib_count}, {elapsed:.1f}s)",
                  file=sys.stderr)

    # Shuffle
    rng.shuffle(examples)

    # Write
    with open(args.out, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. "
          f"Total: {len(examples)} (english={english_count}, jib={jib_count}). "
          f"Output: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
