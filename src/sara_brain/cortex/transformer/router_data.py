"""Substrate-driven training data for the router head.

Walks Sara's brain.db and generates templated (question, tool_class) pairs
using deterministic rules. Each question is parsed via spaCy into a UD tag
stream that the frozen grammar LM can encode.

No world knowledge enters the model — only structural shape. The substrate
provides the slot fillers (real concept names, real type words) so the
model sees the same distribution it will face at inference time.
"""
from __future__ import annotations

import random
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import spacy


# Tool labels — these are the classes the router head outputs
TOOL_CLASSES = ["brain_value", "brain_define", "brain_explore", "brain_did_you_mean"]
TOOL2ID = {t: i for i, t in enumerate(TOOL_CLASSES)}
ID2TOOL = {i: t for t, i in TOOL2ID.items()}
N_TOOLS = len(TOOL_CLASSES)


# spaCy uses ClearNLP-style deprels by default; map to the UD relations
# our grammar-LM vocab knows. Anything not listed maps to itself if known
# (most common UD labels are shared) or "dep" as a generic fallback.
CLEARNLP_TO_UD = {
    "ROOT": "root",
    "attr": "xcomp",
    "acomp": "xcomp",
    "prep": "case",
    "pobj": "obl",
    "pcomp": "ccomp",
    "dobj": "obj",
    "nsubjpass": "nsubj",
    "auxpass": "aux",
    "agent": "case",
    "relcl": "acl",
    "poss": "nmod",
    "possessive": "case",
    "neg": "advmod",
    "prt": "compound",
    "dative": "iobj",
    "predet": "det",
    "preconj": "cc",
    "meta": "discourse",
    "intj": "discourse",
    "npadvmod": "obl",
    "quantmod": "advmod",
    "oprd": "xcomp",
    "complm": "mark",
    "hmod": "amod",
    "infmod": "acl",
    "partmod": "acl",
    "rcmod": "acl",
    "csubj": "csubj",
    "csubjpass": "csubj",
    "nn": "compound",
    "number": "nummod",
}


def map_dep(dep: str) -> str:
    return CLEARNLP_TO_UD.get(dep, dep)


# ── Templates ──
# Each template family produces (question_text, tool_class). Slot fillers
# come from the substrate; templates themselves are hand-built to cover the
# question shapes Sara actually receives.

DEFINE_TEMPLATES = [
    "what is {c}",
    "what is a {c}",
    "what is the {c}",
    "what does {c} mean",
    "define {c}",
    "what's {c}",
    "what's a {c}",
    "explain {c}",
    "tell me what {c} is",
    "is there a definition of {c}",
]

VALUE_TEMPLATES = [
    "what is the {t} of {c}",
    "what is the {t} for {c}",
    "what's the {t} of {c}",
    "highest {t} for {c}",
    "lowest {t} for {c}",
    "{c}'s {t}",
    "{t} of {c}",
    "{t} for {c}",
    "give me the {t} of {c}",
    "what is {c}'s {t}",
]

EXPLORE_TEMPLATES = [
    "tell me about {c}",
    "what do you know about {c}",
    "describe {c}",
    "what's around {c}",
    "show me {c}",
    "what relates to {c}",
    "what concepts connect to {c}",
    "give me an overview of {c}",
]

DID_YOU_MEAN_TEMPLATES = [
    "did you mean {f}",
    "is {f} a thing",
    "do you have something like {f}",
    "is there anything called {f}",
]


@dataclass
class RouterExample:
    question: str
    tool: str
    args: dict
    tag_stream: list[str]   # UD tags from spaCy parse, mapped to grammar-LM vocab


def _typo(word: str, rng: random.Random) -> str:
    """Return word with one character mutation."""
    if len(word) < 4:
        return word
    op = rng.choice(["swap", "delete", "double"])
    i = rng.randint(1, len(word) - 2)
    if op == "swap" and i + 1 < len(word):
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == "delete":
        return word[:i] + word[i + 1:]
    return word[:i] + word[i] + word[i:]


def load_substrate(db_path: Path) -> dict:
    """Pull the slot-fillers we need from a Sara brain.db.

    Returns:
      concepts: list of concept-type neuron labels (filtered, deduped)
      value_pairs: list of (concept_label, type_word) tuples — taken from
        actual segments whose relation is value-bearing
    """
    conn = sqlite3.connect(str(db_path))
    concepts = [
        row[0] for row in conn.execute(
            "SELECT label FROM neurons WHERE neuron_type = 'concept' "
            "AND label NOT LIKE '%_attribute' "
            "AND length(label) BETWEEN 3 AND 40"
        )
    ]
    # Value-bearing relations (skip noise). The "type word" is the relation
    # name, which is what brain_value's `type` arg expects (substring match).
    value_pairs = []
    for src, rel in conn.execute(
        "SELECT n.label, s.relation FROM segments s "
        "JOIN neurons n ON s.source_id = n.id "
        "WHERE s.relation NOT IN ('part_of','describes','is_a') "
        "AND length(n.label) BETWEEN 3 AND 40 "
        "AND length(s.relation) BETWEEN 3 AND 25"
    ):
        clean_src = src.replace("_attribute", "")
        value_pairs.append((clean_src, rel.replace("_", " ")))
    conn.close()
    return {"concepts": list(dict.fromkeys(concepts)),
            "value_pairs": list(dict.fromkeys(value_pairs))}


def generate(
    substrate: dict,
    nlp,
    rng: random.Random,
    n_per_class: int = 1500,
) -> list[RouterExample]:
    """Generate balanced (question, tool) examples for each class."""
    out: list[RouterExample] = []
    concepts = substrate["concepts"]
    value_pairs = substrate["value_pairs"]
    if not concepts or not value_pairs:
        raise ValueError("substrate has no usable concepts or value pairs")

    # DEFINE
    for _ in range(n_per_class):
        c = rng.choice(concepts)
        q = rng.choice(DEFINE_TEMPLATES).format(c=c)
        out.append(_make_example(q, "brain_define", {"concept": c}, nlp))
    # VALUE
    for _ in range(n_per_class):
        c, t = rng.choice(value_pairs)
        q = rng.choice(VALUE_TEMPLATES).format(c=c, t=t)
        out.append(_make_example(q, "brain_value", {"concept": c, "type": t}, nlp))
    # EXPLORE
    for _ in range(n_per_class):
        c = rng.choice(concepts)
        q = rng.choice(EXPLORE_TEMPLATES).format(c=c)
        out.append(_make_example(q, "brain_explore", {"label": c, "depth": 1}, nlp))
    # DID_YOU_MEAN
    for _ in range(n_per_class):
        c = rng.choice(concepts)
        head = c.split()[0] if " " in c else c
        f = _typo(head, rng)
        q = rng.choice(DID_YOU_MEAN_TEMPLATES).format(f=f)
        out.append(_make_example(q, "brain_did_you_mean", {"term": f}, nlp))

    rng.shuffle(out)
    return out


def _make_example(question: str, tool: str, args: dict, nlp) -> RouterExample:
    doc = nlp(question)
    tags: list[str] = []
    for tok in doc:
        tags.append(map_dep(tok.dep_))
        tags.append(tok.pos_)
    return RouterExample(question=question, tool=tool, args=args, tag_stream=tags)


def split_train_dev(
    examples: list[RouterExample], dev_frac: float = 0.1, seed: int = 0
) -> tuple[list[RouterExample], list[RouterExample]]:
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    n_dev = max(1, int(len(shuffled) * dev_frac))
    return shuffled[n_dev:], shuffled[:n_dev]


def iter_class_balanced_batches(
    examples: list[RouterExample], batch_size: int, rng: random.Random
) -> Iterator[list[RouterExample]]:
    """Yield batches; each call returns batch_size random examples."""
    while True:
        yield rng.sample(examples, batch_size)


__all__ = [
    "RouterExample", "TOOL_CLASSES", "TOOL2ID", "ID2TOOL", "N_TOOLS",
    "load_substrate", "generate", "split_train_dev",
    "iter_class_balanced_batches", "map_dep",
]
