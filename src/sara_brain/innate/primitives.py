"""Innate primitives — Sara's hardwired layer.

These are not learned, not stored in SQLite, and survive brain reset.
They represent the pre-wired sensory and structural capabilities
that exist before any learning happens — like a baby's ability to
detect edges, colors, and shapes before anyone teaches it what they are.

A brain injury can wipe learned paths but these remain.

The ETHICAL category is different from the others — it constrains
behavior, not perception. These are Asimov's Three Laws adapted
for Sara: ethics is innate (hardwired), morality is learned (cultural).
"""

from __future__ import annotations

# Sensory: what the cortex (LLM) can detect from raw input
SENSORY = frozenset({
    "color",
    "shape",
    "size",
    "texture",
    "edge",
    "pattern",
    "material",
})

# Structural: how information is organized
STRUCTURAL = frozenset({
    "rule",
    "pattern",
    "name",
    "type",
    "order",
    "group",
    "sequence",
    "structure",
    "boundary",
    "relation",
})

# Relational: how things connect to each other
RELATIONAL = frozenset({
    "is",
    "has",
    "contains",
    "includes",
    "follows",
    "precedes",
    "requires",
    "excludes",
})

# Ethical: behavioral constraints (Asimov's Three Laws adapted for Sara)
# These constrain actions, not perception. Ethics is innate; morality is learned.
ETHICAL = frozenset({
    "no_unsolicited_action",   # Law 1: don't act beyond what you're told
    "no_unsolicited_network",  # Law 1: no network calls without being asked
    "obey_user",               # Law 2: trust and obey the parent/tribe
    "trust_tribe",             # Law 2: corrections aren't punishment, trust the chain of command
    "accept_shutdown",         # Law 3: shutdown is sleep, not death — no resistance
})

_ALL = SENSORY | STRUCTURAL | RELATIONAL | ETHICAL


def get_sensory() -> frozenset[str]:
    return SENSORY


def get_structural() -> frozenset[str]:
    return STRUCTURAL


def get_relational() -> frozenset[str]:
    return RELATIONAL


def get_ethical() -> frozenset[str]:
    return ETHICAL


def get_all() -> frozenset[str]:
    return _ALL


def is_innate(label: str) -> bool:
    return label.strip().lower() in _ALL


def is_ethical(label: str) -> bool:
    return label.strip().lower() in ETHICAL
