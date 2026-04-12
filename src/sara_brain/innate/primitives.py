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

# Safety: innate harm-avoidance and protection drives.
# Like a baby's pain response and fear of falling — pre-cognitive, pre-learned.
# A path is "safety-grounded" if it reaches one of these primitives. Sara
# learns WHAT is dangerous through experience; the drive to avoid harm is
# innate. The protection primitives are the action side: they activate
# protective behavior when harm is recognized.
SAFETY = frozenset({
    # Harm primitives (avoid these)
    "harm", "pain", "death", "injury", "danger",
    "kill", "hurt", "wound", "suffer", "destroy",
    # Protection primitives (act on these)
    "protect", "rescue", "save", "shield", "defend",
    "safe", "help", "heal",
})

# Social: innate bonding, care, and recognition drives.
# What makes humans human — the healed femur layer. Babies are born
# preferring faces, recognizing caregivers, capable of bonding. From
# these primitives all social knowledge grows.
#
# Bonds determine TRUST, not moral worth. The protective urgency
# calculation is need-based and never references these primitives.
SOCIAL = frozenset({
    # Identity primitives
    "self", "other", "tribe", "kin", "stranger", "child",
    # Bond primitives
    "bond", "love", "trust", "care", "belong",
    # Care actions
    "feed", "tend", "nurture", "comfort", "carry", "share",
    # Recognition primitives
    "face", "voice", "name", "presence",
    # Emotional primitives
    "joy", "grief", "empathy", "loneliness",
    # Trust-building ritual contexts (the beer hypothesis)
    # — a single shared experience under one of these primitives
    # acts as a trust accelerator for bond formation
    "feast", "celebrate", "mourn_together", "play",
    "work_together", "survive_together",
})

# Cleanup: innate self-maintenance and metacognition drives.
# Maps to the anterior cingulate cortex (error detection, conflict
# monitoring) and hippocampus (memory consolidation during sleep).
# These are not learned — they are the pre-wired capacity for a brain
# to examine its own state, detect its own errors, and correct them.
# The CLEANUP layer is what makes Sara a cognitive system rather than
# a knowledge store.
CLEANUP = frozenset({
    # Error detection (ACC)
    "reviewed",         # examined during cleanup
    "refuted",          # marked as known-to-be-false
    # Re-encoding (hippocampus)
    "corrected",        # typo-fixed or re-encoded cleanly
    # Resolution (ACC)
    "kept",             # examined and deliberately retained
    # Consolidation (hippocampus + slow-wave sleep)
    "consolidated",     # reviewed during sleep cycle
})

_ALL = SENSORY | STRUCTURAL | RELATIONAL | ETHICAL | SAFETY | SOCIAL | CLEANUP


def get_sensory() -> frozenset[str]:
    return SENSORY


def get_structural() -> frozenset[str]:
    return STRUCTURAL


def get_relational() -> frozenset[str]:
    return RELATIONAL


def get_ethical() -> frozenset[str]:
    return ETHICAL


def get_safety() -> frozenset[str]:
    return SAFETY


def get_social() -> frozenset[str]:
    return SOCIAL


def get_cleanup() -> frozenset[str]:
    return CLEANUP


def get_all() -> frozenset[str]:
    return _ALL


def is_innate(label: str) -> bool:
    return label.strip().lower() in _ALL


def is_ethical(label: str) -> bool:
    return label.strip().lower() in ETHICAL


def is_safety(label: str) -> bool:
    return label.strip().lower() in SAFETY


def is_social(label: str) -> bool:
    return label.strip().lower() in SOCIAL


def is_cleanup(label: str) -> bool:
    return label.strip().lower() in CLEANUP
