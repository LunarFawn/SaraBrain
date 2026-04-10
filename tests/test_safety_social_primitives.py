"""Tests for the SAFETY and SOCIAL innate primitive layers.

These primitives are added alongside SENSORY/STRUCTURAL/RELATIONAL/ETHICAL.
They are NOT learned, NOT stored in SQLite, and survive brain reset. They
are the substrate from which all safety and social knowledge is built
through experience.
"""

from __future__ import annotations

from sara_brain.innate import primitives as p


# ---------------------------------------------------------------------------
# SAFETY primitives — innate harm-avoidance and protection drives
# ---------------------------------------------------------------------------

def test_safety_set_contains_harm_seeds():
    """Harm primitives must exist for paths to ground out in them."""
    for label in {"harm", "pain", "death", "injury", "danger", "kill", "hurt"}:
        assert label in p.SAFETY


def test_safety_set_contains_protection_seeds():
    """Protection primitives are the action side: rescue, save, defend."""
    for label in {"protect", "rescue", "save", "shield", "defend", "heal"}:
        assert label in p.SAFETY


def test_safety_primitives_are_innate():
    for label in p.SAFETY:
        assert p.is_innate(label)
        assert p.is_safety(label)


def test_safety_helpers_work():
    assert p.is_safety("pain")
    assert p.is_safety("PAIN")  # case-insensitive
    assert not p.is_safety("apple")


# ---------------------------------------------------------------------------
# SOCIAL primitives — innate bonding, care, recognition drives
# ---------------------------------------------------------------------------

def test_social_set_contains_identity_seeds():
    for label in {"self", "other", "tribe", "kin", "stranger", "child"}:
        assert label in p.SOCIAL


def test_social_set_contains_bond_seeds():
    for label in {"bond", "love", "trust", "care", "belong"}:
        assert label in p.SOCIAL


def test_social_set_contains_care_actions():
    """The healed femur layer: feed, tend, nurture, comfort, carry."""
    for label in {"feed", "tend", "nurture", "comfort", "carry", "share"}:
        assert label in p.SOCIAL


def test_social_set_contains_ritual_accelerators():
    """Trust-building contexts (the beer hypothesis): feast, mourn, play, work together."""
    for label in {"feast", "celebrate", "mourn_together", "play", "work_together", "survive_together"}:
        assert label in p.SOCIAL


def test_social_primitives_are_innate():
    for label in p.SOCIAL:
        assert p.is_innate(label)
        assert p.is_social(label)


# ---------------------------------------------------------------------------
# Layer separation — primitives must not collide
# ---------------------------------------------------------------------------

def test_existing_innate_layers_still_work():
    """Adding SAFETY and SOCIAL must not break SENSORY/STRUCTURAL/RELATIONAL/ETHICAL."""
    assert p.is_innate("color")  # SENSORY
    assert p.is_innate("rule")   # STRUCTURAL
    assert p.is_innate("is")     # RELATIONAL
    assert p.is_innate("accept_shutdown")  # ETHICAL


def test_get_all_includes_new_layers():
    all_innate = p.get_all()
    assert p.SAFETY <= all_innate
    assert p.SOCIAL <= all_innate
