"""Tests for the epistemic state of a segment.

The contested-vs-fresh bug: a segment with traversals=100, refutations=100
has the same `strength` value (1.0) as a fresh segment with traversals=0,
refutations=0. They are epistemically completely different states. The
fix is to expose `belief` and `evidence_weight` as separate properties
and derive an `epistemic_state` from their combination.
"""

from __future__ import annotations

from sara_brain.models.segment import Segment


def make_segment(traversals: int = 0, refutations: int = 0) -> Segment:
    seg = Segment(id=None, source_id=1, target_id=2, relation="is")
    seg.traversals = traversals
    seg.refutations = refutations
    seg._recalculate()
    return seg


def test_fresh_segment_is_unknown():
    seg = make_segment(traversals=0, refutations=0)
    assert seg.epistemic_state == "unknown"


def test_strongly_validated_segment_is_believed():
    seg = make_segment(traversals=100, refutations=0)
    assert seg.epistemic_state == "believed"
    assert seg.belief > 0
    assert seg.evidence_weight > 1.0


def test_strongly_refuted_segment_is_refuted():
    seg = make_segment(traversals=0, refutations=100)
    assert seg.epistemic_state == "refuted"
    assert seg.belief < 0
    assert seg.evidence_weight > 1.0


def test_contested_segment_is_distinguishable_from_fresh():
    """The bug fix: T=100, R=100 must NOT be confused with T=0, R=0.

    Both have strength == 1.0 (the symmetric formula cancels), but they
    are epistemically completely different states. The contested segment
    has high evidence_weight; the fresh one does not.
    """
    fresh = make_segment(traversals=0, refutations=0)
    contested = make_segment(traversals=100, refutations=100)

    # Both have the same strength under the symmetric formula
    assert abs(fresh.strength - contested.strength) < 1e-9

    # But they are NOT in the same epistemic state
    assert fresh.epistemic_state == "unknown"
    assert contested.epistemic_state == "contested"

    # And the evidence_weight is what distinguishes them
    assert fresh.evidence_weight < 1.0
    assert contested.evidence_weight > 1.0


def test_belief_direction_independent_of_weight():
    """A weakly-believed and a strongly-believed segment differ in evidence_weight, not belief sign."""
    weakly_believed = make_segment(traversals=2, refutations=0)
    strongly_believed = make_segment(traversals=200, refutations=0)
    assert weakly_believed.belief > 0
    assert strongly_believed.belief > 0
    assert strongly_believed.evidence_weight > weakly_believed.evidence_weight
