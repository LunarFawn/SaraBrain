"""IS-A edges must not be traversed by wavefronts.

A wavefront arriving at `nerve_cell` via content edges must not leak
out through the IS-A edge to `cell` and onward to sibling specializations
like `battery_cell`. That would re-introduce the polysemy-by-shared-
head-noun problem the hierarchy is meant to solve. IS-A is
inheritance-only, reached via `Recognizer.inherit_definitions()`.
"""
from __future__ import annotations

from sara_brain.core.brain import Brain
from sara_brain.models.neuron import Neuron, NeuronType
from sara_brain.models.segment import Segment


def _add_node(brain: Brain, label: str) -> int:
    n = brain.neuron_repo.create(
        Neuron(id=None, label=label, neuron_type=NeuronType.CONCEPT)
    )
    return n.id


def _add_segment(brain: Brain, src: int, tgt: int, relation: str) -> None:
    brain.segment_repo.create(
        Segment(id=None, source_id=src, target_id=tgt,
                relation=relation, strength=1.0)
    )


def test_is_a_not_traversed_by_wavefront() -> None:
    """A wavefront at `nerve_cell` must NOT reach `battery_cell` via `cell`."""
    brain = Brain(":memory:")

    # Build: nerve_cell → cell ← battery_cell, nerve_cell carries a unique
    # content edge to `signaling`; battery_cell carries a content edge to
    # `charging`.
    nerve_cell = _add_node(brain, "nerve_cell")
    cell = _add_node(brain, "cell")
    battery_cell = _add_node(brain, "battery_cell")
    signaling = _add_node(brain, "signaling")
    charging = _add_node(brain, "charging")

    _add_segment(brain, nerve_cell, cell, "is_a")
    _add_segment(brain, battery_cell, cell, "is_a")
    _add_segment(brain, nerve_cell, signaling, "does")
    _add_segment(brain, battery_cell, charging, "does")

    results = brain.recognizer.recognize(["nerve_cell"])
    reached_labels = {r.neuron.label for r in results}

    # `signaling` is reachable via content edge
    assert "signaling" in reached_labels, \
        "wavefront must follow non-IS-A content edges"
    # `cell` is NOT reachable because IS-A is filtered
    assert "cell" not in reached_labels, \
        "IS-A edge should not propagate"
    # `battery_cell` is NOT reachable — no cross-contamination
    assert "battery_cell" not in reached_labels, \
        "siblings via shared IS-A head must stay isolated"
    assert "charging" not in reached_labels, \
        "sibling content edges must not leak in via IS-A"


def test_inherit_definitions_walks_is_a() -> None:
    """inherit_definitions() DOES walk IS-A upward — that's its job."""
    brain = Brain(":memory:")

    nerve_cell = _add_node(brain, "nerve_cell")
    biological_cell = _add_node(brain, "biological_cell")
    cell = _add_node(brain, "cell")
    container = _add_node(brain, "container")

    _add_segment(brain, nerve_cell, biological_cell, "is_a")
    _add_segment(brain, biological_cell, cell, "is_a")
    _add_segment(brain, cell, container, "is_a")

    ancestors = brain.recognizer.inherit_definitions(nerve_cell)

    assert biological_cell in ancestors
    assert cell in ancestors
    assert container in ancestors
    # Order is nearest-first
    assert ancestors.index(biological_cell) < ancestors.index(cell)
    assert ancestors.index(cell) < ancestors.index(container)


def test_inherit_definitions_ignores_non_is_a() -> None:
    """Content edges (relation != 'is_a') must be ignored by inherit_definitions."""
    brain = Brain(":memory:")

    nerve_cell = _add_node(brain, "nerve_cell")
    signaling = _add_node(brain, "signaling")

    _add_segment(brain, nerve_cell, signaling, "does")

    ancestors = brain.recognizer.inherit_definitions(nerve_cell)
    assert signaling not in ancestors
    assert ancestors == []
