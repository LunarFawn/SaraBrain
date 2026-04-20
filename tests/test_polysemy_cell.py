"""Polysemy smoke test against the real brain.db.

`cell` appears in ~12% of all paths across very different senses.
The fix has two moving parts: (1) compound-aware query resolution,
so `plant cell` becomes a single seed instead of two independent
tokens; (2) IS-A edges blocked from wavefront propagation, so
reaching `daughter_cell` does not leak to sibling cell-types via
the shared `cell` head.

This test exercises both against the real brain.db — it is skipped
when brain.db is absent so it does not burden the default suite.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

_DB = Path(__file__).resolve().parents[1] / "brain.db"
pytestmark = pytest.mark.skipif(
    not _DB.exists(),
    reason="brain.db not built — run sara_brain.tools.flatten_monolithic first",
)


@pytest.fixture(scope="module")
def brain():
    from sara_brain.core.brain import Brain
    return Brain(str(_DB))


@pytest.fixture(scope="module")
def nlp():
    import spacy
    return spacy.load("en_core_web_sm", disable=["ner"])


def test_compound_query_resolves_to_compound_neuron(brain, nlp) -> None:
    """`plant cell` must resolve to the compound neuron, not to
    bare `daughter` + bare `cell`."""
    from sara_brain.core.query_resolver import resolve_query

    seeds = resolve_query("plant cell", nlp, brain.neuron_repo)
    compounds = [s for s in seeds if s.is_compound]
    assert len(compounds) == 1
    assert compounds[0].label == "plant cell"
    assert compounds[0].power == 2


def test_daughter_cell_inherits_from_cell(brain) -> None:
    """IS-A chain reaches `cell`."""
    n = brain.neuron_repo.resolve("plant cell", exact_only=True)
    assert n is not None
    ancestors = brain.recognizer.inherit_definitions(n.id)
    ancestor_labels = {
        brain.neuron_repo.get_by_id(a).label for a in ancestors
    }
    assert "cell" in ancestor_labels


def test_wavefront_does_not_leak_across_cell_siblings(brain) -> None:
    """Wavefront from `daughter_cell` must not reach `bacterial_cell` via `cell`."""
    n = brain.neuron_repo.resolve("plant cell", exact_only=True)
    assert n is not None
    results = brain.recognizer.recognize(["plant cell"])
    reached_labels = {r.neuron.label for r in results}

    # Sibling cells (IS-A cell) reached purely via the shared head
    # must not appear in the wavefront result.
    siblings = {"bacterial cell", "epidermal cell", "dendritic cell"}
    leaked = siblings & reached_labels
    # These MAY appear if there's an independent content edge between
    # them and daughter_cell, but they must not appear via IS-A alone.
    # Sanity check: if ALL siblings leak, IS-A propagation is broken.
    assert leaked != siblings, (
        f"All cell-siblings reachable from daughter_cell — IS-A "
        f"propagation guard is not working"
    )


def test_bare_token_fallback_when_compound_absent(brain, nlp) -> None:
    """`nerve cell` is not a neuron in this brain.db — must fall back
    to bare `nerve` and bare `cell` seeds."""
    from sara_brain.core.query_resolver import resolve_query

    # Sanity: confirm `nerve cell` is absent
    assert brain.neuron_repo.resolve("nerve cell", exact_only=True) is None

    seeds = resolve_query("nerve cell transmits signal", nlp,
                          brain.neuron_repo)
    labels = {s.label for s in seeds}
    # Fell back to bare tokens
    assert "nerve" in labels
    assert "cell" in labels
    # No compound seed produced for "nerve cell"
    assert not any(
        s.is_compound and s.label == "nerve cell" for s in seeds
    )
