"""Unit test for compound-aware query resolution against brain.db.

`resolve_query("daughter cell", ...)` must emit a single compound
seed bound to the `daughter cell` neuron — not two bare-token seeds
for `daughter` and `cell`. This is the mechanism that makes the
wavefront model's "collapse at the compound" concrete: the
collapse happens at resolution time, before propagation.

Skipped when brain.db is absent or doesn't have a `daughter cell`
neuron to resolve against.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_DB = Path(__file__).resolve().parents[1] / "brain.db"
pytestmark = pytest.mark.skipif(
    not _DB.exists(),
    reason="brain.db not built",
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
    from sara_brain.core.query_resolver import resolve_query

    if brain.neuron_repo.resolve("daughter cell", exact_only=True) is None:
        pytest.skip("`daughter cell` not in brain.db")

    seeds = resolve_query("daughter cell", nlp, brain.neuron_repo)
    compounds = [s for s in seeds if s.is_compound]
    assert len(compounds) == 1
    assert compounds[0].label == "daughter cell"
    assert compounds[0].power == 2
