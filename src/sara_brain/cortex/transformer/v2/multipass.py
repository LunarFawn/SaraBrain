"""Multi-pass extraction filters for richer substrate coverage.

Three passes over the same document, each with a different focus:

  Pass 1 (definitions):   X is_a Y — what IS this thing?
  Pass 2 (relationships): X produces/requires/contains Y — what does it DO?
  Pass 3 (bridges):       connects concepts already in the substrate

The filter functions take a list of extracted triples and return the
subset appropriate for that pass. Pass 3 requires a brain reference
to check substrate membership.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sara_brain.core.brain import Brain

from .extractor_rules import Triple

# The relations that signal a definitional triple.
_DEFINITION_RELATIONS = frozenset({"be", "is_a", "is a"})


def filter_definitions(triples: list[Triple]) -> list[Triple]:
    """Pass 1: keep only definitional triples (X is Y)."""
    return [t for t in triples if t.relation in _DEFINITION_RELATIONS]


def filter_relationships(triples: list[Triple]) -> list[Triple]:
    """Pass 2: keep only action/relationship triples (non-definitional)."""
    return [t for t in triples if t.relation not in _DEFINITION_RELATIONS]


def filter_bridges(triples: list[Triple], brain: "Brain") -> list[Triple]:
    """Pass 3: keep triples where both subject and object already exist
    as neurons in the substrate. These are bridge facts — connecting
    known concepts for better wavefront traversal."""
    out = []
    for t in triples:
        subj_neuron = brain.neuron_repo.resolve(t.subject, exact_only=True)
        obj_neuron = brain.neuron_repo.resolve(t.object, exact_only=True)
        if subj_neuron and obj_neuron:
            out.append(t)
    return out


PASS_NAMES = ("definitions", "relationships", "bridges")
