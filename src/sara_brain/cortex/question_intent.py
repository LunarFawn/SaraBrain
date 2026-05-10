"""Question-intent classifier — picks retrieval shape per question type.

Routing layer asks: *what is X* and *how does X work* are different
questions; their answers live in different shapes of the substrate
graph. A definitional answer is in 1-hop edges around the topic; a
mechanism answer is a chain through causal edges, multiple hops out.
This module classifies the question and dictates a `RetrievalShape`
(depth + relation bias) that the router consumes.

Bias is a *ranking* hint, never a filter. Sara's design holds that
associative noise is signal; non-bias edges still come back, they
just sort below bias-matched edges so the synth reads the answer first.

Reused vocabularies:
  - ``_COP_REL_WORDS`` / ``_CAUSAL_REL_WORDS`` / ``_TEMPORAL_REL_WORDS``
    from :mod:`sara_brain.teaching.openie` (relation-word sets used by
    the OpenIE sensor classifier).
  - ``is_comprehensive_intent`` from
    :mod:`sara_brain.cortex.transformer.dig` (detects "tell me
    everything about X" shape).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from sara_brain.cortex.transformer.dig import is_comprehensive_intent
from sara_brain.teaching.openie import (
    _CAUSAL_REL_WORDS,
    _COP_REL_WORDS,
    _TEMPORAL_REL_WORDS,
)


class QuestionIntent(Enum):
    DEFINITIONAL = "definitional"
    MECHANISM = "mechanism"
    ASSOCIATIVE = "associative"
    RELATIONAL = "relational"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RetrievalShape:
    depth: int
    relation_bias: frozenset[str]
    bidirectional: bool


# Substrate-observed mechanism relations beyond the OpenIE causal set.
# Sourced from the aptamer_rev1 teach scripts; extends the bias list
# for actual relations users have taught (e.g. ``act_as``, ``overcome``,
# ``provide``, ``defined_by``).
_SUBSTRATE_MECHANISM_RELATIONS = frozenset({
    "act_as", "acts_as",
    "overcome", "overcomes",
    "provide", "provides",
    "enable", "enables",
    "produce", "produces",
    "create", "creates",
    "trigger", "triggers",
    "lead_to", "leads_to",
    "generate", "generates",
    "drive", "drives",
    "result_in", "results_in",
    "prevent", "prevents",
    "reduce", "reduces",
    "defined_by",
    "describes_process",
    "contribute_to", "contributes_to",
    "cause_transitions_in", "causes_transitions_in",
})

_MECHANISM_REL_WORDS = _CAUSAL_REL_WORDS | _SUBSTRATE_MECHANISM_RELATIONS

_DEFINITIONAL_REL_WORDS = _COP_REL_WORDS | frozenset({
    "is_a", "instance_of", "kind_of", "type_of", "defined_as",
    "abbreviation_of", "called", "means",
})

# "how does X work" → MECHANISM only when the second word is a real
# manner/action verb. Guards "how big is X" / "how tall" from being
# misread as mechanism questions.
_HOW_MANNER_VERBS = frozenset({
    "does", "do", "did",
    "can", "could", "might",
    "is", "are",
    "work", "works", "working",
    "function", "functions",
    "operate", "operates",
})

# Mechanism-noun trigger when no WH-word is present (e.g. "fulcrum
# mechanism please" → MECHANISM).
_MECHANISM_NOUNS = frozenset({
    "mechanism", "process", "pipeline", "pathway", "chain",
    "operation", "function", "workings",
})

_PUNCT_RE = re.compile(r"[?.!,;:]+")
_WHITESPACE_RE = re.compile(r"\s+")

# MECHANISM > ASSOCIATIVE > RELATIONAL > DEFINITIONAL > UNKNOWN.
# Compound-question dispatch picks the strongest intent across clauses.
_INTENT_PRIORITY = {
    QuestionIntent.UNKNOWN: 0,
    QuestionIntent.DEFINITIONAL: 1,
    QuestionIntent.RELATIONAL: 2,
    QuestionIntent.ASSOCIATIVE: 3,
    QuestionIntent.MECHANISM: 4,
}


def _tokenize(question: str) -> list[str]:
    cleaned = _PUNCT_RE.sub(" ", question.lower())
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned.split() if cleaned else []


def _classify_clause(tokens: list[str]) -> QuestionIntent:
    if not tokens:
        return QuestionIntent.UNKNOWN
    first = tokens[0]
    second = tokens[1] if len(tokens) > 1 else ""

    if first == "how":
        if second in _HOW_MANNER_VERBS:
            return QuestionIntent.MECHANISM
        # "how big is X" / "how tall" / etc. — manner-of-being, not
        # mechanism. Fall through to the mechanism-noun check, then
        # to UNKNOWN.
    if first == "why":
        return QuestionIntent.MECHANISM

    if any(t in _MECHANISM_NOUNS for t in tokens):
        return QuestionIntent.MECHANISM

    if first in {"what", "who", "where"}:
        return QuestionIntent.DEFINITIONAL
    if first in {"which", "when"}:
        return QuestionIntent.RELATIONAL

    return QuestionIntent.UNKNOWN


def classify(question: str) -> QuestionIntent:
    """Classify a question by retrieval-shape intent.

    Order of precedence: comprehensive ("tell me everything") wins
    over wh-classification because that phrase often appears wrapped
    around a what/who question but signals the user wants breadth
    rather than a focused answer. Then per-clause classification with
    compound-aware dispatch.
    """
    if not question:
        return QuestionIntent.UNKNOWN

    if is_comprehensive_intent(question):
        return QuestionIntent.ASSOCIATIVE

    # Compound questions: split on " and ", classify each clause,
    # take the strongest intent. "what is X and how does it work"
    # → MECHANISM (stronger than DEFINITIONAL).
    clauses = [c.strip() for c in re.split(r"\s+and\s+", question.lower()) if c.strip()]
    if len(clauses) > 1:
        best = QuestionIntent.UNKNOWN
        for clause in clauses:
            intent = _classify_clause(_tokenize(clause))
            if _INTENT_PRIORITY[intent] > _INTENT_PRIORITY[best]:
                best = intent
        return best

    return _classify_clause(_tokenize(question))


_SHAPE_TABLE: dict[QuestionIntent, RetrievalShape] = {
    QuestionIntent.DEFINITIONAL: RetrievalShape(
        depth=1,
        relation_bias=_DEFINITIONAL_REL_WORDS,
        bidirectional=True,
    ),
    QuestionIntent.MECHANISM: RetrievalShape(
        depth=2,
        relation_bias=_MECHANISM_REL_WORDS,
        bidirectional=True,
    ),
    QuestionIntent.ASSOCIATIVE: RetrievalShape(
        depth=3,
        relation_bias=frozenset(),
        bidirectional=True,
    ),
    QuestionIntent.RELATIONAL: RetrievalShape(
        depth=1,
        relation_bias=_TEMPORAL_REL_WORDS,
        bidirectional=False,
    ),
    QuestionIntent.UNKNOWN: RetrievalShape(
        depth=1,
        relation_bias=frozenset(),
        bidirectional=True,
    ),
}


def shape_for(intent: QuestionIntent) -> RetrievalShape:
    return _SHAPE_TABLE[intent]


def rank_edges_by_bias(
    edges: list[dict], bias: frozenset[str]
) -> list[dict]:
    """Stable-sort edges so bias-matched relations come first.

    Empty bias is identity. Order within each bucket is preserved.
    Each ``edge`` dict is expected to carry a ``"relation"`` key; rows
    without one rank as non-biased.
    """
    if not bias:
        return list(edges)
    indexed = list(enumerate(edges))
    indexed.sort(key=lambda pair: (
        0 if pair[1].get("relation") in bias else 1,
        pair[0],
    ))
    return [e for _, e in indexed]


# Edge-line regex copy: matches the `'src' --[rel]--> 'tgt'` shape that
# brain_explore emits in the formatted output. Duplicated locally so
# this module avoids a circular import with cortex/transformer/multihop.
_EDGE_LINE_RE = re.compile(
    r"""(?P<sq>['"])(?P<src>.+?)(?P=sq)"""
    r"""\s*--\[(?P<rel>[^\]]+)\]-->\s*"""
    r"""(?P<tq>['"])(?P<tgt>.+?)(?P=tq)"""
)


def apply_bias_to_edge_text(text: str, bias: frozenset[str]) -> str:
    """Re-order edge lines in a brain_explore result string by bias.

    Within each contiguous run of edge lines, bias-matched edges sort
    first; non-edge lines (headers, neuron listings, blank separators)
    keep their position. Empty bias is identity.

    The rewrite is line-level so the surrounding structure of the
    output (``Discovered at depth N:`` sub-headers, ``Neurons
    reachable...`` sections) survives unchanged.
    """
    if not bias or not text:
        return text
    out_lines: list[str] = []
    edge_buffer: list[str] = []

    def _flush() -> None:
        if not edge_buffer:
            return
        # Stable-sort: bias-matched first, original order preserved.
        indexed = list(enumerate(edge_buffer))
        indexed.sort(key=lambda pair: (
            0 if _line_matches_bias(pair[1], bias) else 1,
            pair[0],
        ))
        out_lines.extend(line for _, line in indexed)
        edge_buffer.clear()

    for line in text.splitlines():
        if _EDGE_LINE_RE.search(line):
            edge_buffer.append(line)
        else:
            _flush()
            out_lines.append(line)
    _flush()
    return "\n".join(out_lines)


def _line_matches_bias(line: str, bias: frozenset[str]) -> bool:
    m = _EDGE_LINE_RE.search(line)
    if m is None:
        return False
    return m["rel"].strip() in bias


__all__ = [
    "QuestionIntent",
    "RetrievalShape",
    "classify",
    "shape_for",
    "rank_edges_by_bias",
    "apply_bias_to_edge_text",
]
