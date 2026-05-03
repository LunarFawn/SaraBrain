"""Template-based synthesizer — substrate facts -> prose, no LLM.

Two roles per v024 plan:

1. Inference path: parses the gathered (tool_call, result) list from
   StatelessReader and emits a coherent prose answer from the substrate
   edges directly — closes Sara's no-LLM-in-loop demo when paired with
   the cortex router.

2. Labeler path: walk a substrate edge list and emit one or more
   templated sentences. Used to generate (edge_list, prose) training
   pairs for the eventual neural synthesizer head.

Both share the per-relation template table — one source of truth for
how each edge type renders into English.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Edge:
    src: str
    rel: str
    tgt: str
    refuted: bool = False


# Pattern matches a line like:
#   'instrument' --[is_a]--> 'serena rna analysis tool_attribute'
#   'X' --[rel]--> 'Y' [REFUTED]
_EDGE_RE = re.compile(
    r"'(?P<src>[^']+)'\s*--\[(?P<rel>[^\]]+)\]-->\s*'(?P<tgt>[^']+)'(?P<flags>.*)"
)


def _strip_attr(label: str) -> str:
    return label.replace("_attribute", "")


def parse_edges_from_text(text: str) -> list[Edge]:
    edges: list[Edge] = []
    for line in text.splitlines():
        m = _EDGE_RE.search(line)
        if not m:
            continue
        edges.append(Edge(
            src=_strip_attr(m["src"]),
            rel=m["rel"],
            tgt=_strip_attr(m["tgt"]),
            refuted="[REFUTED]" in (m["flags"] or ""),
        ))
    return edges


def parse_edges_from_gathered(gathered: list[dict]) -> list[Edge]:
    """Pull all edges out of every gathered tool result. Deduped."""
    seen: set[tuple] = set()
    out: list[Edge] = []
    for fact in gathered:
        for e in parse_edges_from_text(fact.get("result", "")):
            key = (e.src, e.rel, e.tgt, e.refuted)
            if key in seen:
                continue
            seen.add(key)
            out.append(e)
    return out


# ── Per-relation rendering templates ──
# Each template fills (src, tgt) into a short English clause. Direction
# matters: "X --[is_a]--> Y" reads as "X is a Y" (subject-verb-object).

_TEMPLATES: dict[str, str] = {
    # identity / definition
    "is_a":              "{src} is a {tgt}",
    "is_an_instance_of": "{src} is an instance of {tgt}",
    "is_subsystem_of":   "{src} is a subsystem of {tgt}",
    "synonym_of":        "{src} is also known as {tgt}",
    "stands_for":        "{src} stands for {tgt}",
    "defined_as":        "{src} is defined as {tgt}",
    "means":             "{src} means {tgt}",

    # measurement
    "measures":          "{src} measures {tgt}",
    "measured_by":       "{src} is measured by {tgt}",
    "evaluates":         "{src} evaluates {tgt}",
    "assesses":          "{src} assesses {tgt}",
    "offers_metric":     "{src} offers a metric for {tgt}",

    # composition / relation
    "part_of":           "{src} is part of {tgt}",
    "incorporates":      "{src} incorporates {tgt}",
    "integrates":        "{src} integrates {tgt}",
    "leverages":         "{src} leverages {tgt}",
    "applies_to":        "{src} applies to {tgt}",
    "focuses_on":        "{src} focuses on {tgt}",
    "related_to":        "{src} is related to {tgt}",
    "analogous_to":      "{src} is analogous to {tgt}",
    "are_analogous_to":  "{src} are analogous to {tgt}",

    # ranges / values
    "optimal_ratio":         "the optimal ratio of {src} is {tgt}",
    "optimal_ratio_range":   "the optimal ratio range of {src} is {tgt}",
    "max_score_ratio":       "the maximum score ratio of {src} is {tgt}",
    "has_dual_mode_kdon":    "{src} has a dual-mode kdon of {tgt}",
}

_FALLBACK = "{src} {rel_pretty} {tgt}"


def _render_edge(e: Edge) -> str:
    tmpl = _TEMPLATES.get(e.rel)
    if tmpl is None:
        rel_pretty = e.rel.replace("_", " ")
        text = _FALLBACK.format(src=e.src, rel_pretty=rel_pretty, tgt=e.tgt)
    else:
        text = tmpl.format(src=e.src, tgt=e.tgt)
    if e.refuted:
        text = f"it is not the case that {text}"
    return text + "."


# Relations we never want to mention in prose (graph plumbing, not facts).
_NOISE_RELATIONS = {"describes"}


def render_edges(edges: list[Edge], topic: str | None = None) -> str:
    """Group edges by subject and render each cluster as a short sentence
    list. If `topic` is given, lift edges where the topic is the subject
    to the front."""
    edges = [e for e in edges if e.rel not in _NOISE_RELATIONS]
    if not edges:
        return ""

    by_src: dict[str, list[Edge]] = defaultdict(list)
    for e in edges:
        by_src[e.src].append(e)

    ordered_keys = list(by_src.keys())
    if topic:
        topic_l = topic.lower().strip()
        ordered_keys.sort(key=lambda k: 0 if topic_l in k.lower() else 1)

    sentences: list[str] = []
    for src in ordered_keys:
        rendered = [_render_edge(e) for e in by_src[src]]
        # Capitalize the first letter of the first sentence in this cluster.
        if rendered:
            rendered[0] = rendered[0][0].upper() + rendered[0][1:]
        sentences.extend(rendered)
    return " ".join(sentences)


def synthesize(question: str, gathered: list[dict]) -> str:
    """Top-level: given the question and gathered facts, produce a prose
    answer using only the substrate edges (no model knowledge)."""
    edges = parse_edges_from_gathered(gathered)
    if not edges:
        # Surface non-edge results verbatim so honest "no neuron matching"
        # / "no fuzzy matches" answers still show.
        bodies = [
            fact.get("result", "").strip()
            for fact in gathered if fact.get("result")
        ]
        return "\n".join(b for b in bodies if b) or (
            "Sara's substrate has nothing to say about this question."
        )

    topic = _topic_hint_from_question(question)
    body = render_edges(edges, topic=topic)
    if not body:
        return "Sara's substrate has nothing to say about this question."
    return body


def _topic_hint_from_question(question: str) -> str | None:
    q = question.lower().strip().rstrip("?.!")
    for prefix in ("what is the ", "what is a ", "what is ",
                   "tell me about ", "what do you know about ",
                   "describe ", "explain ", "define "):
        if q.startswith(prefix):
            return q[len(prefix):].strip() or None
    return None


__all__ = [
    "Edge", "parse_edges_from_text", "parse_edges_from_gathered",
    "render_edges", "synthesize",
]
