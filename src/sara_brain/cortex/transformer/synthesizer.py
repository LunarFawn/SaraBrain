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

from .dig import _STOP_WORDS


@dataclass
class Edge:
    src: str
    rel: str
    tgt: str
    refuted: bool = False
    target_was_attribute: bool = False
    """True when the original target neuron had the '_attribute' suffix.
    Sara's substrate uses the attribute pattern to encode 'src is the
    [rel]-thing of tgt' rather than 'src verb tgt'. The renderer needs
    this flag to know whether to invert."""


# Pattern matches a line like:
#   'instrument' --[is_a]--> 'serena rna analysis tool_attribute'
#   'X' --[rel]--> 'Y' [REFUTED]
# Either quote (single OR double) is accepted on either side; brain output
# switches to double quotes when the label itself contains an apostrophe
# (e.g. "newton's first law_attribute").
_EDGE_RE = re.compile(
    r"""(?P<sq>['"])(?P<src>.+?)(?P=sq)"""
    r"""\s*--\[(?P<rel>[^\]]+)\]-->\s*"""
    r"""(?P<tq>['"])(?P<tgt>.+?)(?P=tq)"""
    r"""(?P<flags>.*)"""
)


def _strip_attr(label: str) -> str:
    return label.replace("_attribute", "")


def parse_edges_from_text(text: str) -> list[Edge]:
    edges: list[Edge] = []
    for line in text.splitlines():
        m = _EDGE_RE.search(line)
        if not m:
            continue
        raw_tgt = m["tgt"]
        edges.append(Edge(
            src=_strip_attr(m["src"]),
            rel=m["rel"],
            tgt=_strip_attr(raw_tgt),
            refuted="[REFUTED]" in (m["flags"] or ""),
            target_was_attribute=raw_tgt.endswith("_attribute"),
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
# Two tables. Sara's substrate uses an "_attribute" target convention to
# encode "src is the [rel]-thing of tgt"; we need inverted templates for
# those edges. For active verb relations (measures / evaluates / ...)
# the substrate direction already reads naturally, so the standard
# template applies even when the target has the attribute suffix.

# Standard templates (subject-verb-object reading).
_TEMPLATES: dict[str, str] = {
    # measurement / activity (these read naturally even when target is _attribute)
    "measures":          "{src} measures {tgt}",
    "measured_by":       "{src} is measured by {tgt}",
    "evaluates":         "{src} evaluates {tgt}",
    "assesses":          "{src} assesses {tgt}",
    "offers_metric":     "{src} offers a metric for {tgt}",
    "incorporates":      "{src} incorporates {tgt}",
    "integrates":        "{src} integrates {tgt}",
    "leverages":         "{src} leverages {tgt}",
    "validate":          "{src} validates {tgt}",
    "validates":         "{src} validates {tgt}",
    "applies_to":        "{src} applies to {tgt}",
    "focuses_on":        "{src} focuses on {tgt}",
    "related_to":        "{src} is related to {tgt}",
    "analogous_to":      "{src} is analogous to {tgt}",
    "are_analogous_to":  "{src} are analogous to {tgt}",
    "synonym_of":        "{src} is also known as {tgt}",
    "stands_for":        "{src} stands for {tgt}",
    "defined_as":        "{src} is defined as {tgt}",
    "means":             "{src} means {tgt}",
    # copula / generic possession (added to avoid fallthrough to
    # "{tgt} has {rel_pretty} of {src}" producing "has is of" garbage)
    "is":                "{src} is {tgt}",
    "have":              "{src} has {tgt}",
    # composition (no attribute inversion needed)
    "part_of":           "{src} is part of {tgt}",
    "is_subsystem_of":   "{src} is a subsystem of {tgt}",
}

# Inverted templates: applied when the original target had the _attribute
# suffix. These read the substrate's direction the right way around for
# attributive relations (is_a / has / scored_by / value-bearing nouns).
_ATTR_TEMPLATES: dict[str, str] = {
    "is":                "{tgt} is {src}",
    "is_a":              "{tgt} is a {src}",
    "is_an_instance_of": "{tgt} is an instance of {src}",
    "has":               "{tgt} has {src}",
    "have":              "{tgt} has {src}",
    "scored_by":         "{tgt} is scored by {src}",
    "described_by":      "{tgt} is described by {src}",
    "also_known_as":     "{tgt} is also known as {src}",
    "abbreviation_of":   "{src} is an abbreviation of {tgt}",
    "abbreviation":      "{tgt}'s abbreviation is {src}",
    "states":            "{tgt} states that {src}",
    "expressed_as":      "{tgt} is expressed as {src}",
    "act_as":            "{src} acts as {tgt}",
    "acts_as":           "{src} acts as {tgt}",
    "applies_to":        "{tgt} applies to {src}",
    "provides_framework_for": "{tgt} provides a framework for {src}",
    "caused_by":         "{tgt} is caused by {src}",
    "results_in":        "{src} results in {tgt}",
    "indicates":         "{src} indicates {tgt}",
    "produces":          "{src} produces {tgt}",
    "associated_with":   "{src} is associated with {tgt}",
    "influence":         "{src} influences {tgt}",
    "simulate":          "{src} simulate {tgt}",
    "provide":           "{src} provide {tgt}",
    "drops_sharply_below": "{tgt} drops sharply below {src}",
    "requires_for_score_100": "{tgt} requires {src} for a score of 100",
    "highest_eterna_total_score": (
        "{tgt}'s highest eterna total score is achieved {src}"
    ),
    # value/range relations: src is the value, tgt is the concept it belongs to
    "optimal_ratio":       "the optimal ratio of {tgt} is {src}",
    "optimal_ratio_range": "the optimal ratio range of {tgt} is {src}",
    "max_score_ratio":     "the maximum score ratio of {tgt} is {src}",
    "has_dual_mode_kdon":  "{tgt} has a dual-mode kdon of {src}",
}

_FALLBACK = "{src} {rel_pretty} {tgt}"
_ATTR_FALLBACK = "{tgt} has {rel_pretty} of {src}"


def _render_edge(e: Edge) -> str:
    if e.target_was_attribute and e.rel in _ATTR_TEMPLATES:
        text = _ATTR_TEMPLATES[e.rel].format(src=e.src, tgt=e.tgt)
    elif e.rel in _TEMPLATES:
        text = _TEMPLATES[e.rel].format(src=e.src, tgt=e.tgt)
    else:
        rel_pretty = e.rel.replace("_", " ")
        if e.target_was_attribute:
            text = _ATTR_FALLBACK.format(src=e.src, tgt=e.tgt, rel_pretty=rel_pretty)
        else:
            text = _FALLBACK.format(src=e.src, tgt=e.tgt, rel_pretty=rel_pretty)
    if e.refuted:
        text = f"it is not the case that {text}"
    return text + "."


# Relations we never want to mention in prose (graph plumbing, not facts).
_NOISE_RELATIONS = {"describes"}


# Verbs the sentence-combiner recognises as a subject/predicate split
# point. Any leading words up to the first whitelisted verb count as
# the rendered subject; everything from the verb on is the predicate.
# Only verbs that appear (or could appear) at the start of a template
# predicate need to be here — extending this is safe and conservative
# (missing entries just disable combining for that template).
_PREDICATE_VERBS = {
    "is", "are", "was", "were", "has", "have", "had",
    "means", "stands", "acts", "applies", "focuses",
    "measures", "evaluates", "assesses", "leverages",
    "incorporates", "integrates", "validates", "validate",
    "indicates", "produces", "results", "influences",
    "simulate", "provide", "provides", "drops", "requires",
    "offers", "states",
}


def _is_decomposition_part_of(e: Edge) -> bool:
    """True when an edge looks like substrate-ingestion decomposition —
    a `part_of` edge whose source is a single content-word token of the
    target's label (e.g. `inertia` --part_of--> `inertia in rna`).
    These are tautological and add no information in prose."""
    if e.rel != "part_of":
        return False
    src = e.src.strip().lower()
    if not src or " " in src:
        return False
    tgt_tokens = {t for t in re.findall(r"[a-z0-9]+", e.tgt.lower())}
    return src in tgt_tokens


def _split_subj_pred(sentence: str) -> tuple[str, str] | None:
    """Split `Cat is a mammal.` into (`Cat`, `is a mammal.`). Returns
    None if no whitelisted verb appears after the first token."""
    body = sentence.rstrip(".").rstrip()
    if not body:
        return None
    words = body.split()
    for i in range(1, len(words)):
        if words[i].lower() in _PREDICATE_VERBS:
            return " ".join(words[:i]), " ".join(words[i:]) + "."
    return None


def _join_predicates(preds: list[str]) -> str:
    """Oxford-comma join: ['is a', 'has b', 'measures c'] → 'is a, has b,
    and measures c'."""
    cleaned = [p.rstrip(".").rstrip() for p in preds]
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return ", ".join(cleaned[:-1]) + ", and " + cleaned[-1]


def render_edges(edges: list[Edge], topic: str | None = None) -> str:
    """Render an edge list as prose. Filters substrate noise (graph
    plumbing relations, stop-word subjects, decomposition `part_of`
    edges), groups sentences by their rendered subject, and joins
    same-subject predicates into one sentence each.

    If `topic` is given, sentences whose subject contains the topic
    are lifted to the front."""
    edges = [e for e in edges if e.rel not in _NOISE_RELATIONS]
    # Drop edges whose subject is a bare stop word ("in", "of", ...) —
    # substrate ingestion emits is_part_of from every constituent token
    # of a multi-word label.
    edges = [e for e in edges if e.src.strip().lower() not in _STOP_WORDS]
    # Drop content-word decomposition edges (same noise pattern, just
    # with a non-stop-word constituent).
    edges = [e for e in edges if not _is_decomposition_part_of(e)]
    if not edges:
        return ""

    rendered: list[str] = []
    for e in edges:
        s = _render_edge(e)
        if s:
            rendered.append(s[0].upper() + s[1:])

    # Cluster by rendered subject so we can combine same-subject sentences.
    # Sentences whose subject can't be cleanly extracted stay standalone.
    by_subject: dict[str, list[str]] = defaultdict(list)
    subject_order: list[str] = []
    standalone: list[str] = []
    for s in rendered:
        sp = _split_subj_pred(s)
        if sp is None:
            standalone.append(s)
            continue
        subj, pred = sp
        if subj not in by_subject:
            subject_order.append(subj)
        by_subject[subj].append(pred)

    if topic:
        topic_l = topic.lower().strip()
        subject_order.sort(key=lambda k: 0 if topic_l in k.lower() else 1)

    out: list[str] = []
    for subj in subject_order:
        preds = by_subject[subj]
        out.append(f"{subj} {_join_predicates(preds)}.")
    out.extend(standalone)
    return " ".join(out)


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
