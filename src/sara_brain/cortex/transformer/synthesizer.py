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


# ── L2-en-anchored article insertion (v032) ──
# Replaces v027 Wave 2 article heuristic. The determiner set is drawn
# from vocab_en's function-word allowlist so that vocab_en stays the
# single source of truth for what counts as a determiner. Applied
# only to bare-copula templates (`is`); templates with built-in
# articles (`is_a`, `is_an_instance_of`, ...) are unaffected.

_DETERMINERS: frozenset[str] = frozenset({
    # vocab_en determiner group
    "a", "an", "the", "this", "that", "these", "those",
    "some", "any", "every", "each", "no",
    # vocab_en pronoun group entries that function as determiners in English
    "his", "her", "its", "their",
})


def _validate_determiners_against_vocab_en() -> None:
    """Loud failure if `_DETERMINERS` and vocab_en drift apart."""
    from .vocab_en import EN_FUNCTION_WORD_SET
    missing = _DETERMINERS - EN_FUNCTION_WORD_SET
    if missing:
        raise RuntimeError(
            f"_DETERMINERS references words not in vocab_en: "
            f"{sorted(missing)} — sync vocab_en.ENGLISH_FUNCTION_WORDS"
        )


_validate_determiners_against_vocab_en()


_MASS_NOUNS: frozenset[str] = frozenset({
    # Generic English mass nouns
    "inertia", "information", "water", "energy", "force", "gravity",
    "mass", "data", "knowledge", "evidence", "feedback", "music",
    "weather", "advice", "money", "research", "love", "happiness",
    "sadness", "anger", "wisdom", "patience", "hope", "luck",
    "air", "light", "heat", "sound", "motion", "matter",
    # Domain-specific (RNA / structural biology) — extend as mis-fires surface
    "rna", "dna", "atp", "gtp", "selex",
})


_COMMON_ADJECTIVES: frozenset[str] = frozenset({
    "happy", "sad", "small", "big", "old", "new", "good", "bad",
    "warm", "cold", "hot", "fast", "slow", "high", "low", "easy",
    "hard", "soft", "loud", "quiet", "kind", "rich", "poor",
    "wet", "dry", "clean", "dirty", "free", "busy", "open", "real",
    "sure", "ready", "alive", "dead", "right", "wrong", "true", "false",
    "long", "short", "wide", "narrow", "deep", "shallow",
    "young", "main", "next", "last", "first", "best",
    "worst", "great", "fine", "full", "empty", "weak", "strong",
    "central", "key", "primary", "secondary", "early", "late",
})


_ADJECTIVE_SUFFIXES: tuple[str, ...] = ("ous", "ful", "able", "ible")


def _looks_plural(word: str) -> bool:
    w = word.lower()
    if not w.endswith("s"):
        return False
    return not w.endswith(("ss", "is", "us", "os"))


def _looks_like_adjective(word: str) -> bool:
    w = word.lower()
    if w in _COMMON_ADJECTIVES:
        return True
    return len(w) > 4 and any(w.endswith(s) for s in _ADJECTIVE_SUFFIXES)


def _maybe_article(target: str) -> str:
    """Return `'a '`, `'an '`, or `''` depending on whether `target`
    looks like a singular indefinite count noun that needs an article.

    Conservative — returns `''` on uncertainty (already-determined,
    mass noun, plural-shaped, adjective-shaped). Vowel onset gives
    `'an '`, otherwise `'a '`."""
    target = target.strip()
    if not target:
        return ""
    words = target.split()
    first = words[0].lower()
    last = words[-1].lower()
    if first in _DETERMINERS:
        return ""
    if first in _MASS_NOUNS:
        return ""
    if _looks_plural(last):
        return ""
    if _looks_like_adjective(first):
        return ""
    return "an " if first[0] in "aeiou" else "a "


# (rel, target_was_attribute) → which slot ('src' or 'tgt') is the
# noun-phrase slot that should receive an indefinite article. Only
# bare-copula relations are listed; templates with built-in articles
# (e.g. `is_a` whose template already says "is a {src}") are not.
_ARTICLE_SLOT: dict[tuple[str, bool], str] = {
    ("is", False): "tgt",   # _TEMPLATES["is"]: "{src} is {tgt}"
    ("is", True):  "src",   # _ATTR_TEMPLATES["is"]: "{tgt} is {src}"
}


def _render_edge(e: Edge) -> str:
    src, tgt = e.src, e.tgt

    # v032: insert indefinite article on the appropriate slot for
    # bare-copula templates only. Templates with built-in articles
    # are left untouched.
    article_slot = _ARTICLE_SLOT.get((e.rel, e.target_was_attribute))
    if article_slot:
        slot_value = src if article_slot == "src" else tgt
        article = _maybe_article(slot_value)
        if article:
            if article_slot == "src":
                src = f"{article}{src}"
            else:
                tgt = f"{article}{tgt}"

    if e.target_was_attribute and e.rel in _ATTR_TEMPLATES:
        text = _ATTR_TEMPLATES[e.rel].format(src=src, tgt=tgt)
    elif e.rel in _TEMPLATES:
        text = _TEMPLATES[e.rel].format(src=src, tgt=tgt)
    else:
        rel_pretty = e.rel.replace("_", " ")
        if e.target_was_attribute:
            text = _ATTR_FALLBACK.format(src=src, tgt=tgt, rel_pretty=rel_pretty)
        else:
            text = _FALLBACK.format(src=src, tgt=tgt, rel_pretty=rel_pretty)
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


# v040 — slot-aware predicate-phrase extraction. Re-used by both
# scripts/build_vocab_brain_en.py (to seed the vocab brain) and the
# slotted training-data renderer below. Strips `{src}`/`{tgt}`
# placeholders, returns whatever predicate phrase remains, collapses
# whitespace.
_TEMPLATE_PLACEHOLDER_RE = re.compile(r"\{(?:src|tgt)\}")


def extract_predicate_phrase(template: str) -> str:
    """`"{tgt} is a {src}"` -> `"is a"`. `"{src} measures {tgt}"` ->
    `"measures"`. For templates with placeholders interspersed in the
    predicate phrase (e.g. `"the optimal ratio of {tgt} is {src}"`),
    returns the concatenation `"the optimal ratio of is"` — which is
    NOT a substring of the original; callers must check before using
    str.replace."""
    phrase = _TEMPLATE_PLACEHOLDER_RE.sub("", template).strip()
    return re.sub(r"\s+", " ", phrase)


def _render_edge_slotted(
    e: Edge,
    content_map: dict[str, str],
    pred_map: dict[str, str],
) -> str:
    """Slotted version of `_render_edge` for v040 training data.
    Substitutes content (src/tgt) with `<Cn>` tokens from
    `content_map` and the predicate phrase with the `<Pn>` token from
    `pred_map`. If a slot mapping isn't available (cluster exceeded
    cap, or predicate phrase isn't substring-replaceable), falls back
    to the literal text — the model just sees the v039-style output
    for that edge."""
    src_repr = content_map.get(e.src.strip().lower(), e.src)
    tgt_repr = content_map.get(e.tgt.strip().lower(), e.tgt)
    pred_repr = pred_map.get(e.rel)

    def _try_slot_predicate(template: str) -> str:
        if pred_repr is None:
            return template
        phrase = extract_predicate_phrase(template)
        if phrase and phrase in template:
            return template.replace(phrase, pred_repr)
        return template  # complex template (placeholder mid-phrase) — leave literal

    if e.target_was_attribute and e.rel in _ATTR_TEMPLATES:
        text = _try_slot_predicate(_ATTR_TEMPLATES[e.rel]).format(
            src=src_repr, tgt=tgt_repr,
        )
    elif e.rel in _TEMPLATES:
        text = _try_slot_predicate(_TEMPLATES[e.rel]).format(
            src=src_repr, tgt=tgt_repr,
        )
    else:
        rel_pretty = pred_repr if pred_repr is not None else e.rel.replace("_", " ")
        if e.target_was_attribute:
            text = _ATTR_FALLBACK.format(src=src_repr, tgt=tgt_repr, rel_pretty=rel_pretty)
        else:
            text = _FALLBACK.format(src=src_repr, tgt=tgt_repr, rel_pretty=rel_pretty)
    if e.refuted:
        text = f"it is not the case that {text}"
    return text + "."


def render_edges_slotted(
    edges: list[Edge],
    content_map: dict[str, str],
    pred_map: dict[str, str],
    topic: str | None = None,
) -> str:
    """Slotted prose for v040 training data. Filters noise like
    `render_edges` does, but emits ONE SENTENCE PER EDGE (no
    same-subject combining) — the slot tokens make the combiner's
    verb-whitelist heuristic moot, and per-edge structure is the
    minimum viable training signal. Combining can return as a future
    polish slice if needed."""
    edges = [e for e in edges if e.rel not in _NOISE_RELATIONS]
    edges = [e for e in edges if e.src.strip().lower() not in _STOP_WORDS]
    edges = [e for e in edges if not _is_decomposition_part_of(e)]
    if not edges:
        return ""

    # Topic ordering: edges whose src or tgt contains the topic come first.
    if topic:
        topic_l = topic.lower().strip()
        edges = sorted(
            edges,
            key=lambda e: 0 if (
                topic_l in e.src.lower() or topic_l in e.tgt.lower()
            ) else 1,
        )

    sentences: list[str] = []
    for e in edges:
        s = _render_edge_slotted(e, content_map, pred_map)
        if s:
            sentences.append(s[0].upper() + s[1:])
    return " ".join(sentences)


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


# v047 slice A.3 — event-node aware extraction.

_EVENT_PREFIX = "event:"
_EVENT_BINDING_RELATIONS: frozenset[str] = frozenset({
    "event_subject", "event_action", "event_object",
    "event_location", "event_start", "event_end", "event_modifier",
})


def _is_event_label(label: str) -> bool:
    return label.strip().lower().startswith(_EVENT_PREFIX)


def find_event_references(edges: list[Edge]) -> set[str]:
    """Return the set of event-node labels referenced by these edges.

    Event references show up either as an event-subject side of an
    `event_*` binding edge, or as an unrelated edge into/out of an
    event node. Used by chat.py to auto-expand events before
    rendering."""
    refs: set[str] = set()
    for e in edges:
        if _is_event_label(e.src):
            refs.add(e.src.strip())
        if _is_event_label(e.tgt):
            refs.add(e.tgt.strip())
    return refs


def extract_event_renderings(
    edges: list[Edge],
) -> tuple[list[str], list[Edge]]:
    """Split edges into (event-prose strings, remaining edges).

    Detects clusters of `event_*` binding edges sharing an `event:`-
    prefixed subject. Each such cluster collapses into ONE readable
    prose sentence ("Alice walked to the cafe at downtown from 3pm
    to 5pm.") so the downstream synthesizer doesn't have to render
    individual binding edges as their own sentences.

    Edges that don't belong to event-node clusters pass through in
    `remaining_edges`."""
    by_event: dict[str, dict[str, str]] = {}
    remaining: list[Edge] = []
    for e in edges:
        src = e.src.strip()
        if src.lower().startswith(_EVENT_PREFIX) and e.rel in _EVENT_BINDING_RELATIONS:
            by_event.setdefault(src, {})[e.rel] = e.tgt.strip()
            continue
        remaining.append(e)
    # Drop "event_subject"-only stub edges (no full event bindings
    # available) — they'd otherwise leak as 'event:X event subject Y.'
    # in the rendered output. The bindings get re-added by the chat
    # layer's auto-expand step before this function runs in practice;
    # if they didn't, we suppress rather than emit the noise.
    remaining = [
        e for e in remaining
        if not (_is_event_label(e.src) or _is_event_label(e.tgt))
    ]

    if not by_event:
        return [], remaining

    # Render each event cluster as one bundled sentence.
    rendered: list[str] = []
    for event_label in sorted(by_event):
        b = by_event[event_label]
        subj = b.get("event_subject")
        act = b.get("event_action")
        if not subj or not act:
            # Incomplete event binding (caller didn't auto-expand).
            # Suppress rather than re-leak the binding edges as raw
            # 'event:X event subject Y.' noise — chat.py's
            # _expand_event_references handles the expansion when the
            # brain is available.
            continue
        obj = b.get("event_object")
        loc = b.get("event_location")
        start = b.get("event_start")
        end = b.get("event_end")
        mod = b.get("event_modifier")
        parts: list[str] = [subj]
        if mod:
            parts.append(mod)
        parts.append(act.replace("_", " "))
        if obj:
            parts.append(obj)
        if loc:
            parts.append(f"at {loc}")
        if start and end:
            parts.append(f"from {start} to {end}")
        elif start:
            parts.append(f"at {start}")
        elif end:
            parts.append(f"until {end}")
        sentence = " ".join(parts).strip()
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]
            if not sentence.endswith("."):
                sentence += "."
            rendered.append(sentence)
    return rendered, remaining


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

    # v047 A.3: extract event-node clusters and render them as bundled
    # sentences before the regular per-edge rendering picks them apart.
    event_prose, edges = extract_event_renderings(edges)

    topic = _topic_hint_from_question(question)
    body = render_edges(edges, topic=topic)
    parts = []
    if event_prose:
        parts.extend(event_prose)
    if body:
        parts.append(body)
    if not parts:
        return "Sara's substrate has nothing to say about this question."
    return " ".join(parts)


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
    "render_edges", "synthesize", "extract_event_renderings",
    "find_event_references",
]
