"""Rule-based argument extractor for the router.

Given (question, predicted_tool, parsed_doc), pull out the tool's
arguments — concept / type / label / term — using the spaCy parse and
optional substrate lookup. This is the deterministic counterpart to the
neural classifier head: it doesn't learn anything, but it knows the
substrate's vocabulary (so it can choose substrate-correct labels and
apply the compound-concept rule).

No model state lives here — everything is rules + substrate queries.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

# These match the wh-words and stop-words we need to strip from arguments.
_WH_WORDS = {
    "what", "what's", "whats", "who", "who's", "whose", "whom",
    "where", "when", "why", "how", "which",
}
_STOP_PREFIX = {
    "tell", "give", "show", "explain", "define", "describe",
    "do", "does", "did", "is", "are", "was", "were",
    "the", "a", "an", "me", "you", "us", "about",
    "anything", "something", "thing", "have",
    "i", "we", "they", "of", "for", "to", "from", "in", "on",
    "know", "means",
}
# Tokens that mark argument boundaries; strip them as fillers ONLY after
# the marker-based split logic has had a chance to use them.
_MARKER_FILLERS = {"called", "like", "mean"}
_QUANTITY_HINTS = {"highest", "lowest", "max", "maximum", "min", "minimum",
                   "best", "worst", "biggest", "smallest"}


@dataclass
class ArgResolution:
    args: dict
    confidence: float       # 0..1; rule strength
    rationale: str          # why these args (for debugging)


class SubstrateIndex:
    """Lazy lookup over the brain's neuron labels.

    Used to decide between "spider" (raw word) vs "ssng1 highest kdoff"
    (substrate-canonical compound label) when both could plausibly be
    the right `concept` for a brain_value call.
    """

    def __init__(self, db_path: Path | None):
        self.labels: set[str] = set()
        self.relations: set[str] = set()
        if db_path is None:
            return
        conn = sqlite3.connect(str(db_path))
        for (label,) in conn.execute("SELECT label FROM neurons"):
            self.labels.add(label.lower().strip())
        for (rel,) in conn.execute("SELECT DISTINCT relation FROM segments"):
            self.relations.add(rel.lower().strip())
        conn.close()

    def has(self, label: str) -> bool:
        return label.lower().strip() in self.labels

    def best_compound(self, base: str, quantity: str | None, type_word: str | None) -> str | None:
        """Try compound labels in priority order; return first hit."""
        base = base.strip().lower()
        type_word = (type_word or "").strip().lower()
        candidates = []
        if quantity and type_word:
            candidates.append(f"{base} {quantity} {type_word}")
            candidates.append(f"{quantity} {base} {type_word}")
        if type_word:
            candidates.append(f"{base} {type_word}")
        for c in candidates:
            if c in self.labels:
                return c
        return None


def _clean(span: str) -> str:
    s = " ".join(span.lower().split())
    s = s.strip(" ,.;:!?\"'")
    # Repair "word - word" → "word-word" (spaCy splits on hyphen).
    s = s.replace(" - ", "-")
    return s


def _is_filler(word: str) -> bool:
    w = word.lower()
    return w in _STOP_PREFIX or w in _WH_WORDS


def _content_span(doc) -> list[str]:
    """Strip leading wh-words and fillers; keep the rest as the content
    span (the thing the question is about)."""
    tokens = [t.text for t in doc if t.pos_ != "PUNCT"]
    while tokens and _is_filler(tokens[0]):
        tokens.pop(0)
    while tokens and _is_filler(tokens[-1]):
        tokens.pop()
    return tokens


def _extract_quantity(tokens: list[str]) -> tuple[str | None, list[str]]:
    """If a quantity hint word leads the span, peel it off. Returns
    (quantity, remaining_tokens)."""
    if tokens and tokens[0].lower() in _QUANTITY_HINTS:
        return tokens[0].lower(), tokens[1:]
    return None, tokens


def _split_on(tokens: list[str], marker: str) -> tuple[list[str], list[str]] | None:
    """Split tokens at the first occurrence of `marker`. Returns
    (before, after) or None if not present."""
    for i, t in enumerate(tokens):
        if t.lower() == marker:
            return tokens[:i], tokens[i + 1:]
    return None


def extract_args(
    question: str,
    tool: str,
    nlp,
    index: SubstrateIndex | None = None,
) -> ArgResolution:
    """Pull tool arguments from a question. spaCy parse drives the rules;
    substrate index (if given) refines `concept` for brain_value via the
    compound-label rule from the original llama prompt."""
    doc = nlp(question)
    tokens = _content_span(doc)
    rationale_parts: list[str] = []

    if tool == "brain_define":
        concept = _clean(" ".join(tokens))
        rationale_parts.append("strip wh/fillers, take rest as concept")
        return ArgResolution({"concept": concept}, 0.9, "; ".join(rationale_parts))

    if tool == "brain_explore":
        # Cut at "about" if present, since the explore templates often say
        # "tell me about X" / "what do you know about X".
        about = _split_on(tokens, "about")
        span = about[1] if about else tokens
        label = _clean(" ".join(span))
        rationale_parts.append("split on 'about' if present, label = remainder")
        return ArgResolution(
            {"label": label, "depth": 1}, 0.9, "; ".join(rationale_parts),
        )

    if tool == "brain_did_you_mean":
        # Marker-based splits MUST run on the unstripped token stream so
        # "did you mean X" doesn't lose the "mean" marker to filler-strip.
        raw_tokens = [t.text for t in doc if t.pos_ != "PUNCT"]
        for marker in ("mean", "called", "like"):
            sp = _split_on(raw_tokens, marker)
            if sp and sp[1]:
                rest = [t for t in sp[1] if t.lower() not in _STOP_PREFIX]
                term = _clean(" ".join(rest))
                if term:
                    rationale_parts.append(f"split on '{marker}', term = right side")
                    return ArgResolution(
                        {"term": term}, 0.9, "; ".join(rationale_parts),
                    )
        # Fallback: whole content span is the candidate
        return ArgResolution(
            {"term": _clean(" ".join(tokens))}, 0.5,
            "fallback: whole span as term",
        )

    if tool == "brain_value":
        # Templates split on "of" or "for": "what is the {type} of {c}",
        # "highest {type} for {c}". We split on either.
        rest = tokens
        quantity, rest = _extract_quantity(rest)
        if quantity:
            rationale_parts.append(f"quantity hint '{quantity}'")

        sp = _split_on(rest, "of") or _split_on(rest, "for")
        if sp:
            type_tokens, concept_tokens = sp
            type_word = _clean(" ".join(type_tokens))
            concept = _clean(" ".join(concept_tokens))
            rationale_parts.append("split on 'of'/'for', type=left, concept=right")
        else:
            # Fallback: assume "{c}'s {type}" or first-word-is-type form
            if rest and rest[-1].endswith("'s"):
                # weird, skip
                concept = _clean(" ".join(rest[:-1]))
                type_word = ""
            else:
                concept, type_word = _clean(" ".join(rest)), ""
                rationale_parts.append("no 'of'/'for' split; whole span as concept")

        # Substrate-aware compound rule: if the substrate has a concept like
        # "{concept} {quantity} {type}", prefer that as the concept and
        # drop type from the call (the substrate label encodes both).
        if index is not None:
            compound = index.best_compound(concept, quantity, type_word)
            if compound:
                rationale_parts.append(f"compound substrate label '{compound}'")
                return ArgResolution({"concept": compound}, 0.95,
                                     "; ".join(rationale_parts))

        args: dict = {"concept": concept}
        if type_word:
            args["type"] = type_word
        return ArgResolution(args, 0.85, "; ".join(rationale_parts))

    raise ValueError(f"unknown tool: {tool}")


__all__ = ["ArgResolution", "SubstrateIndex", "extract_args"]
