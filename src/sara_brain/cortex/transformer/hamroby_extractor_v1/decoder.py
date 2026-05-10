"""Decoder — BIO labels + word array → (subject, relation, object) triples.

The decoder reads tags at word level and slices the original word
array (the conveyor belt) by index. Every output span is a verbatim
substring of the input — no fragmentation, no reconstruction, no
subword reassembly. The model never saw the words in `parsed.words`;
the decoder reads them straight off.

For list-object patterns (one subject + one relation + multiple
objects), emit one Triple per object span. Single-object clauses
produce a single Triple. Partial trios (missing role) yield zero
triples.
"""
from __future__ import annotations

from dataclasses import dataclass

from .feature_extractor import ParsedSentence
from .vocab import (
    TAG_B_O, TAG_B_R, TAG_B_S, TAG_I_O, TAG_I_R, TAG_I_S,
)


@dataclass
class ExtractedTriple:
    subject: str
    relation: str
    object: str
    subject_word_span: tuple[int, int]   # [start, end) word indices
    relation_word_span: tuple[int, int]
    object_word_span: tuple[int, int]


def _collect_all_spans(tags: list[int], start_b: int, cont_i: int
                       ) -> list[tuple[int, int]]:
    """Lenient BIO decoding.

    Rules:
      - B-tag opens a span (closes any open one).
      - I-tag with no open span ALSO opens a span — handles the
        common case where a partially-trained tagger emits I-X
        without the preceding B-X. Standard fix in NER literature
        (e.g. flair, spaCy NER).
      - I-tag with open span extends it.
      - Anything else closes the open span.
    """
    spans: list[tuple[int, int]] = []
    in_span = False
    s_start = -1
    s_end = -1
    for i, t in enumerate(tags):
        if t == start_b:
            if in_span:
                spans.append((s_start, s_end))
            in_span = True
            s_start = i
            s_end = i + 1
        elif t == cont_i:
            if in_span:
                s_end = i + 1
            else:
                # Orphan I — treat as B (lenient decoding).
                in_span = True
                s_start = i
                s_end = i + 1
        elif in_span:
            spans.append((s_start, s_end))
            in_span = False
    if in_span:
        spans.append((s_start, s_end))
    return spans


def _slice_words(words: list[str], span: tuple[int, int]) -> str:
    """Return the verbatim substring of `words[start:end]` joined by
    a single space. The conveyor belt is read here — directly from the
    input word list, never reconstructed from any model output."""
    return " ".join(words[span[0]:span[1]])


def decode(parsed: ParsedSentence, tags: list[int]) -> list[ExtractedTriple]:
    """Decode word-level BIO tags into Triples by slicing parsed.words.

    Args:
      parsed: the ParsedSentence the tags were predicted for.
      tags: per-word BIO label ids, len(tags) == len(parsed.words).

    Returns:
      Zero or more ExtractedTriple. First subject + first relation are
      paired with each object span (handles list-object patterns).
    """
    if len(tags) != len(parsed.words):
        return []

    s_spans = _collect_all_spans(tags, TAG_B_S, TAG_I_S)
    r_spans = _collect_all_spans(tags, TAG_B_R, TAG_I_R)
    o_spans = _collect_all_spans(tags, TAG_B_O, TAG_I_O)
    if not (s_spans and r_spans and o_spans):
        return []

    s_span = s_spans[0]
    r_span = r_spans[0]
    s_text = _slice_words(parsed.words, s_span)
    r_text = _slice_words(parsed.words, r_span)

    triples: list[ExtractedTriple] = []
    for o_span in o_spans:
        triples.append(ExtractedTriple(
            subject=s_text,
            relation=r_text,
            object=_slice_words(parsed.words, o_span),
            subject_word_span=s_span,
            relation_word_span=r_span,
            object_word_span=o_span,
        ))
    return triples


__all__ = ["ExtractedTriple", "decode"]
