"""Adapter — synthetic_pairs.Pair records → grammar-feature training data.

Takes (prose, subject_span, relation_span, object_span) records produced
by `cortex/transformer/v2/synthetic_pairs.py` (which we keep for the
data generation logic; only the BPE-tokenizer-coupled training code
gets dropped) and re-encodes them at WORD level for the grammar
encoder.

Per record output:
  pos_ids:        list[int] per word
  dep_ids:        list[int] per word
  offset_ids:     list[int] per word
  funcword_ids:   list[int] per word
  bio_labels:     list[int] per word — derived from char spans
  words:          list[str] per word — the conveyor belt for decode

Char-to-word span mapping rule: a word receives a B/I tag from a span
if its character range intersects that span. The first word in a span
gets B; subsequent words get I.
"""
from __future__ import annotations

from dataclasses import dataclass

from .feature_extractor import ParsedSentence, parse_sentence
from .vocab import (
    TAG_B_O, TAG_B_R, TAG_B_S, TAG_I_O, TAG_I_R, TAG_I_S, TAG_O,
)


@dataclass
class GrammarTrainExample:
    words: list[str]
    pos_ids: list[int]
    dep_ids: list[int]
    offset_ids: list[int]
    funcword_ids: list[int]
    bio_labels: list[int]


def char_spans_to_word_bio(
    char_offsets: list[tuple[int, int]],
    subject_span: tuple[int, int],
    relation_span: tuple[int, int],
    object_span: tuple[int, int],
    additional_object_spans: list[tuple[int, int]] | None = None,
) -> list[int]:
    """Map character spans (from synthetic_pairs.Pair) to per-word BIO
    labels.

    Each span is treated as its own region — the first overlapping word
    gets the B-tag, subsequent overlapping words get the I-tag. The
    subject and relation spans are single; the object can have multiple
    regions (e.g. conjuncts of a conj-dobj like 'apples and oranges'),
    each producing its OWN B-O. That gives the decoder distinct B-O
    spans so it emits one triple per conjunct.
    """
    # Build the list of independent spans. Each entry is its own region
    # with its own b/i tags and its own emitted-flag (so multiple object
    # spans each emit their own B-O).
    spans: list[tuple[tuple[int, int], int, int]] = [
        (subject_span, TAG_B_S, TAG_I_S),
        (relation_span, TAG_B_R, TAG_I_R),
        (object_span, TAG_B_O, TAG_I_O),
    ]
    for extra in additional_object_spans or []:
        spans.append((extra, TAG_B_O, TAG_I_O))
    emitted = [False] * len(spans)
    out: list[int] = []
    for tok_start, tok_end in char_offsets:
        assigned = TAG_O
        for idx, ((sp_start, sp_end), b_tag, i_tag) in enumerate(spans):
            # Word's char range overlaps the labelled span.
            if tok_end > sp_start and tok_start < sp_end:
                if not emitted[idx]:
                    emitted[idx] = True
                    assigned = b_tag
                else:
                    assigned = i_tag
                break
        out.append(assigned)
    return out


def pair_to_example(pair, nlp) -> GrammarTrainExample | None:
    """Convert one synthetic_pairs.Pair into a GrammarTrainExample.
    Returns None if the parser produces no usable words for the prose.

    If the Pair carries a pre-computed ParsedSentence (e.g. UD-sourced
    real-prose pairs with gold UD features), use it directly and skip
    the spaCy re-parse. This avoids the ~46% conj-position POS noise
    spaCy produces on delexicalized real prose.
    """
    parsed: ParsedSentence = (
        getattr(pair, "pre_parsed", None)
        or parse_sentence(pair.prose, nlp)
    )
    if not parsed.words:
        return None
    bio = char_spans_to_word_bio(
        parsed.char_offsets,
        pair.subject_span, pair.relation_span, pair.object_span,
        additional_object_spans=getattr(pair, "additional_object_spans", None),
    )
    pos_ids, dep_ids, off_ids, fw_ids = zip(*parsed.feature_ids)
    return GrammarTrainExample(
        words=list(parsed.words),
        pos_ids=list(pos_ids),
        dep_ids=list(dep_ids),
        offset_ids=list(off_ids),
        funcword_ids=list(fw_ids),
        bio_labels=bio,
    )


def build_examples(
    pairs: list,
    nlp,
    max_seq: int = 128,
    *,
    progress_every: int = 5000,
    label: str = "examples",
) -> list[GrammarTrainExample]:
    """Run pair_to_example over a list of Pairs, drop too-long ones.

    Prints a heartbeat to stderr every `progress_every` pairs so a
    long feature-generation phase doesn't look frozen.
    """
    import sys
    import time
    out: list[GrammarTrainExample] = []
    started = time.time()
    n = len(pairs)
    for i, p in enumerate(pairs, start=1):
        ex = pair_to_example(p, nlp)
        if ex is not None and len(ex.words) <= max_seq - 2:
            out.append(ex)
        if progress_every and (i % progress_every == 0 or i == n):
            elapsed = time.time() - started
            rate = i / max(1e-3, elapsed)
            eta = (n - i) / rate if rate > 0 else 0.0
            print(
                f"[hamroby-extract] {label} {i}/{n} "
                f"({rate:.0f}/s, eta {eta:.0f}s, kept {len(out)})",
                file=sys.stderr,
            )
    return out


__all__ = [
    "GrammarTrainExample",
    "char_spans_to_word_bio",
    "pair_to_example",
    "build_examples",
]
