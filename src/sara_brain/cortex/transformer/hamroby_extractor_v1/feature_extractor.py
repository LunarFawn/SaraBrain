"""spaCy → per-word grammar-feature tuples + parallel surface array.

The encoder consumes ParsedSentence.feature_ids — a tensor of
(POS_id, dep_id, head_offset_id, funcword_id) per word. The decoder
later slices ParsedSentence.words by index to emit verbatim
substrings; the surface text never enters the model.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..router_data import CLEARNLP_TO_UD
from .vocab import (
    DEP_TO_ID, NONE_FUNCWORD_ID, POS_TO_ID,
    UNK_DEP_ID, UNK_POS_ID,
    encode_funcword, encode_head_offset,
)


# POS tags whose tokens may genuinely be function words. Anything
# outside this set (NOUN, PROPN, VERB, ADJ, ADV, NUM, INTJ, SYM, X)
# gets funcword = NONE regardless of surface form.
_CLOSED_CLASS_POS: frozenset[str] = frozenset({
    "ADP", "AUX", "CCONJ", "DET", "PART", "PRON", "SCONJ",
})


def _normalize_dep(spacy_dep: str) -> str:
    """spaCy's en_core_web_sm uses ClearNLP-style dep labels (prep,
    pobj, dobj, ...). Map them to the UD-canonical set our vocab knows.
    Subtypes like `nsubj:pass` are stripped."""
    base = spacy_dep.split(":", 1)[0]
    return CLEARNLP_TO_UD.get(base, base.lower())


# Tokenizer rule (per Jennifer's instinct, 2026-05-09): if there's no
# whitespace in it, treat it as a single word. This is the simplest
# possible rule and catches the vast majority of scientific notation
# (5'3', kdoff, K_d, mg/mL, ATP, p<0.05, 37°C, ...) without enumerating
# cases.
#
# Implementation: replace spaCy's default tokenizer with one whose
# only splitters are (a) whitespace and (b) trailing/leading sentence
# punctuation (period, comma, semicolon, colon, !, ?, parens, quotes).
# Internal punctuation — apostrophes, hyphens, slashes, underscores,
# equals signs, less-than/greater-than — is preserved as part of the
# token.
#
# Trade-off: spaCy's POS tagger and dep parser were trained on
# standard tokenization. When a domain compound like "5'3'" comes
# through as one token, the tagger may label it POS=X (unknown), and
# the parser may give it a degenerate dependency role. That's still
# better than fragmentation: the head's BIO classifier handles unknown
# POS gracefully, and the conveyor belt preserves the surface form
# for verbatim emission to Sara.

import re as _re


def _make_whitespace_first_tokenizer(nlp):
    """Build a spaCy Tokenizer per the "no whitespace → one word" rule.

    The only splitter is trailing sentence punctuation `.,;:!?` — and
    only at the end of a whitespace-bounded token. Apostrophes,
    quotes, parens, brackets, hyphens, slashes, underscores all stay
    attached because the user's rule is "no space inside means one
    word." Internal characters never trigger a split.
    """
    from spacy.tokenizer import Tokenizer
    suffix_re = _re.compile(r"[.,;:!?]+$")
    return Tokenizer(
        nlp.vocab,
        prefix_search=None,
        suffix_search=suffix_re.search,
        infix_finditer=None,
        token_match=None,
    )


# Per-process registry of paper-specific compound atomic tokens (which
# WOULD have whitespace in surface text but should still be one
# neuron label). These get post-tokenization merging — see usage.
_EXTRA_COMPOUND_TOKENS: list[str] = []


def register_compound_tokens(*tokens: str) -> None:
    """Register paper-specific compound terms that contain whitespace
    but should be merged into one token after tokenization (e.g.
    "molecular snare", "5'3' static stem"). Empty for now; populated
    by paper-specific glossary code if needed."""
    for tok in tokens:
        if tok and tok not in _EXTRA_COMPOUND_TOKENS:
            _EXTRA_COMPOUND_TOKENS.append(tok)


def register_domain_tokenizer_rules(nlp) -> None:
    """Replace spaCy's tokenizer with a whitespace-first one.
    Idempotent — the new tokenizer instance overwrites the old one
    each call."""
    nlp.tokenizer = _make_whitespace_first_tokenizer(nlp)


class _CascadeNLP:
    """spaCy-callable wrapper that runs a fast primary parser first and
    falls back to a more accurate parser when the primary produces a
    degenerate parse (no VERB or AUX token).

    Targets the failure mode we measured on `en_core_web_sm`: terse
    sentences with all-caps acronym subjects (e.g. "DNA and RNA share
    base pairing.") get no verb in the parse at all, breaking the
    extractor. en_core_web_trf handles those reliably. Most sentences
    parse fine with sm (5-10x faster) and never trigger the fallback.

    Both nlp instances share the same domain (whitespace-first)
    tokenizer, so token alignment is consistent across paths.
    """
    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback
        # Expose primary's tokenizer/vocab for callers that touch them.
        self.tokenizer = primary.tokenizer
        self.vocab = primary.vocab

    def __call__(self, text: str):
        doc = self.primary(text)
        has_pred = any(t.pos_ in ("VERB", "AUX") for t in doc)
        if has_pred:
            return doc
        return self.fallback(text)


def load_domain_nlp(
    model: str | None = None,
    *,
    disable=(),
    cascade: bool = True,
    primary_model: str = "en_core_web_sm",
    fallback_model: str = "en_core_web_trf",
):
    """Return a spaCy nlp with the whitespace-first tokenizer.

    Default behavior (`cascade=True`, `model=None`): load both
    `en_core_web_sm` (fast primary) and `en_core_web_trf` (accurate
    fallback), wrap them in a cascade. Most sentences hit only the
    sm path; degenerate parses (no VERB/AUX, e.g. "DNA and RNA share
    base pairing.") transparently retry on trf.

    Pass `model=<name>` (or `cascade=False`) to get a single-model
    nlp. Falls back gracefully to single-model if the fallback model
    isn't installed.
    """
    import spacy
    if model is not None or not cascade:
        chosen = model or primary_model
        nlp = spacy.load(chosen, disable=list(disable))
        register_domain_tokenizer_rules(nlp)
        return nlp
    primary = spacy.load(primary_model, disable=list(disable))
    register_domain_tokenizer_rules(primary)
    try:
        fallback = spacy.load(fallback_model, disable=list(disable))
    except OSError:
        # Fallback model not installed — degrade gracefully.
        return primary
    register_domain_tokenizer_rules(fallback)
    return _CascadeNLP(primary, fallback)


@dataclass
class ParsedSentence:
    """Output of the feature extractor.

    `words`: list of surface strings, one per word (excluding spaCy
        punctuation tokens by default but including function words).
        Used only by the decoder to slice verbatim spans.
    `feature_ids`: list of (pos_id, dep_id, offset_id, funcword_id)
        tuples, one per word, aligned with `words`.
    `char_offsets`: list of (start, end) character spans into the
        original text, aligned with `words`. Used to map char-level
        ground-truth spans (from synthetic_pairs) to per-word BIO
        labels at training time.
    """
    text: str
    words: list[str]
    feature_ids: list[tuple[int, int, int, int]]
    char_offsets: list[tuple[int, int]]


def parse_sentence(text: str, nlp, *, drop_punct: bool = True) -> ParsedSentence:
    """Run spaCy on `text` and emit the feature tuples and surface array.

    Punctuation is dropped by default — it carries no SVO signal and
    breaking on it cleans up downstream BIO tagging. Pass
    `drop_punct=False` to keep punctuation tokens.
    """
    doc = nlp(text)
    words: list[str] = []
    feature_ids: list[tuple[int, int, int, int]] = []
    char_offsets: list[tuple[int, int]] = []

    # Map original token-index in the doc -> output word-index, so we
    # can compute head_offset relative to the kept-words array (after
    # punctuation is dropped).
    keep_mask: list[bool] = []
    for tok in doc:
        keep_mask.append(not (drop_punct and tok.is_punct))
    doc_idx_to_word_idx: dict[int, int] = {}
    next_word_idx = 0
    for i, keep in enumerate(keep_mask):
        if keep:
            doc_idx_to_word_idx[i] = next_word_idx
            next_word_idx += 1

    for tok in doc:
        if not keep_mask[tok.i]:
            continue
        pos_id = POS_TO_ID.get(tok.pos_, UNK_POS_ID)
        dep_norm = _normalize_dep(tok.dep_)
        dep_id = DEP_TO_ID.get(dep_norm, UNK_DEP_ID)
        # Compute head offset relative to the OUTPUT word index. If
        # the head was a dropped punct, fall back to a self-pointing
        # offset (0) — the head signal is degenerate but recoverable.
        head_doc_idx = tok.head.i
        if head_doc_idx in doc_idx_to_word_idx:
            head_offset = (doc_idx_to_word_idx[head_doc_idx]
                           - doc_idx_to_word_idx[tok.i])
        else:
            head_offset = 0
        offset_id = encode_head_offset(head_offset)
        # Funcword stream: only encode for closed-class POS tags where
        # the token is genuinely a function word. Open-class POS tags
        # (NOUN, PROPN, VERB, ADJ, ADV, NUM, INTJ, SYM, X) get NONE
        # regardless of surface form — otherwise a proper noun like
        # "Do" in "Jeet Kune Do" leaks the funcword id for "do" the
        # auxiliary, biasing the BIO classifier.
        if tok.pos_ in _CLOSED_CLASS_POS:
            funcword_id = encode_funcword(tok.text.lower())
        else:
            funcword_id = NONE_FUNCWORD_ID

        words.append(tok.text)
        feature_ids.append((pos_id, dep_id, offset_id, funcword_id))
        char_offsets.append((tok.idx, tok.idx + len(tok.text)))

    return ParsedSentence(
        text=text,
        words=words,
        feature_ids=feature_ids,
        char_offsets=char_offsets,
    )


__all__ = ["ParsedSentence", "parse_sentence"]
