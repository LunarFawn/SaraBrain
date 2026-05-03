"""L2-en vocabulary — extends L1 structural vocab with English function words.

See [docs/v029_vocab_en_plan.md](../../../../docs/v029_vocab_en_plan.md) and
[docs/v028_multi_layer_cortex_architecture.md](../../../../docs/v028_multi_layer_cortex_architecture.md)
for the L1/L2/L3 architecture this fits into.

This module provides a *superset* of L1's vocabulary:
  - IDs 0..VOCAB_SIZE-1: inherited from L1 (vocab.py); every L1 token
    keeps its existing ID so L1 checkpoints stay loadable verbatim.
  - IDs VOCAB_SIZE.. : English closed-class function words appended in a
    fixed, stable order (never reorder; only append).

L1 checkpoints continue to load against vocab.py unchanged.
L2-en checkpoints will load against this module's `VOCAB_SIZE_EN`.
"""
from __future__ import annotations

from .vocab import (
    BOS_ID,
    EOS_ID,
    ID2TOK,
    PAD_ID,
    SEP_ID,
    TOK2ID,
    UNK_ID,
    VOCAB,
    VOCAB_SIZE,
)


# Closed-class English function words. Order is stable: appending new
# tokens at the end is safe; reordering invalidates L2-en checkpoints.
# Categories below are documentation only — flat tuple is the source of
# truth.
ENGLISH_FUNCTION_WORDS: tuple[str, ...] = (
    # Determiners (12)
    "a", "an", "the", "this", "that", "these", "those",
    "some", "any", "every", "each", "no",
    # Prepositions (19)
    "of", "in", "on", "at", "by", "for", "with", "to", "from", "as",
    "into", "over", "under", "between", "through", "against",
    "about", "before", "after",
    # Conjunctions (14)
    "and", "or", "but", "nor", "so", "yet",
    "because", "although", "if", "when", "while", "since",
    "unless", "until",
    # Auxiliaries (22)
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did",
    "can", "could", "will", "would", "shall", "should",
    "may", "might", "must",
    # Pronouns (15)
    "it", "they", "them", "their", "its",
    "which", "who", "whose", "whom",
    "he", "she", "him", "her", "his", "hers",
    # Negation / particles (10)
    "not", "n't", "also", "only", "just",
    "then", "now", "here", "there", "'s",
    # Quantifiers (7)
    "more", "less", "much", "many", "few", "several", "all",
)


# Sanity: no duplicates within the function-word list, and no overlap
# with L1 tokens. These run at import time so a bad edit fails fast
# rather than silently producing a degenerate vocabulary.
assert len(set(ENGLISH_FUNCTION_WORDS)) == len(ENGLISH_FUNCTION_WORDS), (
    "ENGLISH_FUNCTION_WORDS contains duplicates"
)
_overlap = set(ENGLISH_FUNCTION_WORDS) & set(VOCAB)
assert not _overlap, f"function words overlap with L1 vocab: {sorted(_overlap)}"
del _overlap


# Lookup helper used by ud.py to decide between literal form vs UPOS tag
# during lexicalized ingestion.
EN_FUNCTION_WORD_SET: frozenset[str] = frozenset(ENGLISH_FUNCTION_WORDS)


def is_english_function_word(token: str) -> bool:
    """True when `token` is in the L2-en function-word allowlist
    (case-insensitive). Used by `ud.py` to decide whether to keep a UD
    token's literal form or abstract it to its UPOS tag."""
    if not token:
        return False
    return token.lower() in EN_FUNCTION_WORD_SET


# The L2-en vocabulary: L1 tokens at their original IDs, then English
# function words appended above.
VOCAB_EN: list[str] = list(VOCAB) + list(ENGLISH_FUNCTION_WORDS)
TOK2ID_EN: dict[str, int] = {tok: i for i, tok in enumerate(VOCAB_EN)}
ID2TOK_EN: dict[int, str] = {i: tok for tok, i in TOK2ID_EN.items()}
VOCAB_SIZE_EN: int = len(VOCAB_EN)


# Special IDs (PAD=0, BOS=1, EOS=2, SEP=3, UNK=4) are re-exported
# unchanged. They are referenced by name in inference.py and by hardcoded
# integer in some sampling loops; they MUST keep their L1 values.
__all__ = [
    "VOCAB_EN", "TOK2ID_EN", "ID2TOK_EN", "VOCAB_SIZE_EN",
    "ENGLISH_FUNCTION_WORDS", "EN_FUNCTION_WORD_SET",
    "is_english_function_word",
    "PAD_ID", "BOS_ID", "EOS_ID", "SEP_ID", "UNK_ID",
]
