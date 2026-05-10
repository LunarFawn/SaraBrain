"""Vocabulary for the grammar-feature transformer.

Each word is encoded as four parallel feature streams. None of the
streams contain open-class English content; the model literally
cannot embed a domain noun like "aptamer" or "snare".

  1. POS tag        — one of ~17 Universal POS tags
  2. Dep label      — one of ~37 Universal Dependency relation labels
  3. Head offset    — relative position of the dep head, binned
  4. Function-word  — closed-class English glue ("the", "by", "to"),
                      reused from HamRoby v1's `vocab_en`. NONE for
                      open-class words.

The four streams are embedded separately and concatenated into the
encoder's per-word input. Surface text of open-class words rides a
parallel "conveyor belt" array used only at decode time to slice
verbatim spans into Sara.
"""
from __future__ import annotations

from .. import vocab as _grammar_vocab
from .. import vocab_en as _vocab_en


# ── POS (Universal POS tags) ──────────────────────────────────────────
# Same set HamRoby v1 uses, plus PAD at id 0.
PAD_POS = "<pad>"
POS_TAGS: tuple[str, ...] = (PAD_POS,) + tuple(_grammar_vocab.UPOS) + ("UNK_POS",)
POS_TO_ID: dict[str, int] = {tag: i for i, tag in enumerate(POS_TAGS)}
ID_TO_POS: dict[int, str] = {i: tag for tag, i in POS_TO_ID.items()}
PAD_POS_ID = POS_TO_ID[PAD_POS]
UNK_POS_ID = POS_TO_ID["UNK_POS"]
N_POS = len(POS_TAGS)


# ── Dependency labels (UD v2) ─────────────────────────────────────────
PAD_DEP = "<pad>"
DEP_TAGS: tuple[str, ...] = (PAD_DEP,) + tuple(_grammar_vocab.UD_DEPS) + ("UNK_DEP",)
DEP_TO_ID: dict[str, int] = {tag: i for i, tag in enumerate(DEP_TAGS)}
ID_TO_DEP: dict[int, str] = {i: tag for tag, i in DEP_TO_ID.items()}
PAD_DEP_ID = DEP_TO_ID[PAD_DEP]
UNK_DEP_ID = DEP_TO_ID["UNK_DEP"]
N_DEP = len(DEP_TAGS)


# ── Head offset (binned) ──────────────────────────────────────────────
# A word's dependency head can be at any relative position. We bin
# into a small set so the model has a finite vocab. Beyond ±10 the
# precise offset rarely matters; we lump them as FAR_LEFT / FAR_RIGHT.
# id 0 = PAD, id 1 = FAR_LEFT, then -10..-1 (10 ids), 0 (self/root),
# +1..+10 (10 ids), FAR_RIGHT.
PAD_OFFSET_ID = 0
HEAD_OFFSET_BINS = list(range(-10, 11))   # -10 .. +10 inclusive
N_OFFSET = 1 + 1 + len(HEAD_OFFSET_BINS) + 1   # PAD + FAR_LEFT + bins + FAR_RIGHT
FAR_LEFT_ID = 1
FAR_RIGHT_ID = N_OFFSET - 1


def encode_head_offset(offset: int) -> int:
    """Map a signed integer offset to a vocabulary id."""
    if offset < HEAD_OFFSET_BINS[0]:
        return FAR_LEFT_ID
    if offset > HEAD_OFFSET_BINS[-1]:
        return FAR_RIGHT_ID
    return 2 + (offset - HEAD_OFFSET_BINS[0])   # 2 reserves PAD + FAR_LEFT


# ── Function-word vocabulary ──────────────────────────────────────────
# Reuse HamRoby v1's closed-class English function-word list. id 0 is
# NONE (this position is an open-class content word — model sees nothing
# of it). Function words occupy ids 1..N.
NONE_FUNCWORD = "<none>"
PAD_FUNCWORD = "<pad>"
_FUNCWORD_LIST: list[str] = [PAD_FUNCWORD, NONE_FUNCWORD] + list(_vocab_en.ENGLISH_FUNCTION_WORDS)
FUNCWORD_TO_ID: dict[str, int] = {tok: i for i, tok in enumerate(_FUNCWORD_LIST)}
ID_TO_FUNCWORD: dict[int, str] = {i: tok for tok, i in FUNCWORD_TO_ID.items()}
PAD_FUNCWORD_ID = FUNCWORD_TO_ID[PAD_FUNCWORD]
NONE_FUNCWORD_ID = FUNCWORD_TO_ID[NONE_FUNCWORD]
N_FUNCWORDS = len(_FUNCWORD_LIST)
EN_FUNCTION_WORD_SET = _vocab_en.EN_FUNCTION_WORD_SET


def encode_funcword(word_lower: str) -> int:
    """If the word is in the closed-class function-word list, return
    its id. Otherwise return NONE — the model sees nothing of the
    surface content."""
    if word_lower in EN_FUNCTION_WORD_SET:
        return FUNCWORD_TO_ID[word_lower]
    return NONE_FUNCWORD_ID


# ── BIO tag vocabulary ────────────────────────────────────────────────
# Output side — what the extraction head predicts at each word.
TAG_O = 0
TAG_B_S = 1
TAG_I_S = 2
TAG_B_R = 3
TAG_I_R = 4
TAG_B_O = 5
TAG_I_O = 6
N_TAGS = 7
TAG_NAMES: tuple[str, ...] = ("O", "B-S", "I-S", "B-R", "I-R", "B-O", "I-O")


__all__ = [
    "POS_TAGS", "POS_TO_ID", "ID_TO_POS", "PAD_POS_ID", "UNK_POS_ID", "N_POS",
    "DEP_TAGS", "DEP_TO_ID", "ID_TO_DEP", "PAD_DEP_ID", "UNK_DEP_ID", "N_DEP",
    "HEAD_OFFSET_BINS", "N_OFFSET", "PAD_OFFSET_ID",
    "FAR_LEFT_ID", "FAR_RIGHT_ID", "encode_head_offset",
    "FUNCWORD_TO_ID", "ID_TO_FUNCWORD", "PAD_FUNCWORD_ID", "NONE_FUNCWORD_ID",
    "N_FUNCWORDS", "encode_funcword", "EN_FUNCTION_WORD_SET",
    "TAG_O", "TAG_B_S", "TAG_I_S", "TAG_B_R", "TAG_I_R", "TAG_B_O", "TAG_I_O",
    "N_TAGS", "TAG_NAMES",
]
