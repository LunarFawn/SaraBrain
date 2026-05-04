"""HamRobySum vocabulary — extends vocab_en with synthesizer delimiters.

See [docs/v033_reproducible_hamroby_loop.md](../../../../docs/v033_reproducible_hamroby_loop.md)
for the architecture this fits into.

Builds on top of `vocab_en` (which already includes the L1 structural
vocab + English function-word literals). Adds a small set of structural
delimiters used in the synth-head's prompt format:

    <facts>
      <subj> inertia in rna <pred> is <obj> tendency to maintain current state <edge_sep>
      <subj> ribosome <pred> part_of <obj> cell <edge_sep>
    <prose>
    Inertia in rna is a tendency to maintain current state. Ribosome is part of cell.
    </prose>

The model learns to continue from `<prose>` to `</prose>`. Substrate
content is tokenized as a sequence of L1+L2 vocab tokens (lowercased
words); out-of-vocab content tokens fall back to `<unk>` until the
serializer (slice 4b) decides on a real OOV strategy.

Like `vocab_en`, this is a *superset* — every L1 + L2-en token keeps
its existing ID. The synth checkpoint loads against `VOCAB_SIZE_SYNTH`;
older L1/L2 checkpoints continue to load against their own vocab files
unchanged.
"""
from __future__ import annotations

from .vocab_en import (
    BOS_ID,
    EN_FUNCTION_WORD_SET,
    EOS_ID,
    ID2TOK_EN,
    PAD_ID,
    SEP_ID,
    TOK2ID_EN,
    UNK_ID,
    VOCAB_EN,
    VOCAB_SIZE_EN,
)


# Structural delimiters used in the synth-head's prompt format. Order
# is stable: appending new tokens at the end is safe; reordering
# invalidates synth checkpoints.
SYNTH_DELIMITERS: tuple[str, ...] = (
    # Top-level frame markers.
    "<facts>",
    "<prose>",
    "</prose>",
    # Per-edge field markers.
    "<subj>",
    "<pred>",
    "<obj>",
    # Between edges in a cluster.
    "<edge_sep>",
    # Flags carried alongside an edge.
    "<refuted>",      # the edge was refuted in the substrate
    "<attr>",         # target_was_attribute = True
    # Clustering hint (some clusters are tagged with a topic).
    "<topic>",
)


# Punctuation literals the synth head needs to emit in prose. Kept at
# the synth layer (not in vocab_en) so existing L2-en checkpoints
# trained at vocab_size=175 stay loadable; the synth trainer will
# random-init these rows and learn them during synth training.
SYNTH_PUNCTUATION: tuple[str, ...] = (
    ".", ",", ";", ":", "?", "!", "-",
)


_SYNTH_ADDED: tuple[str, ...] = SYNTH_DELIMITERS + SYNTH_PUNCTUATION

# Sanity: no duplicates among additions, and no overlap with vocab_en.
# These run at import time so a bad edit fails fast.
assert len(set(_SYNTH_ADDED)) == len(_SYNTH_ADDED), (
    "SYNTH_DELIMITERS + SYNTH_PUNCTUATION contains duplicates"
)
_overlap = set(_SYNTH_ADDED) & set(VOCAB_EN)
assert not _overlap, f"synth additions overlap with vocab_en: {sorted(_overlap)}"
del _overlap


# Lookup helpers for the serializer.
SYNTH_DELIMITER_SET: frozenset[str] = frozenset(SYNTH_DELIMITERS)
SYNTH_PUNCTUATION_SET: frozenset[str] = frozenset(SYNTH_PUNCTUATION)


# The synth vocabulary: vocab_en at the front, then synth additions.
VOCAB_SYNTH: list[str] = list(VOCAB_EN) + list(_SYNTH_ADDED)
TOK2ID_SYNTH: dict[str, int] = {tok: i for i, tok in enumerate(VOCAB_SYNTH)}
ID2TOK_SYNTH: dict[int, str] = {i: tok for tok, i in TOK2ID_SYNTH.items()}
VOCAB_SIZE_SYNTH: int = len(VOCAB_SYNTH)


# Convenience: pull the structural delimiter IDs by name so the
# serializer and trainer don't carry magic strings.
FACTS_ID:    int = TOK2ID_SYNTH["<facts>"]
PROSE_ID:    int = TOK2ID_SYNTH["<prose>"]
END_PROSE_ID: int = TOK2ID_SYNTH["</prose>"]
SUBJ_ID:     int = TOK2ID_SYNTH["<subj>"]
PRED_ID:     int = TOK2ID_SYNTH["<pred>"]
OBJ_ID:      int = TOK2ID_SYNTH["<obj>"]
EDGE_SEP_ID: int = TOK2ID_SYNTH["<edge_sep>"]
REFUTED_ID:  int = TOK2ID_SYNTH["<refuted>"]
ATTR_ID:     int = TOK2ID_SYNTH["<attr>"]
TOPIC_ID:    int = TOK2ID_SYNTH["<topic>"]


def build_brain_vocab(extra_words) -> tuple[list[str], dict[str, int]]:
    """Return `(vocab_list, tok2id)` for a brain-extended vocabulary:
    `VOCAB_SYNTH` followed by content words from the substrate, in the
    order they're encountered (deduped). Substrate-per-user means the
    synth model's vocab is also per-substrate — L1 and L2 stay
    universal; only the head's emission vocab is brain-specific.

    `extra_words` is any iterable of strings (already lowercased and
    tokenized — e.g. from `_tokenize_text` in synth_data). Words
    already in `VOCAB_SYNTH` are skipped."""
    vocab = list(VOCAB_SYNTH)
    seen = set(vocab)
    for w in extra_words:
        if w in seen:
            continue
        seen.add(w)
        vocab.append(w)
    tok2id = {tok: i for i, tok in enumerate(vocab)}
    return vocab, tok2id


__all__ = [
    "VOCAB_SYNTH", "TOK2ID_SYNTH", "ID2TOK_SYNTH", "VOCAB_SIZE_SYNTH",
    "SYNTH_DELIMITERS", "SYNTH_DELIMITER_SET",
    "SYNTH_PUNCTUATION", "SYNTH_PUNCTUATION_SET",
    "build_brain_vocab",
    # Re-exported special IDs (unchanged from L1).
    "PAD_ID", "BOS_ID", "EOS_ID", "SEP_ID", "UNK_ID",
    # Re-exported L2-en handles.
    "VOCAB_EN", "VOCAB_SIZE_EN", "TOK2ID_EN", "ID2TOK_EN",
    "EN_FUNCTION_WORD_SET",
    # Synth-specific delimiter IDs.
    "FACTS_ID", "PROSE_ID", "END_PROSE_ID",
    "SUBJ_ID", "PRED_ID", "OBJ_ID", "EDGE_SEP_ID",
    "REFUTED_ID", "ATTR_ID", "TOPIC_ID",
]
