"""Grammar-LM dataset: continuous (DEPREL, UPOS) tag streams from UD.

Trains the cortex transformer with a standard next-token language modeling
objective over delexicalized grammar tokens. After training the model can
score, sample, and embed grammatical structure — the foundation that the
later router and synthesizer heads consume.

By default loads all available English UD treebanks (EWT, GUM, LinES,
ParTUT, Atis, ESL). Pass a different treebanks list to focus or expand.

L2 (per-language) training opts in via `lexicalize_function_words`,
`function_word_set`, and `vocab_table` — see v030 plan. Defaults
reproduce the v024-era L1 behaviour byte-for-byte.
"""
from __future__ import annotations

import random
from pathlib import Path

import torch

from . import ud
from .vocab import BOS_ID, EOS_ID, PAD_ID, TOK2ID, UNK_ID, VOCAB_SIZE


def _encode(tokens: list[str]) -> list[int]:
    """Module-level encoder kept for back-compat callers; uses L1 vocab."""
    return [TOK2ID.get(t, UNK_ID) for t in tokens]


class UDStreamDataset:
    """Holds UD sentences as encoded tag streams ready for LM batching.

    L1 (default): pure structural tag streams encoded against `vocab.TOK2ID`.
    L2 (opt-in):  pass `lexicalize_function_words=True`, the matching
                  `function_word_set`, and the L2 `vocab_table` (e.g.
                  `vocab_en.TOK2ID_EN`)."""

    def __init__(
        self,
        split: str = "train",
        treebanks: list[str] | None = None,
        cache_root: Path = ud.DEFAULT_CACHE_ROOT,
        max_tokens_per_sentence: int = 60,
        lexicalize_function_words: bool = False,
        function_word_set: frozenset[str] | None = None,
        vocab_table: dict[str, int] | None = None,
    ):
        if treebanks is None:
            treebanks = ud.ENGLISH_ALL
        self._tok2id: dict[str, int] = vocab_table if vocab_table is not None else TOK2ID
        self.streams: list[list[int]] = []
        per_tb_counts: dict[str, int] = {}
        skipped = 0
        for tb in treebanks:
            try:
                path = ud.ensure_split(tb, split, cache_root)
            except Exception as e:
                print(f"[ud-lm] skip {tb}/{split}: {e}", flush=True)
                continue
            n_before = len(self.streams)
            for sent in ud.parse_conllu(path):
                if not sent.tokens:
                    continue
                tags = ud.to_input_tokens(
                    sent,
                    max_tokens=max_tokens_per_sentence,
                    lexicalize_function_words=lexicalize_function_words,
                    function_word_set=function_word_set,
                )
                ids = [BOS_ID] + self._encode(tags) + [EOS_ID]
                if len(ids) < 4:
                    skipped += 1
                    continue
                self.streams.append(ids)
            per_tb_counts[tb] = len(self.streams) - n_before
        breakdown = ", ".join(f"{k}={v}" for k, v in per_tb_counts.items())
        flavor = "L2" if lexicalize_function_words else "L1"
        print(
            f"[ud-lm/{flavor}] split={split} sentences={len(self.streams)} "
            f"avg_len={sum(len(s) for s in self.streams) / max(1, len(self.streams)):.1f} "
            f"skipped={skipped}  ({breakdown})",
            flush=True,
        )

    def _encode(self, tokens: list[str]) -> list[int]:
        return [self._tok2id.get(t, UNK_ID) for t in tokens]

    def __len__(self) -> int:
        return len(self.streams)


def make_lm_batch(
    dataset: UDStreamDataset,
    batch_size: int,
    max_seq: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample sentences, right-pad to max_seq. Returns (input_ids, target_ids).

    Loss is computed on every position via cross_entropy(ignore_index=PAD_ID),
    so no explicit mask is needed — padded positions contribute zero gradient.
    """
    seqs: list[list[int]] = []
    for _ in range(batch_size):
        s = rng.choice(dataset.streams)
        if len(s) > max_seq:
            start = rng.randint(0, len(s) - max_seq)
            s = s[start:start + max_seq]
        else:
            s = s + [PAD_ID] * (max_seq - len(s))
        seqs.append(s)
    t = torch.tensor(seqs, dtype=torch.long)
    return t, t


__all__ = ["UDStreamDataset", "make_lm_batch", "VOCAB_SIZE"]
