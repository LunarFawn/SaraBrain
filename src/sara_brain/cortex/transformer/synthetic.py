"""Grammar-LM dataset: continuous (DEPREL, UPOS) tag streams from UD.

Trains the cortex transformer with a standard next-token language modeling
objective over delexicalized grammar tokens. After training the model can
score, sample, and embed grammatical structure — the foundation that the
later router and synthesizer heads consume.
"""
from __future__ import annotations

import random
from pathlib import Path

import torch

from . import ud
from .vocab import BOS_ID, EOS_ID, PAD_ID, TOK2ID, UNK_ID, VOCAB_SIZE


def _encode(tokens: list[str]) -> list[int]:
    return [TOK2ID.get(t, UNK_ID) for t in tokens]


class UDStreamDataset:
    """Holds UD sentences as encoded tag streams ready for LM batching."""

    def __init__(
        self,
        split: str = "train",
        cache_dir: Path = ud.DEFAULT_CACHE,
        max_tokens_per_sentence: int = 60,
    ):
        path = ud.ensure_split(split, cache_dir)
        self.streams: list[list[int]] = []
        skipped = 0
        for sent in ud.parse_conllu(path):
            if not sent.tokens:
                continue
            tags = ud.to_input_tokens(sent, max_tokens=max_tokens_per_sentence)
            ids = [BOS_ID] + _encode(tags) + [EOS_ID]
            if len(ids) < 4:
                skipped += 1
                continue
            self.streams.append(ids)
        print(
            f"[ud-lm] split={split} sentences={len(self.streams)} "
            f"avg_len={sum(len(s) for s in self.streams) / max(1, len(self.streams)):.1f} "
            f"skipped={skipped}",
            flush=True,
        )

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
