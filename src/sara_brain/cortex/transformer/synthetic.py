from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch

from . import ud
from .vocab import (
    BOS_ID, EOS_ID, PAD_ID, SEP_ID, TOK2ID, UNK_ID, VOCAB_SIZE,
)

# Hand-built fallback templates (used when UD is disabled or as a small
# auxiliary stream). The grammar transformer's input is a delexicalized
# (DEPREL, UPOS) tag pair stream.

_NSUBJ = ["nsubj", "NOUN"]
_VERB_BE = ["root", "AUX"]
_OBJ_NOUN = ["obj", "NOUN"]
_OBJ_ADJ = ["xcomp", "ADJ"]
_DET = ["det", "DET"]
_NEG = ["advmod", "PART"]

INPUT_TEMPLATES = {
    "TEACH": [
        _DET + _NSUBJ + _VERB_BE + _DET + _OBJ_NOUN,
        _DET + _NSUBJ + _VERB_BE + _OBJ_ADJ,
        _NSUBJ + _VERB_BE + _OBJ_NOUN,
    ],
    "REFUTE": [
        _DET + _NSUBJ + _VERB_BE + _NEG + _OBJ_NOUN,
        _NSUBJ + _VERB_BE + _NEG + _OBJ_ADJ,
    ],
    "QUESTION": [
        ["root", "PRON"] + _VERB_BE + _NSUBJ,
        ["root", "PRON"] + _VERB_BE + _DET + _NSUBJ,
        ["root", "VERB"] + _OBJ_NOUN,
    ],
}


@dataclass
class Example:
    input_tokens: list[str]
    target_tokens: list[str]
    kind: str


def _target_for(kind: str) -> list[str]:
    if kind == "TEACH":
        return ["TEACH", "[SUBJ]", "[VERB]", "[OBJ]"]
    if kind == "REFUTE":
        return ["REFUTE", "[SUBJ]", "[NEG]", "[VERB]", "[OBJ]"]
    if kind == "QUESTION":
        return ["QUESTION", "TOOL:brain_value", "[CONCEPT]", "[TYPE]"]
    raise ValueError(kind)


def make_template_example(rng: random.Random, mode: str = "<router>") -> Example:
    kind = rng.choices(["TEACH", "REFUTE", "QUESTION"], weights=[0.5, 0.2, 0.3])[0]
    inp = [mode] + rng.choice(INPUT_TEMPLATES[kind])
    tgt = _target_for(kind)
    return Example(input_tokens=inp, target_tokens=tgt, kind=kind)


def encode_pair(ex: Example, max_seq: int) -> tuple[list[int], list[int], list[int]]:
    """Build LM training tensors. Layout: [BOS] input... [SEP] target... [EOS]
    Pad to max_seq. Loss mask covers only target tokens and EOS — the model
    learns to emit the target conditioned on the input prefix.

    If the input is too long, truncate the INPUT, not the target — otherwise
    the loss mask would be empty and backward would NaN.
    """
    inp_ids = [TOK2ID.get(t, UNK_ID) for t in ex.input_tokens]
    tgt_ids = [TOK2ID.get(t, UNK_ID) for t in ex.target_tokens]

    overhead = 3  # BOS + SEP + EOS
    budget = max_seq - overhead - len(tgt_ids)
    if budget < 1:
        # target alone exceeds budget; clip target as last resort
        tgt_ids = tgt_ids[: max_seq - overhead - 1]
        budget = 1
    if len(inp_ids) > budget:
        inp_ids = inp_ids[:budget]

    seq = [BOS_ID] + inp_ids + [SEP_ID] + tgt_ids + [EOS_ID]
    target = list(seq)
    mask = [0] * (1 + len(inp_ids) + 1) + [1] * len(tgt_ids) + [1]

    pad = max_seq - len(seq)
    seq += [PAD_ID] * pad
    target += [PAD_ID] * pad
    mask += [0] * pad
    return seq, target, mask


class MixedSource:
    """Mix UD-derived examples with hand-built templates.

    Loads UD on first use; if download fails, falls back to templates only.
    """

    def __init__(
        self,
        ud_ratio: float = 0.6,
        ud_split: str = "train",
        ud_cache: Path = ud.DEFAULT_CACHE,
        max_input_tokens: int = 24,
        seed: int = 0,
    ):
        self.ud_ratio = ud_ratio
        self.max_input_tokens = max_input_tokens
        self._rng = random.Random(seed)
        self._ud_examples: list[Example] = []
        self._ud_loaded = False
        self._ud_split = ud_split
        self._ud_cache = ud_cache

    def _load_ud(self) -> None:
        if self._ud_loaded:
            return
        try:
            raw = ud.load_examples(
                split=self._ud_split,
                cache_dir=self._ud_cache,
                max_tokens=self.max_input_tokens,
            )
            self._ud_examples = [
                Example(input_tokens=["<router>"] + inp, target_tokens=tgt, kind=kind)
                for (inp, tgt, kind) in raw
            ]
            print(f"[mixed] loaded {len(self._ud_examples)} UD examples", flush=True)
        except Exception as e:
            print(f"[mixed] UD load failed ({e}); template-only", flush=True)
            self._ud_examples = []
        self._ud_loaded = True

    def sample(self) -> Example:
        self._load_ud()
        if self._ud_examples and self._rng.random() < self.ud_ratio:
            return self._rng.choice(self._ud_examples)
        return make_template_example(self._rng)


def make_batch(
    batch_size: int,
    max_seq: int,
    rng: random.Random,
    source: MixedSource | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seqs, tgts, masks = [], [], []
    for _ in range(batch_size):
        if source is not None:
            ex = source.sample()
        else:
            ex = make_template_example(rng)
        s, t, m = encode_pair(ex, max_seq)
        seqs.append(s)
        tgts.append(t)
        masks.append(m)
    return (
        torch.tensor(seqs, dtype=torch.long),
        torch.tensor(tgts, dtype=torch.long),
        torch.tensor(masks, dtype=torch.long),
    )


__all__ = [
    "Example", "MixedSource", "make_template_example", "encode_pair",
    "make_batch", "VOCAB_SIZE",
]
