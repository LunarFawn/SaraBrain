from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from .vocab import (
    BOS_ID, EOS_ID, PAD_ID, SEP_ID, TOK2ID, VOCAB_SIZE,
)

# Structural input templates per "kind". The grammar transformer's input is a
# POS+dep tag stream (delexicalized). Here we hand-build small templates so the
# smoke loop has signal without needing UD treebanks downloaded yet.

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


def make_example(rng: random.Random, mode: str = "<router>") -> Example:
    kind = rng.choices(["TEACH", "REFUTE", "QUESTION"], weights=[0.5, 0.2, 0.3])[0]
    inp = [mode] + rng.choice(INPUT_TEMPLATES[kind])
    tgt = _target_for(kind)
    return Example(input_tokens=inp, target_tokens=tgt, kind=kind)


def encode_pair(ex: Example, max_seq: int) -> tuple[list[int], list[int], list[int]]:
    """Returns (input_ids, target_ids_shifted_for_lm, loss_mask).

    Sequence layout: [BOS] input... [SEP] target... [EOS]  (padded to max_seq)
    Loss mask is 1 for the target tokens (and EOS), 0 elsewhere — the model
    only learns to predict the target conditioned on the input prefix.
    """
    inp_ids = [TOK2ID[t] for t in ex.input_tokens]
    tgt_ids = [TOK2ID[t] for t in ex.target_tokens]

    seq = [BOS_ID] + inp_ids + [SEP_ID] + tgt_ids + [EOS_ID]
    if len(seq) > max_seq:
        seq = seq[:max_seq]

    target = list(seq)  # autoregressive: target == input shifted in model.forward
    mask = [0] * (1 + len(inp_ids) + 1) + [1] * len(tgt_ids) + [1]
    mask = mask[:len(seq)]

    pad = max_seq - len(seq)
    seq += [PAD_ID] * pad
    target += [PAD_ID] * pad
    mask += [0] * pad
    return seq, target, mask


def make_batch(
    batch_size: int, max_seq: int, rng: random.Random
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seqs, tgts, masks = [], [], []
    for _ in range(batch_size):
        ex = make_example(rng)
        s, t, m = encode_pair(ex, max_seq)
        seqs.append(s)
        tgts.append(t)
        masks.append(m)
    return (
        torch.tensor(seqs, dtype=torch.long),
        torch.tensor(tgts, dtype=torch.long),
        torch.tensor(masks, dtype=torch.long),
    )


__all__ = ["Example", "make_example", "encode_pair", "make_batch", "VOCAB_SIZE"]
