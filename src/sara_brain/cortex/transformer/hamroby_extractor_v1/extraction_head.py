"""Word-level BIO span tagger on top of GrammarEncoder.

Output is one tag per word (not per subword). 7 classes:

    O      = outside any span
    B-S, I-S = subject span
    B-R, I-R = relation span
    B-O, I-O = object span

Tags are predicted at word level by construction — there are no
subword positions in this architecture. The decoder reads BIO labels
plus the original word array (the conveyor belt) and emits verbatim
multi-word spans.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import GrammarEncoder
from .vocab import N_TAGS


class ExtractionHead(nn.Module):
    """Classifier on per-word hidden states. Always trained alongside
    the encoder — no freeze option, since the encoder is already
    content-free by construction."""

    def __init__(self, encoder: GrammarEncoder, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        d = encoder.cfg.d_model
        self.tagger = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, N_TAGS),
        )

    def forward(
        self,
        pos_ids: torch.Tensor,
        dep_ids: torch.Tensor,
        offset_ids: torch.Tensor,
        funcword_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.encoder(pos_ids, dep_ids, offset_ids, funcword_ids)
        logits = self.tagger(h)   # [B, T, N_TAGS]
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, N_TAGS),
                labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def predict_tags(
        self,
        pos_ids: torch.Tensor,
        dep_ids: torch.Tensor,
        offset_ids: torch.Tensor,
        funcword_ids: torch.Tensor,
    ) -> torch.Tensor:
        logits, _ = self.forward(pos_ids, dep_ids, offset_ids, funcword_ids)
        return logits.argmax(dim=-1)


__all__ = ["ExtractionHead"]
