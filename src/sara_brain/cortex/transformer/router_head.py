"""Router head — small classifier over the frozen Grammar LM encoder.

Takes a question's UD tag stream, encodes it via the frozen grammar
transformer, mean-pools the hidden states across non-pad positions, and
projects to one of N tool classes (brain_value / brain_define /
brain_explore / brain_did_you_mean).

Argument extraction (concept, type, label, term) is a separate
rule-based step — this head only emits the tool selection.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import GrammarModel


class RouterHead(nn.Module):
    def __init__(self, encoder: GrammarModel, n_classes: int, freeze_encoder: bool = True):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in encoder.parameters():
                p.requires_grad_(False)
        d = encoder.cfg.d_model
        self.classifier = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d, n_classes),
        )
        self.pad_id = encoder.cfg.pad_id

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Mean-pool the encoder's final hidden states over non-pad
        positions. Returns [B, d_model]."""
        ctx = torch.no_grad() if self.freeze_encoder else torch.enable_grad()
        with ctx:
            h = self.encoder.encode_hidden(input_ids)  # [B, T, d]
        mask = (input_ids != self.pad_id).float().unsqueeze(-1)  # [B, T, 1]
        summed = (h * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        pooled = self.encode(input_ids)
        logits = self.classifier(pooled)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return logits, loss

    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(input_ids)
        return logits.argmax(dim=-1)

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
