"""Grammar-feature transformer.

Per-word input is four parallel feature ids (POS, dep, head-offset,
function-word). Each is embedded separately, concatenated, projected
to `d_model`, and fed through a stack of bidirectional transformer
blocks. The model never embeds open-class content tokens — those
ride a parallel "conveyor belt" array used only at decode time.

Architecture mirrors v2's EncoderBlock for the body; the only thing
that's different is the input layer.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .vocab import (
    N_DEP, N_FUNCWORDS, N_OFFSET, N_POS,
    PAD_DEP_ID, PAD_FUNCWORD_ID, PAD_OFFSET_ID, PAD_POS_ID,
)


@dataclass
class ExtractorConfig:
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    max_seq: int = 128
    dropout: float = 0.1
    # Per-feature embedding dims. Sum must equal d_model after the
    # concat → linear projection. We let each stream learn its own
    # representation rather than forcing identical embedding spaces.
    pos_dim: int = 64
    dep_dim: int = 96
    offset_dim: int = 32
    funcword_dim: int = 64

    @classmethod
    def tiny(cls) -> "ExtractorConfig":
        return cls(d_model=128, n_heads=4, n_layers=2, d_ff=256, max_seq=64,
                   pos_dim=32, dep_dim=48, offset_dim=16, funcword_dim=32)

    @classmethod
    def base(cls) -> "ExtractorConfig":
        return cls()

    @classmethod
    def large(cls) -> "ExtractorConfig":
        return cls(d_model=384, n_heads=6, n_layers=6, d_ff=1536,
                   pos_dim=96, dep_dim=128, offset_dim=48, funcword_dim=112)


class EncoderBlock(nn.Module):
    """Pre-LN bidirectional transformer block — same shape as v2."""

    def __init__(self, cfg: ExtractorConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True,
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(
            h, h, h,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x


class FeatureInputEmbedding(nn.Module):
    """Four parallel embedding tables, concatenated, projected to d_model."""

    def __init__(self, cfg: ExtractorConfig):
        super().__init__()
        self.pos_emb = nn.Embedding(N_POS, cfg.pos_dim, padding_idx=PAD_POS_ID)
        self.dep_emb = nn.Embedding(N_DEP, cfg.dep_dim, padding_idx=PAD_DEP_ID)
        self.offset_emb = nn.Embedding(N_OFFSET, cfg.offset_dim, padding_idx=PAD_OFFSET_ID)
        self.funcword_emb = nn.Embedding(N_FUNCWORDS, cfg.funcword_dim,
                                         padding_idx=PAD_FUNCWORD_ID)
        concat_dim = (cfg.pos_dim + cfg.dep_dim + cfg.offset_dim
                      + cfg.funcword_dim)
        self.project = nn.Linear(concat_dim, cfg.d_model)
        self.position_emb = nn.Embedding(cfg.max_seq, cfg.d_model)

    def forward(
        self,
        pos_ids: torch.Tensor,
        dep_ids: torch.Tensor,
        offset_ids: torch.Tensor,
        funcword_ids: torch.Tensor,
    ) -> torch.Tensor:
        b, t = pos_ids.shape
        feats = torch.cat([
            self.pos_emb(pos_ids),
            self.dep_emb(dep_ids),
            self.offset_emb(offset_ids),
            self.funcword_emb(funcword_ids),
        ], dim=-1)
        x = self.project(feats)
        positions = torch.arange(t, device=pos_ids.device).unsqueeze(0).expand(b, t)
        x = x + self.position_emb(positions)
        return x


class GrammarEncoder(nn.Module):
    """Stack of bidirectional EncoderBlocks operating on grammar
    features. Returns hidden states `[B, T, d_model]`."""

    def __init__(self, cfg: ExtractorConfig):
        super().__init__()
        self.cfg = cfg
        self.input_emb = FeatureInputEmbedding(cfg)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self,
        pos_ids: torch.Tensor,
        dep_ids: torch.Tensor,
        offset_ids: torch.Tensor,
        funcword_ids: torch.Tensor,
    ) -> torch.Tensor:
        x = self.input_emb(pos_ids, dep_ids, offset_ids, funcword_ids)
        x = self.drop(x)
        # Padding mask: any position where pos_id is PAD treats as pad.
        key_padding_mask = (pos_ids == PAD_POS_ID)
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        return self.ln_f(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


__all__ = [
    "ExtractorConfig", "FeatureInputEmbedding", "GrammarEncoder",
    "EncoderBlock", "count_params",
]
