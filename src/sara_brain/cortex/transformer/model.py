from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GrammarConfig:
    vocab_size: int
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 18
    d_ff: int = 3072
    max_seq: int = 256
    dropout: float = 0.1
    pad_id: int = 0

    @classmethod
    def tiny(cls, vocab_size: int) -> "GrammarConfig":
        return cls(vocab_size=vocab_size, d_model=256, n_heads=4,
                   n_layers=4, d_ff=1024, max_seq=256, dropout=0.1)

    @classmethod
    def base_125m(cls, vocab_size: int) -> "GrammarConfig":
        return cls(vocab_size=vocab_size, d_model=768, n_heads=12,
                   n_layers=18, d_ff=3072, max_seq=256, dropout=0.1)

    @classmethod
    def prod_300m(cls, vocab_size: int) -> "GrammarConfig":
        return cls(vocab_size=vocab_size, d_model=1024, n_heads=16,
                   n_layers=24, d_ff=4096, max_seq=256, dropout=0.1)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GrammarConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x


class GrammarModel(nn.Module):
    def __init__(self, cfg: GrammarConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_embed = nn.Embedding(cfg.max_seq, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_embed.weight  # weight tying

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.full((cfg.max_seq, cfg.max_seq), float("-inf")), diagonal=1),
            persistent=False,
        )
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
        input_ids: torch.Tensor,
        target_ids: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        b, t = input_ids.shape
        assert t <= self.cfg.max_seq, f"seq {t} > max_seq {self.cfg.max_seq}"

        pos = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, t)
        x = self.drop(self.tok_embed(input_ids) + self.pos_embed(pos))

        mask = self.causal_mask[:t, :t]
        for block in self.blocks:
            x = block(x, mask)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if target_ids is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = target_ids[:, 1:].contiguous()
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_targets = shift_targets.view(-1)
            if loss_mask is not None:
                shift_mask = loss_mask[:, 1:].contiguous().view(-1).bool()
                loss = F.cross_entropy(
                    flat_logits[shift_mask], flat_targets[shift_mask],
                    ignore_index=self.cfg.pad_id,
                )
            else:
                loss = F.cross_entropy(
                    flat_logits, flat_targets, ignore_index=self.cfg.pad_id,
                )
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
