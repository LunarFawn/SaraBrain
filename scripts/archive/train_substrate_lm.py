"""Phase 1: Train a substrate language model (next-token prediction).

Teaches the model to understand substrate format — what tokens follow
what, how wavefront output is structured. No task labels needed.

Usage:
    python scripts/train_substrate_lm.py \
        --data training_data/substrate_lm_100k.txt \
        --out models/sara-cortex-lm-v1 \
        --steps 20000
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW


class Config:
    vocab_size: int = 4096
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    max_seq: int = 256
    dropout: float = 0.1


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout))

    def forward(self, x, mask=None):
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + h
        x = x + self.ff(self.ln2(x))
        return x


class SubstrateLM(nn.Module):
    """Causal language model for substrate format."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Weight tying
        self.head.weight = self.embed.weight

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        x = self.embed(input_ids) + self.pos(torch.arange(T, device=input_ids.device))
        x = self.drop(x)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        for block in self.blocks:
            x = block(x, mask=mask)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
        return logits, loss

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


class Tokenizer:
    def __init__(self):
        self.tok2id = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}

    def fit(self, text, max_vocab=4096):
        from collections import Counter
        counts = Counter(self._split(text))
        for tok, _ in counts.most_common(max_vocab - len(self.tok2id)):
            if tok not in self.tok2id:
                self.tok2id[tok] = len(self.tok2id)

    def encode(self, text):
        return [self.tok2id.get(t, 3) for t in self._split(text)]

    def _split(self, text):
        return re.findall(r"[a-z_]+|[0-9]+\.[0-9]+|[0-9]+|[^\s]", text.lower())

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.tok2id, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Raw substrate text file")
    ap.add_argument("--out", default="models/sara-cortex-lm-v1")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load text
    print(f"Loading: {args.data}")
    with open(args.data) as f:
        raw_text = f.read()
    print(f"  {len(raw_text):,} chars")

    # Build tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit(raw_text, max_vocab=4096)
    vocab_size = len(tokenizer.tok2id)
    print(f"  Vocab: {vocab_size}")

    # Tokenize all text
    all_ids = tokenizer.encode(raw_text)
    print(f"  Tokens: {len(all_ids):,}")

    # Model
    cfg = Config()
    cfg.vocab_size = vocab_size
    model = SubstrateLM(cfg).to(device)
    print(f"  Params: {model.param_count():,} ({model.param_count()/1e6:.1f}M)")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    # Training
    seq_len = cfg.max_seq
    rng = random.Random(42)
    t0 = time.time()
    best_loss = float("inf")

    model.train()
    for step in range(1, args.steps + 1):
        # LR schedule
        warmup = args.steps // 10
        if step < warmup:
            lr = args.lr * step / warmup
        else:
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * (step - warmup) / (args.steps - warmup)))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Random batch of sequences from the corpus
        batch_ids = []
        for _ in range(args.batch_size):
            start = rng.randint(0, len(all_ids) - seq_len - 1)
            batch_ids.append(all_ids[start:start + seq_len + 1])

        x = torch.tensor([b[:-1] for b in batch_ids], dtype=torch.long, device=device)
        y = torch.tensor([b[1:] for b in batch_ids], dtype=torch.long, device=device)

        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 500 == 0:
            ppl = math.exp(loss.item()) if loss.item() < 10 else float("inf")
            print(f"  step={step}/{args.steps} loss={loss.item():.4f} ppl={ppl:.1f} lr={lr:.6f} ({time.time()-t0:.0f}s)")
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save({
                    "config": cfg.__dict__,
                    "model": model.state_dict(),
                    "tokenizer": tokenizer.tok2id,
                    "step": step, "loss": best_loss,
                }, f"{args.out}/best.pt")

    # Save final
    torch.save({
        "config": cfg.__dict__,
        "model": model.state_dict(),
        "tokenizer": tokenizer.tok2id,
        "step": args.steps, "loss": loss.item(),
    }, f"{args.out}/final.pt")
    tokenizer.save(f"{args.out}/tokenizer.json")
    print(f"\nDone in {time.time()-t0:.0f}s. Best loss: {best_loss:.4f} (ppl={math.exp(best_loss):.1f})")


if __name__ == "__main__":
    main()
