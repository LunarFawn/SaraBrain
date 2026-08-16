"""Phase 2: Train a bidirectional masked language model on substrate text.

The model learns TOKEN IDENTITY — "this token here is the same as
that token there" — by predicting randomly masked tokens from
bidirectional context. This is the BERT pretraining recipe applied
to substrate format.

After this, the model can be fine-tuned for MCQ elimination because
it understands that "zorpak" in the facts and "zorpak" in the choices
are the same concept.

Uses the same 100k substrate text data from Phase 1, but with masked
LM objective instead of causal LM.

Usage:
    python scripts/train_masked_lm.py \
        --data training_data/substrate_lm_100k.txt \
        --out models/sara-masked-lm \
        --steps 100000
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


def tokenize(text):
    return re.findall(r"[a-zA-Z_]+(?:'[a-z]+)?|[0-9]+\.[0-9]+|[0-9]+|[^\s]", text.lower())


MASK_ID = 4  # special mask token


class MaskedLM(nn.Module):
    """Bidirectional transformer for masked language modeling."""
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=8, max_seq=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq, d_model)
        self.drop = nn.Dropout(0.1)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4,
                                          dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.predict = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, pad_mask=None):
        B, T = input_ids.shape
        x = self.embed(input_ids) + self.pos(torch.arange(T, device=input_ids.device))
        x = self.drop(x)
        # No causal mask — full bidirectional attention
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.ln(x)
        return self.predict(x)  # (B, T, vocab_size)

    def encode(self, input_ids, pad_mask=None):
        """Get hidden representations (for downstream tasks)."""
        B, T = input_ids.shape
        x = self.embed(input_ids) + self.pos(torch.arange(T, device=input_ids.device))
        x = self.drop(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.ln(x)


def build_vocab(texts, max_vocab=8000):
    """Build vocabulary from training text."""
    from collections import Counter
    counts = Counter()
    for text in texts:
        counts.update(tokenize(text))
    tok2id = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "<mask>": 4, "<sep>": 5, "<choice>": 6}
    for tok, _ in counts.most_common(max_vocab - len(tok2id)):
        if tok not in tok2id:
            tok2id[tok] = len(tok2id)
    return tok2id


def mask_tokens(ids, tok2id, rng, mask_prob=0.15):
    """Randomly mask 15% of tokens (BERT-style).
    Of masked: 80% → <mask>, 10% → random token, 10% → keep original.
    Returns masked_ids, labels (original token at masked positions, -100 elsewhere).
    """
    masked = list(ids)
    labels = [-100] * len(ids)
    vocab_size = len(tok2id)

    for i in range(len(ids)):
        if ids[i] <= 6:  # don't mask special tokens
            continue
        if rng.random() < mask_prob:
            labels[i] = ids[i]  # target is the original
            r = rng.random()
            if r < 0.8:
                masked[i] = MASK_ID  # 80%: replace with <mask>
            elif r < 0.9:
                masked[i] = rng.randint(7, vocab_size - 1)  # 10%: random token
            # else 10%: keep original (masked[i] stays the same)

    return masked, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Raw substrate text file")
    ap.add_argument("--out", default="models/sara-masked-lm")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-seq", type=int, default=256)
    ap.add_argument("--checkpoint-every", type=int, default=10000)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load text and build vocab
    print(f"Loading: {args.data}")
    with open(args.data) as f:
        raw_text = f.read()
    # Split into chunks for vocab building
    chunks = [c.strip() for c in raw_text.split("\n\n") if c.strip()]
    print(f"  {len(chunks)} text chunks, {len(raw_text):,} chars")

    tok2id = build_vocab(chunks[:10000], max_vocab=8000)
    vocab_size = len(tok2id)
    print(f"  Vocab: {vocab_size}")

    # Tokenize all chunks
    all_token_ids = []
    for chunk in chunks:
        ids = [tok2id.get(t, 3) for t in tokenize(chunk)]
        if len(ids) >= 10:  # skip very short chunks
            all_token_ids.append(ids)
    print(f"  Usable chunks: {len(all_token_ids)}")

    # Model
    model = MaskedLM(vocab_size, d_model=512, n_heads=8, n_layers=8,
                     max_seq=args.max_seq).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {params:,} ({params/1e6:.0f}M)")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)
    best_loss = float("inf")
    t0 = time.time()
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

        # Build batch
        batch_masked, batch_labels = [], []
        for _ in range(args.batch_size):
            # Pick a random chunk, take a random slice
            chunk = rng.choice(all_token_ids)
            start = rng.randint(0, max(0, len(chunk) - args.max_seq))
            ids = chunk[start:start + args.max_seq]
            masked, labels = mask_tokens(ids, tok2id, rng)
            batch_masked.append(masked)
            batch_labels.append(labels)

        # Pad
        max_len = max(len(x) for x in batch_masked)
        padded_masked = [x + [0] * (max_len - len(x)) for x in batch_masked]
        padded_labels = [x + [-100] * (max_len - len(x)) for x in batch_labels]
        pad_mask = [[False] * len(x) + [True] * (max_len - len(x)) for x in batch_masked]

        input_ids = torch.tensor(padded_masked, dtype=torch.long, device=device)
        labels_t = torch.tensor(padded_labels, dtype=torch.long, device=device)
        pm = torch.tensor(pad_mask, dtype=torch.bool, device=device)

        logits = model(input_ids, pm)
        loss = F.cross_entropy(logits.view(-1, vocab_size), labels_t.view(-1), ignore_index=-100)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 500 == 0:
            # Compute masked token accuracy
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                mask_positions = labels_t != -100
                if mask_positions.any():
                    acc = (preds[mask_positions] == labels_t[mask_positions]).float().mean().item()
                else:
                    acc = 0.0
            print(f"  step={step}/{args.steps} loss={loss.item():.4f} mask_acc={acc:.2f} ({time.time()-t0:.0f}s)")

        if step % args.checkpoint_every == 0 or step == args.steps:
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save({
                    "model": model.state_dict(),
                    "tok2id": tok2id,
                    "config": {"vocab_size": vocab_size, "d_model": 512, "n_heads": 8,
                               "n_layers": 8, "max_seq": args.max_seq},
                    "step": step,
                    "loss": best_loss,
                }, f"{args.out}/best.pt")
            # Save checkpoint
            torch.save({"model": model.state_dict(), "step": step},
                       f"{args.out}/checkpoint_{step:06d}.pt")
            for old in sorted(Path(args.out).glob("checkpoint_*.pt"))[:-3]:
                old.unlink()

    print(f"\nDone in {time.time()-t0:.0f}s. Best loss: {best_loss:.4f}")
    print(f"Now fine-tune for MCQ elimination with: train_masked_eliminator.py")


if __name__ == "__main__":
    main()
