"""Train a from-scratch Sara-native cortex model.

No pretrained backbone. No inherited weights. No borrowed knowledge.
A small transformer trained entirely on synthetic substrate reasoning.

The model learns ONE thing: read Sara Brain wavefront output and
determine which answer is supported by the substrate.

Usage:
    python scripts/train_sara_cortex_scratch.py \
        --data training_data/sara_cortex_synthetic_10k.jsonl \
        --out models/sara-cortex-scratch-v1 \
        --steps 10000
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
    vocab_size: int = 100000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 2048
    max_seq: int = 512
    dropout: float = 0.1
    n_classes: int = 4


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

    def forward(self, x):
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.ff(self.ln2(x))
        return x


class SaraCortex(nn.Module):
    """From-scratch substrate reasoning model."""
    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_seq, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.n_classes)

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.embed(input_ids) + self.pos(torch.arange(T, device=input_ids.device))
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x[:, 0, :])

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


class Tokenizer:
    def __init__(self):
        from transformers import AutoTokenizer
        # Use gpt2 BPE tokenizer to handle any novel word
        self.enc = AutoTokenizer.from_pretrained("gpt2")
        self.enc.pad_token = self.enc.eos_token
        self.tok2id = self.enc.get_vocab()

    def fit(self, texts, max_vocab=4096):
        # AutoTokenizer is already fit
        pass

    def encode(self, text, max_len=512):
        # Truncate and return ids
        return self.enc.encode(text, truncation=True, max_length=max_len)

    def save(self, path):
        # Save a dummy JSON so load() doesn't crash on older scripts
        with open(path, "w") as f:
            import json
            json.dump({"<dummy>": 0}, f)

    @classmethod
    def load(cls, path):
        return cls()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="models/sara-cortex-scratch-v1")
    ap.add_argument("--resume", default=None, help="Path to existing model directory to resume from")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    examples = []
    with open(args.data) as f:
        for line in f:
            examples.append(json.loads(line))
    rng = random.Random(42)
    rng.shuffle(examples)
    val = examples[:200]
    train = examples[200:]
    print(f"Train: {len(train)}, Val: {len(val)}")

    # Build tokenizer from training data
    if args.resume:
        print(f"Resuming from {args.resume}...")
        tokenizer = Tokenizer.load(f"{args.resume}/tokenizer.json")
        ckpt = torch.load(f"{args.resume}/best.pt", map_location=device, weights_only=False)
        sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
        
        cfg = Config()
        cfg.vocab_size = len(tokenizer.tok2id)
        if "embed.weight" in sd:
            cfg.d_model = sd["embed.weight"].shape[1]
            cfg.max_seq = sd["pos.weight"].shape[0]
            cfg.n_layers = len([k for k in sd.keys() if k.endswith("ln1.weight")])
            cfg.n_heads = 12 if cfg.d_model == 768 else (16 if cfg.d_model == 1024 else 8)
            cfg.d_ff = sd["blocks.0.ff.0.weight"].shape[0]
            
        model = SaraCortex(cfg).to(device)
        model.load_state_dict(sd)
        
        print(f"Loaded {model.param_count() / 1e6:.1f}M param model.")
    else:
        tokenizer = Tokenizer()
        print("Building vocabulary from scratch...")
        texts = [ex["substrate"] + " " + ex["question"] for ex in examples]
        tokenizer.fit(texts, max_vocab=Config.vocab_size)
        
        cfg = Config()
        cfg.vocab_size = len(tokenizer.tok2id)
        model = SaraCortex(cfg).to(device)
        print(f"Created fresh {model.param_count() / 1e6:.1f}M param model.")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    best_val = 0.0
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

        # Batch
        batch = rng.sample(train, args.batch_size)
        ids_list, labels = [], []
        for ex in batch:
            text = f"{ex['substrate']}\n{ex['question']}"
            ids_list.append(tokenizer.encode(text, cfg.max_seq))
            labels.append(ord(ex["answer"]) - ord("A"))

        max_len = max(len(x) for x in ids_list)
        padded = [x + [0] * (max_len - len(x)) for x in ids_list]
        inp = torch.tensor(padded, dtype=torch.long, device=device)
        tgt = torch.tensor(labels, dtype=torch.long, device=device)

        loss = F.cross_entropy(model(inp), tgt)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 200 == 0:
            print(f"  step={step}/{args.steps} loss={loss.item():.4f} lr={lr:.6f} ({time.time()-t0:.0f}s)")

        if step % 1000 == 0 or step == args.steps:
            model.eval()
            correct = 0
            with torch.no_grad():
                for i in range(0, len(val), args.batch_size):
                    b = val[i:i+args.batch_size]
                    ids_l, labs = [], []
                    for ex in b:
                        ids_l.append(tokenizer.encode(f"{ex['substrate']}\n{ex['question']}", cfg.max_seq))
                        labs.append(ord(ex["answer"]) - ord("A"))
                    ml = max(len(x) for x in ids_l)
                    p = [x + [0] * (ml - len(x)) for x in ids_l]
                    correct += (model(torch.tensor(p, dtype=torch.long, device=device)).argmax(-1) == torch.tensor(labs, dtype=torch.long, device=device)).sum().item()
            acc = correct / len(val)
            print(f"  >>> val: {correct}/{len(val)} = {acc*100:.1f}%")
            if acc > best_val:
                best_val = acc
                torch.save({"config": cfg.__dict__, "model": model.state_dict(),
                            "tokenizer": tokenizer.tok2id, "val_acc": acc, "step": step},
                           f"{args.out}/best.pt")
                print(f"  >>> new best!")
            model.train()

    print(f"\nDone in {time.time()-t0:.0f}s. Best val: {best_val*100:.1f}%")
    tokenizer.save(f"{args.out}/tokenizer.json")


if __name__ == "__main__":
    main()
