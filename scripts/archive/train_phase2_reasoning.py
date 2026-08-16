"""Phase 2: Train reasoning head on top of Phase 1 substrate LM.

Loads the substrate language model checkpoint, replaces the LM head
with a 4-way classifier, partially freezes the backbone, and trains
on the MCQ substrate reasoning task.

Usage:
    python scripts/train_phase2_reasoning.py \
        --lm-checkpoint models/sara-cortex-lm-v1/best.pt \
        --data training_data/sara_cortex_synthetic_10k.jsonl \
        --out models/sara-cortex-final-v1 \
        --steps 5000
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

from train_substrate_lm import SubstrateLM, Config, Block, Tokenizer


class SubstrateReasoner(nn.Module):
    """Phase 1 LM backbone + classification head."""
    def __init__(self, lm: SubstrateLM):
        super().__init__()
        self.embed = lm.embed
        self.pos = lm.pos
        self.drop = lm.drop
        self.blocks = lm.blocks
        self.ln_f = lm.ln_f
        self.cls_head = nn.Linear(lm.cfg.d_model, 4)

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.embed(input_ids) + self.pos(torch.arange(T, device=input_ids.device))
        x = self.drop(x)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.ln_f(x)
        return self.cls_head(x.mean(dim=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm-checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="models/sara-cortex-final-v1")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--freeze-layers", type=int, default=4,
                    help="Freeze first N transformer layers (default: 4 of 6)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Phase 1 checkpoint
    print(f"Loading LM checkpoint: {args.lm_checkpoint}")
    ckpt = torch.load(args.lm_checkpoint, map_location="cpu", weights_only=False)
    cfg = Config()
    for k, v in ckpt["config"].items():
        setattr(cfg, k, v)

    lm = SubstrateLM(cfg)
    lm.load_state_dict(ckpt["model"])
    print(f"  Loaded. Loss was: {ckpt.get('loss', '?')}")

    # Build reasoner
    model = SubstrateReasoner(lm).to(device)

    # Freeze early layers
    for i, block in enumerate(model.blocks):
        if i < args.freeze_layers:
            for p in block.parameters():
                p.requires_grad = False
    # Freeze embeddings (they learned substrate format in Phase 1)
    for p in model.embed.parameters():
        p.requires_grad = False
    for p in model.pos.parameters():
        p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # Load tokenizer
    tok2id = ckpt["tokenizer"]

    def encode(text, max_len=256):
        tokens = re.findall(r"[a-z_]+|[0-9]+\.[0-9]+|[0-9]+|[^\s]", text.lower())[:max_len - 2]
        ids = [1]  # bos
        for t in tokens:
            ids.append(tok2id.get(t, 3))
        ids.append(2)  # eos
        return ids

    # Load MCQ data
    print(f"Loading data: {args.data}")
    examples = []
    with open(args.data) as f:
        for line in f:
            examples.append(json.loads(line))
    rng = random.Random(42)
    rng.shuffle(examples)
    val = examples[:200]
    train_ex = examples[200:]
    print(f"  Train: {len(train_ex)}, Val: {len(val)}")

    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=args.lr, weight_decay=0.01)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    best_val = 0.0
    t0 = time.time()
    max_seq = cfg.max_seq
    model.train()

    for step in range(1, args.steps + 1):
        warmup = args.steps // 10
        if step < warmup:
            lr = args.lr * step / warmup
        else:
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * (step - warmup) / (args.steps - warmup)))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        batch = rng.sample(train_ex, args.batch_size)
        ids_list, labels = [], []
        for ex in batch:
            text = f"{ex['question']}\n{ex['substrate']}"
            ids_list.append(encode(text, max_seq))
            labels.append(ord(ex["answer"]) - ord("A"))

        ml = max(len(x) for x in ids_list)
        padded = [x + [0] * (ml - len(x)) for x in ids_list]
        inp = torch.tensor(padded, dtype=torch.long, device=device)
        tgt = torch.tensor(labels, dtype=torch.long, device=device)

        loss = F.cross_entropy(model(inp), tgt)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 200 == 0:
            print(f"  step={step}/{args.steps} loss={loss.item():.4f} ({time.time()-t0:.0f}s)")

        if step % 1000 == 0 or step == args.steps:
            model.eval()
            correct = 0
            with torch.no_grad():
                for i in range(0, len(val), args.batch_size):
                    b = val[i:i+args.batch_size]
                    ids_l, labs = [], []
                    for ex in b:
                        ids_l.append(encode(f"{ex['question']}\n{ex['substrate']}", max_seq))
                        labs.append(ord(ex["answer"]) - ord("A"))
                    ml2 = max(len(x) for x in ids_l)
                    p = [x + [0] * (ml2 - len(x)) for x in ids_l]
                    preds = model(torch.tensor(p, dtype=torch.long, device=device)).argmax(-1)
                    correct += (preds == torch.tensor(labs, dtype=torch.long, device=device)).sum().item()
            acc = correct / len(val)
            print(f"  >>> val: {correct}/{len(val)} = {acc*100:.1f}%")
            if acc > best_val:
                best_val = acc
                torch.save({"config": cfg.__dict__, "model": model.state_dict(),
                            "tokenizer": tok2id, "val_acc": acc},
                           f"{args.out}/best.pt")
                print(f"  >>> new best!")
            model.train()

    print(f"\nDone in {time.time()-t0:.0f}s. Best val: {best_val*100:.1f}%")


if __name__ == "__main__":
    main()
