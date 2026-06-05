"""Phase 3: Fine-tune the masked LM for MCQ elimination.

Loads the pretrained masked LM (which has 99% token identity accuracy)
and adds a choice elimination head. The model now uses its understanding
of token identity to match concepts between facts and choices.

Usage:
    python scripts/train_masked_eliminator.py \
        --lm-checkpoint models/sara-masked-lm/best.pt \
        --data training_data/eliminator_500k.jsonl \
        --out models/sara-masked-eliminator \
        --steps 50000
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

from train_masked_lm import MaskedLM


MAX_CHOICES = 8
SEP_ID = 5
CHOICE_ID = 6


class MaskedEliminator(nn.Module):
    """Masked LM backbone + elimination head."""
    def __init__(self, masked_lm: MaskedLM, d_model=512):
        super().__init__()
        self.encoder = masked_lm  # reuse the full encoder
        self.choice_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 3),  # CONTRADICTED, CONSISTENT, UNKNOWN
        )

    def forward(self, input_ids, choice_positions, pad_mask=None):
        # Get bidirectional representations
        hidden = self.encoder.encode(input_ids, pad_mask)  # (B, T, d_model)

        B = input_ids.shape[0]
        logits = torch.zeros(B, MAX_CHOICES, 3, device=input_ids.device)
        for i in range(MAX_CHOICES):
            pos = choice_positions[:, i]
            valid = pos >= 0
            if valid.any():
                idx = pos.clamp(min=0).unsqueeze(1).unsqueeze(2).expand(-1, 1, hidden.shape[2])
                repr = hidden.gather(1, idx).squeeze(1)
                logits[valid, i] = self.choice_head(repr[valid])

        return logits


def tokenize(text):
    return re.findall(r"[a-zA-Z_]+(?:'[a-z]+)?|[0-9]+\.[0-9]+|[0-9]+|[^\s]", text.lower())


def encode_with_oov(text, tok2id, max_len):
    tokens = tokenize(text)[:max_len]
    ids = []
    for t in tokens:
        ids.append(tok2id.get(t, 3))  # 3 = <unk>
    return ids


def encode_example(ex, tok2id, max_seq=256):
    facts_ids = encode_with_oov(ex["facts"], tok2id, 140)
    q_ids = encode_with_oov(ex["question"], tok2id, 30)

    ids = facts_ids + [SEP_ID] + q_ids
    choice_positions = [-1] * MAX_CHOICES

    choices = ex["choices_list"]
    for i, choice in enumerate(choices):
        if i >= MAX_CHOICES:
            break
        ids.append(CHOICE_ID)
        choice_positions[i] = len(ids) - 1
        c_ids = encode_with_oov(choice, tok2id, 20)
        ids.extend(c_ids)

    ids = ids[:max_seq]
    labels = ex["labels"][:MAX_CHOICES] + [-1] * (MAX_CHOICES - len(ex["labels"]))
    return ids, choice_positions, labels, ex["correct_idx"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm-checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="models/sara-masked-eliminator")
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)  # low LR for fine-tuning
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load masked LM
    print(f"Loading masked LM: {args.lm_checkpoint}")
    ckpt = torch.load(args.lm_checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    tok2id = ckpt["tokenizer"] if "tokenizer" in ckpt else ckpt["tok2id"]
    print(f"  Vocab: {cfg['vocab_size']}, d_model: {cfg['d_model']}, layers: {cfg['n_layers']}")

    lm = MaskedLM(cfg["vocab_size"], cfg["d_model"], cfg["n_heads"],
                  cfg["n_layers"], cfg["max_seq"])
    lm.load_state_dict(ckpt["model"])
    print(f"  Loaded pretrained masked LM (loss was {ckpt.get('loss', '?')})")

    # Build eliminator
    model = MaskedEliminator(lm, d_model=cfg["d_model"]).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {params:,} ({params/1e6:.0f}M)")

    # Load data
    examples = []
    with open(args.data) as f:
        for line in f:
            examples.append(json.loads(line))
    rng = random.Random(42)
    rng.shuffle(examples)
    val = examples[:200]
    train_ex = examples[200:]
    print(f"Train: {len(train_ex)}, Val: {len(val)}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    t0 = time.time()
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
        all_ids, all_pos, all_labels, all_correct = [], [], [], []
        for ex in batch:
            ids, pos, labels, correct = encode_example(ex, tok2id, cfg["max_seq"])
            all_ids.append(ids)
            all_pos.append(pos)
            all_labels.append(labels)
            all_correct.append(correct)

        max_len = max(len(x) for x in all_ids)
        padded = [x + [0] * (max_len - len(x)) for x in all_ids]
        pad_mask = [[False] * len(x) + [True] * (max_len - len(x)) for x in all_ids]

        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        positions = torch.tensor(all_pos, dtype=torch.long, device=device)
        labels_t = torch.tensor(all_labels, dtype=torch.long, device=device)
        pm = torch.tensor(pad_mask, dtype=torch.bool, device=device)

        logits = model(input_ids, positions, pm)

        loss = 0.0
        n_valid = 0
        for i in range(MAX_CHOICES):
            valid = labels_t[:, i] >= 0
            if valid.any():
                loss += F.cross_entropy(logits[valid, i], labels_t[valid, i], reduction="sum")
                n_valid += valid.sum().item()
        loss = loss / max(n_valid, 1)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 500 == 0:
            consistent_scores = logits[:, :, 1]
            preds = consistent_scores.argmax(dim=-1)
            correct_t = torch.tensor(all_correct, device=device)
            acc = (preds == correct_t).float().mean().item()
            print(f"  step={step}/{args.steps} loss={loss.item():.4f} pick_acc={acc:.2f} ({time.time()-t0:.0f}s)")

        if step % 5000 == 0 or step == args.steps:
            model.eval()
            correct_count = 0
            with torch.no_grad():
                for ex in val[:100]:
                    ids, pos, labels, correct = encode_example(ex, tok2id, cfg["max_seq"])
                    inp = torch.tensor([ids], dtype=torch.long, device=device)
                    p = torch.tensor([pos], dtype=torch.long, device=device)
                    logits = model(inp, p)
                    if logits[0, :, 1].argmax().item() == correct:
                        correct_count += 1
            pick_acc = correct_count / 100
            print(f"  >>> val pick_acc: {correct_count}/100 = {pick_acc*100:.0f}%")
            if pick_acc > best_val_acc:
                best_val_acc = pick_acc
                torch.save({"model": model.state_dict(), "tok2id": tok2id,
                            "config": cfg, "step": step, "val_acc": pick_acc},
                           f"{args.out}/best.pt")
                print(f"  >>> new best!")
            model.train()

    print(f"\nDone in {time.time()-t0:.0f}s. Best val pick accuracy: {best_val_acc*100:.0f}%")


if __name__ == "__main__":
    main()
