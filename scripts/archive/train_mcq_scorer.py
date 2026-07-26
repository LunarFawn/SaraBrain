"""Train a from-scratch MCQ scorer model.

Bidirectional encoder reads [facts + question + choices], scores each
choice by how well it's supported by the facts. Variable number of
choices (2-8). Uses copy-style attention to compare choices against facts.

Usage:
    python scripts/train_mcq_scorer.py \
        --data training_data/mcq_scorer_500k.jsonl \
        --out models/sara-mcq-scorer \
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


class ScorerEncoder(nn.Module):
    """Bidirectional encoder for facts + question + choices."""
    def __init__(self, vocab_size, d_model=768, n_heads=12, n_layers=8, max_seq=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq, d_model)
        self.drop = nn.Dropout(0.1)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4,
                                          dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, ids, pad_mask=None):
        B, T = ids.shape
        x = self.embed(ids) + self.pos(torch.arange(T, device=ids.device))
        x = self.drop(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.ln(x)


class MCQScorer(nn.Module):
    """Scores each choice against the encoded facts.

    Input: [facts SEP question SEP choice_A SEP choice_B SEP ...]
    Output: score per choice (softmax → pick highest)
    """
    def __init__(self, vocab_size, d_model=768, n_heads=12, n_layers=8,
                 max_seq=512, max_choices=8):
        super().__init__()
        self.encoder = ScorerEncoder(vocab_size, d_model, n_heads, n_layers, max_seq)
        self.choice_proj = nn.Linear(d_model, 1)  # score per position
        self.max_choices = max_choices
        self.sep_token = 4  # <sep> token ID

    def forward(self, input_ids, choice_positions, pad_mask=None):
        """
        input_ids: (B, T) - full sequence
        choice_positions: (B, max_choices) - token position of each choice start (-1 if unused)
        """
        B = input_ids.shape[0]
        enc = self.encoder(input_ids, pad_mask)  # (B, T, d_model)

        # Extract representation at each choice position
        scores = torch.full((B, self.max_choices), -1e9, device=input_ids.device)
        for i in range(self.max_choices):
            pos = choice_positions[:, i]  # (B,)
            valid = pos >= 0
            if valid.any():
                # Gather the hidden state at each choice's start position
                idx = pos.clamp(min=0).unsqueeze(1).unsqueeze(2).expand(-1, 1, enc.shape[2])
                choice_repr = enc.gather(1, idx).squeeze(1)  # (B, d_model)
                score = self.choice_proj(choice_repr).squeeze(1)  # (B,)
                scores[valid, i] = score[valid]

        return scores  # (B, max_choices) — apply softmax externally


def build_vocab():
    """Minimal vocab — same as extractor base vocab."""
    BASE = [
        "<pad>", "<bos>", "<eos>", "<unk>", "<sep>",
        ".", ",", "|", "\n", "-",
        "is_a", "contains", "produces", "requires", "involves",
        "causes", "prevents", "occurs_in", "part_of", "enables",
        "interacts_with", "transforms_into", "regulates", "provides",
        "a", "an", "the", "of", "and", "in", "to", "by", "for", "with",
        "what", "which", "does", "is", "are", "how", "that",
    ]
    return {t: i for i, t in enumerate(BASE)}


def encode_with_oov(text, tok2id, max_len):
    tokens = tokenize(text)[:max_len]
    ids, oov_map = [], {}
    for t in tokens:
        if t in tok2id:
            ids.append(tok2id[t])
        else:
            if t not in oov_map:
                oov_map[t] = len(tok2id) + len(oov_map)
            ids.append(oov_map[t])
    return ids, oov_map


MAX_CHOICES = 8
SEP_ID = 4


def encode_example(ex, tok2id, max_seq=400):
    """Encode: [facts SEP question SEP choiceA SEP choiceB SEP ...]"""
    facts_ids, oov = encode_with_oov(ex["facts"], tok2id, 200)
    q_ids, oov2 = encode_with_oov(ex["question"], tok2id, 50)
    oov.update(oov2)

    # Build full sequence
    ids = facts_ids + [SEP_ID] + q_ids + [SEP_ID]
    choice_positions = [-1] * MAX_CHOICES

    # Parse choices from the choice string
    choices_raw = ex["choices"].split(" | ")
    for i, choice in enumerate(choices_raw):
        if i >= MAX_CHOICES:
            break
        choice_positions[i] = len(ids)  # mark start position
        c_text = choice[3:] if len(choice) > 3 else choice  # strip "A. "
        c_ids, oov3 = encode_with_oov(c_text, tok2id, 30)
        oov.update(oov3)
        ids.extend(c_ids)
        ids.append(SEP_ID)

    ids = ids[:max_seq]
    return ids, choice_positions, ex["correct_idx"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="models/sara-mcq-scorer")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--checkpoint-every", type=int, default=10000)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tok2id = build_vocab()
    ext_vocab = len(tok2id) + 500  # OOV space
    print(f"Vocab: {len(tok2id)} base + 500 OOV = {ext_vocab}")

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

    # Model
    model = MCQScorer(ext_vocab, d_model=768, n_heads=12, n_layers=8,
                      max_seq=400, max_choices=MAX_CHOICES).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,} ({params/1e6:.0f}M)")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
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
        batch = rng.sample(train_ex, args.batch_size)
        all_ids, all_positions, all_targets = [], [], []
        for ex in batch:
            ids, positions, target = encode_example(ex, tok2id, 400)
            all_ids.append(ids)
            all_positions.append(positions)
            all_targets.append(target)

        # Pad
        max_len = max(len(x) for x in all_ids)
        padded = [x + [0] * (max_len - len(x)) for x in all_ids]
        pad_mask = [[False] * len(x) + [True] * (max_len - len(x)) for x in all_ids]

        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        positions = torch.tensor(all_positions, dtype=torch.long, device=device)
        targets = torch.tensor(all_targets, dtype=torch.long, device=device)
        pm = torch.tensor(pad_mask, dtype=torch.bool, device=device)

        scores = model(input_ids, positions, pm)
        loss = F.cross_entropy(scores, targets)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 500 == 0:
            # Quick train accuracy
            preds = scores.argmax(dim=-1)
            acc = (preds == targets).float().mean().item()
            print(f"  step={step}/{args.steps} loss={loss.item():.4f} acc={acc:.2f} ({time.time()-t0:.0f}s)")

        if step % 5000 == 0 or step == args.steps:
            model.eval()
            correct = 0
            with torch.no_grad():
                for ex in val[:100]:
                    ids, positions, target = encode_example(ex, tok2id, 400)
                    inp = torch.tensor([ids], dtype=torch.long, device=device)
                    pos = torch.tensor([positions], dtype=torch.long, device=device)
                    scores = model(inp, pos)
                    if scores.argmax(dim=-1).item() == target:
                        correct += 1
            val_acc = correct / 100
            print(f"  >>> val accuracy: {correct}/100 = {val_acc*100:.0f}%")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({"model": model.state_dict(), "tok2id": tok2id,
                            "step": step, "val_acc": val_acc},
                           f"{args.out}/best.pt")
                print(f"  >>> new best!")
            model.train()

        if step % args.checkpoint_every == 0:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": step}, f"{args.out}/checkpoint_{step:06d}.pt")
            ckpts = sorted(Path(args.out).glob("checkpoint_*.pt"))
            for old in ckpts[:-3]:
                old.unlink()

    print(f"\nDone in {time.time()-t0:.0f}s. Best val accuracy: {best_val_acc*100:.0f}%")


if __name__ == "__main__":
    main()
