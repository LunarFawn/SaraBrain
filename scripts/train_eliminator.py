"""Train the elimination-based reasoner.

For each choice, outputs: CONTRADICTED (0), CONSISTENT (1), or UNKNOWN (2).
The correct answer is the one labeled CONSISTENT. Wrong answers are
labeled CONTRADICTED because the facts explicitly conflict with them.

This is how humans reason: eliminate what's wrong, what's left is right.

Usage:
    python scripts/train_eliminator.py \
        --data training_data/eliminator_500k.jsonl \
        --out models/sara-eliminator \
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


BASE_VOCAB = [
    "<pad>", "<bos>", "<eos>", "<unk>", "<sep>", "<choice>",
    ".", ",", "|", "\n", "-",
    "is_a", "contains", "produces", "requires", "involves",
    "causes", "prevents", "occurs_in", "part_of", "enables",
    "interacts_with", "transforms_into", "regulates", "provides",
    "a", "an", "the", "of", "and", "in", "to", "by", "for", "with",
    "what", "which", "does", "is", "are", "how", "that",
]

MAX_CHOICES = 8
SEP_ID = 4
CHOICE_ID = 5


def build_vocab():
    return {t: i for i, t in enumerate(BASE_VOCAB)}


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


class Eliminator(nn.Module):
    """Bidirectional encoder that labels each choice as
    CONTRADICTED (0), CONSISTENT (1), or UNKNOWN (2).
    """
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=8, max_seq=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq, d_model)
        self.drop = nn.Dropout(0.1)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4,
                                          dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.ln = nn.LayerNorm(d_model)
        # Per-choice classification: 3 classes (CONTRADICTED, CONSISTENT, UNKNOWN)
        self.choice_classifier = nn.Linear(d_model, 3)

    def forward(self, input_ids, choice_positions, pad_mask=None):
        B, T = input_ids.shape
        x = self.embed(input_ids) + self.pos(torch.arange(T, device=input_ids.device))
        x = self.drop(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = self.ln(x)

        # Classify each choice
        # choice_positions: (B, MAX_CHOICES) — position of <choice> token for each choice
        logits = torch.zeros(B, MAX_CHOICES, 3, device=input_ids.device)
        for i in range(MAX_CHOICES):
            pos = choice_positions[:, i]
            valid = pos >= 0
            if valid.any():
                idx = pos.clamp(min=0).unsqueeze(1).unsqueeze(2).expand(-1, 1, x.shape[2])
                repr = x.gather(1, idx).squeeze(1)
                logits[valid, i] = self.choice_classifier(repr[valid])

        return logits  # (B, MAX_CHOICES, 3)

    def predict(self, input_ids, choice_positions, pad_mask=None):
        """Returns per-choice labels and picks the CONSISTENT one."""
        logits = self.forward(input_ids, choice_positions, pad_mask)
        labels = logits.argmax(dim=-1)  # (B, MAX_CHOICES) — 0/1/2 per choice
        # Pick the choice labeled CONSISTENT (1)
        # If multiple are consistent, pick highest logit for class 1
        consistent_scores = logits[:, :, 1]  # score for "CONSISTENT" class
        return labels, consistent_scores.argmax(dim=-1)


def encode_example(ex, tok2id, max_seq=400):
    """Encode: [facts SEP question CHOICE choiceA CHOICE choiceB ...]"""
    facts_ids, oov = encode_with_oov(ex["facts"], tok2id, 180)
    q_ids, oov2 = encode_with_oov(ex["question"], tok2id, 40)
    oov.update(oov2)

    ids = facts_ids + [SEP_ID] + q_ids
    choice_positions = [-1] * MAX_CHOICES

    choices = ex["choices_list"]
    for i, choice in enumerate(choices):
        if i >= MAX_CHOICES:
            break
        ids.append(CHOICE_ID)
        choice_positions[i] = len(ids) - 1  # position of <choice> marker
        c_ids, oov3 = encode_with_oov(choice, tok2id, 25)
        oov.update(oov3)
        ids.extend(c_ids)

    ids = ids[:max_seq]

    # Labels: pad unused choices with -1
    labels = ex["labels"][:MAX_CHOICES] + [-1] * (MAX_CHOICES - len(ex["labels"]))

    return ids, choice_positions, labels, ex["correct_idx"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="models/sara-eliminator")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--checkpoint-every", type=int, default=10000)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tok2id = build_vocab()
    ext_vocab = len(tok2id) + 500
    print(f"Vocab: {ext_vocab}")

    examples = []
    with open(args.data) as f:
        for line in f:
            examples.append(json.loads(line))
    rng = random.Random(42)
    rng.shuffle(examples)
    val = examples[:200]
    train_ex = examples[200:]
    print(f"Train: {len(train_ex)}, Val: {len(val)}")

    model = Eliminator(ext_vocab, d_model=512, n_heads=8, n_layers=8, max_seq=400).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,} ({params/1e6:.0f}M)")

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
            ids, pos, labels, correct = encode_example(ex, tok2id)
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

        logits = model(input_ids, positions, pm)  # (B, MAX_CHOICES, 3)

        # Loss: per-choice cross entropy, ignoring positions with label -1
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
            # Check: does it pick the CONSISTENT choice correctly?
            consistent_scores = logits[:, :, 1]  # logit for class 1
            preds = consistent_scores.argmax(dim=-1)
            correct_t = torch.tensor(all_correct, device=device)
            acc = (preds == correct_t).float().mean().item()
            print(f"  step={step}/{args.steps} loss={loss.item():.4f} pick_acc={acc:.2f} ({time.time()-t0:.0f}s)")

        if step % 5000 == 0 or step == args.steps:
            model.eval()
            correct_count = 0
            label_acc = 0
            label_total = 0
            with torch.no_grad():
                for ex in val[:100]:
                    ids, pos, labels, correct = encode_example(ex, tok2id)
                    inp = torch.tensor([ids], dtype=torch.long, device=device)
                    p = torch.tensor([pos], dtype=torch.long, device=device)
                    logits = model(inp, p)
                    # Pick accuracy
                    consistent_scores = logits[0, :, 1]
                    if consistent_scores.argmax().item() == correct:
                        correct_count += 1
                    # Label accuracy
                    for i in range(len(ex["labels"])):
                        pred_label = logits[0, i].argmax().item()
                        if pred_label == ex["labels"][i]:
                            label_acc += 1
                        label_total += 1

            pick_acc = correct_count / 100
            lab_acc = label_acc / max(label_total, 1)
            print(f"  >>> val pick_acc: {correct_count}/100 = {pick_acc*100:.0f}%")
            print(f"  >>> val label_acc: {lab_acc*100:.0f}%")
            if pick_acc > best_val_acc:
                best_val_acc = pick_acc
                torch.save({"model": model.state_dict(), "tok2id": tok2id,
                            "step": step, "val_acc": pick_acc},
                           f"{args.out}/best.pt")
                print(f"  >>> new best!")
            model.train()

        if step % args.checkpoint_every == 0:
            torch.save({"model": model.state_dict(), "step": step},
                       f"{args.out}/checkpoint_{step:06d}.pt")
            for old in sorted(Path(args.out).glob("checkpoint_*.pt"))[:-3]:
                old.unlink()

    print(f"\nDone in {time.time()-t0:.0f}s. Best val pick accuracy: {best_val_acc*100:.0f}%")


if __name__ == "__main__":
    main()
