"""Train MCQ eliminator on top of the grammar backbone (unfrozen).

Uses the L2-en grammar model (125M, trained on UD treebanks + English
function words) as the starting point. Extends positional embeddings
to handle longer sequences. Unfrozen — the whole model adapts.

The grammar backbone understands sentence structure (subject/verb/object
positions). This structural knowledge should help it match concepts
between facts and choices — something the pure from-scratch models
couldn't learn.

Usage:
    python scripts/train_grammar_eliminator.py \
        --data training_data/eliminator_500k.jsonl \
        --out models/sara-grammar-eliminator \
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sara_brain.cortex.transformer.model import GrammarConfig, GrammarModel


MAX_CHOICES = 8
MAX_SEQ = 400


def tokenize(text):
    return re.findall(r"[a-zA-Z_]+(?:'[a-z]+)?|[0-9]+\.[0-9]+|[0-9]+|[^\s]", text.lower())


class GrammarEliminator(nn.Module):
    """Grammar backbone + choice elimination head."""

    def __init__(self, vocab_size, d_model=768, n_heads=12, n_layers=18, max_seq=400):
        super().__init__()
        # Build a fresh model with extended vocab and positions
        cfg = GrammarConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_model * 4,
            max_seq=max_seq,
            dropout=0.1,
        )
        self.backbone = GrammarModel(cfg)
        self.d_model = d_model
        # Classification head: per-choice → 3 classes
        self.choice_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 3),
        )

    def forward(self, input_ids, choice_positions, pad_mask=None):
        B, T = input_ids.shape
        # Run backbone encoder (no causal mask — bidirectional for this task)
        x = self.backbone.tok_embed(input_ids)
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = x + self.backbone.pos_embed(pos)
        x = self.backbone.drop(x)
        # Use self-attention without causal mask (bidirectional)
        for block in self.backbone.blocks:
            h = block.ln1(x)
            # No attn_mask = bidirectional attention
            h, _ = block.attn(h, h, h, key_padding_mask=pad_mask, need_weights=False)
            x = x + h
            x = x + block.ff(block.ln2(x))
        x = self.backbone.ln_f(x)

        # Score each choice position
        logits = torch.zeros(B, MAX_CHOICES, 3, device=input_ids.device)
        for i in range(MAX_CHOICES):
            pos_idx = choice_positions[:, i]
            valid = pos_idx >= 0
            if valid.any():
                idx = pos_idx.clamp(min=0).unsqueeze(1).unsqueeze(2).expand(-1, 1, self.d_model)
                repr = x.gather(1, idx).squeeze(1)
                logits[valid, i] = self.choice_head(repr[valid])

        return logits


SEP_ID = 4
CHOICE_ID = 5

# Extended vocab: L2 base (175) + substrate/choice OOV
BASE_VOCAB_TOKENS = [
    "<pad>", "<bos>", "<eos>", "<unk>", "<sep>", "<choice>",
]


def build_extended_vocab():
    """Start with the 6 special tokens, rest are OOV (copied concepts)."""
    return {t: i for i, t in enumerate(BASE_VOCAB_TOKENS)}


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


def encode_example(ex, tok2id, max_seq=400):
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
        choice_positions[i] = len(ids) - 1
        c_ids, oov3 = encode_with_oov(choice, tok2id, 25)
        oov.update(oov3)
        ids.extend(c_ids)

    ids = ids[:max_seq]
    labels = ex["labels"][:MAX_CHOICES] + [-1] * (MAX_CHOICES - len(ex["labels"]))
    return ids, choice_positions, labels, ex["correct_idx"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="models/sara-grammar-eliminator")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)  # lower LR for pretrained backbone
    ap.add_argument("--checkpoint-every", type=int, default=10000)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tok2id = build_extended_vocab()
    ext_vocab = len(tok2id) + 500
    print(f"Vocab: {ext_vocab}")

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

    # Build model with grammar backbone dimensions
    model = GrammarEliminator(ext_vocab, d_model=768, n_heads=12, n_layers=18,
                              max_seq=MAX_SEQ).to(device)

    # Load L2 weights (partial — copy what fits)
    l2_path = "src/sara_brain/cortex/checkpoints/l2_en_003000.pt"
    l2_ckpt = torch.load(l2_path, map_location="cpu", weights_only=False)
    l2_state = l2_ckpt.get("model_state_dict", l2_ckpt)
    model_state = model.backbone.state_dict()

    loaded = 0
    for name, p in l2_state.items():
        if name in model_state:
            mp = model_state[name]
            if "tok_embed" in name or "pos_embed" in name:
                # Copy rows that fit, leave rest at random init
                rows = min(p.shape[0], mp.shape[0])
                mp[:rows].copy_(p[:rows])
                loaded += 1
            elif mp.shape == p.shape:
                mp.copy_(p)
                loaded += 1
    model.backbone.load_state_dict(model_state)
    print(f"Loaded {loaded} params from L2 grammar backbone")

    params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {params:,} ({params/1e6:.0f}M)")

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
                    ids, pos, labels, correct = encode_example(ex, tok2id)
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
                            "step": step, "val_acc": pick_acc},
                           f"{args.out}/best.pt")
                print(f"  >>> new best!")
            model.train()

    print(f"\nDone in {time.time()-t0:.0f}s. Best val pick accuracy: {best_val_acc*100:.0f}%")


if __name__ == "__main__":
    main()
