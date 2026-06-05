"""Character-level MCQ eliminator.

Two phases in one script:
1. Pretrain a character-level masked LM (learns token identity at char level)
2. Fine-tune elimination head (picks the CONSISTENT choice)

Character vocab: 26 letters + digits + punctuation + specials = ~45 tokens.
No word is ever OOV. Every concept is fully visible as a char sequence.

Usage:
    # Full run (pretrain + fine-tune)
    python scripts/train_char_eliminator.py \
        --pretrain-data training_data/substrate_lm_100k.txt \
        --finetune-data training_data/eliminator_500k.jsonl \
        --out models/sara-char-eliminator \
        --pretrain-steps 50000 \
        --finetune-steps 50000
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW


# Character vocabulary
CHARS = list("abcdefghijklmnopqrstuvwxyz0123456789 .,;:!?-_'\"()/\n")
SPECIAL = ["<pad>", "<mask>", "<sep>", "<choice>"]
CHAR_VOCAB = SPECIAL + CHARS
CHAR2ID = {c: i for i, c in enumerate(CHAR_VOCAB)}
VOCAB_SIZE = len(CHAR_VOCAB)
PAD_ID = 0
MASK_ID = 1
SEP_ID = 2
CHOICE_ID = 3
MAX_CHOICES = 8


def char_encode(text, max_len=512):
    """Encode text as character IDs."""
    ids = []
    for c in text.lower()[:max_len]:
        ids.append(CHAR2ID.get(c, CHAR2ID[" "]))  # unknown chars → space
    return ids


class CharTransformer(nn.Module):
    """Character-level bidirectional transformer."""
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=256, n_heads=8,
                 n_layers=6, max_seq=1024):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq, d_model)
        self.drop = nn.Dropout(0.1)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4,
                                          dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.ln = nn.LayerNorm(d_model)
        # MLM head
        self.mlm_head = nn.Linear(d_model, vocab_size)

    def encode(self, input_ids, pad_mask=None):
        B, T = input_ids.shape
        x = self.embed(input_ids) + self.pos(torch.arange(T, device=input_ids.device))
        x = self.drop(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.ln(x)

    def forward_mlm(self, input_ids, pad_mask=None):
        hidden = self.encode(input_ids, pad_mask)
        return self.mlm_head(hidden)


class CharEliminator(nn.Module):
    """Character transformer + elimination head."""
    def __init__(self, backbone: CharTransformer):
        super().__init__()
        self.backbone = backbone
        self.choice_head = nn.Sequential(
            nn.Linear(backbone.d_model, backbone.d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(backbone.d_model // 2, 3),
        )

    def forward(self, input_ids, choice_positions, pad_mask=None):
        hidden = self.backbone.encode(input_ids, pad_mask)
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


def pretrain_mlm(model, data_path, steps, batch_size, device, lr=3e-4, max_seq=512):
    """Phase 1: Masked LM pretraining at character level."""
    print("\n=== Phase 1: Character Masked LM ===")
    with open(data_path) as f:
        raw = f.read()
    # Split into chunks
    chunks = [c.strip() for c in raw.split("\n\n") if len(c.strip()) > 50]
    print(f"  {len(chunks)} chunks")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = random.Random(42)
    t0 = time.time()
    model.train()

    for step in range(1, steps + 1):
        warmup = steps // 10
        cur_lr = lr * (step / warmup if step < warmup else
                       0.5 * (1 + math.cos(math.pi * (step - warmup) / (steps - warmup))))
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        # Build batch
        batch_masked, batch_labels = [], []
        for _ in range(batch_size):
            chunk = rng.choice(chunks)
            ids = char_encode(chunk, max_seq)
            # Mask 15%
            masked, labels = list(ids), [-100] * len(ids)
            for i in range(len(ids)):
                if ids[i] < 4:
                    continue
                if rng.random() < 0.15:
                    labels[i] = ids[i]
                    r = rng.random()
                    if r < 0.8:
                        masked[i] = MASK_ID
                    elif r < 0.9:
                        masked[i] = rng.randint(4, VOCAB_SIZE - 1)
            batch_masked.append(masked)
            batch_labels.append(labels)

        max_len = max(len(x) for x in batch_masked)
        padded = [x + [0] * (max_len - len(x)) for x in batch_masked]
        lab_padded = [x + [-100] * (max_len - len(x)) for x in batch_labels]
        pm = [[False] * len(x) + [True] * (max_len - len(x)) for x in batch_masked]

        inp = torch.tensor(padded, dtype=torch.long, device=device)
        lab = torch.tensor(lab_padded, dtype=torch.long, device=device)
        mask = torch.tensor(pm, dtype=torch.bool, device=device)

        logits = model.forward_mlm(inp, mask)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), lab.view(-1), ignore_index=-100)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 1000 == 0:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                valid = lab != -100
                acc = (preds[valid] == lab[valid]).float().mean().item() if valid.any() else 0
            print(f"  step={step}/{steps} loss={loss.item():.4f} mask_acc={acc:.2f} ({time.time()-t0:.0f}s)")

    print(f"  Pretrain done in {time.time()-t0:.0f}s")
    return model


def finetune_eliminator(model, data_path, steps, batch_size, device, out_path, lr=1e-4, max_seq=512):
    """Phase 2: Fine-tune elimination head."""
    print("\n=== Phase 2: Elimination Fine-tune ===")
    eliminator = CharEliminator(model).to(device)

    examples = []
    with open(data_path) as f:
        for line in f:
            examples.append(json.loads(line))
    rng = random.Random(42)
    rng.shuffle(examples)
    val = examples[:200]
    train_ex = examples[200:]
    print(f"  Train: {len(train_ex)}, Val: {len(val)}")

    optimizer = AdamW(eliminator.parameters(), lr=lr, weight_decay=0.01)
    best_val_acc = 0.0
    t0 = time.time()
    eliminator.train()

    for step in range(1, steps + 1):
        warmup = steps // 10
        cur_lr = lr * (step / warmup if step < warmup else
                       0.5 * (1 + math.cos(math.pi * (step - warmup) / (steps - warmup))))
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        batch = rng.sample(train_ex, batch_size)
        all_ids, all_pos, all_labels, all_correct = [], [], [], []

        for ex in batch:
            # Encode as characters: [facts SEP question CHOICE choiceA CHOICE choiceB ...]
            text = ex["facts"]
            ids = char_encode(text, max_seq - 100)
            ids.append(SEP_ID)
            ids.extend(char_encode(ex["question"], 60))

            choice_positions = [-1] * MAX_CHOICES
            for i, choice in enumerate(ex["choices_list"][:MAX_CHOICES]):
                ids.append(CHOICE_ID)
                choice_positions[i] = len(ids) - 1
                ids.extend(char_encode(choice, 40))

            ids = ids[:max_seq]
            labels = ex["labels"][:MAX_CHOICES] + [-1] * (MAX_CHOICES - len(ex["labels"]))
            all_ids.append(ids)
            all_pos.append(choice_positions)
            all_labels.append(labels)
            all_correct.append(ex["correct_idx"])

        max_len = max(len(x) for x in all_ids)
        padded = [x + [0] * (max_len - len(x)) for x in all_ids]
        pm = [[False] * len(x) + [True] * (max_len - len(x)) for x in all_ids]

        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        positions = torch.tensor(all_pos, dtype=torch.long, device=device)
        labels_t = torch.tensor(all_labels, dtype=torch.long, device=device)
        pad_mask = torch.tensor(pm, dtype=torch.bool, device=device)

        logits = eliminator(input_ids, positions, pad_mask)

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
        nn.utils.clip_grad_norm_(eliminator.parameters(), 1.0)
        optimizer.step()

        if step % 500 == 0:
            consistent_scores = logits[:, :, 1]
            preds = consistent_scores.argmax(dim=-1)
            correct_t = torch.tensor(all_correct, device=device)
            acc = (preds == correct_t).float().mean().item()
            print(f"  step={step}/{steps} loss={loss.item():.4f} pick_acc={acc:.2f} ({time.time()-t0:.0f}s)")

        if step % 5000 == 0 or step == steps:
            eliminator.eval()
            correct_count = 0
            with torch.no_grad():
                for ex in val[:100]:
                    text = ex["facts"]
                    ids = char_encode(text, max_seq - 100)
                    ids.append(SEP_ID)
                    ids.extend(char_encode(ex["question"], 60))
                    choice_positions = [-1] * MAX_CHOICES
                    for i, choice in enumerate(ex["choices_list"][:MAX_CHOICES]):
                        ids.append(CHOICE_ID)
                        choice_positions[i] = len(ids) - 1
                        ids.extend(char_encode(choice, 40))
                    ids = ids[:max_seq]

                    inp = torch.tensor([ids], dtype=torch.long, device=device)
                    p = torch.tensor([choice_positions], dtype=torch.long, device=device)
                    out = eliminator(inp, p)
                    if out[0, :, 1].argmax().item() == ex["correct_idx"]:
                        correct_count += 1
            pick_acc = correct_count / 100
            print(f"  >>> val pick_acc: {correct_count}/100 = {pick_acc*100:.0f}%")
            if pick_acc > best_val_acc:
                best_val_acc = pick_acc
                torch.save({"model": eliminator.state_dict(), "step": step,
                            "val_acc": pick_acc}, f"{out_path}/best.pt")
                print(f"  >>> new best!")
            eliminator.train()

    print(f"\n  Fine-tune done in {time.time()-t0:.0f}s. Best val: {best_val_acc*100:.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrain-data", required=True)
    ap.add_argument("--finetune-data", required=True)
    ap.add_argument("--out", default="models/sara-char-eliminator")
    ap.add_argument("--pretrain-steps", type=int, default=50000)
    ap.add_argument("--finetune-steps", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-seq", type=int, default=512)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Char vocab: {VOCAB_SIZE} tokens")

    Path(args.out).mkdir(parents=True, exist_ok=True)

    # Build model
    model = CharTransformer(VOCAB_SIZE, d_model=256, n_heads=8, n_layers=6,
                            max_seq=args.max_seq).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,} ({params/1e6:.0f}M)")

    # Phase 1: Pretrain
    model = pretrain_mlm(model, args.pretrain_data, args.pretrain_steps,
                         args.batch_size, device, max_seq=args.max_seq)

    # Save pretrained
    torch.save({"model": model.state_dict()}, f"{args.out}/pretrained.pt")

    # Phase 2: Fine-tune
    finetune_eliminator(model, args.finetune_data, args.finetune_steps,
                        args.batch_size, device, args.out, max_seq=args.max_seq)


if __name__ == "__main__":
    main()
