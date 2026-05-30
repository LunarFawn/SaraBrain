"""Train L3-substrate — substrate-reading layer on top of frozen L1/L2.

Same pattern as train_l2.py: loads the grammar backbone, extends the
vocabulary with substrate-format tokens, freezes the transformer blocks,
and trains only the new embedding rows + a classification head.

The model learns to read Sara Brain wavefront output through the same
grammar backbone that understands English structure. Substrate format
is treated as another language layered on the universal grammar.

Architecture:
    L1 (76 UD tags) — universal grammar, frozen
    L2 (175 tokens) — English function words, frozen
    L3 (L2 + substrate tokens) — wavefront format, TRAINED
    Classification head — 4-way MCQ, TRAINED

Usage:
    .venv/bin/python scripts/train_hamroby_substrate.py \
        --grammar-ckpt src/sara_brain/cortex/checkpoints/grammar_base_020000.pt \
        --data training_data/sara_cortex_synthetic_10k.jsonl \
        --out models/hamroby-substrate-v1 \
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sara_brain.cortex.transformer.model import GrammarConfig, GrammarModel


# ---- Substrate vocabulary ----

# Structural tokens for substrate format
SUBSTRATE_STRUCTURAL = [
    "<substrate>", "<question>", "<intersection>", "<reached>",
    "<strength>", "<seeds>", "<wavefront>",
    "(", ")", ",", ".", ":", "'", '"', "\n",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "strength=", "sources=",
    "intersection", "intersections", "reached", "neuron", "neurons",
    "seed", "seeds", "wavefront", "from", "convergence",
    "result", "map", "top", "full",
    "question", "which", "what", "does", "following",
    "a", "b", "c", "d",
]


def build_substrate_vocab(training_texts: list[str], base_vocab_size: int,
                          max_new_tokens: int = 3000) -> dict[str, int]:
    """Build substrate vocabulary extending the L1/L2 base.

    Tokenizes substrate text, counts frequencies, keeps top tokens.
    IDs start at base_vocab_size to not collide with L1/L2.
    """
    from collections import Counter
    counts = Counter()
    for text in training_texts:
        tokens = _tokenize_substrate(text)
        counts.update(tokens)

    # Start with structural tokens
    vocab = {}
    idx = base_vocab_size
    for tok in SUBSTRATE_STRUCTURAL:
        if tok not in vocab:
            vocab[tok] = idx
            idx += 1

    # Add most common tokens from training data
    for tok, _ in counts.most_common(max_new_tokens):
        if tok not in vocab:
            vocab[tok] = idx
            idx += 1
            if len(vocab) >= max_new_tokens:
                break

    return vocab


def _tokenize_substrate(text: str) -> list[str]:
    """Split substrate text into tokens."""
    return re.findall(r"[a-z_]+|[0-9]+\.[0-9]+|[0-9]+|[^\s]", text.lower())


# ---- Data ----

def load_examples(path: str) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def encode_example(ex: dict, l2_tok2id: dict, substrate_vocab: dict,
                   max_seq: int) -> tuple[list[int], int]:
    """Encode a training example into token IDs."""
    text = f"{ex['substrate']}\n\n{ex['question']}"
    tokens = _tokenize_substrate(text)[:max_seq - 2]

    # Map tokens: try substrate vocab first, then L2, then UNK
    unk_id = 4  # <unk> in L1 vocab
    ids = [1]  # <bos>
    for tok in tokens:
        if tok in substrate_vocab:
            ids.append(substrate_vocab[tok])
        elif tok in l2_tok2id:
            ids.append(l2_tok2id[tok])
        else:
            ids.append(unk_id)
    ids.append(2)  # <eos>

    label = ord(ex["answer"]) - ord("A")
    return ids, label


# ---- Training ----

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load L1/L2 vocab
    from sara_brain.cortex.transformer.vocab import TOK2ID as L1_TOK2ID, VOCAB_SIZE as L1_SIZE
    try:
        from sara_brain.cortex.transformer.vocab_en import TOK2ID_EN as L2_TOK2ID, VOCAB_SIZE_EN as L2_SIZE
    except ImportError:
        L2_TOK2ID = L1_TOK2ID
        L2_SIZE = L1_SIZE

    # Load grammar checkpoint
    print(f"Loading grammar backbone: {args.grammar_ckpt}")
    ckpt = torch.load(args.grammar_ckpt, map_location="cpu", weights_only=False)
    l1_config = ckpt["config"]
    print(f"  L1 config: d_model={l1_config['d_model']}, layers={l1_config['n_layers']}")

    # Load training data
    print(f"Loading data: {args.data}")
    examples = load_examples(args.data)
    rng = random.Random(42)
    rng.shuffle(examples)
    val_size = min(200, len(examples) // 10)
    val_examples = examples[:val_size]
    train_examples = examples[val_size:]
    print(f"  Train: {len(train_examples)}, Val: {val_size}")

    # Build substrate vocabulary
    print("Building substrate vocabulary...")
    texts = [f"{ex['substrate']}\n\n{ex['question']}" for ex in train_examples]
    substrate_vocab = build_substrate_vocab(texts, L2_SIZE, max_new_tokens=3000)
    total_vocab = L2_SIZE + len(substrate_vocab)
    print(f"  L1/L2 base: {L2_SIZE}, Substrate tokens: {len(substrate_vocab)}, Total: {total_vocab}")

    # Create L3 model (extended vocab)
    cfg = GrammarConfig(
        vocab_size=total_vocab,
        d_model=l1_config["d_model"],
        n_heads=l1_config["n_heads"],
        n_layers=l1_config["n_layers"],
        d_ff=l1_config["d_ff"],
        max_seq=args.max_seq,
        dropout=0.1,
    )
    model = GrammarModel(cfg)

    # Copy L1 weights into L3 (same as L2 projection pattern)
    l1_state = ckpt.get("model_state_dict", ckpt)
    l3_state = model.state_dict()
    for name, p in l1_state.items():
        if name not in l3_state:
            continue
        l3_p = l3_state[name]
        if "tok_embed" in name or "head" in name:
            # Copy L1 rows, leave new rows at random init
            rows_to_copy = min(p.shape[0], l3_p.shape[0])
            l3_p[:rows_to_copy].copy_(p[:rows_to_copy])
        elif "pos_embed" in name:
            # L1 has 96 positions, we need more — copy what exists
            rows_to_copy = min(p.shape[0], l3_p.shape[0])
            l3_p[:rows_to_copy].copy_(p[:rows_to_copy])
        elif l3_p.shape == p.shape:
            l3_p.copy_(p)
    model.load_state_dict(l3_state)

    # Add classification head (not part of GrammarModel)
    class L3WithHead(nn.Module):
        def __init__(self, backbone, d_model):
            super().__init__()
            self.backbone = backbone
            self.cls_head = nn.Linear(d_model, 4)

        def forward(self, input_ids):
            # Run backbone forward (returns logits, loss)
            # We need hidden states, so replicate the forward pass
            x = self.backbone.tok_embed(input_ids)
            B, T = input_ids.shape
            pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
            if T <= self.backbone.pos_embed.weight.shape[0]:
                x = x + self.backbone.pos_embed(pos)
            x = self.backbone.drop(x)
            attn_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            for block in self.backbone.blocks:
                x = block(x, attn_mask=attn_mask)
            x = self.backbone.ln_f(x)
            # Mean pool and classify
            pooled = x.mean(dim=1)
            return self.cls_head(pooled)

    full_model = L3WithHead(model, cfg.d_model).to(device)

    # Freeze backbone transformer blocks, only train embeddings + head
    for name, param in full_model.backbone.named_parameters():
        if "tok_embed" in name:
            param.requires_grad = True  # train new substrate embeddings
        else:
            param.requires_grad = False  # freeze grammar backbone
    for param in full_model.cls_head.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in full_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in full_model.parameters())
    print(f"  Trainable: {trainable:,} / {total_params:,} ({100*trainable/total_params:.1f}%)")

    # Optimizer (only trainable params)
    optimizer = AdamW(
        [p for p in full_model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    # Training loop
    print(f"\nStarting training: {args.steps} steps")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    t0 = time.time()

    full_model.train()
    for step in range(1, args.steps + 1):
        # Cosine LR
        warmup = args.steps // 10
        if step < warmup:
            lr = args.lr * step / warmup
        else:
            progress = (step - warmup) / (args.steps - warmup)
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Make batch
        batch = rng.sample(train_examples, min(args.batch_size, len(train_examples)))
        ids_list, labels = [], []
        for ex in batch:
            ids, label = encode_example(ex, L2_TOK2ID, substrate_vocab, args.max_seq)
            ids_list.append(ids)
            labels.append(label)

        max_len = min(max(len(ids) for ids in ids_list), args.max_seq)
        padded = [ids[:max_len] + [0] * (max_len - len(ids[:max_len])) for ids in ids_list]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        targets = torch.tensor(labels, dtype=torch.long, device=device)

        logits = full_model(input_ids)
        loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(full_model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0:
            elapsed = time.time() - t0
            print(f"  step={step}/{args.steps} loss={loss.item():.4f} lr={lr:.6f} ({elapsed:.0f}s)")

        # Validation
        if step % 500 == 0 or step == args.steps:
            full_model.eval()
            correct = 0
            with torch.no_grad():
                for i in range(0, len(val_examples), args.batch_size):
                    batch = val_examples[i:i+args.batch_size]
                    ids_list, labs = [], []
                    for ex in batch:
                        ids, label = encode_example(ex, L2_TOK2ID, substrate_vocab, args.max_seq)
                        ids_list.append(ids)
                        labs.append(label)
                    max_len = min(max(len(ids) for ids in ids_list), args.max_seq)
                    padded = [ids[:max_len] + [0] * (max_len - len(ids[:max_len])) for ids in ids_list]
                    inp = torch.tensor(padded, dtype=torch.long, device=device)
                    tgt = torch.tensor(labs, dtype=torch.long, device=device)
                    preds = full_model(inp).argmax(dim=-1)
                    correct += (preds == tgt).sum().item()
            val_acc = correct / len(val_examples)
            print(f"  >>> val accuracy: {correct}/{len(val_examples)} = {val_acc*100:.1f}%")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "step": step,
                    "config": cfg.__dict__,
                    "model_state_dict": full_model.state_dict(),
                    "substrate_vocab": substrate_vocab,
                    "val_acc": val_acc,
                }, f"{args.out}/best.pt")
                print(f"  >>> new best! saved.")
            full_model.train()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. Best val accuracy: {best_val_acc*100:.1f}%")
    print(f"Saved to: {args.out}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--grammar-ckpt", required=True, help="L1 grammar checkpoint")
    ap.add_argument("--data", required=True, help="Training data .jsonl")
    ap.add_argument("--out", default="models/hamroby-substrate-v1", help="Output dir")
    ap.add_argument("--steps", type=int, default=5000, help="Training steps")
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size")
    ap.add_argument("--max-seq", type=int, default=512, help="Max sequence length")
    ap.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
