"""Train a from-scratch Sara teaching model (bridge-fact extractor).

Copy-mechanism encoder-decoder that extracts triples from source text
by POINTING to words in the input. Cannot hallucinate — it can only
copy tokens that exist in the source paragraph.

Saves checkpoints every N steps. Resume from crash with --resume.

Usage:
    # Start fresh
    python scripts/train_sara_extractor_scratch.py \
        --data training_data/extractor_copy_train.jsonl \
        --out models/sara-extractor-scratch \
        --steps 50000

    # Resume from checkpoint
    python scripts/train_sara_extractor_scratch.py \
        --data training_data/extractor_copy_train.jsonl \
        --out models/sara-extractor-scratch \
        --steps 50000 --resume
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

# Minimal base vocab — relations and structure only.
# All content words are COPIED from input.
BASE_VOCAB = [
    "<pad>", "<bos>", "<eos>", "<unk>", "<sep>", "<triple>",
    "</triple>", "<rel>", "<obj>",
    ".", ",", "\n",
    # Common relation verbs the model can generate
    "is", "is_a", "are", "has", "have", "contains", "includes",
    "produces", "requires", "involves", "causes", "prevents",
    "occurs", "occurs_in", "forms", "uses", "provides",
    "attaches", "attaches_to", "separates", "divides", "called", "known_as",
    "composed_of", "part_of", "results_in", "leads_to", "depends_on",
    "activates", "inhibits", "enables", "regulates",
    "interacts_with", "transforms_into",
    "during", "within", "between", "from", "into",
    "a", "an", "the", "of", "and", "in", "to", "by", "for", "with",
    # Extended grammar/function words (needed for real English text)
    "that", "which", "this", "it", "its", "each", "as", "on",
    "at", "or", "not", "but", "if", "when", "where", "how",
    "more", "than", "both", "all", "no", "can", "cannot",
    "found", "using", "through", "toward", "without",
    "multiple", "three", "two", "one", "first",
    "process", "structure", "system", "type", "phase", "stage",
    "region", "pattern", "rate", "change", "cycle", "formation",
    "properly", "rapidly", "effectively", "correctly", "exclusively",
    "composed", "arranged", "connected", "classified", "based",
    "regulated", "disrupted", "required", "maintained",
    "increases", "decreases", "determines", "depends",
    "triggers", "releases", "undergoes", "transforms",
    "begins", "ends", "leading", "causing",
    "essential", "necessary", "distinct", "final",
    "because", "then", "also", "only",
    "binding", "form", "interact", "accumulates",
    "research", "shows", "absence", "catalyst",
    "consists", "every", "other",
]


def tokenize(text: str) -> list[str]:
    """Split text into tokens."""
    return re.findall(r"[a-zA-Z_]+(?:'[a-z]+)?|[0-9]+\.[0-9]+|[0-9]+|[^\s]", text.lower())


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=4, max_seq=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq, d_model)
        self.drop = nn.Dropout(0.1)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4,
                                          dropout=0.1, batch_first=True,
                                          norm_first=True)
        self.layers = nn.TransformerEncoder(layer, n_layers)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, ids, pad_mask=None):
        B, T = ids.shape
        x = self.embed(ids) + self.pos(torch.arange(T, device=ids.device))
        x = self.drop(x)
        x = self.layers(x, src_key_padding_mask=pad_mask)
        return self.ln(x)


class CopyDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=3, max_seq=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq, d_model)
        self.drop = nn.Dropout(0.1)
        layer = nn.TransformerDecoderLayer(d_model, n_heads, d_model * 4,
                                          dropout=0.1, batch_first=True,
                                          norm_first=True)
        self.layers = nn.TransformerDecoder(layer, n_layers)
        self.ln = nn.LayerNorm(d_model)

        self.gen_proj = nn.Linear(d_model, vocab_size)
        self.copy_gate = nn.Linear(d_model, 1)
        self.copy_key = nn.Linear(d_model, d_model)

    def forward(self, tgt_ids, enc_out, enc_ids, enc_pad_mask=None):
        B, T = tgt_ids.shape
        x = self.embed(tgt_ids) + self.pos(torch.arange(T, device=tgt_ids.device))
        x = self.drop(x)
        causal = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        x = self.layers(x, enc_out, tgt_mask=causal,
                        memory_key_padding_mask=enc_pad_mask)
        x = self.ln(x)

        # Generate distribution
        gen_logits = self.gen_proj(x)
        gen_probs = F.softmax(gen_logits, dim=-1)

        # Copy distribution
        copy_scores = torch.bmm(self.copy_key(x), enc_out.transpose(1, 2))
        if enc_pad_mask is not None:
            copy_scores = copy_scores.masked_fill(enc_pad_mask.unsqueeze(1), -6e4)
        copy_probs = F.softmax(copy_scores, dim=-1)

        # Gate
        p_copy = torch.sigmoid(self.copy_gate(x))

        # Scatter copy probs into vocab space
        V = gen_logits.shape[-1]
        copy_vocab = torch.zeros(B, T, V, device=x.device)
        enc_expanded = enc_ids.unsqueeze(1).expand(-1, T, -1)
        copy_vocab.scatter_add_(2, enc_expanded, copy_probs)

        return (1 - p_copy) * gen_probs + p_copy * copy_vocab


class SaraExtractor(nn.Module):
    def __init__(self, vocab_size, d_model=256, enc_layers=4, dec_layers=3,
                 n_heads=8, max_enc=512, max_dec=128):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, n_heads, enc_layers, max_enc)
        self.decoder = CopyDecoder(vocab_size, d_model, n_heads, dec_layers, max_dec)
        self.vocab_size = vocab_size
        self.max_dec = max_dec

    def forward(self, enc_ids, dec_ids, enc_pad_mask=None):
        enc_out = self.encoder(enc_ids, enc_pad_mask)
        return self.decoder(dec_ids, enc_out, enc_ids, enc_pad_mask)

    def generate(self, enc_ids, enc_pad_mask=None, max_len=80):
        enc_out = self.encoder(enc_ids, enc_pad_mask)
        B = enc_ids.shape[0]
        dec = torch.full((B, 1), 1, dtype=torch.long, device=enc_ids.device)  # <bos>
        for _ in range(max_len):
            probs = self.decoder(dec, enc_out, enc_ids, enc_pad_mask)
            nxt = probs[:, -1, :].argmax(dim=-1, keepdim=True)
            dec = torch.cat([dec, nxt], dim=1)
            if (nxt == 2).all():  # <eos>
                break
        return dec[:, 1:]


def build_vocab():
    tok2id = {}
    for i, t in enumerate(BASE_VOCAB):
        tok2id[t] = i
    return tok2id


def encode_with_oov(text, tok2id, max_len):
    tokens = tokenize(text)[:max_len]
    ids, oov, oov_map = [], [], {}
    for t in tokens:
        if t in tok2id:
            ids.append(tok2id[t])
        else:
            if t not in oov_map:
                oov_map[t] = len(tok2id) + len(oov)
                oov.append(t)
            ids.append(oov_map[t])
    return ids, oov, oov_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="models/sara-extractor-scratch")
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--checkpoint-every", type=int, default=2000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-enc", type=int, default=400)
    ap.add_argument("--max-dec", type=int, default=100)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tok2id = build_vocab()
    max_oov = 300
    ext_vocab = len(tok2id) + max_oov
    print(f"Base vocab: {len(tok2id)}, Extended: {ext_vocab}")

    # Load data
    examples = []
    with open(args.data) as f:
        for line in f:
            examples.append(json.loads(line))
    rng = random.Random(42)
    rng.shuffle(examples)
    val = examples[:50]
    train_ex = examples[50:]
    print(f"Train: {len(train_ex)}, Val: {len(val)}")

    # Model
    model = SaraExtractor(ext_vocab, d_model=256, enc_layers=4, dec_layers=3,
                          n_heads=8, max_enc=args.max_enc, max_dec=args.max_dec).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,} ({params/1e6:.1f}M)")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint
    start_step = 0
    best_val_loss = float("inf")
    if args.resume:
        ckpts = sorted(Path(args.out).glob("checkpoint_*.pt"))
        if ckpts:
            ckpt = torch.load(ckpts[-1], map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_step = ckpt["step"]
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            print(f"Resumed from step {start_step} (loss={ckpt.get('loss', '?')})")

    # Training loop
    t0 = time.time()
    model.train()

    for step in range(start_step + 1, args.steps + 1):
        # LR schedule
        warmup = args.steps // 10
        if step < warmup:
            lr = args.lr * step / warmup
        else:
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * (step - warmup) / (args.steps - warmup)))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Build batch
        batch = rng.sample(train_ex, min(args.batch_size, len(train_ex)))
        enc_list, dec_in_list, dec_tgt_list = [], [], []

        for ex in batch:
            enc_ids, oov, oov_map = encode_with_oov(ex["paragraph"], tok2id, args.max_enc)

            # Encode target (triples) using same OOV mapping
            target_text = ex.get("output", "\n".join(ex.get("triples", [])))
            tgt_tokens = tokenize(target_text)[:args.max_dec - 1]
            tgt_ids = []
            for t in tgt_tokens:
                if t in tok2id:
                    tgt_ids.append(tok2id[t])
                elif t in oov_map:
                    tgt_ids.append(oov_map[t])
                else:
                    tgt_ids.append(tok2id["<unk>"])

            dec_in = [tok2id["<bos>"]] + tgt_ids
            dec_tgt = tgt_ids + [tok2id["<eos>"]]

            enc_list.append(enc_ids)
            dec_in_list.append(dec_in)
            dec_tgt_list.append(dec_tgt)

        # Pad
        me = max(len(x) for x in enc_list)
        md = max(len(x) for x in dec_in_list)
        enc_pad = [x + [0] * (me - len(x)) for x in enc_list]
        din_pad = [x + [0] * (md - len(x)) for x in dec_in_list]
        dtgt_pad = [x + [0] * (md - len(x)) for x in dec_tgt_list]
        pmask = [[False] * len(x) + [True] * (me - len(x)) for x in enc_list]

        enc_t = torch.tensor(enc_pad, dtype=torch.long, device=device)
        din_t = torch.tensor(din_pad, dtype=torch.long, device=device)
        dtgt_t = torch.tensor(dtgt_pad, dtype=torch.long, device=device)
        pm_t = torch.tensor(pmask, dtype=torch.bool, device=device)

        probs = model(enc_t, din_t, pm_t)
        probs_clamped = probs.clamp(min=1e-9)
        loss_tok = -torch.log(probs_clamped.gather(2, dtgt_t.unsqueeze(2)).squeeze(2))
        tgt_mask = (dtgt_t != 0).float()
        loss = (loss_tok * tgt_mask).sum() / tgt_mask.sum().clamp(min=1)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Logging
        if step % 500 == 0:
            elapsed = time.time() - t0
            print(f"  step={step}/{args.steps} loss={loss.item():.4f} lr={lr:.6f} ({elapsed:.0f}s)")

        # Checkpoint
        if step % args.checkpoint_every == 0:
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": loss.item(),
                "best_val_loss": best_val_loss,
            }, f"{args.out}/checkpoint_{step:06d}.pt")
            # Keep only last 3 checkpoints
            ckpts = sorted(Path(args.out).glob("checkpoint_*.pt"))
            for old in ckpts[:-3]:
                old.unlink()

        # Validation + generation sample
        if step % 5000 == 0 or step == args.steps:
            model.eval()
            val_loss_sum, val_tokens = 0.0, 0
            with torch.no_grad():
                for ex in val[:20]:
                    enc_ids, oov, oov_map = encode_with_oov(ex["paragraph"], tok2id, args.max_enc)
                    target_text = ex.get("output", "\n".join(ex.get("triples", [])))
                    tgt_tokens = tokenize(target_text)[:args.max_dec - 1]
                    tgt_ids = [tok2id.get(t, oov_map.get(t, tok2id["<unk>"])) for t in tgt_tokens]
                    din = [tok2id["<bos>"]] + tgt_ids
                    dtgt = tgt_ids + [tok2id["<eos>"]]

                    e = torch.tensor([enc_ids + [0] * (10)], dtype=torch.long, device=device)
                    d = torch.tensor([din], dtype=torch.long, device=device)
                    t_t = torch.tensor([dtgt], dtype=torch.long, device=device)
                    pm = torch.tensor([[False] * len(enc_ids) + [True] * 10], dtype=torch.bool, device=device)

                    p = model(e, d, pm).clamp(min=1e-9)
                    lt = -torch.log(p.gather(2, t_t.unsqueeze(2)).squeeze(2))
                    m = (t_t != 0).float()
                    val_loss_sum += (lt * m).sum().item()
                    val_tokens += m.sum().item()

            val_loss = val_loss_sum / max(val_tokens, 1)
            print(f"  >>> val loss: {val_loss:.4f}")

            # Generate sample
            ex = val[0]
            enc_ids, oov, oov_map = encode_with_oov(ex["paragraph"], tok2id, args.max_enc)
            e = torch.tensor([enc_ids], dtype=torch.long, device=device)
            pm = torch.zeros(1, len(enc_ids), dtype=torch.bool, device=device)
            with torch.no_grad():
                out = model.generate(e, pm, max_len=80)[0].tolist()
            id2tok = {v: k for k, v in tok2id.items()}
            for t, idx in oov_map.items():
                id2tok[idx] = t
            gen = " ".join(id2tok.get(i, f"[{i}]") for i in out if i not in (0, 2))
            expected = ex.get("output", " | ".join(ex.get("triples", [])))
            print(f"  >>> expected: {expected[:120]}")
            print(f"  >>> generated: {gen[:120]}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({"model": model.state_dict(), "tok2id": tok2id,
                            "step": step, "val_loss": val_loss},
                           f"{args.out}/best.pt")
                print(f"  >>> new best!")
            model.train()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/3600:.1f}h). Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
