"""Sara Cortex — from-scratch model with copy mechanism.

A small transformer that reads rendered wavefront facts and produces
answers by COPYING tokens from the input (pointer network) rather
than generating from vocabulary. This solves the rare-token problem:
nonsense concept labels don't need to be in the vocabulary — the
model points to them in the input.

Architecture:
  - Encoder: reads [facts + question] bidirectionally
  - Decoder: generates answer tokens by either:
    (a) copying a token from the input (pointer), or
    (b) generating from vocabulary (for relation verbs, punctuation)

This is the architecture for extractive QA from substrate facts.

Usage:
    python scripts/train_sara_cortex_copy.py \
        --data training_data/sara_cortex_srctext_2000.jsonl \
        --out models/sara-cortex-copy-v1 \
        --steps 15000
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


# Small fixed vocab for structure tokens (relations, punctuation)
# Concept labels are COPIED from input, not generated from vocab
BASE_VOCAB = ["<pad>", "<bos>", "<eos>", "<unk>", "<copy>",
              ".", ",", ":", "'", "(", ")", "\n", "-",
              # Relation verbs (the model generates these)
              "is", "is_a", "has", "has_property", "are", "was",
              "involves", "includes", "contains", "requires",
              "produces", "creates", "forms", "generates",
              "causes", "prevents", "reduces", "increases",
              "begins", "starts", "ends", "follows", "precedes",
              "opposes", "supports", "enables", "activates",
              "describes", "explains", "defines", "means",
              "within", "during", "after", "before",
              "part_of", "role_in", "act_within", "interacts_with",
              "provides", "holds", "joins", "breaks", "gets",
              "detects", "binds", "triggers", "associates",
              "becomes", "comprises", "encodes", "occurs",
              "predicts", "emphasizes", "contributes",
              # Common structure words
              "the", "a", "of", "and", "in", "to", "for", "from",
              "based", "on", "substrate", "facts", "about",
              ]


def tokenize(text):
    return re.findall(r"[a-z_']+|[0-9]+\.[0-9]+|[0-9]+|[^\s]", text.lower())


class CopyEncoder(nn.Module):
    """Bidirectional encoder for input facts + question."""
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=4, max_seq=300):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq, d_model)
        self.drop = nn.Dropout(0.1)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model*4,
                                          dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, input_ids, pad_mask=None):
        B, T = input_ids.shape
        x = self.embed(input_ids) + self.pos(torch.arange(T, device=input_ids.device))
        x = self.drop(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.ln(x)


class CopyDecoder(nn.Module):
    """Decoder with copy mechanism — can point to input tokens."""
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=2, max_seq=50):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq, d_model)
        self.drop = nn.Dropout(0.1)
        layer = nn.TransformerDecoderLayer(d_model, n_heads, d_model*4,
                                          dropout=0.1, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, n_layers)
        self.ln = nn.LayerNorm(d_model)

        # Generate probability (vocab) vs copy probability (pointer)
        self.gen_proj = nn.Linear(d_model, vocab_size)
        self.copy_gate = nn.Linear(d_model, 1)  # sigmoid → p(copy)
        self.copy_attn = nn.Linear(d_model, d_model)  # for pointer attention

    def forward(self, tgt_ids, enc_out, enc_input_ids, tgt_mask=None, enc_pad_mask=None):
        B, T_dec = tgt_ids.shape
        T_enc = enc_out.shape[1]

        x = self.embed(tgt_ids) + self.pos(torch.arange(T_dec, device=tgt_ids.device))
        x = self.drop(x)
        causal_mask = torch.triu(torch.ones(T_dec, T_dec, device=x.device), diagonal=1).bool()
        x = self.decoder(x, enc_out, tgt_mask=causal_mask,
                         memory_key_padding_mask=enc_pad_mask)
        x = self.ln(x)

        # Generate distribution over vocab
        gen_logits = self.gen_proj(x)  # (B, T_dec, vocab_size)
        gen_probs = F.softmax(gen_logits, dim=-1)

        # Copy distribution over input positions
        copy_scores = torch.bmm(self.copy_attn(x), enc_out.transpose(1, 2))  # (B, T_dec, T_enc)
        if enc_pad_mask is not None:
            copy_scores = copy_scores.masked_fill(enc_pad_mask.unsqueeze(1), -1e9)
        copy_probs = F.softmax(copy_scores, dim=-1)  # (B, T_dec, T_enc)

        # Gate: probability of copying vs generating
        p_copy = torch.sigmoid(self.copy_gate(x))  # (B, T_dec, 1)

        # Scatter copy probs into vocab-sized tensor using input token IDs
        vocab_size = gen_logits.shape[-1]
        copy_vocab = torch.zeros(B, T_dec, vocab_size, device=x.device)
        enc_ids_expanded = enc_input_ids.unsqueeze(1).expand(-1, T_dec, -1)  # (B, T_dec, T_enc)
        copy_vocab.scatter_add_(2, enc_ids_expanded, copy_probs)

        # Final distribution: mix of generate and copy
        final_probs = (1 - p_copy) * gen_probs + p_copy * copy_vocab
        return final_probs


class SaraCortexCopy(nn.Module):
    """Full encoder-decoder with copy mechanism."""
    def __init__(self, vocab_size, d_model=256, enc_layers=4, dec_layers=2,
                 n_heads=8, max_enc=300, max_dec=50):
        super().__init__()
        self.encoder = CopyEncoder(vocab_size, d_model, n_heads, enc_layers, max_enc)
        self.decoder = CopyDecoder(vocab_size, d_model, n_heads, dec_layers, max_dec)
        self.vocab_size = vocab_size

    def forward(self, enc_ids, dec_ids, enc_pad_mask=None):
        enc_out = self.encoder(enc_ids, enc_pad_mask)
        probs = self.decoder(dec_ids, enc_out, enc_ids, enc_pad_mask=enc_pad_mask)
        return probs

    def generate(self, enc_ids, enc_pad_mask=None, max_len=30, bos_id=1, eos_id=2):
        enc_out = self.encoder(enc_ids, enc_pad_mask)
        B = enc_ids.shape[0]
        dec_ids = torch.full((B, 1), bos_id, dtype=torch.long, device=enc_ids.device)
        for _ in range(max_len):
            probs = self.decoder(dec_ids, enc_out, enc_ids, enc_pad_mask=enc_pad_mask)
            next_id = probs[:, -1, :].argmax(dim=-1, keepdim=True)
            dec_ids = torch.cat([dec_ids, next_id], dim=1)
            if (next_id == eos_id).all():
                break
        return dec_ids[:, 1:]  # strip bos


def build_vocab():
    """Build the fixed vocab (relations + structure). Concept labels are copied."""
    tok2id = {}
    for i, tok in enumerate(BASE_VOCAB):
        tok2id[tok] = i
    return tok2id


def encode_with_oov(text, tok2id, max_len=300):
    """Encode text, mapping OOV tokens to unique IDs beyond base vocab.
    Returns (ids, oov_list) where oov_list maps extended IDs back to tokens."""
    tokens = tokenize(text)[:max_len]
    ids = []
    oov = []
    oov_map = {}
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
    ap.add_argument("--out", default="models/sara-cortex-copy-v1")
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tok2id = build_vocab()
    base_vocab_size = len(tok2id)
    print(f"Base vocab: {base_vocab_size}")

    # Load data
    examples = []
    with open(args.data) as f:
        for line in f:
            ex = json.loads(line)
            examples.append(ex)
    rng = random.Random(42)
    rng.shuffle(examples)
    val = examples[:100]
    train_ex = examples[100:]
    print(f"Train: {len(train_ex)}, Val: {len(val)}")

    # Extended vocab size (base + max OOV per example)
    # We'll use a fixed extended size to keep tensors uniform
    max_oov = 200
    ext_vocab_size = base_vocab_size + max_oov

    model = SaraCortexCopy(ext_vocab_size, d_model=256, enc_layers=4,
                           dec_layers=2, n_heads=8, max_enc=280, max_dec=40).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,} ({params/1e6:.1f}M)")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    t0 = time.time()
    MAX_ENC = 280
    MAX_DEC = 35

    for step in range(1, args.steps + 1):
        w = args.steps // 10
        lr = args.lr * (step/w if step < w else 0.5*(1+math.cos(math.pi*(step-w)/(args.steps-w))))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        batch = rng.sample(train_ex, args.batch_size)
        enc_batch, dec_in_batch, dec_tgt_batch, pad_masks = [], [], [], []

        for ex in batch:
            input_text = ex.get("facts", ex.get("rendered_facts", "")) + " " + ex["question"]
            enc_ids, oov, oov_map = encode_with_oov(input_text, tok2id, MAX_ENC)

            # Encode answer using same OOV mapping
            ans_tokens = tokenize(ex["answer"])[:MAX_DEC]
            ans_ids = []
            for t in ans_tokens:
                if t in tok2id:
                    ans_ids.append(tok2id[t])
                elif t in oov_map:
                    ans_ids.append(oov_map[t])
                else:
                    ans_ids.append(tok2id["<unk>"])

            dec_in = [tok2id["<bos>"]] + ans_ids
            dec_tgt = ans_ids + [tok2id["<eos>"]]

            enc_batch.append(enc_ids)
            dec_in_batch.append(dec_in)
            dec_tgt_batch.append(dec_tgt)

        # Pad
        max_enc_len = max(len(x) for x in enc_batch)
        max_dec_len = max(len(x) for x in dec_in_batch)
        enc_padded = [x + [0]*(max_enc_len-len(x)) for x in enc_batch]
        dec_in_padded = [x + [0]*(max_dec_len-len(x)) for x in dec_in_batch]
        dec_tgt_padded = [x + [0]*(max_dec_len-len(x)) for x in dec_tgt_batch]
        pad_mask = [[False]*len(x) + [True]*(max_enc_len-len(x)) for x in enc_batch]

        enc_t = torch.tensor(enc_padded, dtype=torch.long, device=device)
        dec_in_t = torch.tensor(dec_in_padded, dtype=torch.long, device=device)
        dec_tgt_t = torch.tensor(dec_tgt_padded, dtype=torch.long, device=device)
        pad_t = torch.tensor(pad_mask, dtype=torch.bool, device=device)

        probs = model(enc_t, dec_in_t, pad_t)
        # Loss: NLL on target tokens
        probs_clamped = probs.clamp(min=1e-9)
        loss_per_tok = -torch.log(probs_clamped.gather(2, dec_tgt_t.unsqueeze(2)).squeeze(2))
        # Mask padding in target
        tgt_mask = (dec_tgt_t != 0).float()
        loss = (loss_per_tok * tgt_mask).sum() / tgt_mask.sum().clamp(min=1)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 500 == 0:
            print(f"  step={step}/{args.steps} loss={loss.item():.4f} ({time.time()-t0:.0f}s)")

        if step % 3000 == 0 or step == args.steps:
            model.eval()
            # Generate on val example
            ex = val[0]
            input_text = ex.get("facts", ex.get("rendered_facts", "")) + " " + ex["question"]
            enc_ids, oov, oov_map = encode_with_oov(input_text, tok2id, MAX_ENC)
            enc_t = torch.tensor([enc_ids], dtype=torch.long, device=device)
            pad_t = torch.zeros(1, len(enc_ids), dtype=torch.bool, device=device)

            with torch.no_grad():
                out_ids = model.generate(enc_t, pad_t, max_len=25)[0].tolist()

            # Decode: base vocab + OOV
            id2tok_ext = {v: k for k, v in tok2id.items()}
            for t, idx in oov_map.items():
                id2tok_ext[idx] = t
            gen_text = " ".join(id2tok_ext.get(i, f"[{i}]") for i in out_ids if i not in (0, 2))

            print(f"  Q: {ex['question'][:60]}")
            print(f"  Expected: {ex['answer'][:60]}")
            print(f"  Generated: {gen_text[:60]}")

            if loss.item() < best_val:
                best_val = loss.item()
                torch.save({"model": model.state_dict(), "tok2id": tok2id,
                            "step": step, "loss": best_val},
                           f"{args.out}/best.pt")
                print(f"  >>> new best!")
            model.train()

    print(f"\nDone in {time.time()-t0:.0f}s. Best loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
