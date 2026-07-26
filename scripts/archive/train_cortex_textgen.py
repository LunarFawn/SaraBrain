"""Train from-scratch Sara cortex on text generation.

The model learns: given (question + substrate), generate the answer.
This IS a next-token prediction task — but conditioned on substrate
context. The Phase 1 LM already learned substrate format. Now we
fine-tune it to generate answers from substrate facts.

Usage:
    python scripts/train_cortex_textgen.py \
        --lm-checkpoint models/sara-cortex-lm-v1/best.pt \
        --data training_data/sara_cortex_textgen_2500.jsonl \
        --out models/sara-cortex-textgen-v1 \
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from train_substrate_lm import SubstrateLM, Config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm-checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="models/sara-cortex-textgen-v1")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Phase 1 LM
    print(f"Loading LM: {args.lm_checkpoint}")
    ckpt = torch.load(args.lm_checkpoint, map_location="cpu", weights_only=False)
    cfg = Config()
    for k, v in ckpt["config"].items():
        setattr(cfg, k, v)

    # Load training data first to extend vocabulary
    print(f"Loading: {args.data}")
    examples = []
    with open(args.data) as f:
        for line in f:
            examples.append(json.loads(line))
    rng = random.Random(42)
    rng.shuffle(examples)
    val = examples[:100]
    train_ex = examples[100:]
    print(f"  Train: {len(train_ex)}, Val: {len(val)}")

    # Extend tokenizer with answer vocabulary
    tok2id = dict(ckpt["tokenizer"])
    all_answer_text = " ".join(ex["answer"] for ex in examples)
    answer_tokens = re.findall(r"[a-z_]+|[0-9]+\.[0-9]+|[0-9]+|[^\s]", all_answer_text.lower())
    for t in set(answer_tokens):
        if t not in tok2id:
            tok2id[t] = len(tok2id)
    id2tok = {v: k for k, v in tok2id.items()}
    print(f"  Vocab extended: {ckpt['config']['vocab_size']} → {len(tok2id)}")

    # Rebuild model with extended vocab
    cfg.vocab_size = len(tok2id)
    model = SubstrateLM(cfg).to(device)
    # Load Phase 1 weights (partial — old vocab rows)
    old_state = ckpt["model"]
    new_state = model.state_dict()
    for name, p in old_state.items():
        if name in new_state:
            if "embed" in name or "head" in name:
                rows = min(p.shape[0], new_state[name].shape[0])
                new_state[name][:rows] = p[:rows]
            elif new_state[name].shape == p.shape:
                new_state[name].copy_(p)
    model.load_state_dict(new_state)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    # Special tokens
    SEP_ID = tok2id.get("<eos>", 2)

    def tokenize(text, max_len=None):
        tokens = re.findall(r"[a-z_]+|[0-9]+\.[0-9]+|[0-9]+|[^\s]", text.lower())
        ids = [tok2id.get(t, 3) for t in tokens]
        if max_len:
            ids = ids[:max_len]
        return ids

    # Training: input = [question + substrate + SEP + answer]
    # Loss only on the answer tokens (teacher forcing)
    max_ctx = cfg.max_seq - 30  # leave room for answer
    max_ans = 30

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
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
        batch_ids = []
        batch_loss_masks = []

        for ex in batch:
            ctx = f"{ex['question']}\n{ex['substrate']}"
            ctx_ids = tokenize(ctx, max_ctx)
            ans_ids = tokenize(ex["answer"], max_ans)
            # Full sequence: context + SEP + answer
            full = ctx_ids + [SEP_ID] + ans_ids
            full = full[:cfg.max_seq]
            # Loss mask: 0 for context, 1 for answer tokens
            mask = [0] * (len(ctx_ids) + 1) + [1] * len(ans_ids)
            mask = mask[:cfg.max_seq]
            batch_ids.append(full)
            batch_loss_masks.append(mask)

        # Pad
        ml = max(len(x) for x in batch_ids)
        padded = [x + [0] * (ml - len(x)) for x in batch_ids]
        masks_padded = [m + [0] * (ml - len(m)) for m in batch_loss_masks]

        inp = torch.tensor([x[:-1] for x in padded], dtype=torch.long, device=device)
        tgt = torch.tensor([x[1:] for x in padded], dtype=torch.long, device=device)
        loss_mask = torch.tensor([m[1:] for m in masks_padded], dtype=torch.float, device=device)

        logits, _ = model(inp)
        # Compute loss only on answer tokens
        loss_per_token = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none")
        loss_per_token = loss_per_token.reshape(tgt.shape)
        masked_loss = (loss_per_token * loss_mask).sum() / loss_mask.sum().clamp(min=1)

        optimizer.zero_grad()
        masked_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 200 == 0:
            print(f"  step={step}/{args.steps} loss={masked_loss.item():.4f} ({time.time()-t0:.0f}s)")

        if step % 2000 == 0 or step == args.steps:
            # Validate
            model.eval()
            val_loss_sum = 0.0
            val_tokens = 0
            with torch.no_grad():
                for ex in val[:50]:
                    ctx = f"{ex['question']}\n{ex['substrate']}"
                    ctx_ids = tokenize(ctx, max_ctx)
                    ans_ids = tokenize(ex["answer"], max_ans)
                    full = ctx_ids + [SEP_ID] + ans_ids
                    full = full[:cfg.max_seq]
                    mask = [0] * (len(ctx_ids) + 1) + [1] * len(ans_ids)
                    mask = mask[:cfg.max_seq]
                    inp_v = torch.tensor([full[:-1]], dtype=torch.long, device=device)
                    tgt_v = torch.tensor([full[1:]], dtype=torch.long, device=device)
                    mask_v = torch.tensor([mask[1:]], dtype=torch.float, device=device)
                    logits_v, _ = model(inp_v)
                    lpt = F.cross_entropy(
                        logits_v.reshape(-1, logits_v.size(-1)), tgt_v.reshape(-1), reduction="none")
                    lpt = lpt.reshape(tgt_v.shape)
                    val_loss_sum += (lpt * mask_v).sum().item()
                    val_tokens += mask_v.sum().item()
            val_loss = val_loss_sum / max(val_tokens, 1)
            val_ppl = math.exp(val_loss) if val_loss < 10 else float("inf")
            print(f"  >>> val loss={val_loss:.4f} ppl={val_ppl:.1f}")

            # Generate a sample
            ex = val[0]
            ctx = f"{ex['question']}\n{ex['substrate']}"
            ctx_ids = tokenize(ctx, max_ctx) + [SEP_ID]
            inp_g = torch.tensor([ctx_ids], dtype=torch.long, device=device)
            generated = []
            for _ in range(max_ans):
                logits_g, _ = model(inp_g[:, -cfg.max_seq:])
                next_id = logits_g[0, -1].argmax().item()
                if next_id == 0 or next_id == SEP_ID:
                    break
                generated.append(next_id)
                inp_g = torch.cat([inp_g, torch.tensor([[next_id]], device=device)], dim=1)
            gen_text = " ".join(id2tok.get(i, "?") for i in generated)
            print(f"  >>> sample Q: {ex['question'][:60]}")
            print(f"  >>> expected: {ex['answer'][:60]}")
            print(f"  >>> generated: {gen_text[:60]}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({"config": cfg.__dict__, "model": model.state_dict(),
                            "tokenizer": tok2id, "val_loss": val_loss},
                           f"{args.out}/best.pt")
                print(f"  >>> new best!")
            model.train()

    print(f"\nDone in {time.time()-t0:.0f}s. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
