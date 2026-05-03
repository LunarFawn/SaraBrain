"""Train the router-classification head on top of a frozen Grammar LM.

Usage:
  python -m sara_brain.cortex.transformer.train_router \\
      --grammar-ckpt src/sara_brain/cortex/checkpoints/grammar_base_015000.pt \\
      --brain aptamer_full.db.bak \\
      --steps 2000
"""
from __future__ import annotations

import argparse
import math
import random
import time
from datetime import datetime
from pathlib import Path

import spacy
import torch
import torch.nn as nn
from torch.optim import AdamW

from .model import GrammarConfig, GrammarModel
from .router_data import (
    N_TOOLS, TOOL2ID, generate, load_substrate, split_train_dev,
)
from .router_head import RouterHead
from .vocab import BOS_ID, EOS_ID, PAD_ID, TOK2ID, UNK_ID, VOCAB_SIZE


def encode_tags(tags: list[str], max_seq: int) -> list[int]:
    ids = [BOS_ID] + [TOK2ID.get(t, UNK_ID) for t in tags] + [EOS_ID]
    if len(ids) > max_seq:
        ids = ids[:max_seq]
    return ids + [PAD_ID] * (max_seq - len(ids))


def make_batch(
    examples, batch_size: int, max_seq: int, rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    picked = rng.sample(examples, batch_size)
    input_ids = torch.tensor(
        [encode_tags(ex.tag_stream, max_seq) for ex in picked], dtype=torch.long
    )
    labels = torch.tensor([TOOL2ID[ex.tool] for ex in picked], dtype=torch.long)
    return input_ids, labels


@torch.no_grad()
def eval_accuracy(
    head: RouterHead, examples, device: torch.device, max_seq: int, batch_size: int,
) -> tuple[float, dict]:
    head.eval()
    correct = 0
    total = 0
    per_class_correct: dict[str, list[int]] = {}
    for i in range(0, len(examples), batch_size):
        batch = examples[i:i + batch_size]
        if not batch:
            continue
        x = torch.tensor(
            [encode_tags(ex.tag_stream, max_seq) for ex in batch],
            dtype=torch.long, device=device,
        )
        y = torch.tensor([TOOL2ID[ex.tool] for ex in batch], dtype=torch.long, device=device)
        pred = head.predict(x)
        eq = (pred == y)
        correct += int(eq.sum().item())
        total += len(batch)
        for ex, ok in zip(batch, eq.tolist()):
            per_class_correct.setdefault(ex.tool, [0, 0])
            per_class_correct[ex.tool][0] += int(ok)
            per_class_correct[ex.tool][1] += 1
    head.train()
    summary = {tool: f"{c}/{n} ({c/n:.1%})" for tool, (c, n) in per_class_correct.items()}
    return correct / max(1, total), summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--grammar-ckpt", type=Path, required=True)
    p.add_argument("--brain", type=Path, required=True)
    p.add_argument("--out", type=Path,
                   default=Path("src/sara_brain/cortex/checkpoints/router_head.pt"))
    p.add_argument("--n-per-class", type=int, default=2000)
    p.add_argument("--max-seq", type=int, default=64)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--unfreeze-encoder", action="store_true",
                   help="Fine-tune the grammar LM together with the head (default: frozen)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)

    print(f"[load] grammar checkpoint: {args.grammar_ckpt}", flush=True)
    ck = torch.load(args.grammar_ckpt, map_location=device, weights_only=False)
    cfg = GrammarConfig(**ck["config"])
    cfg.max_seq = max(cfg.max_seq, args.max_seq)
    encoder = GrammarModel(cfg).to(device)
    encoder.load_state_dict(ck["state_dict"])
    print(f"[load] grammar params: {encoder.num_params()/1e6:.2f}M  "
          f"(trained dev_ppl={ck.get('dev_ppl'):.3f})", flush=True)

    print(f"[data] loading substrate {args.brain}", flush=True)
    nlp = spacy.load("en_core_web_sm")
    substrate = load_substrate(args.brain)
    print(f"[data] {len(substrate['concepts'])} concepts, "
          f"{len(substrate['value_pairs'])} value pairs", flush=True)
    examples = generate(substrate, nlp, rng, n_per_class=args.n_per_class)
    train_ex, dev_ex = split_train_dev(examples, dev_frac=0.1, seed=args.seed)
    print(f"[data] {len(train_ex)} train, {len(dev_ex)} dev", flush=True)

    head = RouterHead(encoder, N_TOOLS, freeze_encoder=not args.unfreeze_encoder).to(device)
    print(f"[head] trainable params: {head.num_trainable_params()/1e6:.3f}M  "
          f"(encoder {'unfrozen' if args.unfreeze_encoder else 'frozen'})", flush=True)

    opt = AdamW([p for p in head.parameters() if p.requires_grad],
                lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    print("=" * 70, flush=True)
    print("  step    loss    train_acc   lr", flush=True)
    print("-" * 70, flush=True)

    head.train()
    last_loss = float("nan")
    last_acc = 0.0
    t0 = time.time()

    for step in range(1, args.steps + 1):
        if step <= args.warmup:
            lr = args.lr * step / args.warmup
        else:
            progress = (step - args.warmup) / max(1, args.steps - args.warmup)
            lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = max(lr, args.lr * 0.01)
        for g in opt.param_groups:
            g["lr"] = lr

        inp, labels = make_batch(train_ex, args.batch, args.max_seq, rng)
        inp = inp.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        logits, loss = head(inp, labels=labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in head.parameters() if p.requires_grad], 1.0
        )
        opt.step()

        last_loss = loss.item()
        last_acc = (logits.argmax(-1) == labels).float().mean().item()

        if step % args.log_every == 0 or step == 1:
            print(f"{step:6d}  {last_loss:6.4f}    {last_acc:.3f}     {lr:.2e}", flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            acc, breakdown = eval_accuracy(head, dev_ex, device, args.max_seq, args.batch)
            print(f"[eval] step={step}  dev_acc={acc:.3f}  per-class={breakdown}", flush=True)

    final_acc, final_breakdown = eval_accuracy(head, dev_ex, device, args.max_seq, args.batch)
    print("=" * 70, flush=True)
    print(f"done  {datetime.now().isoformat(timespec='seconds')}  "
          f"final dev_acc={final_acc:.3f}  elapsed={time.time()-t0:.1f}s", flush=True)
    print(f"per-class: {final_breakdown}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "head_state_dict": head.classifier.state_dict(),
        "encoder_ckpt_path": str(args.grammar_ckpt),
        "encoder_unfrozen": args.unfreeze_encoder,
        "n_classes": N_TOOLS,
        "tool_classes": list(TOOL2ID.keys()),
        "max_seq": args.max_seq,
        "dev_acc": final_acc,
        "dev_per_class": final_breakdown,
    }, args.out)
    print(f"[save] {args.out}", flush=True)


if __name__ == "__main__":
    main()
