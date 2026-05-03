from __future__ import annotations

import argparse
import math
import random
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.optim import AdamW

from .model import GrammarConfig, GrammarModel
from .synthetic import UDStreamDataset, make_lm_batch
from .vocab import PAD_ID, VOCAB_SIZE


PRESETS = {
    "tiny": GrammarConfig.tiny,
    "base": GrammarConfig.base_125m,
    "prod": GrammarConfig.prod_300m,
}


def cosine_lr(step: int, warmup: int, total: int, base_lr: float, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def fmt_mem(device: torch.device) -> str:
    if device.type != "cuda":
        return "—"
    used = torch.cuda.memory_allocated(device) / 1e6
    peak = torch.cuda.max_memory_allocated(device) / 1e6
    return f"{used:6.0f}MB (peak {peak:.0f}MB)"


@torch.no_grad()
def eval_perplexity(
    model: GrammarModel,
    dataset: UDStreamDataset,
    device: torch.device,
    batch_size: int,
    max_seq: int,
    n_batches: int,
    rng: random.Random,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for _ in range(n_batches):
        inp, tgt = make_lm_batch(dataset, batch_size, max_seq, rng)
        inp = inp.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            _, loss = model(inp, target_ids=tgt)
        total_loss += loss.item()
        n += 1
    model.train()
    avg = total_loss / max(1, n)
    return math.exp(avg)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=list(PRESETS), default="tiny")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max-seq", type=int, default=96)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=500,
                   help="Steps between dev-set perplexity evaluations")
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--ckpt-dir", type=Path, default=Path("src/sara_brain/cortex/checkpoints"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--compile", action="store_true",
                   help="Enable torch.compile (requires a C compiler installed)")
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    eval_rng = random.Random(args.seed + 1)
    torch.manual_seed(args.seed)

    train_ds = UDStreamDataset(split="train")
    dev_ds = UDStreamDataset(split="dev")

    cfg = PRESETS[args.preset](VOCAB_SIZE)
    cfg.max_seq = args.max_seq
    cfg.pad_id = PAD_ID

    device = torch.device(args.device)
    model = GrammarModel(cfg).to(device)
    n_params = model.num_params()

    opt = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    use_amp = (device.type == "cuda") and not args.no_amp
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    if args.compile and device.type == "cuda":
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"[warn] torch.compile disabled: {e}", flush=True)

    print("=" * 78, flush=True)
    print(f"start  {datetime.now().isoformat(timespec='seconds')}", flush=True)
    print(f"preset={args.preset}  params={n_params/1e6:.2f}M  vocab={VOCAB_SIZE}  "
          f"d={cfg.d_model} h={cfg.n_heads} L={cfg.n_layers} ff={cfg.d_ff} seq={cfg.max_seq}", flush=True)
    print(f"device={device}  amp={use_amp}  batch={args.batch}  steps={args.steps}  "
          f"lr={args.lr:g}->{args.min_lr:g} warmup={args.warmup}", flush=True)
    print(f"data: UD English EWT  train={len(train_ds)} sentences  dev={len(dev_ds)}", flush=True)
    print(f"ckpts -> {args.ckpt_dir}", flush=True)
    print("=" * 78, flush=True)
    header = "  step    loss     ppl     lr       tok/s     gpu"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    model.train()
    t_log = time.time()
    tokens_since_log = 0
    last_loss = float("nan")
    last_dev_ppl = float("nan")

    for step in range(1, args.steps + 1):
        lr = cosine_lr(step, args.warmup, args.steps, args.lr, args.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr

        inp, tgt = make_lm_batch(train_ds, args.batch, cfg.max_seq, rng)
        inp = inp.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            _, loss = model(inp, target_ids=tgt)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        last_loss = loss.item()
        tokens_since_log += inp.numel()

        if step % args.log_every == 0 or step == 1:
            dt = time.time() - t_log
            tps = tokens_since_log / max(dt, 1e-6)
            train_ppl = math.exp(min(20.0, last_loss))
            print(f"{step:6d}  {last_loss:6.4f}  {train_ppl:6.2f}  {lr:7.2e}  {tps:8.0f}  {fmt_mem(device)}",
                  flush=True)
            t_log = time.time()
            tokens_since_log = 0

        if step % args.eval_every == 0 or step == args.steps:
            last_dev_ppl = eval_perplexity(
                model, dev_ds, device, args.batch, cfg.max_seq,
                args.eval_batches, eval_rng, use_amp, amp_dtype,
            )
            print(f"[eval] step={step}  dev_ppl={last_dev_ppl:.3f}", flush=True)

        if step % args.ckpt_every == 0 or step == args.steps:
            path = args.ckpt_dir / f"grammar_{args.preset}_{step:06d}.pt"
            sd = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
            torch.save({
                "step": step,
                "loss": last_loss,
                "dev_ppl": last_dev_ppl,
                "config": cfg.__dict__,
                "preset": args.preset,
                "state_dict": sd,
            }, path)
            print(f"[ckpt] {path}", flush=True)

    print("=" * 78, flush=True)
    print(f"done   {datetime.now().isoformat(timespec='seconds')}  "
          f"final_loss={last_loss:.4f}  final_dev_ppl={last_dev_ppl:.3f}", flush=True)


if __name__ == "__main__":
    main()
