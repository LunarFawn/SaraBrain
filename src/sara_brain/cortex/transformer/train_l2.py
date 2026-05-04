"""Train L2-en — function-word adapter on top of frozen L1 grammar LM.

Loads an L1 checkpoint (vocab_size=76, see vocab.py), instantiates a
matching architecture sized for the L2-en vocabulary (vocab_size=175,
see vocab_en.py), copies L1 weights verbatim, leaves the new
function-word embedding rows at random init, and trains them on
lexicalized UD English.

The transformer blocks, position embedding, and final layernorm stay
frozen — L1's universal grammatical capacity is preserved untouched.
Only the token embedding (which is weight-tied to the LM head, see
model.py:71) is updated. That single matrix carries the entire L2-en
overlay.

Output: `l2_en_{step:06d}.pt` checkpoints next to the L1 checkpoints,
loadable for L2 inference.

Usage:
    .venv/bin/python -m sara_brain.cortex.transformer.train_l2 \\
        --grammar-ckpt src/sara_brain/cortex/checkpoints/grammar_base_015000.pt \\
        --steps 3000
"""
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
from .train import cosine_lr, eval_perplexity, fmt_mem
from .vocab import VOCAB_SIZE as L1_VOCAB_SIZE
from .vocab_en import EN_FUNCTION_WORD_SET, TOK2ID_EN, VOCAB_SIZE_EN


def project_l1_into_l2(l1_state_dict: dict, l2_model: GrammarModel) -> dict:
    """Copy every L1 parameter into L2's matching slot. For `tok_embed`,
    copy rows [0, L1_VOCAB_SIZE) and leave rows [L1_VOCAB_SIZE,
    VOCAB_SIZE_EN) at the L2 random init.

    `head.weight` is tied to `tok_embed.weight` (see model.py:71), so it
    needs no separate projection — but the L1 checkpoint may have it
    saved redundantly; we skip that key on the L2 side."""
    l2_state = l2_model.state_dict()
    skipped: list[str] = []
    copied: list[str] = []
    padded: list[str] = []
    for name, p in l1_state_dict.items():
        if name not in l2_state:
            skipped.append(name)
            continue
        l2_p = l2_state[name]
        if name == "tok_embed.weight":
            assert p.shape[0] == L1_VOCAB_SIZE, (
                f"L1 tok_embed has {p.shape[0]} rows, expected {L1_VOCAB_SIZE}"
            )
            assert l2_p.shape[0] == VOCAB_SIZE_EN, (
                f"L2 tok_embed has {l2_p.shape[0]} rows, expected {VOCAB_SIZE_EN}"
            )
            assert p.shape[1] == l2_p.shape[1], (
                f"d_model mismatch: L1 {p.shape[1]} vs L2 {l2_p.shape[1]}"
            )
            l2_p[:L1_VOCAB_SIZE].copy_(p)
            padded.append(name)
        elif name == "head.weight":
            # Tied to tok_embed; the L2 model already shares this matrix.
            skipped.append(name)
        elif l2_p.shape == p.shape:
            l2_p.copy_(p)
            copied.append(name)
        else:
            raise ValueError(
                f"shape mismatch on {name}: L1 {p.shape} vs L2 {l2_p.shape}"
            )
    l2_model.load_state_dict(l2_state)
    return {"copied": copied, "padded": padded, "skipped": skipped}


def freeze_l1_params(model: GrammarModel) -> tuple[list, list]:
    """Freeze everything except tok_embed (which is tied to the LM head,
    so training it trains both at once). Returns (trainable, frozen)
    parameter lists for the optimizer / logging."""
    trainable: list = []
    frozen: list = []
    for name, p in model.named_parameters():
        if name == "tok_embed.weight":
            p.requires_grad = True
            trainable.append((name, p))
        else:
            p.requires_grad = False
            frozen.append((name, p))
    return trainable, frozen


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--grammar-ckpt", type=Path, required=True,
        help="L1 grammar checkpoint to start from (vocab_size=76)",
    )
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max-seq", type=int, default=96)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument(
        "--ckpt-every", type=int, default=5000,
        help="Steps between ckpt saves. v035 default 5000 — single-final "
             "for typical L2 adapter runs (we always save at args.steps too).",
    )
    p.add_argument(
        "--ckpt-dir", type=Path,
        default=Path("src/sara_brain/cortex/checkpoints"),
    )
    p.add_argument(
        "--lang", default="en",
        help="Language tag for checkpoint naming (l2_<lang>_<step>.pt). "
             "Currently only 'en' has a vocab module.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--no-amp", action="store_true")
    p.add_argument(
        "--unfreeze-l1", action="store_true",
        help="Train all parameters (slower, may degrade L1's universal "
             "structural priors). Default keeps L1 layers frozen and "
             "trains only the embedding/LM-head adapter.",
    )
    args = p.parse_args()

    if args.lang != "en":
        raise SystemExit(
            f"--lang={args.lang!r} not supported yet — add vocab_{args.lang}.py first"
        )

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    eval_rng = random.Random(args.seed + 1)
    torch.manual_seed(args.seed)

    # Load L1 checkpoint and reconstruct its config.
    print(f"[l2] loading L1 checkpoint: {args.grammar_ckpt}", flush=True)
    ck = torch.load(args.grammar_ckpt, map_location="cpu", weights_only=False)
    l1_cfg = GrammarConfig(**ck["config"])
    if l1_cfg.vocab_size != L1_VOCAB_SIZE:
        raise SystemExit(
            f"L1 checkpoint vocab_size={l1_cfg.vocab_size} but "
            f"vocab.VOCAB_SIZE={L1_VOCAB_SIZE}; refusing to project"
        )

    # Build L2 model with the same architecture but L2-en vocab size.
    l2_cfg = GrammarConfig(
        vocab_size=VOCAB_SIZE_EN,
        d_model=l1_cfg.d_model,
        n_heads=l1_cfg.n_heads,
        n_layers=l1_cfg.n_layers,
        d_ff=l1_cfg.d_ff,
        max_seq=max(l1_cfg.max_seq, args.max_seq),
        dropout=l1_cfg.dropout,
        pad_id=l1_cfg.pad_id,
    )
    device = torch.device(args.device)
    model = GrammarModel(l2_cfg).to(device)

    # Project L1 weights into L2 model.
    l1_state = ck["state_dict"]
    proj_report = project_l1_into_l2(l1_state, model)
    print(
        f"[l2] projected L1 -> L2: copied={len(proj_report['copied'])} "
        f"padded={len(proj_report['padded'])} skipped={len(proj_report['skipped'])}",
        flush=True,
    )

    # Freeze L1 layers unless --unfreeze-l1.
    if args.unfreeze_l1:
        trainable = list(model.named_parameters())
        frozen: list = []
    else:
        trainable, frozen = freeze_l1_params(model)
    n_trainable = sum(p.numel() for _, p in trainable)
    n_frozen = sum(p.numel() for _, p in frozen)
    print(
        f"[l2] trainable params: {n_trainable:,}  frozen: {n_frozen:,}  "
        f"({'all' if args.unfreeze_l1 else 'tok_embed only'})",
        flush=True,
    )

    # L2 dataset: lexicalized UD English encoded against TOK2ID_EN.
    train_ds = UDStreamDataset(
        split="train",
        lexicalize_function_words=True,
        function_word_set=EN_FUNCTION_WORD_SET,
        vocab_table=TOK2ID_EN,
    )
    dev_ds = UDStreamDataset(
        split="dev",
        lexicalize_function_words=True,
        function_word_set=EN_FUNCTION_WORD_SET,
        vocab_table=TOK2ID_EN,
    )

    opt = AdamW(
        [p for _, p in trainable], lr=args.lr,
        betas=(0.9, 0.95), weight_decay=0.1,
    )

    use_amp = (device.type == "cuda") and not args.no_amp
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    print("=" * 78, flush=True)
    print(f"start  {datetime.now().isoformat(timespec='seconds')}", flush=True)
    print(
        f"L2-{args.lang}  vocab={VOCAB_SIZE_EN}  d={l2_cfg.d_model} h={l2_cfg.n_heads} "
        f"L={l2_cfg.n_layers} ff={l2_cfg.d_ff} seq={l2_cfg.max_seq}",
        flush=True,
    )
    print(
        f"device={device}  amp={use_amp}  batch={args.batch}  steps={args.steps}  "
        f"lr={args.lr:g}->{args.min_lr:g} warmup={args.warmup}",
        flush=True,
    )
    print(
        f"data: lexicalized UD English  train={len(train_ds)} sentences  "
        f"dev={len(dev_ds)}",
        flush=True,
    )
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

    # Initial dev ppl (sanity: L2 with random function-word rows should be
    # noticeably worse than L1's structural ppl — that's the headroom we're
    # training away).
    init_ppl = eval_perplexity(
        model, dev_ds, device, args.batch, l2_cfg.max_seq,
        args.eval_batches, eval_rng, use_amp, amp_dtype,
    )
    print(f"[eval] step=0 (pre-train)  dev_ppl={init_ppl:.3f}", flush=True)

    for step in range(1, args.steps + 1):
        lr = cosine_lr(step, args.warmup, args.steps, args.lr, args.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr

        inp, tgt = make_lm_batch(train_ds, args.batch, l2_cfg.max_seq, rng)
        inp = inp.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            _, loss = model(inp, target_ids=tgt)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for _, p in trainable], 1.0,
        )
        opt.step()

        last_loss = loss.item()
        tokens_since_log += inp.numel()

        if step % args.log_every == 0 or step == 1:
            dt = time.time() - t_log
            tps = tokens_since_log / max(dt, 1e-6)
            train_ppl = math.exp(min(20.0, last_loss))
            print(
                f"{step:6d}  {last_loss:6.4f}  {train_ppl:6.2f}  "
                f"{lr:7.2e}  {tps:8.0f}  {fmt_mem(device)}",
                flush=True,
            )
            t_log = time.time()
            tokens_since_log = 0

        if step % args.eval_every == 0 or step == args.steps:
            last_dev_ppl = eval_perplexity(
                model, dev_ds, device, args.batch, l2_cfg.max_seq,
                args.eval_batches, eval_rng, use_amp, amp_dtype,
            )
            print(f"[eval] step={step}  dev_ppl={last_dev_ppl:.3f}", flush=True)

        if step % args.ckpt_every == 0 or step == args.steps:
            path = args.ckpt_dir / f"l2_{args.lang}_{step:06d}.pt"
            sd = (
                model._orig_mod.state_dict() if hasattr(model, "_orig_mod")
                else model.state_dict()
            )
            torch.save({
                "step": step,
                "loss": last_loss,
                "dev_ppl": last_dev_ppl,
                "config": l2_cfg.__dict__,
                "lang": args.lang,
                "l1_ckpt": str(args.grammar_ckpt),
                "frozen_l1": not args.unfreeze_l1,
                "state_dict": sd,
                "optimizer_state": opt.state_dict(),
                "rng_state": rng.getstate(),
            }, path)
            print(f"[ckpt] {path}", flush=True)

    print("=" * 78, flush=True)
    print(
        f"done   {datetime.now().isoformat(timespec='seconds')}  "
        f"final_loss={last_loss:.4f}  final_dev_ppl={last_dev_ppl:.3f}  "
        f"(pre-train ppl was {init_ppl:.3f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
