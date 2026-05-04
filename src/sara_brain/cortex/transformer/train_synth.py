"""Train HamRobySum — the synthesizer head on top of L1 + L2-en.

Loads an L2-en checkpoint (which already carries L1 weights inside),
projects into a brain-extended synth model whose vocabulary is
`VOCAB_SYNTH` (192 base tokens) + per-brain content words (from the
sidecar .vocab.json produced by synth_data.py --serialize-out).

By default freezes L1+L2 transformer blocks, position embedding, and
final layernorm — trains only the token embedding (weight-tied to
the LM head, so head trains automatically). The new content-word and
delimiter rows are random-init at the start; the existing 175 L2-en
rows are projected verbatim.

Standard next-token cross-entropy with a per-position loss mask: only
positions whose next token is part of the prose continuation
contribute to loss. The facts prefix is conditioning context, not a
prediction target.

See docs/v033_reproducible_hamroby_loop.md for the architecture and
docs/v031_train_l2_plan.md for the L2 trainer this mirrors.

Usage:
    .venv/bin/python -m sara_brain.cortex.transformer.synth_data \\
        --brain /tmp/sara_demo.db \\
        --serialize-out /tmp/synth_pairs.jsonl
    .venv/bin/python -m sara_brain.cortex.transformer.train_synth \\
        --l2-ckpt src/sara_brain/cortex/checkpoints/l2_en_003000.pt \\
        --pairs /tmp/synth_pairs.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.optim import AdamW

from .model import GrammarConfig, GrammarModel
from .train import cosine_lr, fmt_mem
from .vocab_en import VOCAB_SIZE_EN
from .vocab_synth import PAD_ID, VOCAB_SIZE_SYNTH


def project_base_into_synth(
    base_state_dict: dict,
    synth_model: GrammarModel,
    base_vocab_size: int,
) -> dict:
    """Copy every base parameter into the synth model. The token
    embedding is padded — first `base_vocab_size` rows from base, the
    rest stay at the synth model's random init. The tied LM head
    follows automatically."""
    synth_state = synth_model.state_dict()
    copied: list[str] = []
    padded: list[str] = []
    skipped: list[str] = []
    for name, p in base_state_dict.items():
        if name not in synth_state:
            skipped.append(name)
            continue
        target = synth_state[name]
        if name == "tok_embed.weight":
            assert p.shape[0] == base_vocab_size, (
                f"base tok_embed has {p.shape[0]} rows, expected {base_vocab_size}"
            )
            assert p.shape[1] == target.shape[1], (
                f"d_model mismatch: base {p.shape[1]} vs synth {target.shape[1]}"
            )
            target[:base_vocab_size].copy_(p)
            padded.append(name)
        elif name == "pos_embed.weight":
            # Same row-padding pattern as tok_embed: synth may have a
            # larger max_seq than the base; copy the rows that fit and
            # leave the rest at the synth model's random init.
            n_rows = min(p.shape[0], target.shape[0])
            assert p.shape[1] == target.shape[1], (
                f"d_model mismatch on pos_embed: base {p.shape[1]} vs "
                f"synth {target.shape[1]}"
            )
            target[:n_rows].copy_(p[:n_rows])
            padded.append(name)
        elif name == "head.weight":
            # Tied to tok_embed; the synth model already shares this matrix.
            skipped.append(name)
        elif target.shape == p.shape:
            target.copy_(p)
            copied.append(name)
        else:
            raise ValueError(
                f"shape mismatch on {name}: base {p.shape} vs synth {target.shape}"
            )
    synth_model.load_state_dict(synth_state)
    return {"copied": copied, "padded": padded, "skipped": skipped}


def freeze_base_params(model: GrammarModel) -> tuple[list, list]:
    """Freeze everything except the token embedding (which is tied to
    the LM head). Same strategy as train_l2.py."""
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


def load_serialized_pairs(path: Path) -> list[dict]:
    """Load a JSONL where each line is {input_ids, loss_mask, ...}."""
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def make_batch(
    rows: list[dict],
    indices: list[int],
    max_seq: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad selected rows to the longest in batch (capped at max_seq).
    Returns (input_ids, target_ids, loss_mask) — input == target for
    next-token LM loss."""
    selected = [rows[i] for i in indices]
    seq_lens = [min(len(r["input_ids"]), max_seq) for r in selected]
    pad_to = max(seq_lens)
    input_ids = torch.full((len(selected), pad_to), PAD_ID, dtype=torch.long)
    loss_mask = torch.zeros((len(selected), pad_to), dtype=torch.long)
    for i, r in enumerate(selected):
        n = min(len(r["input_ids"]), max_seq)
        input_ids[i, :n] = torch.tensor(r["input_ids"][:n], dtype=torch.long)
        loss_mask[i, :n] = torch.tensor(r["loss_mask"][:n], dtype=torch.long)
    return input_ids, input_ids.clone(), loss_mask


@torch.no_grad()
def eval_loss(
    model: GrammarModel,
    rows: list[dict],
    device: torch.device,
    batch_size: int,
    max_seq: int,
    n_batches: int,
    rng: random.Random,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    for _ in range(n_batches):
        idx = [rng.randrange(len(rows)) for _ in range(batch_size)]
        inp, tgt, mask = make_batch(rows, idx, max_seq)
        inp = inp.to(device); tgt = tgt.to(device); mask = mask.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            _, loss = model(inp, target_ids=tgt, loss_mask=mask)
        if loss is not None and torch.isfinite(loss):
            total += loss.item(); n += 1
    model.train()
    return total / max(1, n)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--l2-ckpt", type=Path, required=True,
        help="L2-en checkpoint (vocab_size=VOCAB_SIZE_EN). Carries the "
             "frozen L1 transformer blocks plus the trained L2-en token "
             "embedding rows.",
    )
    p.add_argument(
        "--pairs", type=Path, required=True,
        help="JSONL produced by synth_data.py --serialize-out. Expects "
             "a sidecar <pairs>.vocab.json with the brain-extended vocab.",
    )
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--max-seq", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-batches", type=int, default=10)
    p.add_argument("--ckpt-every", type=int, default=2000)
    p.add_argument("--dev-frac", type=float, default=0.1)
    p.add_argument(
        "--ckpt-dir", type=Path,
        default=Path("src/sara_brain/cortex/checkpoints"),
    )
    p.add_argument(
        "--ckpt-name", default=None,
        help="Stem for the synth checkpoint (default: hamroby_sum_<brain-stem>)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--no-amp", action="store_true")
    p.add_argument(
        "--unfreeze-base", action="store_true",
        help="Train all parameters (loses L1+L2 universality). Default "
             "freezes everything except the embedding.",
    )
    args = p.parse_args()

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    eval_rng = random.Random(args.seed + 1)
    torch.manual_seed(args.seed)

    # Load brain-extended vocab.
    vocab_path = args.pairs.with_suffix(args.pairs.suffix + ".vocab.json")
    with vocab_path.open() as f:
        brain_vocab = json.load(f)
    brain_vocab_size = brain_vocab["vocab_size"]
    print(
        f"[synth] brain vocab loaded: {brain_vocab_size} tokens "
        f"({brain_vocab_size - VOCAB_SIZE_SYNTH} brain content words above the "
        f"VOCAB_SIZE_SYNTH={VOCAB_SIZE_SYNTH} base)",
        flush=True,
    )

    # Load serialized rows.
    rows = load_serialized_pairs(args.pairs)
    rng.shuffle(rows)
    n_dev = max(1, int(len(rows) * args.dev_frac))
    dev_rows = rows[:n_dev]
    train_rows = rows[n_dev:]
    print(
        f"[synth] loaded {len(rows)} rows ({len(train_rows)} train / {len(dev_rows)} dev)",
        flush=True,
    )
    if not train_rows or not dev_rows:
        raise SystemExit("not enough data — increase --dev-frac or generate more pairs")

    # Load L2-en checkpoint and reconstruct its config.
    print(f"[synth] loading L2 base: {args.l2_ckpt}", flush=True)
    ck = torch.load(args.l2_ckpt, map_location="cpu", weights_only=False)
    base_cfg = GrammarConfig(**ck["config"])
    if base_cfg.vocab_size != VOCAB_SIZE_EN:
        raise SystemExit(
            f"L2 checkpoint vocab_size={base_cfg.vocab_size} but "
            f"VOCAB_SIZE_EN={VOCAB_SIZE_EN}; pass an L2-en checkpoint"
        )

    # Build synth model.
    synth_cfg = GrammarConfig(
        vocab_size=brain_vocab_size,
        d_model=base_cfg.d_model,
        n_heads=base_cfg.n_heads,
        n_layers=base_cfg.n_layers,
        d_ff=base_cfg.d_ff,
        max_seq=max(base_cfg.max_seq, args.max_seq),
        dropout=base_cfg.dropout,
        pad_id=base_cfg.pad_id,
    )
    device = torch.device(args.device)
    model = GrammarModel(synth_cfg).to(device)

    proj = project_base_into_synth(ck["state_dict"], model, base_cfg.vocab_size)
    print(
        f"[synth] projected L2 -> synth: copied={len(proj['copied'])} "
        f"padded={len(proj['padded'])} skipped={len(proj['skipped'])}",
        flush=True,
    )

    if args.unfreeze_base:
        trainable = list(model.named_parameters())
        frozen: list = []
    else:
        trainable, frozen = freeze_base_params(model)
    n_trainable = sum(p.numel() for _, p in trainable)
    n_frozen = sum(p.numel() for _, p in frozen)
    print(
        f"[synth] trainable params: {n_trainable:,}  frozen: {n_frozen:,}  "
        f"({'all' if args.unfreeze_base else 'tok_embed only'})",
        flush=True,
    )

    opt = AdamW(
        [p for _, p in trainable], lr=args.lr,
        betas=(0.9, 0.95), weight_decay=0.1,
    )

    use_amp = (device.type == "cuda") and not args.no_amp
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    ckpt_name = args.ckpt_name or f"hamroby_sum_{args.pairs.stem}"

    print("=" * 78, flush=True)
    print(f"start  {datetime.now().isoformat(timespec='seconds')}", flush=True)
    print(
        f"HamRobySum  vocab={brain_vocab_size}  d={synth_cfg.d_model} "
        f"h={synth_cfg.n_heads} L={synth_cfg.n_layers} ff={synth_cfg.d_ff} "
        f"seq={synth_cfg.max_seq}",
        flush=True,
    )
    print(
        f"device={device}  amp={use_amp}  batch={args.batch}  steps={args.steps}  "
        f"lr={args.lr:g}->{args.min_lr:g} warmup={args.warmup}",
        flush=True,
    )
    print(
        f"data: {args.pairs}  train={len(train_rows)} rows  dev={len(dev_rows)} rows",
        flush=True,
    )
    print(f"ckpts -> {args.ckpt_dir}/{ckpt_name}_*.pt", flush=True)
    print("=" * 78, flush=True)
    header = "  step    loss     ppl     lr       tok/s     gpu"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    model.train()
    t_log = time.time()
    tokens_since_log = 0
    last_loss = float("nan")
    last_dev_loss = float("nan")

    init_loss = eval_loss(
        model, dev_rows, device, args.batch, synth_cfg.max_seq,
        args.eval_batches, eval_rng, use_amp, amp_dtype,
    )
    print(
        f"[eval] step=0 (pre-train)  dev_loss={init_loss:.4f}  "
        f"dev_ppl={math.exp(min(20.0, init_loss)):.3f}",
        flush=True,
    )

    for step in range(1, args.steps + 1):
        lr = cosine_lr(step, args.warmup, args.steps, args.lr, args.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr

        idx = [rng.randrange(len(train_rows)) for _ in range(args.batch)]
        inp, tgt, mask = make_batch(train_rows, idx, synth_cfg.max_seq)
        inp = inp.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            _, loss = model(inp, target_ids=tgt, loss_mask=mask)

        if loss is None or not torch.isfinite(loss):
            # All positions in this batch were padding/masked — skip.
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
        opt.step()

        last_loss = loss.item()
        tokens_since_log += int(mask.sum().item())

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
            last_dev_loss = eval_loss(
                model, dev_rows, device, args.batch, synth_cfg.max_seq,
                args.eval_batches, eval_rng, use_amp, amp_dtype,
            )
            print(
                f"[eval] step={step}  dev_loss={last_dev_loss:.4f}  "
                f"dev_ppl={math.exp(min(20.0, last_dev_loss)):.3f}",
                flush=True,
            )

        if step % args.ckpt_every == 0 or step == args.steps:
            path = args.ckpt_dir / f"{ckpt_name}_{step:06d}.pt"
            sd = (
                model._orig_mod.state_dict() if hasattr(model, "_orig_mod")
                else model.state_dict()
            )
            torch.save({
                "step": step,
                "loss": last_loss,
                "dev_loss": last_dev_loss,
                "config": synth_cfg.__dict__,
                "brain_vocab": brain_vocab,   # full vocab list, for inference
                "pairs_path": str(args.pairs),
                "l2_ckpt": str(args.l2_ckpt),
                "frozen_base": not args.unfreeze_base,
                "state_dict": sd,
                "optimizer_state": opt.state_dict(),
                "rng_state": rng.getstate(),
            }, path)
            print(f"[ckpt] {path}", flush=True)

    print("=" * 78, flush=True)
    print(
        f"done   {datetime.now().isoformat(timespec='seconds')}  "
        f"final_loss={last_loss:.4f}  final_dev_loss={last_dev_loss:.4f}  "
        f"(pre-train dev_loss was {init_loss:.4f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
