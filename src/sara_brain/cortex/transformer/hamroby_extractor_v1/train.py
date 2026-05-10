"""Trainer for the grammar-feature transformer.

End-to-end supervised training. Loads synthetic Pairs (from
v2/synthetic_pairs.py), converts to grammar-feature examples, trains
GrammarEncoder + ExtractionHead jointly, evaluates triple_em on a
held-out set.

Usage:
    python -m sara_brain.cortex.transformer.hamroby_extractor_v1.train \\
        --out src/sara_brain/cortex/checkpoints/hamroby_extractor_v1.pt \\
        --size base --steps 5000 --batch-size 32 --lr 5e-4
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from ..v2 import synthetic_pairs
from .decoder import decode
from .extraction_head import ExtractionHead
from .feature_extractor import ParsedSentence
from .model import ExtractorConfig, GrammarEncoder, count_params
from .synthetic_features import GrammarTrainExample, build_examples
from .vocab import (
    PAD_DEP_ID, PAD_FUNCWORD_ID, PAD_OFFSET_ID, PAD_POS_ID,
)


@dataclass
class TrainConfig:
    out_path: Path
    n_train_scenes: int = 4000
    n_eval_scenes: int = 500
    qualifier_prob: float = 0.6
    size: str = "base"
    steps: int = 2000
    batch_size: int = 32
    max_seq: int = 64
    lr: float = 5e-4
    log_every: int = 50
    seed: int = 42
    real_prose_max_sentences: int = 0    # 0 = no real-prose mixing
    real_prose_treebanks: tuple[str, ...] | None = None  # None = all English UD


def _collate(
    batch: list[GrammarTrainExample],
    max_seq: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    B = len(batch)
    pos_ids = torch.full((B, max_seq), PAD_POS_ID, dtype=torch.long, device=device)
    dep_ids = torch.full((B, max_seq), PAD_DEP_ID, dtype=torch.long, device=device)
    off_ids = torch.full((B, max_seq), PAD_OFFSET_ID, dtype=torch.long, device=device)
    fw_ids = torch.full((B, max_seq), PAD_FUNCWORD_ID, dtype=torch.long, device=device)
    labels = torch.full((B, max_seq), -100, dtype=torch.long, device=device)
    for b, ex in enumerate(batch):
        L = min(len(ex.words), max_seq)
        pos_ids[b, :L] = torch.tensor(ex.pos_ids[:L], dtype=torch.long, device=device)
        dep_ids[b, :L] = torch.tensor(ex.dep_ids[:L], dtype=torch.long, device=device)
        off_ids[b, :L] = torch.tensor(ex.offset_ids[:L], dtype=torch.long, device=device)
        fw_ids[b, :L] = torch.tensor(ex.funcword_ids[:L], dtype=torch.long, device=device)
        labels[b, :L] = torch.tensor(ex.bio_labels[:L], dtype=torch.long, device=device)
    return {
        "pos_ids": pos_ids, "dep_ids": dep_ids,
        "offset_ids": off_ids, "funcword_ids": fw_ids,
        "labels": labels,
    }


def _evaluate(
    head: ExtractionHead,
    eval_pairs: list,
    eval_examples: list[GrammarTrainExample],
    nlp,
    max_seq: int,
    device: torch.device,
) -> tuple[float, float]:
    """Returns (token_accuracy_on_supervised_positions, triple_em).

    Triple_em compares the decoded triple's text to the original Pair's
    (subject, relation, object). Pair and example are aligned 1:1 by
    construction in build_examples (pair_to_example returns None on
    parse failure, those are dropped from BOTH lists).
    """
    head.eval()
    correct_tokens = total_tokens = 0
    triple_em = 0
    n = 0
    with torch.no_grad():
        for pair, ex in zip(eval_pairs, eval_examples):
            batch = _collate([ex], max_seq, device)
            logits, _ = head(
                batch["pos_ids"], batch["dep_ids"],
                batch["offset_ids"], batch["funcword_ids"],
                labels=batch["labels"],
            )
            preds = logits.argmax(dim=-1)[0].tolist()
            labs = batch["labels"][0].tolist()
            for p, l in zip(preds, labs):
                if l == -100:
                    continue
                total_tokens += 1
                if p == l:
                    correct_tokens += 1
            # Decode and compare. Need a ParsedSentence with the same
            # words / char_offsets the example was built from. Since
            # the example carries words verbatim, fabricate a minimal
            # ParsedSentence — the decoder only needs `.words`.
            ps = ParsedSentence(
                text=pair.prose, words=ex.words,
                feature_ids=list(zip(ex.pos_ids, ex.dep_ids,
                                     ex.offset_ids, ex.funcword_ids)),
                char_offsets=[(0, 0)] * len(ex.words),
            )
            triples = decode(ps, preds[:len(ex.words)])
            n += 1
            for t in triples:
                if (t.subject.lower().strip() == pair.subject.lower().strip()
                        and t.relation.lower().strip() == pair.relation.lower().strip()
                        and t.object.lower().strip() == pair.obj.lower().strip()):
                    triple_em += 1
                    break
    head.train()
    tok_acc = correct_tokens / max(1, total_tokens)
    em = triple_em / max(1, n)
    return tok_acc, em


def train(cfg: TrainConfig) -> Path:
    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[hamroby-extract] device={device}", file=sys.stderr)

    from .feature_extractor import load_domain_nlp
    nlp = load_domain_nlp(disable=["ner"])

    train_pairs = synthetic_pairs.generate_pairs(
        n_scenes=cfg.n_train_scenes, seed=cfg.seed,
        qualifier_prob=cfg.qualifier_prob,
    )
    eval_pairs = synthetic_pairs.generate_pairs(
        n_scenes=cfg.n_eval_scenes, seed=cfg.seed + 9999,
        qualifier_prob=cfg.qualifier_prob,
    )

    # Optional real-prose pairs from delexicalized UD treebanks. Mixed
    # into the training pool alongside the synthetic pairs so the head
    # sees real syntactic distributions (subordinate clauses, gerunds,
    # parentheticals, anaphora) at their natural frequency. Content is
    # delexicalized — same words in original prose map to consistent
    # nonsense substitutes — so content-orthogonality is preserved.
    if cfg.real_prose_max_sentences > 0:
        from .real_prose_pairs import generate_real_prose_pairs
        print(f"[hamroby-extract] generating real-prose pairs from UD "
              f"(max_sentences={cfg.real_prose_max_sentences})...",
              file=sys.stderr)
        treebanks = (list(cfg.real_prose_treebanks)
                     if cfg.real_prose_treebanks else None)
        real_pairs = generate_real_prose_pairs(
            nlp,
            treebanks=treebanks,
            max_sentences=cfg.real_prose_max_sentences,
            seed=cfg.seed,
        )
        train_pairs = train_pairs + real_pairs
        print(f"[hamroby-extract] real-prose pairs: {len(real_pairs)}; "
              f"total train_pairs={len(train_pairs)}", file=sys.stderr)

    print(f"[hamroby-extract] generating features (train {len(train_pairs)} "
          f"+ eval {len(eval_pairs)} pairs)...", file=sys.stderr)
    train_examples = build_examples(
        train_pairs, nlp, max_seq=cfg.max_seq, label="train",
    )
    # Re-align eval_pairs with successfully-parsed examples so triple_em
    # has matching ground truth.
    from .synthetic_features import pair_to_example
    eval_examples_full: list[GrammarTrainExample] = []
    eval_pairs_aligned: list = []
    eval_started = time.time()
    eval_n = len(eval_pairs)
    for i, p in enumerate(eval_pairs, start=1):
        ex = pair_to_example(p, nlp)
        if ex is not None and len(ex.words) <= cfg.max_seq - 2:
            eval_examples_full.append(ex)
            eval_pairs_aligned.append(p)
        if i % 5000 == 0 or i == eval_n:
            elapsed = time.time() - eval_started
            rate = i / max(1e-3, elapsed)
            eta = (eval_n - i) / rate if rate > 0 else 0.0
            print(
                f"[hamroby-extract] eval {i}/{eval_n} "
                f"({rate:.0f}/s, eta {eta:.0f}s, kept {len(eval_examples_full)})",
                file=sys.stderr,
            )
    print(f"[hamroby-extract] usable train_examples={len(train_examples)} "
          f"eval_examples={len(eval_examples_full)}", file=sys.stderr)

    size_factory = {
        "tiny": ExtractorConfig.tiny,
        "base": ExtractorConfig.base,
        "large": ExtractorConfig.large,
    }[cfg.size]
    enc_cfg = size_factory()
    if cfg.max_seq > enc_cfg.max_seq:
        enc_cfg.max_seq = cfg.max_seq
    encoder = GrammarEncoder(enc_cfg).to(device)
    head = ExtractionHead(encoder).to(device)
    print(f"[hamroby-extract] size={cfg.size} d_model={enc_cfg.d_model} "
          f"n_layers={enc_cfg.n_layers} params={count_params(head):,}",
          file=sys.stderr)

    optim = torch.optim.AdamW(
        head.parameters(),
        lr=cfg.lr, betas=(0.9, 0.98), weight_decay=0.01,
    )

    head.train()
    start = time.time()
    losses: list[float] = []
    for step in range(1, cfg.steps + 1):
        batch = [rng.choice(train_examples) for _ in range(cfg.batch_size)]
        b = _collate(batch, cfg.max_seq, device)
        logits, loss = head(
            b["pos_ids"], b["dep_ids"], b["offset_ids"], b["funcword_ids"],
            labels=b["labels"],
        )
        optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optim.step()
        losses.append(float(loss.item()))
        if step % cfg.log_every == 0 or step == 1:
            recent = sum(losses[-cfg.log_every:]) / max(1, len(losses[-cfg.log_every:]))
            elapsed = time.time() - start
            print(f"[hamroby-extract] step={step:5d}/{cfg.steps} "
                  f"loss={recent:.4f} elapsed={elapsed:.1f}s",
                  file=sys.stderr)

    tok_acc, em = _evaluate(
        head, eval_pairs_aligned, eval_examples_full, nlp, cfg.max_seq, device,
    )
    print(f"[hamroby-extract] eval token_acc={tok_acc:.3f} triple_em={em:.3f}",
          file=sys.stderr)

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "head_state": head.state_dict(),
        "encoder_cfg": enc_cfg.__dict__,
        "eval_token_accuracy": tok_acc,
        "eval_triple_em": em,
    }, cfg.out_path)
    print(f"[hamroby-extract] saved {cfg.out_path}", file=sys.stderr)
    return cfg.out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hamroby-extract-train")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scenes", type=int, default=4000)
    p.add_argument("--eval-scenes", type=int, default=500)
    p.add_argument("--qualifier-prob", type=float, default=0.6)
    p.add_argument("--size", choices=["tiny", "base", "large"], default="base")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-seq", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--real-prose-max-sentences", type=int, default=0,
        help=("Number of UD sentences to sample for real-prose "
              "training pairs (delexicalized). 0 = synthetic-only. "
              "Try 10000 for a meaningful mix."),
    )
    p.add_argument(
        "--real-prose-treebanks", type=str, default="",
        help=("Comma-separated UD treebanks to use for real-prose "
              "pairs (default = all English: ewt,gum,lines,partut,atis,esl)"),
    )
    args = p.parse_args(argv)

    real_prose_treebanks = None
    if args.real_prose_treebanks:
        real_prose_treebanks = tuple(
            x.strip() for x in args.real_prose_treebanks.split(",") if x.strip()
        )

    cfg = TrainConfig(
        out_path=args.out,
        n_train_scenes=args.scenes,
        n_eval_scenes=args.eval_scenes,
        qualifier_prob=args.qualifier_prob,
        size=args.size,
        steps=args.steps,
        batch_size=args.batch_size,
        max_seq=args.max_seq,
        lr=args.lr,
        log_every=args.log_every,
        seed=args.seed,
        real_prose_max_sentences=args.real_prose_max_sentences,
        real_prose_treebanks=real_prose_treebanks,
    )
    train(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
