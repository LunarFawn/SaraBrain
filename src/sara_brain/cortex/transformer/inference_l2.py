"""Sample / score with an L2-en checkpoint.

Mirrors `inference.py` but uses the L2-en vocabulary (vocab_en.py).
The model architecture is identical to L1 — only the vocabulary
(and therefore the embedding / LM head shape) differs.

Usage:
    .venv/bin/python -m sara_brain.cortex.transformer.inference_l2 \\
        --ckpt src/sara_brain/cortex/checkpoints/l2_en_003000.pt \\
        --sample 5
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch

from . import ud
from .model import GrammarConfig, GrammarModel
from .vocab_en import (
    BOS_ID,
    EN_FUNCTION_WORD_SET,
    EOS_ID,
    ID2TOK_EN,
    PAD_ID,
    TOK2ID_EN,
    UNK_ID,
    VOCAB_SIZE_EN,
)


def load_checkpoint(path: Path, device: torch.device) -> GrammarModel:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = GrammarConfig(**ck["config"])
    if cfg.vocab_size != VOCAB_SIZE_EN:
        raise SystemExit(
            f"checkpoint vocab_size={cfg.vocab_size} but VOCAB_SIZE_EN={VOCAB_SIZE_EN}; "
            f"is this an L2-en checkpoint?"
        )
    model = GrammarModel(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(
        f"[load] {path.name}  step={ck.get('step')}  "
        f"loss={ck.get('loss'):.4f}  dev_ppl={ck.get('dev_ppl'):.3f}  "
        f"lang={ck.get('lang')}  frozen_l1={ck.get('frozen_l1')}",
        flush=True,
    )
    return model


@torch.no_grad()
def sample(
    model: GrammarModel,
    device: torch.device,
    rng: random.Random,
    max_len: int = 80,
    temperature: float = 1.0,
    top_k: int = 0,
) -> list[str]:
    ids = [BOS_ID]
    for _ in range(max_len):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits, _ = model(x)
        next_logits = logits[0, -1] / max(1e-6, temperature)
        if top_k > 0:
            v, _ = torch.topk(next_logits, top_k)
            next_logits[next_logits < v[-1]] = -float("inf")
        probs = torch.softmax(next_logits.float(), dim=-1).cpu().numpy()
        nxt = rng.choices(range(len(probs)), weights=probs.tolist(), k=1)[0]
        ids.append(nxt)
        if nxt == EOS_ID:
            break
    return [ID2TOK_EN.get(i, "<unk>") for i in ids[1:-1] if i != EOS_ID]


def format_sample(tokens: list[str]) -> str:
    """Pretty-print a (dep, upos-or-form) tag stream as paired columns
    so the function-word literals stand out from the structural slots."""
    out = []
    for i in range(0, len(tokens) - 1, 2):
        dep, slot = tokens[i], tokens[i + 1]
        is_lex = slot in EN_FUNCTION_WORD_SET
        marker = "*" if is_lex else " "
        out.append(f"{dep:>10s} {marker}{slot:<10s}")
    return "  ".join(out)


@torch.no_grad()
def score_ids(model: GrammarModel, ids: list[int], device: torch.device) -> float:
    x = torch.tensor([ids], dtype=torch.long, device=device)
    _, loss = model(x, target_ids=x)
    return float(loss.item())


def encode_sentence(sent: ud.UDSentence, max_tokens: int = 60) -> list[int]:
    """Encode a UD sentence as the L2-en model expects it: lexicalized
    function words plus structural tokens for content."""
    tags = ud.to_input_tokens(
        sent,
        max_tokens=max_tokens,
        lexicalize_function_words=True,
        function_word_set=EN_FUNCTION_WORD_SET,
    )
    return [BOS_ID] + [TOK2ID_EN.get(t, UNK_ID) for t in tags] + [EOS_ID]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True,
                   help="Path to an l2_<lang>_*.pt checkpoint")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--max-len", type=int, default=80)

    p.add_argument("--score-dev", type=int, default=0,
                   help="Score N sentences from UD English-EWT dev split (lexicalized)")
    args = p.parse_args()

    device = torch.device(args.device)
    rng = random.Random(args.seed)
    model = load_checkpoint(args.ckpt, device)

    if args.sample:
        print()
        print(f"=== SAMPLE  temperature={args.temperature}  top_k={args.top_k} ===")
        print("(* marks lexicalized function-word tokens; everything else is UPOS)")
        for i in range(args.sample):
            toks = sample(model, device, rng, args.max_len, args.temperature, args.top_k)
            print(f"\n[{i+1}] {len(toks)//2} tokens")
            print("   " + format_sample(toks) if toks else "   (empty)")

    if args.score_dev:
        print()
        print(f"=== SCORE  {args.score_dev} EWT dev sentences (lexicalized) ===")
        path = ud.ensure_split("ewt", "dev")
        scored = []
        max_seq = model.cfg.max_seq
        for sent in ud.parse_conllu(path):
            ids = encode_sentence(sent)
            if len(ids) > max_seq:
                continue
            loss = score_ids(model, ids, device)
            scored.append((math.exp(loss), len(ids), sent))
            if len(scored) >= args.score_dev:
                break
        for ppl, n, sent in sorted(scored, key=lambda x: x[0])[:10]:
            tags = ud.to_input_tokens(
                sent, max_tokens=60,
                lexicalize_function_words=True,
                function_word_set=EN_FUNCTION_WORD_SET,
            )
            print(f"  ppl={ppl:6.2f}  n_tags={n}  {' '.join(tags[:20])}{'...' if len(tags) > 20 else ''}")
        if scored:
            avg = sum(p for p, _, _ in scored) / len(scored)
            print(f"  ---  mean ppl over {len(scored)} sentences = {avg:.3f}")


if __name__ == "__main__":
    main()
