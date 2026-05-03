"""Use a trained Grammar Cortex checkpoint without training.

Three modes:
  --sample N      generate N plausible (DEPREL, UPOS) tag sequences
  --score-dev N   score N held-out UD dev sentences and show ppl per example
  --score "..."   score one explicit tag stream you supply as a quoted string

All operate on the structural tag stream — words don't enter (per v024).
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch

from . import ud
from .model import GrammarConfig, GrammarModel
from .vocab import BOS_ID, EOS_ID, ID2TOK, PAD_ID, TOK2ID, UNK_ID, VOCAB_SIZE


def load_checkpoint(path: Path, device: torch.device) -> GrammarModel:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = GrammarConfig(**ck["config"])
    model = GrammarModel(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"[load] {path.name}  step={ck.get('step')}  "
          f"train_loss={ck.get('loss'):.4f}  dev_ppl={ck.get('dev_ppl'):.3f}",
          flush=True)
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
    tokens = [ID2TOK.get(i, "<unk>") for i in ids[1:-1] if i != EOS_ID]
    return tokens


@torch.no_grad()
def score_ids(model: GrammarModel, ids: list[int], device: torch.device) -> float:
    x = torch.tensor([ids], dtype=torch.long, device=device)
    _, loss = model(x, target_ids=x)
    return float(loss.item())


def encode_tag_stream(tokens: list[str]) -> list[int]:
    seq = [BOS_ID] + [TOK2ID.get(t, UNK_ID) for t in tokens] + [EOS_ID]
    return seq


def encode_sentence(sent: ud.UDSentence, max_tokens: int = 60) -> list[int]:
    return encode_tag_stream(ud.to_input_tokens(sent, max_tokens=max_tokens))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True,
                   help="Path to a grammar_*.pt checkpoint")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--sample", type=int, default=0,
                   help="Number of tag sequences to sample")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--max-len", type=int, default=80)

    p.add_argument("--score-dev", type=int, default=0,
                   help="Score N sentences from UD English-EWT dev split")
    p.add_argument("--score", type=str, default=None,
                   help='Score one explicit tag stream, e.g. "nsubj NOUN root AUX xcomp ADJ"')
    args = p.parse_args()

    device = torch.device(args.device)
    rng = random.Random(args.seed)
    model = load_checkpoint(args.ckpt, device)

    if args.sample:
        print()
        print(f"=== SAMPLE  temperature={args.temperature}  top_k={args.top_k} ===")
        for i in range(args.sample):
            toks = sample(model, device, rng, args.max_len, args.temperature, args.top_k)
            paired = [f"{toks[j]:>10s} {toks[j+1]:<6s}"
                      for j in range(0, len(toks) - 1, 2)]
            print(f"\n[{i+1}] {len(toks)//2} tokens")
            print("   " + "  ".join(paired) if paired else "   (empty)")

    if args.score_dev:
        print()
        print(f"=== SCORE  {args.score_dev} EWT dev sentences ===")
        path = ud.ensure_split("ewt", "dev")
        scored = []
        for sent in ud.parse_conllu(path):
            ids = encode_sentence(sent)
            if len(ids) > 256:
                continue
            loss = score_ids(model, ids, device)
            scored.append((math.exp(loss), len(ids), sent))
            if len(scored) >= args.score_dev:
                break
        for ppl, n, sent in sorted(scored, key=lambda x: x[0]):
            tags = ud.to_input_tokens(sent, max_tokens=60)
            print(f"  ppl={ppl:6.2f}  n_tags={n}  {' '.join(tags[:20])}{'...' if len(tags) > 20 else ''}")
        if scored:
            avg = sum(p for p, _, _ in scored) / len(scored)
            print(f"  ---  mean ppl over {len(scored)} sentences = {avg:.3f}")

    if args.score:
        tokens = args.score.split()
        ids = encode_tag_stream(tokens)
        loss = score_ids(model, ids, device)
        print()
        print(f"=== SCORE  custom ===")
        print(f"  tokens: {' '.join(tokens)}")
        print(f"  ppl:    {math.exp(loss):.3f}   loss: {loss:.4f}")


if __name__ == "__main__":
    main()
