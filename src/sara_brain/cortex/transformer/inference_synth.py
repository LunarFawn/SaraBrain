"""HamRobySum inference — render edge clusters to prose with the
trained synthesizer head.

Loads a synth checkpoint (which embeds the brain-extended vocab),
formats an edge cluster into the same `<facts>...<prose>` prefix the
trainer saw, and decodes greedily (or with sampling) to `</prose>`.
The decoded prose tokens are detokenized back to a string.

Falls back gracefully:
- If no synth ckpt is loaded → caller should use `synthesizer.render_edges`
- If the facts prefix exceeds `max_seq` → truncates edges (oldest first)
  with a warning printed once
- If decoding never emits `</prose>` within `max_new_tokens` → returns
  what was generated up to that point

Usage:
    .venv/bin/python -m sara_brain.cortex.transformer.inference_synth \\
        --ckpt src/sara_brain/cortex/checkpoints/hamroby_sum_synth_pairs_002000.pt \\
        --brain /tmp/sara_demo.db \\
        --topic "ribosome"
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from .model import GrammarConfig, GrammarModel
from .synth_data import (
    Edge, _tokenize_text, cluster_by_subject, load_substrate_edges,
    serialize_example, SynthExample,
)
from .vocab_synth import (
    BOS_ID,
    EDGE_SEP_ID,
    END_PROSE_ID,
    EOS_ID,
    FACTS_ID,
    OBJ_ID,
    PAD_ID,
    PRED_ID,
    PROSE_ID,
    REFUTED_ID,
    ATTR_ID,
    SUBJ_ID,
    UNK_ID,
)


_DETOKENIZE_NO_LEAD_SPACE = {".", ",", ";", ":", "?", "!", "'s", "n't", ")"}
_DETOKENIZE_NO_TRAIL_SPACE = {"(",}


def _detokenize(tokens: list[str]) -> str:
    """Turn a list of prose tokens back into a readable string. Glues
    punctuation to the previous word; capitalizes the first letter."""
    if not tokens:
        return ""
    out: list[str] = []
    prev_no_trail = False
    for tok in tokens:
        if not out:
            out.append(tok)
        elif tok in _DETOKENIZE_NO_LEAD_SPACE:
            out.append(tok)
        elif prev_no_trail:
            out.append(tok)
        else:
            out.append(" ")
            out.append(tok)
        prev_no_trail = tok in _DETOKENIZE_NO_TRAIL_SPACE
    text = "".join(out)
    if text:
        text = text[0].upper() + text[1:]
    return text


def load_synth_checkpoint(
    path: Path, device: torch.device,
) -> tuple[GrammarModel, list[str], dict[str, int]]:
    """Returns (model, vocab_list, tok2id)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = GrammarConfig(**ck["config"])
    model = GrammarModel(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    vocab = ck["brain_vocab"]["vocab"]
    tok2id = {tok: i for i, tok in enumerate(vocab)}
    print(
        f"[load] {path.name}  step={ck.get('step')}  "
        f"loss={ck.get('loss', float('nan')):.4f}  "
        f"dev_loss={ck.get('dev_loss', float('nan')):.4f}  "
        f"vocab={cfg.vocab_size}  frozen_base={ck.get('frozen_base')}",
        flush=True,
    )
    return model, vocab, tok2id


def _facts_prefix(
    edges: list[Edge], tok2id: dict[str, int],
) -> list[int]:
    """Just the facts portion + leading <prose> marker; the model will
    decode from there."""
    ids: list[int] = [BOS_ID, FACTS_ID]
    for e in edges:
        ids.append(SUBJ_ID)
        for t in _tokenize_text(e.src):
            ids.append(tok2id.get(t, UNK_ID))
        ids.append(PRED_ID)
        for t in _tokenize_text(e.rel.replace("_", " ")):
            ids.append(tok2id.get(t, UNK_ID))
        ids.append(OBJ_ID)
        for t in _tokenize_text(e.tgt):
            ids.append(tok2id.get(t, UNK_ID))
        if e.refuted:
            ids.append(REFUTED_ID)
        if e.target_was_attribute:
            ids.append(ATTR_ID)
        ids.append(EDGE_SEP_ID)
    ids.append(PROSE_ID)
    return ids


@torch.no_grad()
def synthesize_cluster(
    model: GrammarModel,
    vocab: list[str],
    tok2id: dict[str, int],
    edges: list[Edge],
    device: torch.device,
    max_new_tokens: int = 80,
    temperature: float = 0.0,
    top_k: int = 0,
    rng: random.Random | None = None,
) -> str:
    """Render `edges` as prose. `temperature=0` is greedy."""
    if not edges:
        return ""
    rng = rng or random.Random(0)
    max_seq = model.cfg.max_seq

    # Build facts prefix; truncate edges if it overflows.
    prefix = _facts_prefix(edges, tok2id)
    truncated = 0
    while len(prefix) >= max_seq - 4 and len(edges) > 1:
        edges = edges[1:]   # drop oldest first
        prefix = _facts_prefix(edges, tok2id)
        truncated += 1
    if truncated:
        print(f"[synth] truncated {truncated} edges to fit max_seq={max_seq}",
              flush=True)

    ids = list(prefix)
    out_ids: list[int] = []
    for _ in range(max_new_tokens):
        if len(ids) >= max_seq:
            break
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits, _ = model(x)
        next_logits = logits[0, -1]
        if temperature > 0.0:
            next_logits = next_logits / temperature
            if top_k > 0:
                v, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < v[-1]] = -float("inf")
            probs = torch.softmax(next_logits.float(), dim=-1).cpu().numpy()
            nxt = rng.choices(range(len(probs)), weights=probs.tolist(), k=1)[0]
        else:
            nxt = int(next_logits.argmax().item())
        ids.append(nxt)
        if nxt == END_PROSE_ID or nxt == EOS_ID:
            break
        out_ids.append(nxt)

    # Strip any structural delimiters that leaked into output.
    structural_ids = {
        FACTS_ID, PROSE_ID, END_PROSE_ID, SUBJ_ID, PRED_ID, OBJ_ID,
        EDGE_SEP_ID, REFUTED_ID, ATTR_ID, BOS_ID, EOS_ID, PAD_ID,
    }
    prose_tokens = [vocab[i] for i in out_ids if i not in structural_ids]
    return _detokenize(prose_tokens)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True,
                   help="Path to a hamroby_sum_*.pt checkpoint")
    p.add_argument("--brain", type=Path, required=True,
                   help="brain.db whose edges to synthesize from. The vocab "
                        "should match what the ckpt was trained on.")
    p.add_argument("--topic", type=str, default=None,
                   help="If set, only render the cluster whose subject "
                        "matches the topic (substring, case-insensitive)")
    p.add_argument("--n", type=int, default=5,
                   help="Number of clusters to render (when no --topic given)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 = greedy, >0 = sampling")
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = p.parse_args()

    device = torch.device(args.device)
    rng = random.Random(args.seed)
    model, vocab, tok2id = load_synth_checkpoint(args.ckpt, device)

    edges = load_substrate_edges(args.brain)
    clusters = cluster_by_subject(edges)
    print(f"loaded {len(edges)} edges, {len(clusters)} clusters from {args.brain}")

    if args.topic:
        topic_l = args.topic.lower()
        matching = [(s, c) for s, c in clusters.items() if topic_l in s.lower()]
        if not matching:
            print(f"no cluster matches topic {args.topic!r}")
            return
        for subject, cluster in matching[:args.n]:
            print()
            print(f"=== {subject!r}  ({len(cluster)} edges) ===")
            for e in cluster[:8]:
                print(f"   {e.src} --[{e.rel}]--> {e.tgt}{' [attr]' if e.target_was_attribute else ''}")
            if len(cluster) > 8:
                print(f"   ... +{len(cluster) - 8} more")
            prose = synthesize_cluster(
                model, vocab, tok2id, cluster, device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_k=args.top_k, rng=rng,
            )
            print(f"   PROSE: {prose}")
    else:
        items = list(clusters.items())
        rng.shuffle(items)
        for subject, cluster in items[:args.n]:
            print()
            print(f"=== {subject!r}  ({len(cluster)} edges) ===")
            for e in cluster[:6]:
                print(f"   {e.src} --[{e.rel}]--> {e.tgt}{' [attr]' if e.target_was_attribute else ''}")
            if len(cluster) > 6:
                print(f"   ... +{len(cluster) - 6} more")
            prose = synthesize_cluster(
                model, vocab, tok2id, cluster, device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_k=args.top_k, rng=rng,
            )
            print(f"   PROSE: {prose}")


if __name__ == "__main__":
    main()
