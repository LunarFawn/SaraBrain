"""Substrate-driven labeler for the HamRobySum synthesizer head.

Walks Sara's brain.db edge by edge, clusters edges by subject concept,
and emits (edge_list, prose) training pairs using the same template
table the template synthesizer uses for inference. The neural
synthesizer (HamRobySum) learns this mapping with sentence-shape
variety from the grammar LM — replacing the templates with a small
generative head.

**v035 architecture: substrate content NEVER enters the model's
vocabulary.** Each example's substrate strings (subjects, objects)
are replaced with `<C0>`...`<C31>` slot tokens before serialization,
with the inverse mapping carried alongside. The model learns
structural composition over a small fixed vocabulary; substrate
content rides through the model purely as a position marker.

Two emission paths:

- `--output pairs.jsonl` — one (edges, prose, subject) JSON per line.
  Useful for inspection and for downstream tools that want to
  rewrite the prose. This was the original output format.
- `--serialize-out tokens.jsonl` — one tokenized training row per
  line: `{input_ids, loss_mask, slot_mapping, n_facts, n_prose}`.
  Ready to feed `train_synth.py`. The loss mask is 1 only on the
  prose continuation so the model isn't penalized for failing to
  predict the facts prefix it was conditioned on.
"""
from __future__ import annotations

import re
import random
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from .synthesizer import Edge, render_edges, render_edges_slotted
from .vocab_synth import (
    BOS_ID,
    EOS_ID,
    EDGE_SEP_ID,
    END_PROSE_ID,
    FACTS_ID,
    N_PRED_SLOTS,
    N_SLOTS,
    OBJ_ID,
    PRED_ID,
    PROSE_ID,
    REFUTED_ID,
    ATTR_ID,
    SUBJ_ID,
    TOK2ID_SYNTH,
    UNK_ID,
    pred_slot_token,
    slot_token,
)


@dataclass
class SynthExample:
    edges: list[Edge]
    prose: str
    subject: str        # the concept this cluster is about

    def to_dict(self) -> dict:
        return {
            "edges": [asdict(e) for e in self.edges],
            "prose": self.prose,
            "subject": self.subject,
        }


_NOISE_RELATIONS_FOR_LABELER = {"describes"}
# `part_of` used to be blanket-dropped here because substrate-ingestion
# decomposition edges (e.g. `inertia` --part_of--> `inertia in rna`)
# produced tautological prose. `render_edges` now drops only the
# decomposition cases; useful `part_of` ("RNA is part of cell") flows
# through to both inference and training labels.


def load_substrate_edges(db_path: Path) -> list[Edge]:
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT n1.label, s.relation, n2.label, s.strength "
        "FROM segments s "
        "JOIN neurons n1 ON s.source_id = n1.id "
        "JOIN neurons n2 ON s.target_id = n2.id"
    ).fetchall()
    conn.close()
    out: list[Edge] = []
    for src, rel, tgt, strength in rows:
        if rel in _NOISE_RELATIONS_FOR_LABELER:
            continue
        if rel == "synonym_of":
            continue
        out.append(Edge(
            src=src.replace("_attribute", ""),
            rel=rel,
            tgt=tgt.replace("_attribute", ""),
            refuted=(strength is not None and strength < 0),
        ))
    return out


def cluster_by_subject(edges: list[Edge]) -> dict[str, list[Edge]]:
    by_src: dict[str, list[Edge]] = defaultdict(list)
    for e in edges:
        by_src[e.src].append(e)
    return by_src


def generate_examples(
    db_path: Path,
    min_edges_per_cluster: int = 1,
    max_edges_per_cluster: int = 12,
    rng: random.Random | None = None,
    augment_multiplier: int = 1,
) -> list[SynthExample]:
    """Build (edge_cluster, prose) pairs from a Sara substrate. Each
    cluster is the edges around one subject concept; prose is generated
    by the same template renderer the inference path uses.

    `augment_multiplier > 1` emits N variants per cluster (≥2 edges) by
    shuffling edge order before rendering. The rendered prose changes
    because `render_edges`'s sentence-combining and ordering depend on
    edge order — this gives the trainer multiple legitimate (edges,
    prose) pairs from the same source cluster."""
    rng = rng or random.Random(0)
    edges = load_substrate_edges(db_path)
    clusters = cluster_by_subject(edges)
    examples: list[SynthExample] = []

    def _render_chunk(chunk: list[Edge], subject: str) -> None:
        n_variants = augment_multiplier if len(chunk) >= 2 else 1
        seen_prose: set[str] = set()
        for v in range(n_variants):
            if v == 0:
                ordered = list(chunk)
            else:
                ordered = list(chunk)
                rng.shuffle(ordered)
            prose = render_edges(ordered, topic=subject)
            if prose and prose not in seen_prose:
                seen_prose.add(prose)
                examples.append(SynthExample(
                    edges=ordered, prose=prose, subject=subject,
                ))

    for subject, cluster in clusters.items():
        if len(cluster) < min_edges_per_cluster:
            continue
        if len(cluster) > max_edges_per_cluster:
            # Sub-sample large clusters into multiple smaller examples so
            # the model sees varied edge-set sizes, not just the full
            # neighborhood.
            rng.shuffle(cluster)
            for i in range(0, len(cluster), max_edges_per_cluster):
                _render_chunk(cluster[i:i + max_edges_per_cluster], subject)
        else:
            _render_chunk(cluster, subject)
    return examples


def generate_examples_multi(
    db_paths: list[Path],
    min_edges_per_cluster: int = 1,
    max_edges_per_cluster: int = 12,
    rng: random.Random | None = None,
    augment_multiplier: int = 1,
) -> list[SynthExample]:
    """Concatenate `generate_examples` output across multiple brains.
    Useful for training a HamRobySum that's seen content from more than
    one substrate (per v034)."""
    rng = rng or random.Random(0)
    all_examples: list[SynthExample] = []
    for path in db_paths:
        before = len(all_examples)
        all_examples.extend(generate_examples(
            path,
            min_edges_per_cluster=min_edges_per_cluster,
            max_edges_per_cluster=max_edges_per_cluster,
            rng=rng,
            augment_multiplier=augment_multiplier,
        ))
        print(f"  {path}: +{len(all_examples) - before} examples")
    return all_examples


def write_jsonl(examples: list[SynthExample], path: Path) -> None:
    import json
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict()) + "\n")


# ── Tokenizer for the synth head ──
# Whitespace + punctuation split, lowercase, vocab_synth lookup.
# OOV tokens map to UNK_ID. Substrate labels with characters outside
# the vocab (e.g. apostrophes, slashes inside `5'3' static stem`) leak
# as UNK — acceptable for v0; a per-brain BPE pass would tighten this
# in a later slice if needed.

_PROSE_SPLIT_RE = re.compile(r"([.,;:?!\-])|\s+")


def _tokenize_text(text: str) -> list[str]:
    """Split free text on whitespace and punctuation. Punctuation
    becomes its own token; everything else is lowercased. Angle-bracket
    tokens (`<facts>`, `<C0>`, `<prose>`, ...) keep their case so they
    match the structural delimiters and slot tokens in vocab_synth."""
    out: list[str] = []
    for piece in _PROSE_SPLIT_RE.split(text):
        if not piece:
            continue
        if piece.isspace():
            continue
        # Preserve case for structural / slot tokens.
        if piece.startswith("<") and piece.endswith(">"):
            out.append(piece)
        else:
            out.append(piece.lower())
    return out


def _encode_text(text: str, tok2id: dict[str, int] | None = None) -> list[int]:
    if tok2id is None:
        tok2id = TOK2ID_SYNTH
    return [tok2id.get(t, UNK_ID) for t in _tokenize_text(text)]


def build_slot_mapping(ex: SynthExample) -> dict[str, str]:
    """Per-example mapping `{slot_token: substrate_string}`. Each
    distinct substrate label in the example's edges gets a slot id in
    encounter order. Caps at `N_SLOTS` — clusters with more distinct
    labels than slots will lose the trailing labels (caller should
    chunk those upstream)."""
    mapping: dict[str, str] = {}
    inverse: dict[str, str] = {}   # substrate_string -> slot_token
    next_idx = 0
    for e in ex.edges:
        for s in (e.src, e.tgt):
            s_norm = s.strip()
            if not s_norm or s_norm in inverse:
                continue
            if next_idx >= N_SLOTS:
                break
            tok = slot_token(next_idx)
            mapping[tok] = s_norm
            inverse[s_norm] = tok
            next_idx += 1
        if next_idx >= N_SLOTS:
            break
    return mapping


def slot_substitute(text: str, mapping: dict[str, str]) -> str:
    """Replace each substrate string (mapping value) with its slot
    token (mapping key) in `text`. Case-insensitive. Sorts by length
    descending so longer phrases match before sub-phrases (`rna in
    cell` before `rna`)."""
    if not text or not mapping:
        return text
    out = text
    pairs = sorted(mapping.items(), key=lambda x: -len(x[1]))
    for tok, content in pairs:
        if not content:
            continue
        out = re.sub(re.escape(content), tok, out, flags=re.IGNORECASE)
    return out


def build_pred_mapping(ex: SynthExample) -> dict[str, str]:
    """Per-example mapping `{relation_name: <Pn>}`. Each distinct
    relation in the cluster gets a slot in encounter order. Caps at
    `N_PRED_SLOTS`. Used by v040 to slot the predicate position
    parallel to the content slot mechanism in `build_slot_mapping`."""
    mapping: dict[str, str] = {}
    next_idx = 0
    for e in ex.edges:
        if e.rel in mapping:
            continue
        if next_idx >= N_PRED_SLOTS:
            break
        mapping[e.rel] = pred_slot_token(next_idx)
        next_idx += 1
    return mapping


def serialize_example(ex: SynthExample) -> dict:
    """Convert a SynthExample to a token-id training row using v040
    dual-slot substitution: substrate CONTENT rides as `<C0>`...`<C31>`
    tokens, substrate RELATIONS ride as `<P0>`...`<P15>` tokens. The
    model sees structure only; both content and verbs round-trip
    through substrate at inference time.

    Returns:
      {
        "input_ids":    list[int]
        "loss_mask":   list[int]   — 1 on prose-continuation positions
        "slot_mapping": dict        — {<Cn>: substrate_str}
        "pred_mapping": dict        — {relation_name: <Pn>}
        "n_facts":      int
        "n_prose":      int
      }
    """
    content_mapping = build_slot_mapping(ex)
    pred_mapping = build_pred_mapping(ex)
    content_inv = {v.lower(): k for k, v in content_mapping.items()}

    # Slotted prose: re-render the example's prose using the slot tokens
    # for both content AND predicates. This is the v040 swap that
    # replaces the v039 string-substitution-after-render path.
    prose_with_slots = render_edges_slotted(
        ex.edges, content_inv, pred_mapping, topic=ex.subject,
    )

    def _slot_or_encode(s: str) -> list[int]:
        s_norm = s.strip()
        if s_norm.lower() in content_inv:
            return [TOK2ID_SYNTH[content_inv[s_norm.lower()]]]
        return _encode_text(s_norm)

    ids: list[int] = [BOS_ID, FACTS_ID]
    for e in ex.edges:
        ids.append(SUBJ_ID)
        ids.extend(_slot_or_encode(e.src))
        ids.append(PRED_ID)
        # v040: facts prefix uses the predicate slot when available.
        if e.rel in pred_mapping:
            ids.append(TOK2ID_SYNTH[pred_mapping[e.rel]])
        else:
            ids.extend(_encode_text(e.rel.replace("_", " ")))
        ids.append(OBJ_ID)
        ids.extend(_slot_or_encode(e.tgt))
        if e.refuted:
            ids.append(REFUTED_ID)
        if e.target_was_attribute:
            ids.append(ATTR_ID)
        ids.append(EDGE_SEP_ID)
    n_facts = len(ids)
    ids.append(PROSE_ID)
    prose_start = len(ids)
    ids.extend(_encode_text(prose_with_slots))
    ids.append(END_PROSE_ID)
    ids.append(EOS_ID)
    n_prose = len(ids) - prose_start

    loss_mask = [0] * len(ids)
    for i in range(prose_start - 1, len(ids) - 1):
        loss_mask[i] = 1

    return {
        "input_ids":    ids,
        "loss_mask":   loss_mask,
        "slot_mapping": content_mapping,
        "pred_mapping": pred_mapping,
        "n_facts":      n_facts,
        "n_prose":      n_prose,
    }


def write_serialized_jsonl(
    examples: list[SynthExample], path: Path,
    max_seq: int | None = None,
) -> dict:
    """Serialize each example against the fixed VOCAB_SYNTH (v035) and
    write training-ready JSONL. Each row carries its own per-example
    slot mapping; no sidecar vocab file is needed because the synth
    vocab is universal."""
    import json

    written = 0
    skipped_too_long = 0
    total_loss_positions = 0
    seq_lens: list[int] = []
    unk_count = 0
    slot_use_total = 0
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            row = serialize_example(ex)
            if max_seq is not None and len(row["input_ids"]) > max_seq:
                skipped_too_long += 1
                continue
            f.write(json.dumps(row) + "\n")
            written += 1
            total_loss_positions += sum(row["loss_mask"])
            seq_lens.append(len(row["input_ids"]))
            unk_count += sum(1 for t in row["input_ids"] if t == UNK_ID)
            slot_use_total += len(row["slot_mapping"])

    summary = {
        "written": written,
        "skipped_too_long": skipped_too_long,
        "avg_seq_len": (sum(seq_lens) / max(1, len(seq_lens))),
        "max_seq_len": (max(seq_lens) if seq_lens else 0),
        "min_seq_len": (min(seq_lens) if seq_lens else 0),
        "total_loss_positions": total_loss_positions,
        "avg_slots_per_example": slot_use_total / max(1, written),
        "unk_count": unk_count,
    }
    return summary


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(
        description="Generate (edge_list, prose) training pairs from a "
                    "Sara brain.db for the HamRobySum synthesizer head.")
    p.add_argument(
        "--brain", type=Path, action="append", required=True,
        help="brain.db path. Repeatable: pass multiple --brain flags to "
             "build a multi-brain corpus. With v035's slot-based vocab, "
             "this is fully safe — the model's vocab is fixed regardless "
             "of how many brains you mix in.",
    )
    p.add_argument("--output", type=Path, default=None,
                   help="JSONL output path for human-readable pairs "
                        "(one {edges, prose, subject} per line)")
    p.add_argument("--serialize-out", type=Path, default=None,
                   help="JSONL output for tokenized training rows "
                        "({input_ids, loss_mask, slot_mapping, ...}). "
                        "Ready for train_synth.py. v035: no sidecar "
                        "vocab file — vocab is the fixed VOCAB_SYNTH.")
    p.add_argument("--max-seq", type=int, default=512,
                   help="Drop serialized examples longer than this "
                        "(only with --serialize-out)")
    p.add_argument("--min-edges", type=int, default=1)
    p.add_argument("--max-edges", type=int, default=12)
    p.add_argument(
        "--augment-multiplier", type=int, default=1,
        help="For clusters with ≥2 edges, emit N variants by shuffling "
             "edge order before rendering. 1 = no augmentation. 2-3 is "
             "the v034 default for training the synthesizer.",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.output is None and args.serialize_out is None:
        p.error("must pass at least one of --output / --serialize-out")

    if len(args.brain) == 1:
        examples = generate_examples(
            args.brain[0],
            min_edges_per_cluster=args.min_edges,
            max_edges_per_cluster=args.max_edges,
            rng=random.Random(args.seed),
            augment_multiplier=args.augment_multiplier,
        )
        print(f"generated {len(examples)} examples from {args.brain[0]}")
    else:
        print(f"generating from {len(args.brain)} brains:")
        examples = generate_examples_multi(
            args.brain,
            min_edges_per_cluster=args.min_edges,
            max_edges_per_cluster=args.max_edges,
            rng=random.Random(args.seed),
            augment_multiplier=args.augment_multiplier,
        )
        print(f"total: {len(examples)} examples across {len(args.brain)} brains")
    if examples:
        avg = sum(len(e.edges) for e in examples) / len(examples)
        n_subjects = len({e.subject for e in examples})
        print(f"  {n_subjects} unique subjects, avg {avg:.1f} edges/cluster")

    if args.output:
        write_jsonl(examples, args.output)
        print(f"wrote human-readable pairs -> {args.output}")

    if args.serialize_out:
        from .vocab_synth import VOCAB_SIZE_SYNTH
        summary = write_serialized_jsonl(
            examples, args.serialize_out, max_seq=args.max_seq,
        )
        print(f"wrote serialized rows -> {args.serialize_out}")
        print(
            f"  written={summary['written']}  "
            f"skipped_too_long={summary['skipped_too_long']}  "
            f"avg_seq_len={summary['avg_seq_len']:.1f}  "
            f"min={summary['min_seq_len']}  max={summary['max_seq_len']}\n"
            f"  vocab_size={VOCAB_SIZE_SYNTH} (fixed, generic — v035)  "
            f"loss_positions={summary['total_loss_positions']}\n"
            f"  avg_slots_per_example={summary['avg_slots_per_example']:.1f}  "
            f"unk_in_corpus={summary['unk_count']}"
        )


if __name__ == "__main__":
    main()
