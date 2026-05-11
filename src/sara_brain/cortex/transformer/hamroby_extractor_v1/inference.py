"""Drop-in trained-head extractor with the same shape as the rule stub.

`extract_triples(clause, nlp) -> list[Triple]` matches the signature of
[extractor_rules.extract_triples](../v2/extractor_rules.py). The
underlying pipeline is:

    parse_sentence(clause, nlp)  ->  ParsedSentence (POS/dep/offset/funcword)
        |
    head.predict_tags(...)       ->  per-word BIO tag ids
        |
    decode(parsed, tags)         ->  list[ExtractedTriple]
        |
    -> list[Triple(subject, relation, object, source_clause=clause)]

The checkpoint is loaded once per process (lazy singleton). Pass a
different path via the `HAMROBY_CHECKPOINT` env var if you want to
point at a non-canonical checkpoint.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

from ..v2.extractor_rules import Triple
from .decoder import decode
from .extraction_head import ExtractionHead
from .feature_extractor import parse_sentence
from .model import ExtractorConfig, GrammarEncoder


_CHECKPOINT_DEFAULT = (
    Path(__file__).resolve().parents[5]
    / "src/sara_brain/cortex/checkpoints/hamroby_extractor_v1.pt"
)


_HEAD: ExtractionHead | None = None
_HEAD_CHECKPOINT_PATH: Path | None = None


def _load_head(path: Path | None = None) -> ExtractionHead:
    """Load (and cache) the trained head from `path` or the canonical
    checkpoint. Same checkpoint format used by training and the
    diagnostic scripts."""
    global _HEAD, _HEAD_CHECKPOINT_PATH
    target = Path(path or os.environ.get(
        "HAMROBY_CHECKPOINT", str(_CHECKPOINT_DEFAULT),
    ))
    if _HEAD is not None and _HEAD_CHECKPOINT_PATH == target:
        return _HEAD
    raw = torch.load(target, map_location="cpu", weights_only=False)
    cfg = ExtractorConfig(**raw["encoder_cfg"])
    head = ExtractionHead(GrammarEncoder(cfg))
    head.load_state_dict(raw["head_state"])
    head.eval()
    _HEAD = head
    _HEAD_CHECKPOINT_PATH = target
    return head


def extract_triples(clause: str, nlp) -> list[Triple]:
    """Trained-head extractor in the rule-stub's API shape.

    Returns a list of `Triple(subject, relation, object, source_clause)`
    so callers (e.g. cli_teach_book) can swap rule stub ↔ trained head
    without other changes.
    """
    head = _load_head()
    parsed = parse_sentence(clause, nlp)
    if not parsed.words:
        return []
    pos_ids = torch.tensor(
        [[fid[0] for fid in parsed.feature_ids]], dtype=torch.long,
    )
    dep_ids = torch.tensor(
        [[fid[1] for fid in parsed.feature_ids]], dtype=torch.long,
    )
    off_ids = torch.tensor(
        [[fid[2] for fid in parsed.feature_ids]], dtype=torch.long,
    )
    fw_ids = torch.tensor(
        [[fid[3] for fid in parsed.feature_ids]], dtype=torch.long,
    )
    with torch.no_grad():
        tag_ids = head.predict_tags(pos_ids, dep_ids, off_ids, fw_ids)[0].tolist()
    out: list[Triple] = []
    for tri in decode(parsed, tag_ids):
        # Lowercase the surface text to match the rule stub's behavior
        # (its _normalize lowercases triples for substrate ingest).
        out.append(Triple(
            subject=tri.subject.lower(),
            relation=tri.relation.lower(),
            object=tri.object.lower(),
            source_clause=clause,
        ))
    return out


__all__ = ["extract_triples"]
