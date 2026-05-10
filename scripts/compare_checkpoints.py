"""Compare two hamroby_extractor_v1 checkpoints on the failure battery.

Runs the diagnostic harness against two checkpoints (default: the
baseline at `hamroby_extractor_v1.pt` and a candidate at
`hamroby_extractor_v1_aug.pt`) and prints per-sentence (baseline →
candidate) deltas. Used after retraining to see which failures the
augmentations fixed and whether anything regressed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))

from sara_brain.cortex.transformer.hamroby_extractor_v1.feature_extractor import (
    load_domain_nlp,
    parse_sentence,
)
from sara_brain.cortex.transformer.hamroby_extractor_v1.extraction_head import (
    ExtractionHead,
)
from sara_brain.cortex.transformer.hamroby_extractor_v1.model import (
    ExtractorConfig,
    GrammarEncoder,
)
from sara_brain.cortex.transformer.hamroby_extractor_v1.decoder import decode


BATTERY = [
    ("K_d for the binding is 1.2nM.", "failure"),
    ("Marker theory predicts kdoff with p<0.05.", "failure"),
    ("The 5'3' static stem provides stability.", "failure"),
    ("Noticing a limitation shows the way.", "failure"),
    ("Bruce Lee created Jeet Kune Do.", "control"),
    ("She built a house.", "control"),
    ("The protein folds into hairpins.", "control"),
    ("The system processes books and papers.", "control"),
    ("The molecular snare is critical.", "control"),
    ("Lee Smith leads the team.", "control"),
    ("The frog catches flies.", "control"),
    ("Strong proteins persist.", "control"),
    ("John and Mary went home.", "control"),
]


def load_head(path: Path) -> ExtractionHead:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ExtractorConfig(**raw["encoder_cfg"])
    head = ExtractionHead(GrammarEncoder(cfg))
    head.load_state_dict(raw["head_state"])
    head.eval()
    return head


def run(head: ExtractionHead, parsed) -> list:
    pos_ids = torch.tensor([[fid[0] for fid in parsed.feature_ids]], dtype=torch.long)
    dep_ids = torch.tensor([[fid[1] for fid in parsed.feature_ids]], dtype=torch.long)
    off_ids = torch.tensor([[fid[2] for fid in parsed.feature_ids]], dtype=torch.long)
    fw_ids = torch.tensor([[fid[3] for fid in parsed.feature_ids]], dtype=torch.long)
    tags = head.predict_tags(pos_ids, dep_ids, off_ids, fw_ids)[0].tolist()
    return decode(parsed, tags)


def fmt(triples) -> str:
    if not triples:
        return "NO TRIPLE"
    return " | ".join(
        f"({t.subject!r}, {t.relation!r}, {t.object!r})" for t in triples
    )


def main() -> int:
    base_path = Path(os.environ.get(
        "BASELINE",
        str(REPO / "src/sara_brain/cortex/checkpoints/hamroby_extractor_v1.pt"),
    ))
    cand_path = Path(os.environ.get(
        "CANDIDATE",
        str(REPO / "src/sara_brain/cortex/checkpoints/hamroby_extractor_v1_aug.pt"),
    ))
    if not base_path.exists():
        print(f"missing baseline: {base_path}", file=sys.stderr)
        return 1
    if not cand_path.exists():
        print(f"missing candidate: {cand_path}", file=sys.stderr)
        return 1

    print(f"baseline:  {base_path.name}", file=sys.stderr)
    print(f"candidate: {cand_path.name}", file=sys.stderr)
    base = load_head(base_path)
    cand = load_head(cand_path)
    nlp = load_domain_nlp()

    print(f"# Checkpoint comparison\n")
    print(f"- baseline: `{base_path.relative_to(REPO)}`")
    print(f"- candidate: `{cand_path.relative_to(REPO)}`\n")

    for kind in ("failure", "control"):
        sentences = [s for s, k in BATTERY if k == kind]
        print(f"## {kind.upper()} cases\n")
        for sent in sentences:
            parsed = parse_sentence(sent, nlp)
            base_t = run(base, parsed)
            cand_t = run(cand, parsed)
            same = fmt(base_t) == fmt(cand_t)
            arrow = "=" if same else "→"
            print(f"### `{sent}`\n")
            print(f"- baseline:  {fmt(base_t)}")
            print(f"- candidate: {fmt(cand_t)} {'(unchanged)' if same else arrow + ' CHANGED'}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
