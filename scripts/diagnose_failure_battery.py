"""Diagnostic harness — run the failure battery through both extractors.

Lays out, side-by-side per sentence:
- spaCy POS/dep tags per word
- Trained-head BIO tag per word
- Trained-head decoded triples
- Rule-stub triples

Diagnostic only — no assertions. Output is a markdown report to stdout
for human review. Used to classify each failure as input-side (spaCy
mistag), model-side (model emits wrong BIO given correct features), or
decoder-side (correct tags, wrong slicing).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

# Repo imports.
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
from sara_brain.cortex.transformer.hamroby_extractor_v1.vocab import TAG_NAMES
from sara_brain.cortex.transformer.v2.extractor_rules import (
    extract_triples as rule_extract_triples,
)


import os

CHECKPOINT = Path(os.environ.get(
    "HAMROBY_CHECKPOINT",
    str(REPO / "src/sara_brain/cortex/checkpoints/hamroby_extractor_v1.pt"),
))


BATTERY = [
    # 4 originally named failures.
    ("K_d for the binding is 1.2nM.", "failure"),
    ("Marker theory predicts kdoff with p<0.05.", "failure"),
    ("The 5'3' static stem provides stability.", "failure"),
    ("Noticing a limitation shows the way.", "failure"),
    # Remaining failures after aug3 — added 2026-05-10 for round-3 diagnosis.
    ("DNA and RNA share base pairing.", "failure"),
    ("Cluster analysis groups proteins by similarity.", "failure"),
    ("She bought apples and oranges.", "failure"),
    ("Researchers compared yeast and mammalian cells.", "failure"),
    # 9 controls (mix of clean SVOs, intransitives, copulars, conjunctions).
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


def load_trained_head() -> ExtractionHead:
    raw = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = ExtractorConfig(**raw["encoder_cfg"])
    encoder = GrammarEncoder(cfg)
    head = ExtractionHead(encoder)
    head.load_state_dict(raw["head_state"])
    head.eval()
    return head


def run_trained_head(head: ExtractionHead, parsed) -> list[int]:
    """Run the head over a single ParsedSentence; return per-word tag ids."""
    pos_ids = torch.tensor(
        [[fid[0] for fid in parsed.feature_ids]], dtype=torch.long
    )
    dep_ids = torch.tensor(
        [[fid[1] for fid in parsed.feature_ids]], dtype=torch.long
    )
    offset_ids = torch.tensor(
        [[fid[2] for fid in parsed.feature_ids]], dtype=torch.long
    )
    funcword_ids = torch.tensor(
        [[fid[3] for fid in parsed.feature_ids]], dtype=torch.long
    )
    tags = head.predict_tags(pos_ids, dep_ids, offset_ids, funcword_ids)
    return tags[0].tolist()


def render_spacy_parse(text: str, nlp) -> list[tuple[str, str, str, str]]:
    """Return (word, POS, dep, head) tuples from a regular spaCy parse,
    not the trained-head feature extractor's view. This is the raw
    parse rule_extract_triples sees."""
    doc = nlp(text)
    return [(t.text, t.pos_, t.dep_, t.head.text) for t in doc]


def fmt_triples(triples) -> str:
    if not triples:
        return "NO TRIPLE"
    parts = []
    for t in triples:
        if hasattr(t, "subject"):
            parts.append(f"({t.subject!r}, {t.relation!r}, {t.object!r})")
        else:
            parts.append(repr(t))
    return " | ".join(parts)


def main() -> int:
    print(f"# Failure-battery diagnostic\n")
    print(f"Checkpoint: `{CHECKPOINT.relative_to(REPO)}`\n")

    print("Loading trained head...", file=sys.stderr)
    head = load_trained_head()

    print("Loading spaCy (domain tokenizer)...", file=sys.stderr)
    nlp_domain = load_domain_nlp()

    print("Loading spaCy (default tokenizer for rule stub)...", file=sys.stderr)
    import spacy
    # Use the same model load_domain_nlp uses so the rule-stub view in
    # the report matches what the production rule stub sees.
    try:
        nlp_default = spacy.load("en_core_web_trf")
    except OSError:
        nlp_default = spacy.load("en_core_web_sm")

    failures = []
    controls = []

    for sentence, kind in BATTERY:
        # Trained head input — domain tokenizer.
        parsed = parse_sentence(sentence, nlp_domain)
        tags = run_trained_head(head, parsed)
        trained_triples = decode(parsed, tags)

        # Rule stub input — default tokenizer, since the rule stub
        # itself is the in-tree pipeline and it expects standard spaCy
        # tokenization.
        rule_triples = rule_extract_triples(sentence, nlp_default)

        section = []
        section.append(f"## {kind.upper()}: `{sentence}`\n")
        section.append("**spaCy parse (default tokenizer, rule-stub view):**\n")
        section.append("| word | POS | dep | head |")
        section.append("|---|---|---|---|")
        for w, pos, dep, head_w in render_spacy_parse(sentence, nlp_default):
            section.append(f"| `{w}` | {pos} | {dep} | `{head_w}` |")
        section.append("")

        section.append("**Trained head input (domain tokenizer):**\n")
        section.append("| word | POS_id | dep_id | offset_id | funcword_id | BIO tag |")
        section.append("|---|---|---|---|---|---|")
        for word, fids, tag in zip(parsed.words, parsed.feature_ids, tags):
            tag_name = TAG_NAMES[tag] if 0 <= tag < len(TAG_NAMES) else f"?{tag}"
            section.append(
                f"| `{word}` | {fids[0]} | {fids[1]} | {fids[2]} | {fids[3]} | "
                f"**{tag_name}** |"
            )
        section.append("")

        section.append(f"**Trained head triples:** {fmt_triples(trained_triples)}\n")
        section.append(f"**Rule stub triples:** {fmt_triples(rule_triples)}\n")
        section.append("---\n")

        block = "\n".join(section)
        if kind == "failure":
            failures.append(block)
        else:
            controls.append(block)

    print("# Failure cases\n")
    for block in failures:
        print(block)

    print("# Control cases\n")
    for block in controls:
        print(block)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
