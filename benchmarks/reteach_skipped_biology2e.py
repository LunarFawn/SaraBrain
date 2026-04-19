#!/usr/bin/env python3
"""Retry parser-skipped sentences from the Biology2e ingest with
aggressive simplification.

Reads each chapter's facts file, identifies sentences that Sara's
statement parser rejected on the first pass (because a segment was
not created for them), and applies a second, more aggressive set of
simplifications:

  - split on "which" / "that" / "because" / "when" / "where" / "if"
    clause-boundary markers into independent sentences.
  - promote relative-clause content to standalone sentences.
  - strip trailing subordinate clauses that don't carry the main
    predicate.
  - split coordinated verbs ("X and Y") into separate sentences.

After simplification, each variant is presented to Sara's parser
again. Anything that NOW parses gets taught into the same chapter
region it would originally have gone into.

Usage:
    .venv/bin/python benchmarks/reteach_skipped_biology2e.py \\
        --db biology2e.db --facts-dir benchmarks/biology2e_facts

The facts files carry ALL candidates (taught + skipped). The script
re-tries each one — already-taught facts are no-ops (get_or_create +
strengthen is idempotent for the graph structure).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import spacy

from sara_brain.core.brain import Brain


_CLAUSE_SPLIT_RE = re.compile(
    r"\s+(?:which|that|because|when|where|if|although|though|while|since|until|as\s+long\s+as)\s+",
    re.IGNORECASE,
)
_COORDINATE_VERB_RE = re.compile(
    r"\s+and\s+(?=[a-z][a-z]+\s+)",
    re.IGNORECASE,
)
_PAREN_RE = re.compile(r"\s*\([^)]*\)")
_BRACKET_RE = re.compile(r"\s*\[[^]]*\]")


def aggressive_simplify(sent_text: str) -> list[str]:
    """Produce a list of simpler variants from one complex sentence."""
    s = sent_text.strip()
    s = _PAREN_RE.sub("", s)
    s = _BRACKET_RE.sub("", s)
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return []

    variants: set[str] = {s}
    # Split at clause-boundary markers.
    for m in _CLAUSE_SPLIT_RE.finditer(s):
        left = s[:m.start()].strip(" ,.")
        right = s[m.end():].strip(" ,.")
        if left:
            variants.add(left + "." if not left.endswith(".") else left)
        if right:
            variants.add(right + "." if not right.endswith(".") else right)
    # Coordinated verbs — split on ", and " and " and " when the right
    # side begins with a plausible subject (already handled partially
    # in the earlier pipeline; repeat here so the aggressive pass
    # catches sentences that slipped past).
    parts = re.split(r",\s+and\s+|,\s+or\s+", s)
    for p in parts:
        p = p.strip(" .")
        if p:
            variants.add(p + "." if not p.endswith(".") else p)
    # Strip leading "After" / "During" / "In" clauses up to the first
    # comma — the main predicate follows.
    lead = re.match(
        r"^\s*(?:After|During|In|When|Before|By|With|Without|Through)\s+[^,]*,\s*",
        s, flags=re.IGNORECASE,
    )
    if lead:
        tail = s[lead.end():].strip()
        if tail:
            variants.add(tail)
    return sorted(variants, key=len)


def _spacy_ok(sent) -> bool:
    root = None
    for tok in sent:
        if tok.dep_ == "ROOT":
            root = tok
            break
    if root is None:
        return False
    if root.pos_ not in {"VERB", "AUX", "NOUN", "ADJ"}:
        return False
    has_nsubj = any(
        c.dep_ in {"nsubj", "nsubjpass"} and c.pos_ in {"NOUN", "PROPN", "PRON"}
        for c in root.children
    )
    has_comp = any(
        c.dep_ in {"dobj", "attr", "acomp", "prep", "pobj", "xcomp",
                   "ccomp", "agent", "advmod"}
        for c in root.children
    )
    return has_nsubj and has_comp


def build_regional_learner(brain: Brain, region: str):
    from sara_brain.storage.neuron_repo import NeuronRepo
    from sara_brain.storage.segment_repo import SegmentRepo
    from sara_brain.storage.path_repo import PathRepo
    from sara_brain.core.learner import Learner
    from sara_brain.parsing.statement_parser import StatementParser
    from sara_brain.parsing.taxonomy import Taxonomy

    brain.db.create_region(region)
    nr = NeuronRepo(brain.conn, prefix=region)
    sr = SegmentRepo(brain.conn, prefix=region)
    pr = PathRepo(brain.conn, prefix=region)
    parser = StatementParser(taxonomy=Taxonomy())
    return Learner(parser, nr, sr, pr)


def retry_chapter(brain: Brain, region: str, facts_file: Path,
                  nlp) -> tuple[int, int]:
    """Return (newly_taught, still_skipped)."""
    if not facts_file.exists():
        return 0, 0
    learner = build_regional_learner(brain, region)
    taught = skipped = 0
    lines = facts_file.read_text().splitlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # First, try the original line — if it parses now, great.
        try:
            r = learner.learn(line, apply_filter=False)
        except Exception:
            r = None
        if r is not None:
            # Already counted in first-pass; treat as idempotent win.
            continue
        # Otherwise simplify aggressively and try each variant.
        for variant in aggressive_simplify(line):
            if variant == line:
                continue
            try:
                doc = nlp(variant)
            except Exception:
                continue
            if not any(_spacy_ok(s) for s in doc.sents):
                continue
            try:
                r = learner.learn(variant, apply_filter=False)
            except Exception:
                r = None
            if r is not None:
                taught += 1
                break
        else:
            skipped += 1
    brain.conn.commit()
    return taught, skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--facts-dir", required=True)
    args = ap.parse_args()

    facts_dir = Path(args.facts_dir)
    nlp = spacy.load("en_core_web_sm")
    brain = Brain(args.db)

    total_new = 0
    total_skip = 0
    for i in range(1, 48):
        region = f"ch{i}"
        facts_file = facts_dir / f"ch{i:02d}_facts.txt"
        if not facts_file.exists():
            continue
        t, s = retry_chapter(brain, region, facts_file, nlp)
        total_new += t
        total_skip += s
        print(f"Ch{i:02d}: +{t} newly taught, {s} still skipped")

    print(f"\nTotal newly taught via aggressive simplification: {total_new}")
    print(f"Still unparseable: {total_skip}")
    brain.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
