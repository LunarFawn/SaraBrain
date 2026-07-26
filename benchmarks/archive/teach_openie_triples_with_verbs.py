#!/usr/bin/env python3
"""Feed OpenIE triples into Sara, two-pass: verb-teach first, then fact-teach.

Pass 1: walk every triple, teach its relation head-verb as a verb
via brain.teach_verb(). Unknown verbs become known.

Pass 2: walk triples again and teach each as a fact. With verbs now
registered, the parser accepts statements it previously rejected.

Usage:
    .venv/bin/python benchmarks/teach_openie_triples_with_verbs.py \\
        --db brain.db \\
        --triples benchmarks/openie_ch10/triples.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sara_brain.core.brain import Brain


def _head_verb(relation: str) -> str:
    """Pick the first content token of a multi-word relation as the
    verb head. 'can influence' -> 'influence'; 'are called' -> 'called'.
    Modals and auxiliaries are skipped."""
    _SKIP = {
        "can", "could", "may", "might", "will", "would", "shall",
        "should", "must", "do", "does", "did", "be", "am", "is", "are",
        "was", "were", "been", "being", "have", "has", "had", "not",
    }
    tokens = relation.strip().lower().split()
    for t in tokens:
        if t and t not in _SKIP and t.isalpha():
            return t
    return tokens[-1] if tokens else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--triples", required=True, type=Path)
    ap.add_argument("--min-engines", type=int, default=1)
    args = ap.parse_args()

    if args.db.exists():
        raise FileExistsError(f"Refusing to overwrite {args.db}")

    brain = Brain(str(args.db))
    data = json.load(args.triples.open())

    # Pass 1 — harvest head-verbs, register each
    head_verbs: Counter = Counter()
    for rec in data:
        for t in rec["triples"]:
            if len(t.get("engines", [])) < args.min_engines:
                continue
            head = _head_verb(t["relation"])
            if head:
                head_verbs[head] += 1
    print(f"pass 1 — registering {len(head_verbs)} distinct head verbs")
    for verb in head_verbs:
        brain.teach_verb(verb)

    # Pass 2 — teach each triple as a fact
    taught = 0
    rejected = 0
    reject_reasons: Counter = Counter()
    for rec in data:
        for t in rec["triples"]:
            if len(t.get("engines", [])) < args.min_engines:
                continue
            stmt = f"{t['subject']} {t['relation']} {t['obj']}".strip()
            result = brain.teach(stmt)
            if result is None:
                rejected += 1
                parsed = brain.parser.parse(stmt)
                if parsed is None:
                    reject_reasons["unparseable"] += 1
                elif parsed.verb_unknown:
                    reject_reasons[f"verb_unknown:{parsed.verb_unknown}"] += 1
                else:
                    reject_reasons["other"] += 1
            else:
                taught += 1
    brain.conn.commit()

    print()
    print(f"pass 2 — teach:")
    print(f"  taught:   {taught}")
    print(f"  rejected: {rejected}")
    print()
    print(f"rejection reasons (top 15):")
    for reason, n in reject_reasons.most_common(15):
        print(f"  {n:>4}  {reason}")
    print()
    print(f"neurons:  {brain.conn.execute('SELECT COUNT(*) FROM neurons').fetchone()[0]}")
    print(f"segments: {brain.conn.execute('SELECT COUNT(*) FROM segments').fetchone()[0]}")
    print(f"is_a:     {brain.conn.execute('SELECT COUNT(*) FROM segments WHERE relation=?', ('is_a',)).fetchone()[0]}")
    print(f"paths:    {brain.conn.execute('SELECT COUNT(*) FROM paths').fetchone()[0]}")


if __name__ == "__main__":
    main()
