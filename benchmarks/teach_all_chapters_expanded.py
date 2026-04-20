#!/usr/bin/env python3
"""Teach every biology2e chapter into a fresh brain.db via grammar expansion.

Walks every benchmarks/biology2e_facts/ch*_facts.txt file in numeric
order, calls Brain.teach_expanded on every sentence. Produces a brain
with full-textbook coverage so benchmarks spanning multiple chapters
have content to reason about.

Usage:
    .venv/bin/python benchmarks/teach_all_chapters_expanded.py --db brain.db
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from sara_brain.core.brain import Brain

_CH_RE = re.compile(r"ch(\d+)_facts\.txt$")


def _chapter_files(facts_dir: Path) -> list[Path]:
    files = []
    for p in facts_dir.glob("ch*_facts.txt"):
        m = _CH_RE.search(p.name)
        if m:
            files.append((int(m.group(1)), p))
    files.sort(key=lambda x: x[0])
    return [p for _, p in files]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--facts-dir", type=Path,
                    default=Path("benchmarks/biology2e_facts"))
    args = ap.parse_args()

    if args.db.exists():
        raise FileExistsError(f"Refusing to overwrite {args.db}")
    brain = Brain(str(args.db))

    chapter_files = _chapter_files(args.facts_dir)
    print(f"{len(chapter_files)} chapter files")

    grand_sentences = 0
    grand_sub_facts = 0
    t0 = time.time()
    for path in chapter_files:
        sentences = [
            l.strip() for l in path.read_text().splitlines()
            if l.strip() and not l.startswith("#")
        ]
        ch_sub_facts = 0
        for s in sentences:
            ch_sub_facts += brain.teach_expanded(s)
        grand_sentences += len(sentences)
        grand_sub_facts += ch_sub_facts
        print(f"  {path.name}: {len(sentences)} sentences → {ch_sub_facts} sub-facts")

    elapsed = time.time() - t0
    print()
    print(f"total sentences: {grand_sentences}")
    print(f"total sub-facts: {grand_sub_facts}")
    print(f"neurons:         {brain.conn.execute('SELECT COUNT(*) FROM neurons').fetchone()[0]}")
    print(f"paths:           {brain.conn.execute('SELECT COUNT(*) FROM paths').fetchone()[0]}")
    print(f"segments:        {brain.conn.execute('SELECT COUNT(*) FROM segments').fetchone()[0]}")
    verb_count = brain.conn.execute(
        "SELECT COUNT(*) FROM segments s JOIN neurons n2 "
        "ON n2.id=s.target_id WHERE s.relation='is_a' "
        "AND n2.label='verb'"
    ).fetchone()[0]
    print(f"learned verbs:   {verb_count}")
    print(f"elapsed:         {elapsed:.1f}s")


if __name__ == "__main__":
    main()
