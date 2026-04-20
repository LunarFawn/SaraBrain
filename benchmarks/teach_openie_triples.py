#!/usr/bin/env python3
"""Feed OpenIE triples into Sara via brain.teach().

Reads a triples.json produced by run_openie_cascade.py. For each
triple, reconstructs a sentence "{subject} {relation} {object}" and
offers it to Sara's parser. Accepted triples become taught facts;
rejected ones are logged per-reason so you can see which verbs or
shapes the parser couldn't handle.

No filtering on engine consensus or sensor class — the caller can
pre-filter the JSON before feeding it here if they want to gate
by witness count.

Usage:
    .venv/bin/python benchmarks/teach_openie_triples.py \\
        --db brain.db \\
        --triples benchmarks/openie_ch10/triples.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sara_brain.core.brain import Brain


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--triples", required=True, type=Path)
    ap.add_argument("--min-engines", type=int, default=1,
                    help="minimum number of engines that must have "
                         "produced a triple for it to be taught")
    ap.add_argument("--sensors",
                    help="comma-separated sensor names to keep (e.g. "
                         "'definition,process'). Default: all.")
    args = ap.parse_args()

    if args.db.exists():
        raise FileExistsError(f"Refusing to overwrite {args.db}")

    brain = Brain(str(args.db))
    data = json.load(args.triples.open())

    wanted_sensors = None
    if args.sensors:
        wanted_sensors = {s.strip() for s in args.sensors.split(",")}

    taught = 0
    skipped_filter = 0
    skipped_parser = 0
    rejected_verb: Counter = Counter()
    for rec in data:
        sentence = rec["sentence"]
        for t in rec["triples"]:
            engines = t.get("engines", [])
            sensors = set(t.get("sensors", []))
            if len(engines) < args.min_engines:
                skipped_filter += 1
                continue
            if wanted_sensors and not (sensors & wanted_sensors):
                skipped_filter += 1
                continue
            # Reconstruct: "{subject} {relation} {object}"
            stmt = f"{t['subject']} {t['relation']} {t['obj']}".strip()
            result = brain.teach(stmt)
            if result is None:
                skipped_parser += 1
                rejected_verb[t["relation"].lower()] += 1
            else:
                taught += 1
    brain.conn.commit()

    print(f"taught:        {taught}")
    print(f"filtered out:  {skipped_filter}")
    print(f"parser reject: {skipped_parser}")
    print()
    print("top rejected relations (likely unknown verbs):")
    for rel, n in rejected_verb.most_common(15):
        print(f"  {n:>4}  {rel}")
    print()
    print(f"neurons:  {brain.conn.execute('SELECT COUNT(*) FROM neurons').fetchone()[0]}")
    print(f"segments: {brain.conn.execute('SELECT COUNT(*) FROM segments').fetchone()[0]}")
    print(f"is_a:     {brain.conn.execute('SELECT COUNT(*) FROM segments WHERE relation=?', ('is_a',)).fetchone()[0]}")
    print(f"paths:    {brain.conn.execute('SELECT COUNT(*) FROM paths').fetchone()[0]}")


if __name__ == "__main__":
    main()
