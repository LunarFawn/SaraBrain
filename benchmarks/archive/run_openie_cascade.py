#!/usr/bin/env python3
"""Run the rule-based OpenIE cascade over a file of source sentences.

Uses spaCy dep-parse extraction + five purpose-filter sensors
(definition, process, causation, temporal, datetime). Every output
triple is a source-span copy — no generation, no hallucination.

Output: pretty JSON array at {outdir}/triples.json with full
provenance: source sentence → extracted triples → which sensors
classified each triple.

Usage:
    .venv/bin/python benchmarks/run_openie_cascade.py \\
        --input benchmarks/biology2e_facts/ch10_facts.txt \\
        --outdir benchmarks/openie_out_ch10
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import spacy

from sara_brain.teaching.openie import extract_and_classify


def _split_lines(value):
    if isinstance(value, str) and "\n" in value:
        return value.splitlines()
    if isinstance(value, dict):
        return {k: _split_lines(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_split_lines(x) for x in value]
    return value


def _read_sentences(path: Path) -> list[str]:
    return [
        line.strip() for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    sentences = _read_sentences(args.input)
    if args.limit:
        sentences = sentences[: args.limit]
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_path = args.outdir / "triples.json"

    print(f"{len(sentences)} source sentences from {args.input}")
    print("loading spaCy…")
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    records: list[dict] = []
    t0 = time.time()
    total_triples = 0
    sensor_counts: dict[str, int] = {}
    for i, sentence in enumerate(sentences, 1):
        triples = extract_and_classify(sentence, nlp)
        total_triples += len(triples)
        for t in triples:
            for s in t.sensors:
                sensor_counts[s] = sensor_counts.get(s, 0) + 1
        records.append({
            "sentence": sentence,
            "triple_count": len(triples),
            "triples": [asdict(t) for t in triples],
        })
        if i % 25 == 0 or i == len(sentences):
            print(f"  [{i}/{len(sentences)}] triples so far: {total_triples}")

    with out_path.open("w") as f:
        json.dump(_split_lines(records), f, indent=2, ensure_ascii=False)
        f.write("\n")

    elapsed = time.time() - t0
    print()
    print(f"sentences: {len(sentences)}")
    print(f"triples:   {total_triples}")
    print(f"per-sensor counts:")
    for name, n in sorted(sensor_counts.items(), key=lambda x: -x[1]):
        print(f"  {name:<12} {n}")
    print(f"elapsed: {elapsed:.1f}s")
    print(f"output: {out_path}")


if __name__ == "__main__":
    main()
