#!/usr/bin/env python3
"""Run the teacher cascade over a file of source sentences.

Three phases:
  1a. 5 claim 1B sensors  → {outdir}/claim_out.json
  1b. 3 prose 1B sensors  → {outdir}/prose_out.json
  2.  3B integrators + synthesizer → {outdir}/teach_out.json

Output is auditable JSONL — every sensor's raw output is preserved
so a bad claim can be traced back to the exact prompt that produced it.

Usage:
    .venv/bin/python benchmarks/run_teacher_cascade.py \\
        --input benchmarks/biology2e_facts/ch10_facts.txt \\
        --outdir benchmarks/cascade_out_ch10 \\
        --model-1b llama3.2:1b --model-3b llama3.2:3b

Point `OLLAMA_HOST=http://<box>:11434` to run inference on a remote
machine. Defaults to localhost.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from sara_brain.teaching import cascade


def _read_sentences(path: Path) -> list[str]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--model-1b", default="llama3.2:1b")
    ap.add_argument("--model-3b", default="llama3.2:3b")
    ap.add_argument("--limit", type=int,
                    help="only process the first N sentences (for testing)")
    ap.add_argument("--skip-claim", action="store_true",
                    help="skip phase 1a (claim sensors)")
    ap.add_argument("--skip-prose", action="store_true",
                    help="skip phase 1b (prose sensors)")
    ap.add_argument("--skip-integration", action="store_true",
                    help="skip phase 2 (integrators + synthesizer)")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    sentences = _read_sentences(args.input)
    if args.limit:
        sentences = sentences[: args.limit]
    print(f"{len(sentences)} source sentences from {args.input}")
    print(f"output dir: {args.outdir}")

    claim_path = args.outdir / "claim_out.json"
    prose_path = args.outdir / "prose_out.json"
    teach_path = args.outdir / "teach_out.json"

    t0 = time.time()

    if not args.skip_claim:
        print(f"\n=== Phase 1a — claim sensors ({args.model_1b}) ===")
        t = time.time()
        cascade.run_claim_phase(sentences, args.model_1b, claim_path)
        print(f"  claim phase: {time.time() - t:.1f}s")

    if not args.skip_prose:
        print(f"\n=== Phase 1b — prose sensors ({args.model_1b}) ===")
        t = time.time()
        cascade.run_prose_phase(sentences, args.model_1b, prose_path)
        print(f"  prose phase: {time.time() - t:.1f}s")

    if not args.skip_integration:
        print(f"\n=== Phase 2 — integrators + synthesizer "
              f"({args.model_3b}) ===")
        t = time.time()
        cascade.run_integration_phase(
            claim_path, prose_path, args.model_3b, teach_path,
        )
        print(f"  integration phase: {time.time() - t:.1f}s")

    print(f"\ntotal: {time.time() - t0:.1f}s")
    print(f"outputs:")
    import json as _json
    for p in (claim_path, prose_path, teach_path):
        if p.exists():
            try:
                n = len(_json.loads(p.read_text()))
            except Exception:
                n = 0
            print(f"  {p}  ({n} records)")


if __name__ == "__main__":
    main()
