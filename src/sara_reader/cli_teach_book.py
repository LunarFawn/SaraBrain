"""cli_teach_book — ingest books and papers into Sara Brain via grammar parse.

Auto-commit pipeline:

    source (file path or URL)
        |
    format reader (txt/md/pdf/epub/html)  -> TextSegment(text, provenance)
        |
    spaCy sentence segmentation
        |
    EnhancedParser._split_compound  -> per-clause strings
        |
    extract_triples (rule-based stub) -> [Triple(s, r, o, source_clause), ...]
        |
    brain.teach_triple(subject, relation, object, source_text=clause, source_label=provenance)
        |
    log to stderr + optional SARA_AUDIT_LOG row

No LLM/API call. Run on any TXT/Markdown/PDF/EPUB/HTML/URL source.
Mistakes are corrected later via the existing /refute path in
cli_stateless_chat.

Usage:
    python -m sara_reader.cli_teach_book SOURCE [--brain DB] [--format FMT]
                                                [--max-clauses N] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

from sara_brain.core.brain import Brain
from sara_brain.cortex.parser import EnhancedParser
from sara_brain.cortex.transformer.v2.extractor_rules import extract_triples
from sara_brain.cortex.transformer.v2.format_readers import detect_format, read


def _audit(line: str) -> None:
    log_path = os.environ.get("SARA_AUDIT_LOG")
    if not log_path:
        return
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line.rstrip() + "\n")
    except OSError:
        pass


def _segment_sentences(nlp, text: str) -> list[str]:
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="cli_teach_book",
        description="Auto-ingest a book/paper into Sara Brain via grammar parse.",
    )
    p.add_argument("source", help="File path or URL")
    p.add_argument(
        "--brain", default="/tmp/sara_book.db",
        help="Brain DB path (created if absent). Default: /tmp/sara_book.db",
    )
    p.add_argument(
        "--format", default=None,
        help="Override format auto-detection: txt|md|pdf|epub|html|url",
    )
    p.add_argument(
        "--max-clauses", type=int, default=0,
        help="Stop after this many clauses (0 = no limit). Useful for smoke tests.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Extract and log triples but do not write to the brain.",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-triple stderr lines.",
    )
    args = p.parse_args(argv)

    fmt = args.format or detect_format(args.source)
    print(f"[teach-book] source={args.source} format={fmt} brain={args.brain}",
          file=sys.stderr)

    try:
        import spacy
    except ImportError:
        print("error: spaCy is not installed (.venv/bin/pip install spacy)",
              file=sys.stderr)
        return 2
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("error: en_core_web_sm model not found "
              "(.venv/bin/python -m spacy download en_core_web_sm)",
              file=sys.stderr)
        return 2

    if not args.dry_run:
        Path(args.brain).parent.mkdir(parents=True, exist_ok=True)
    brain = Brain(args.brain)

    segments_seen = 0
    sentences_seen = 0
    clauses_seen = 0
    triples_committed = 0
    relation_counts: Counter[str] = Counter()
    started = time.time()

    try:
        for seg in read(args.source, format=fmt):
            segments_seen += 1
            for sentence in _segment_sentences(nlp, seg.text):
                sentences_seen += 1
                for clause in EnhancedParser._split_compound(sentence):
                    if not clause.strip():
                        continue
                    clauses_seen += 1
                    if args.max_clauses and clauses_seen > args.max_clauses:
                        raise StopIteration
                    triples = extract_triples(clause, nlp)
                    for tri in triples:
                        relation_counts[tri.relation] += 1
                        if not args.quiet:
                            print(
                                f"[teach-book] {seg.provenance} :: "
                                f"({tri.subject!r}, {tri.relation!r}, {tri.object!r})",
                                file=sys.stderr,
                            )
                        _audit(
                            f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\tteach_triple\t"
                            f"{seg.provenance}\t{tri.subject}\t{tri.relation}\t"
                            f"{tri.object}\t{tri.source_clause}"
                        )
                        if not args.dry_run:
                            try:
                                brain.teach_triple(
                                    tri.subject,
                                    tri.relation,
                                    tri.object,
                                    source_text=tri.source_clause,
                                    source_label=seg.provenance,
                                )
                                triples_committed += 1
                            except Exception as e:  # noqa: BLE001 - surface and skip
                                print(
                                    f"[teach-book] WARN teach failed: {e} "
                                    f"on ({tri.subject!r}, {tri.relation!r}, {tri.object!r})",
                                    file=sys.stderr,
                                )
    except StopIteration:
        pass
    finally:
        brain.close()

    elapsed = time.time() - started
    top_rels = ", ".join(f"{r}={c}" for r, c in relation_counts.most_common(8))
    print(file=sys.stderr)
    print(
        f"[teach-book] done in {elapsed:.1f}s. "
        f"segments={segments_seen} sentences={sentences_seen} "
        f"clauses={clauses_seen} triples_committed={triples_committed}",
        file=sys.stderr,
    )
    if top_rels:
        print(f"[teach-book] top relations: {top_rels}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
