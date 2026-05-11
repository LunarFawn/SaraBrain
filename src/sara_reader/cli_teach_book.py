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
from sara_brain.cortex.transformer.v2.extractor_rules import (
    extract_triples as extract_triples_rules,
)
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
    p.add_argument(
        "--no-dictionary", action="store_true",
        help="Skip auto-loading Moby Thesaurus II synonym scaffolding. "
             "By default, new brains get the dictionary bootstrap "
             "(~13s one-time cost, ~30k synonym groups, ~800k edges) so "
             "wavefront propagation can bridge question vocabulary "
             "('tallest') to substrate labels ('extreme phenotype'). "
             "Pass this flag for prose-only brains where the dictionary "
             "isn't needed and the load time matters.",
    )
    p.add_argument(
        "--extractor", default="rules",
        choices=("rules", "trained", "hybrid"),
        help="Triple extractor: 'rules' = deterministic spaCy+rules stub "
             "(default, fast); 'trained' = hamroby_extractor_v1 trained head "
             "(loads canonical .pt checkpoint, uses spaCy sm+trf cascade); "
             "'hybrid' = run BOTH and feed combined triples into the brain. "
             "Hybrid captures clean SVO (from trained head) AND compound-NP "
             "associations like 'jkd part_of \"the foundations of jkd\"' "
             "(from rule stub) — useful when terms appear mostly as POBJ "
             "or inside compound NPs.",
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
    # Select the extractor(s) and load the appropriate nlp. The trained
    # head and hybrid modes use the cascade nlp (sm + trf fallback);
    # rule-only mode uses plain sm. `extractors` is a list of (name, fn)
    # tuples; each runs on every clause and all returned triples are
    # committed. Hybrid runs both — clean SVO from trained head plus
    # compound-NP associations from rule stub.
    extractors: list[tuple[str, callable]] = []
    if args.extractor in ("trained", "hybrid"):
        from sara_brain.cortex.transformer.hamroby_extractor_v1.feature_extractor import (
            load_domain_nlp,
        )
        from sara_brain.cortex.transformer.hamroby_extractor_v1.inference import (
            extract_triples as extract_triples_trained,
        )
        try:
            nlp = load_domain_nlp()
        except OSError as e:
            print(f"error: failed to load cascade nlp: {e}", file=sys.stderr)
            return 2
        extractors.append(("trained", extract_triples_trained))
        if args.extractor == "hybrid":
            extractors.append(("rules", extract_triples_rules))
    else:
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("error: en_core_web_sm model not found "
                  "(.venv/bin/python -m spacy download en_core_web_sm)",
                  file=sys.stderr)
            return 2
        extractors.append(("rules", extract_triples_rules))

    print(
        f"[teach-book] extractor={args.extractor} "
        f"(running: {', '.join(name for name, _ in extractors)})",
        file=sys.stderr,
    )

    if not args.dry_run:
        Path(args.brain).parent.mkdir(parents=True, exist_ok=True)
    brain = Brain(args.brain)

    # Auto-bootstrap the dictionary on new brains so the wavefront has
    # synonym scaffolding by default. Idempotent — skipped if the brain
    # already has synonym_of segments. Opt out with --no-dictionary.
    if not args.no_dictionary and not args.dry_run:
        from sara_brain.bootstrap import ensure_dictionary
        ensure_dictionary(brain)

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
                    # Run each configured extractor. Hybrid mode runs
                    # both; rules/trained mode runs one. All triples are
                    # committed; deduplication is left to the brain
                    # (teach_triple is idempotent on identical edges).
                    triples = []
                    for _name, _fn in extractors:
                        triples.extend(_fn(clause, nlp))
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
