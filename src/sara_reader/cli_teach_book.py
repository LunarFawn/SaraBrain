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
    if nlp is None:
        # Simple sentence splitting without spaCy (for sara extractor)
        import re
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
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
        "--multipass", action="store_true",
        help="Run 3 focused passes over the document: "
             "(1) definitions (is_a), (2) relationships (action verbs), "
             "(3) bridges (connecting concepts already in substrate). "
             "Produces richer substrate coverage than a single pass.",
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
        choices=("rules", "trained", "hybrid", "sara"),
        help="Triple extractor: 'rules' = deterministic spaCy+rules stub "
             "(default, fast); 'trained' = hamroby_extractor_v1 trained head "
             "(loads canonical .pt checkpoint, uses spaCy sm+trf cascade); "
             "'hybrid' = run BOTH and feed combined triples into the brain; "
             "'sara' = from-scratch 115M copy-mechanism extractor (no spaCy, "
             "definitions-first, trained on synthetic data). "
             "Hybrid captures clean SVO (from trained head) AND compound-NP "
             "associations like 'jkd part_of \"the foundations of jkd\"' "
             "(from rule stub) — useful when terms appear mostly as POBJ "
             "or inside compound NPs.",
    )
    args = p.parse_args(argv)

    fmt = args.format or detect_format(args.source)
    print(f"[teach-book] source={args.source} format={fmt} brain={args.brain}",
          file=sys.stderr)

    # Select the extractor(s) and load the appropriate nlp. The trained
    # head and hybrid modes use the cascade nlp (sm + trf fallback);
    # rule-only mode uses plain sm; sara mode uses the from-scratch 115M model.
    extractors: list[tuple[str, callable]] = []
    nlp = None  # only needed for spacy-based extractors

    if args.extractor == "sara":
        # From-scratch 115M extractor — no spaCy needed
        import torch
        import re as _re
        from pathlib import Path as _Path
        sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent / "scripts"))
        from train_sara_extractor_scratch import SaraExtractor, build_vocab, encode_with_oov

        _ckpt_path = str(_Path(__file__).resolve().parent.parent.parent / "models" / "sara-extractor-115m-v2" / "best.pt")
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _tok2id = build_vocab()
        _ext_vocab = len(_tok2id) + 300
        _model = SaraExtractor(_ext_vocab, d_model=768, enc_layers=8, dec_layers=6,
                               n_heads=12, max_enc=300, max_dec=150).to(_device)
        _ckpt = torch.load(_ckpt_path, map_location=_device, weights_only=False)
        _model.load_state_dict(_ckpt["model"])
        _model.eval()
        print(f"[teach-book] loaded sara extractor from {_ckpt_path}", file=sys.stderr)

        def _sara_extract(clause, _nlp_unused):
            """Extract triples using the 115M from-scratch model."""
            enc_ids, oov, oov_map = encode_with_oov(clause, _tok2id, 300)
            enc_t = torch.tensor([enc_ids], dtype=torch.long, device=_device)
            pm = torch.zeros(1, len(enc_ids), dtype=torch.bool, device=_device)
            with torch.no_grad():
                out_ids = _model.generate(enc_t, pm, max_len=100)[0].tolist()
            id2tok = {v: k for k, v in _tok2id.items()}
            for t, idx in oov_map.items():
                id2tok[idx] = t
            gen = " ".join(id2tok.get(i, "?") for i in out_ids if i not in (0, 2))

            # Parse structured output
            from sara_brain.cortex.transformer.v2.extractor_rules import Triple
            triples = []
            for part in gen.split("t_end"):
                if "t_start" in part and "t_rel" in part and "t_obj" in part:
                    try:
                        after = part.split("t_start")[1]
                        subj = after.split("t_rel")[0].strip()
                        rel = after.split("t_rel")[1].split("t_obj")[0].strip()
                        obj = after.split("t_obj")[1].strip()
                        if subj and rel and obj and len(subj) > 1 and len(obj) > 1 and subj != obj:
                            triples.append(Triple(subject=subj, relation=rel, object=obj, source_clause=clause))
                    except (IndexError, ValueError):
                        pass
            return triples

        extractors.append(("sara", _sara_extract))

    elif args.extractor in ("trained", "hybrid"):
        try:
            import spacy
        except ImportError:
            print("error: spaCy is not installed", file=sys.stderr)
            return 2
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
            import spacy
        except ImportError:
            print("error: spaCy is not installed", file=sys.stderr)
            return 2
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("error: en_core_web_sm model not found",
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

    def _run_pass(pass_filter=None, pass_label: str = "single"):
        """Run one extraction pass over the source. If pass_filter is
        provided, only triples passing the filter are committed."""
        nonlocal segments_seen, sentences_seen, clauses_seen, triples_committed
        pass_committed = 0
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
                        triples = []
                        for _name, _fn in extractors:
                            triples.extend(_fn(clause, nlp))
                        if pass_filter is not None:
                            triples = pass_filter(triples)
                        for tri in triples:
                            relation_counts[tri.relation] += 1
                            if not args.quiet:
                                print(
                                    f"[teach-book] [{pass_label}] {seg.provenance} :: "
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
                                    pass_committed += 1
                                except Exception as e:  # noqa: BLE001
                                    print(
                                        f"[teach-book] WARN teach failed: {e} "
                                        f"on ({tri.subject!r}, {tri.relation!r}, {tri.object!r})",
                                        file=sys.stderr,
                                    )
        except StopIteration:
            pass
        return pass_committed

    if args.multipass:
        from sara_brain.cortex.transformer.v2.multipass import (
            filter_bridges,
            filter_definitions,
            filter_relationships,
        )

        print("[teach-book] multi-pass mode: 3 passes over document", file=sys.stderr)

        print("[teach-book] === Pass 1/3: definitions ===", file=sys.stderr)
        n = _run_pass(filter_definitions, "definitions")
        print(f"[teach-book] pass 1 committed {n} definition triples", file=sys.stderr)

        # Reset clause counter for pass 2 (max-clauses applies per pass)
        clauses_seen = 0
        print("[teach-book] === Pass 2/3: relationships ===", file=sys.stderr)
        n = _run_pass(filter_relationships, "relationships")
        print(f"[teach-book] pass 2 committed {n} relationship triples", file=sys.stderr)

        # Reset clause counter for pass 3
        clauses_seen = 0
        print("[teach-book] === Pass 3/3: bridges ===", file=sys.stderr)
        n = _run_pass(lambda ts: filter_bridges(ts, brain), "bridges")
        print(f"[teach-book] pass 3 committed {n} bridge triples", file=sys.stderr)
    else:
        _run_pass()

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
