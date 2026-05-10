"""Deprecated. The v2 BPE-tokenized extraction pipeline turned out to
be a paradigm misalignment — content-aware subword tokenization
fragmented words at output time and undermined Sara's whole-label
substrate (the "del"/"ight" problem from the 2026-05-09 real-prose
eval). The neural pieces (tokenizer, model, encoder, BIO head,
trainers, eval scripts) have been removed.

The active extractor is now `hamroby_extractor_v1/` — a grammar-feature
transformer that operates at word level, never embeds open-class
content, and produces verbatim multi-word spans. See
`docs/v047_reified_events_and_narrative_corpus.md` for the surrounding
narrative-event design and `hamroby_extractor_v1/__init__.py` for the
new extractor's docstring.

Three modules remain here as shared utilities used by both the active
extractor and the ingest CLI / MCP server:

  - `format_readers/` — TXT, Markdown, PDF, EPUB, HTML/URL readers.
    No BPE involvement, just file I/O. Reused by `cli_teach_book` and
    `mcp_server.brain_ingest(extractor="grammar")`.
  - `extractor_rules.py` — pure spaCy rule-based subject/relation/object
    extractor. Stub used as a fallback when the trained head produces
    no clean triple, and as the default extractor before any model is
    trained. Whole-word output by spaCy construction.
  - `synthetic_pairs.py` — nonsense-substrate (prose, triple) generator
    with content-orthogonality by construction. Used by
    `hamroby_extractor_v1` to build training data.
"""
