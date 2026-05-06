# v049 — code knowledge substrate (Sara as a coding-context source)

**Date:** 2026-05-06
**Branch:** `feature/grammar-cortex`
**Builds on:** [v047_reified_events_and_narrative_corpus.md](v047_reified_events_and_narrative_corpus.md)
(reified events as the "multi-valued fact" pattern), the slot-based
synthesizer architecture (v035-v048).

## Context

The reified-event pattern that ships a narrative brain works for
any **multi-valued fact** — not just events in a story. A function
definition is a multi-valued fact:

```
function: download_file
  signature: download_file(url: str, dest: Path) -> Path
  returns: Path
  parameters: url (str), dest (Path)
  raises: NetworkError, IOError
  defined_in: src/net/downloader.py
  calls: urlopen, write_atomic, validate_url
  example: download_file('https://x.com/a.zip', Path('/tmp/a.zip'))
  docstring: "Download URL to dest atomically; returns dest on success."
```

Same shape as a v047 event: one canonical node ("function:download_file")
plus binding edges to its signature, return type, parameters,
caller-callees, examples. Every fact traces to substrate.

**Use case:** an LLM writing code asks Sara for grounded info on
a function it needs to call. Sara returns the substrate-bound facts
(signature + behaviour + relationships). The LLM can't hallucinate
because every claim it makes about the function comes from a real
substrate edge — same hallucination guarantee that v047 gives for
narrative facts, applied to code.

## What ships in v049

A code-graph ingestion path that mirrors the narrative ingestion
path. v047's reified-event tools work unchanged — code "facts" are
just another shape of multi-valued node.

### Slice A — code triples + reified function nodes

Extend `event_tools.py` with a sibling `code_tools.py` that
defines the code-fact convention:

```
function:<module>.<name>   neuron_type='function'
  --[has_signature]-->     <signature string>
  --[returns]-->           <type label>
  --[takes_param]-->       parameter:<name>
  --[raises]-->            <exception type>
  --[defined_in]-->        <file path>
  --[calls]-->             function:<other>
  --[uses_library]-->      <library label>
  --[has_docstring]-->     <docstring>
  --[has_example]-->       example:<id>
```

Plus parameter sub-nodes for type info:

```
parameter:url   --[has_type]-->         str
parameter:url   --[is_optional]-->      false
parameter:url   --[has_default]-->      <default value if any>
```

Tools (mirroring brain_query_event_at, brain_query_events):
- `brain_query_function(name)` — returns the full signature +
  docstring + parameters + examples in one bundled rendering.
- `brain_query_callers(name)` — who calls this function?
- `brain_query_callees(name)` — what does this function call?
- `brain_query_by_returns(type)` — find functions that return X.
- `brain_query_by_param(type)` — find functions taking X as a param.

These are read-only query tools (sibling to the v047 read tools).
Write tools (`teach_function`, `teach_parameter`) live in
`code_tools.py` for the ingestion script to call.

### Slice B — coding-guide ingestion

A new script `scripts/ingest_coding_guide.py` parallels
`ingest_narrative_chapter.py`:

**Two-pass extract / apply workflow:**

```
extract: ingest_coding_guide.py extract \
           --input api_docs.md \
           --out /tmp/api_draft.tsv

# user reviews/edits the TSV

apply:   ingest_coding_guide.py apply \
           --tsv /tmp/api_draft.tsv \
           --brain /tmp/code_kb.db
```

Extraction strategies (in order of cost):

1. **Markdown / RST docstring scrape** — regex parse for function
   signatures (`def foo(...)`, `function foo(...)`) and the
   following paragraph as docstring. Fast, gets ~70% of common
   doc layouts right.

2. **Python AST parse** for `.py` files — `ast.parse` extracts
   actual function defs, signatures, decorators, calls within the
   body. Most accurate path for Python source.

3. **TS / Go / Rust** parsers — out of scope for v049; same shape
   as Python AST when added.

The TSV format matches narrative ingestion: `keep, paragraph_n,
kind, subject, action, object, location, time, source_text` with
new `kind` values:

```
function   subject=function:download_file   action=has_signature   object=<sig>
function   subject=function:download_file   action=returns          object=Path
parameter  subject=parameter:url            action=has_type         object=str
example    subject=example:dl_001           action=demonstrates     object=function:download_file
```

### Slice C — chat REPL slash commands

Mirroring v047's `/where-is`, `/list-events`:

```
/find-function FOO           — same as brain_query_function(foo)
/callers FOO                  — who calls foo?
/callees FOO                  — what does foo call?
/returns-type T               — functions that return T
/takes-type T                 — functions that take T as a param
```

Plus `/explain FOO` — gather full context (signature + docstring +
callers + callees + examples) and synthesise as multi-paragraph
prose suitable for pasting into a coding-LLM prompt.

### Slice D — output format optimised for LLM consumption

The synthesiser by default produces narrative prose. For coding
context, an LLM benefits from a *structured* format:

```
function: download_file
signature: download_file(url: str, dest: Path) -> Path
returns: Path

parameters:
  - url (str): the source URL
  - dest (Path): where to write

calls: urlopen, write_atomic, validate_url
raises: NetworkError, IOError
defined in: src/net/downloader.py

docstring:
  Download URL to dest atomically; returns dest on success.

example:
  download_file('https://x.com/a.zip', Path('/tmp/a.zip'))
```

A new render-mode flag (`/format=code`) produces this layout
instead of bundled narrative prose. The flag flips the
synthesiser into "structured fields per line" rather than
"sentences with substrate-bound clauses."

## Honest assessment of difficulty

This is **architecturally cheap** (the v047 machinery generalises)
but **operationally expensive** in curation effort. A useful code
brain needs:

- Several hundred function definitions to be useful for typical
  coding tasks.
- Accurate signatures, including types — wrong types are worse
  than no types because they confidently mislead.
- Up-to-date docstrings — stale ones harm more than they help.
- Example calls — most function uses are pattern-matched from
  examples, not docstrings.

The Python AST extraction (Slice B path 2) gets you 80% of the
data automatically for any Python project. The remaining 20% is
docstring quality and example curation — same problem any
documentation effort has.

## Why this matters

Three real benefits over current LLM-coding workflows:

1. **No fabricated signatures.** The biggest LLM coding failure
   mode is inventing functions that don't exist or signatures
   that look plausible but aren't right. Sara's substrate-bound
   property makes this structurally impossible — the LLM only
   sees signatures that came from real substrate edges.

2. **Substrate updates without retraining.** New API ships?
   Re-ingest. The coding LLM uses the new signatures immediately,
   no fine-tuning, no RAG vector database to rebuild.

3. **Aliases work the way they should.** `download_file` vs
   `downloadFile` vs `dl_file` — teach an `also_known_as` edge
   and the brain resolves both forms to the same node. No more
   "the LLM kept calling it the wrong case."

## Files

**New:**
- `docs/v049_code_knowledge_substrate.md` — this plan.
- `src/sara_reader/code_tools.py` — code-fact convention + read
  tools. ~250 lines.
- `scripts/ingest_coding_guide.py` — markdown + AST extraction
  + TSV apply path. ~300 lines.

**Modified:**
- `src/sara_brain/cortex/transformer/synthesizer.py` — extend
  `extract_event_renderings` (or sibling) to handle function-
  shaped reified nodes the same way it handles event-shaped
  ones. Optional `/format=code` flag.
- `src/sara_brain/cortex/transformer/chat.py` — `/find-function`,
  `/callers`, `/callees`, `/explain` slash commands.

**Reused unchanged:**
- v047 reified-event machinery (the convention generalises to any
  multi-valued node type — events, functions, recipes, anything).
- Slot-based synth pipeline (substrate-bound rendering works on
  function nodes the same as event nodes).
- ingest_narrative_chapter.py's TSV apply infrastructure (kind=
  triple already supported; just needs new kind=function /
  kind=parameter handlers).

## Order of operations

1. Save plan + commit (this commit).
2. Slice A: code_tools.py — write the convention + read tools.
   Smoke-test with hand-written triples.
3. Slice B: ingest_coding_guide.py with the markdown extractor
   first (simpler, useful for any docs format). AST extractor
   later as a follow-up if Python source ingestion becomes
   common.
4. Slice C: chat REPL slash commands. Smoke test.
5. Slice D: code-format renderer. The default narrative
   rendering should still work for code substrate — Slice D
   just adds a structured alternative.
6. Real-world test: ingest the `sara_reader.tools` module's
   functions into a code brain, query them, see if the output
   looks usable as LLM context.

## Verification

End-to-end pass criteria:

1. Hand-teach a function (e.g. download_file) via /teach-event
   style + /teach kind=function commands. /find-function returns
   signature + parameters + return type + docstring as one
   bundled rendering.
2. /callers returns functions that call download_file (after
   teaching at least one).
3. Ingestion of a markdown API doc produces a draft TSV with
   correct signature extractions for ~70%+ of functions in a
   sample.
4. /explain produces output that a coding LLM can use directly
   as in-context grounding (manual quality check).
5. Architecture: every function-related claim still traces to a
   substrate edge. No invented signatures.

## Out of scope for v049

- Multi-language source parsing (Go, Rust, TS) — Python only for
  v049. Patterns generalise; just add per-language extractors.
- Code-execution sandbox to verify examples compile/run. Worth
  its own slice; the substrate just stores claims about code,
  doesn't execute it.
- Type-system-aware queries ("functions that return Iterable[T]
  for some T"). The substrate stores type strings as labels;
  parametric type matching is a future slice.
- Coupling Sara directly to an external coding LLM. v049 ships
  the substrate + queries; the user pastes Sara's output into
  whatever coding LLM they prefer.
- Auto-ingestion via git hooks or CI. Manual ingestion for v049.

## Status

PLANNED. Implementation begins after this plan commits.
