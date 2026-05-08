# v050 — return to the two-layer architecture from Pearl 2026a / rev8

**Date:** 2026-05-07
**Branch:** `feature/grammar-cortex`
**Builds on (and supersedes):** the v035–v048.1 series, which built
HamRobySum-EN as a renderer between Sara and the user. This realignment
returns the codebase to the two-layer architecture the foundational
papers actually specify.

## Context

The papers define a **two-layer architecture**:

1. **Cortex** — frontier LLM (frozen, stateless, does language).
2. **Hippocampus** — Sara Brain (persistent path-graph store of
   structured triples).

**Pearl 2026a §7.3 (foundational, verbatim):**

> "transformers are the best sensory processing system ever engineered.
> They are not whole brains. **They process; they do not store. They
> are stateless; they do not accumulate. They infer; they do not
> remember.**
>
> Sara Brain is an attempt to build the cognitive system that sensory
> processing feeds. **Not to replace LLMs**, but to give them a
> persistent memory, a traceable knowledge store, and a hardwired
> ethical layer.
>
> The two systems together — **LLM as cortex, Sara Brain as
> hippocampus** — implement the biological architecture that evolved
> precisely because stateless sensory processing is not sufficient for
> intelligence."

**Pearl 2026 rev8 §2.4 (instrument paper, verbatim):**

> "Sara's retrieval uses parallel wavefront propagation across the
> graph rather than embedding-similarity search. Seeded neurons emit
> wavefronts; convergence points become recognition results. For
> instrument purposes this matters in one specific way: the output is
> intentionally *associative* rather than narrowed. **A reader LLM
> receives not a single 'best' answer but a structured neighborhood
> of related triples. The reader must do its own selection.** This is
> a feature: the instrument exposes *how the reader selects*, which is
> one of the behaviors the instrument is designed to measure."

## What drifted

v035–v048.1 built HamRobySum-EN, a 30M-param slot-based LLM whose job
was to render substrate edges as natural English prose. This put a
language-production model **inside** the hippocampus, between Sara
and the consumer.

The papers don't have this layer. They explicitly say language
production is the cortex's job.

The drift produced visible symptoms:

| period | added patch / training shape | reason |
|---|---|---|
| v039 | article fixer (`is a engineer` → `is an engineer`) | model couldn't do article agreement |
| v039 | predicate vocab brain (relation → English phrase) | model couldn't render unknown relations |
| v044 | same-subject Oxford-comma combiner | model produced wall of similar sentences |
| v046 | cluster-size chunking (split clusters > 8 edges) | model degenerated on large clusters |
| v047 | event-binding extractor (collapse event_* edges) | model couldn't render reified-event clusters |
| v048 | complex-grammar training corpus + new ckpt | simple training corpus didn't cover compound/temporal/etc. |
| v048.1 | full-qualifier templates + subject-arc generator + new ckpt | v048 still mangled multi-edge qualifier clusters |
| v049 (post-ship) | homogeneous-cluster splitter | LLM mangled 3+ same-relation edges |
| v049 (post-ship) | preposition-stutter detector | object/location head-collision |
| v049 (post-ship) | discourse-slot expansion table | `<R0>` token leaked through unexpanded |

Each patch was the architecture protesting that the language layer
doesn't belong inside the hippocampus. The patching pile **is** the
empirical evidence: a small renderer (30M params, slot-trained,
synthetic data) can't substitute for a frontier cortex.

## What this realignment does

1. **`chat.py --format {raw,prose}` is required.** No silent default.
   Forces every caller to choose explicitly between cortex-consumption
   format and direct-human-reading format.

2. **`--format raw`** emits the structured triple neighborhood
   directly — the same edge listings `brain_explore` returns and the
   same shape an MCP client receives. This is the paper-aligned
   default for any LLM-consumer workflow ("freeze Claude Dev, query
   Sara, paste raw output, let the cortex synthesize").

3. **`--format prose`** runs the v032 template synthesizer, which
   emits substrate-bound prose without a model. Acceptable as a
   developer comfort layer for direct terminal reading.

4. **`--use-hamrobysum`** stays as research-mode access to the v048.1
   ckpt. Its docstring and CLI help are reframed: this is preserved
   research, not a production path.

5. **`inference_synth.py`** module docstring rewritten to mark the
   research-artifact status explicitly.

6. **The user guide** is updated to teach the two-layer architecture
   first: hippocampus + cortex, with `--format raw` / MCP as the
   canonical hand-off.

## What stays unchanged

Most of the codebase is paper-aligned. This realignment touches only
the path that drifted:

- **Substrate** (SQLite path graph): paper-aligned, untouched.
- **MCP server** (`src/sara_brain/mcp_server.py`): emits structured
  edge text via `@mcp.tool()` functions; paper-aligned, untouched.
- **Reader tools** (`brain_explore`, `brain_value`, `brain_define`,
  `brain_why`, `brain_trace`, `brain_recognize`,
  `brain_did_you_mean`): paper-aligned hippocampus surface,
  untouched.
- **v047 reified events** (`event_tools.py`): hippocampus extension
  (multi-valued facts as bound nodes); paper-aligned, untouched.
- **v049 reified functions** (`code_tools.py`): same pattern applied
  to code-knowledge domain; paper-aligned, untouched.
- **Synthetic substrate generators**
  (`generate_synthetic_substrate.py`,
  `generate_complex_substrate.py`): training-orthogonality
  generators per rev8 §3, untouched.
- **Ingestion scripts** (`ingest_narrative_chapter.py`,
  `ingest_coding_guide.py`): Session-A teaching tooling per rev8 §4,
  untouched.
- **Template synthesizer** (`synthesizer.py`): emits substrate-bound
  prose without a model; acceptable as developer comfort under
  `--format prose`.
- **All HamRobySum infrastructure** (training scripts, checkpoints,
  vocab brain, the v035–v048.1 plan docs): preserved as research
  artifact.

## What this realignment does NOT do

- Does not delete any HamRobySum code. The work is preserved.
- Does not strip the inference-side patches (homogeneous-cluster
  splitter, etc.). They only run when `--use-hamrobysum` is set;
  harmless dead code in the default path.
- Does not refactor the MCP server. It's already paper-aligned.
- Does not add a JSON output mode (defer; `--format raw` text is
  sufficient for piping into a frontier LLM today).
- Does not remove `--use-hamrobysum`. It stays as research access.

## Files

**Modified:**
- `src/sara_brain/cortex/transformer/chat.py` — required `--format`
  flag; `ChatSession.__init__` requires `output_format`;
  `--use-hamrobysum` help text reframed.
- `src/sara_brain/cortex/transformer/inference_synth.py` — module
  docstring marks research-artifact status.
- `docs/user_guide_v049.md` — TL;DR and architecture diagram replaced
  with the two-layer architecture; `--format raw` shown as the
  recommended cortex-consumer interface.

**New:**
- `docs/v050_two_layer_realignment.md` — this doc.

**Audited, not modified:**
- `src/sara_brain/mcp_server.py` — confirmed paper-aligned;
  `@mcp.tool()` functions return structured edge text.

## Verification

End-to-end pass criteria:

1. `chat.py --brain X.db` (no `--format`) exits with usage error.
2. `chat.py --brain X.db --format raw` prints the gathered tool
   result verbatim; matches `brain_explore label=X` output.
3. `chat.py --brain X.db --format prose` prints
   template-synthesized prose (same as the pre-realignment default
   when `--use-hamrobysum` was off).
4. `chat.py --brain X.db --format prose --use-hamrobysum` loads the
   v048.1 ckpt and renders via HamRobySum (research-mode behaviour
   preserved).
5. `inference_synth.py` opens with the research-artifact docstring;
   importing it still works for research-mode users.
6. The user guide's TL;DR matches Pearl 2026a's two-layer language;
   `grep -n "five-layer" docs/user_guide_v049.md` returns nothing.
7. The MCP server still emits structured edge text (no regression).

## Architectural compass for future work

For any new feature added after this realignment, the test is:

- Does it **store** facts or expose stored facts? → hippocampus, ✓
  build it.
- Does it **produce language** from facts? → cortex's job. The
  frontier LLM does this. Don't build it inside Sara.
- Does it **measure** how the cortex uses Sara's output? →
  instrument concern (rev8 territory), build it as evaluation
  tooling.
- Does it **render prose** between Sara and a downstream consumer?
  → architectural error. Don't build it.

This compass keeps the codebase aligned with the papers' two-layer
claim and avoids re-creating the v035–v048.1 patching pressure on
some new layer.

## Status

SHIPPED 2026-05-07.
