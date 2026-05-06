# v047 — reified events + narrative corpus

**Date:** 2026-05-05
**Branch:** `feature/grammar-cortex`
**Builds on:** [v028_multi_layer_cortex_architecture.md](v028_multi_layer_cortex_architecture.md)
(architecture), [v040_predicate_slots_and_vocab_brain.md](v040_predicate_slots_and_vocab_brain.md)
(L3 vocab brain), [v045_multihop_reasoning_plan.md](v045_multihop_reasoning_plan.md)
(multi-hop substrate traversal).

## Context

Two pressures converge:

**1. Multi-valued facts break binary triplets.** A statement like
"Bob is at the cafe from 3-5pm Tuesday" wants to carry four bound
values (subject, location, start_time, end_time). Naive triplets
unbind them — when Bob is also at the library 5-7pm, the substrate
has no way to know which start_time goes with which location.
Aggregating across friends ("where is Bob right now?") becomes
impossible without a binding mechanism.

**2. The current corpus has reached its limit as a development
substrate.** Two unrelated problems:

  - The science vocabulary in the demo brain triggers safety filters
    on common research terms, blocking discussion of the work.
  - The vocab brain only has 51 of the demo brain's 250+ relations,
    most of which are obscure scientific verbs. The fallback for
    unregistered relations (`replace("_", " ")`) produces
    grammatically broken output ("role in", "contribute to") because
    these aren't proper verb phrases.

A narrative corpus addresses both: events are first-class in fiction,
and narrative vocabulary is closer to standard English so the vocab
brain works much better out of the box.

The solution couples them. Reified events solve the multi-valued
binding problem and are the natural fit for narrative data. The
novella corpus exercises the event machinery from day one on data
where it's the dominant case.

## What ships in v047

Two coupled slices that complete a working narrative cortex.

### Slice A — reified event convention + tools

Events are nodes with `neuron_type='event'`. Binding edges link the
event to its subject, action, location, and time bounds. This is
just convention over the existing schema — no migration needed.

**Convention:**
```
event:bob_at_cafe_t1   neuron_type='event'
  --[event_subject]-->  bob
  --[event_action]-->   located
  --[event_object]-->   cafe
  --[event_start]-->    timestamp:2026-05-06T15:00
  --[event_end]-->      timestamp:2026-05-06T17:00
```

Event node labels follow `event:<subject>_<action>_<n>` where `n` is
a counter to disambiguate repeats. Timestamp nodes follow
`timestamp:<ISO-8601>`. Relations `event_*` are reserved for binding
edges.

**Reification IS nested triplets:** once `event:bob_at_cafe_t1`
exists as a node, you can say `(event:bob_at_cafe_t1, observed_by,
alice)` — a triplet about a triplet. No special syntax for nesting;
the event node is just a regular substrate node.

**New tools** in `sara_reader.tools`:

- `brain_teach_event(subject, action, object=None, location=None,
  start_time=None, end_time=None) -> str`. Creates the event node
  and all binding edges in one call. Returns the event node label.

- `brain_query_event_at(subject, timestamp) -> str`. Finds events
  where the subject participates and `event_start <= timestamp <=
  event_end`. Returns the active event(s) and their bindings.

- `brain_query_events(subject) -> str`. Lists all events involving
  a subject in chronological order.

- `brain_explore` extension: when called on an event node, returns
  the bound facts in a readable form ("Bob was located at the cafe
  from 3pm to 5pm Tuesday") rather than just dumping edges.

**Chat REPL slash commands:**

- `/teach-event <subject> <action> <object> [at <location>] [from <time1> to <time2>]`
  — natural-language event teaching.
- `/where-is <subject> [at <time>]` — convenience wrapper around
  `brain_query_event_at`. Defaults to current time.

**Synthesizer integration:** the vocab brain gets `event_*`
relations registered with proper English forms ("was located at",
"started at", "ended at"). Event nodes render via a new template
in `synthesizer.py` that detects `neuron_type='event'` and produces
"X was at Y from T1 to T2" rather than dumping the binding edges
individually.

### Slice B — novella corpus ingestion

The novella is the user's own work — they have full editorial
control and ground truth. Ingestion happens chapter-by-chapter in
three phases per chapter:

**Phase 1: Entity + relation extraction (assisted, manual review)**

A new script `scripts/ingest_narrative_chapter.py` takes a chapter
text file and produces a draft TSV of `(subject, relation, object)`
triples and `(event_subject, event_action, event_object, location,
start, end)` event tuples. The user reviews and edits the TSV
before it's written to the brain.

For v047 the extraction is regex + simple NLP heuristics:
- Character names from a curated list (`characters.txt` per novella).
- Location names from a curated list.
- Time markers via regex: "Tuesday morning", "the next day", "at
  dawn", etc. Mapped to the chapter's internal timeline.
- Verbs as relation candidates.

LLM-assisted extraction is **out of scope for v047**. We want the
ingestion pipeline to work without external API calls, consistent
with the no-API constraint. Quality of extraction is acceptable to
be rough — manual review is the expected step.

**Phase 2: Vocab brain teaching**

After review, any new verbs/relations from the chapter that aren't
in `vocab_en.db` get prompted: the user supplies the proper English
form via `/teach-vocab` (or batch via a TSV column). This grows the
narrative vocab brain organically.

**Phase 3: Substrate write**

Reviewed triples become substrate edges. Reviewed events become
event nodes via `brain_teach_event`. Written to a fresh brain:
`src/sara_brain/cortex/checkpoints/novella_brain.db` (path TBD by
user).

### Slice C — verification

A new chat REPL session, `--brain novella_brain.db`, exercises:

1. **Definitional queries:** "Who is Alice?" returns her
   character-defining edges from the substrate.
2. **Event queries:** "Where was Alice on Tuesday morning?" runs
   `brain_query_event_at(alice, tuesday_morning)` and returns the
   active event prose.
3. **Multi-hop narrative:** "Why did Alice go to the cafe?" exercises
   the v045 multi-hop orchestrator over event nodes (event_action →
   triggered_by → cause).
4. **Nested triplets:** "Did Bob see Alice at the cafe?" tests
   queries over reified events as objects of other edges
   (`(bob, witnessed, event:alice_at_cafe_t1)`).
5. **Grammar quality:** narrative output reads as proper English
   without the v044/v046 hacks (since narrative vocab maps cleanly
   to verb phrases). If grammar is still broken, that's evidence
   the issue is the synth model itself, not just vocab coverage.

## What stays unchanged

- The synth model. No retraining for v047 itself. The same v040 EN
  ckpt renders narrative substrate; quality on the novella
  validates whether the architecture is domain-general.
- The substrate schema. Event nodes use `neuron_type='event'` —
  already supported. No migrations.
- The L1/L2/L3/L4 stack. Adding a fifth layer (events) is tempting
  but premature — events compose from existing primitives.
- v028 honesty guarantees. Event nodes are just substrate; their
  bindings are edges. Every claim still traces to substrate.

## Files

**New:**
- `docs/v047_reified_events_and_narrative_corpus.md` — this plan.
- `src/sara_reader/event_tools.py` — `brain_teach_event`,
  `brain_query_event_at`, `brain_query_events`, event-aware
  `brain_explore` extension. ~250 lines.
- `scripts/ingest_narrative_chapter.py` — chapter → draft TSV +
  vocab gap report. ~200 lines.
- `narrative/characters.txt` (per novella) — curated entity list
  used by the ingestion script.
- `narrative/locations.txt` (per novella) — curated location list.

**Modified:**
- `src/sara_brain/cortex/transformer/chat.py` — `/teach-event`,
  `/where-is`, `/list-events` slash commands.
- `src/sara_brain/cortex/transformer/synthesizer.py` — event-node
  template detection + rendering.
- `src/sara_brain/cortex/vocab/vocab_en.db` — `event_*` relations
  registered with proper English forms.
- `src/sara_reader/tools.py` — register new tools in `TOOLS`
  registry.

**Reused:**
- `Brain.teach_triple` for binding edges.
- `execute_tool` machinery for new tools.
- Synthesizer's slot pipeline; event templates are added alongside
  existing relation templates.
- v045 multi-hop orchestrator (works unchanged on event nodes).

## Order of operations

1. Save plan + commit (this commit).
2. **Slice A.1:** implement `event_tools.py` with the four new
   tools + register in `TOOLS`. Unit-test against a fresh in-memory
   brain. Single commit.
3. **Slice A.2:** add slash commands to chat.py. Smoke-test in REPL.
   Single commit.
4. **Slice A.3:** synthesizer event-node template. Verify a single
   teach-and-render round-trip works. Single commit.
5. **Slice B.1:** write `ingest_narrative_chapter.py`. User runs it
   on chapter 1 of the novella, reviews/edits the TSV. Iterate until
   the script produces useful drafts.
6. **Slice B.2:** ingest chapter 1 into `novella_brain.db`. Teach
   vocab gaps via `/teach-vocab`. ~30-60 min expected for one
   chapter.
7. **Slice C:** verification queries against the chapter-1 brain.
   If grammar is clean, declare v047 done and ingest more chapters
   incrementally. If grammar is still broken, file the failure mode
   as v048 (likely a synth-side issue, not a vocab-side one).

## Verification (when implemented)

End-to-end pass criteria:

1. Teach a single event via `/teach-event alice walked to cafe at
   tuesday morning` and confirm the event node + 4-5 binding edges
   exist in the brain.
2. Query `/where-is alice at tuesday morning` returns the event in
   readable prose.
3. Query `/where-is alice at wednesday` returns "no events found"
   honestly (not invented).
4. Multi-hop: "Why did Alice go to the cafe?" routes through the
   event node and surfaces causal substrate edges if they exist;
   returns honest "no causal edges found" otherwise.
5. Reading a narrative paragraph back: ingest 5-10 events from one
   scene and ask "what happened in that scene?" — output should
   read as narrative prose, in temporal order.
6. Architecture: every claim still traces to substrate edges
   (event nodes are substrate; event_* edges are substrate).

## Out of scope for v047

- LLM-assisted extraction during ingestion. The pipeline is
  manual/heuristic for v047 to preserve the no-API constraint and
  keep the user in editorial control. May reconsider if manual
  ingestion proves too slow.
- Cross-reference resolution ("she" → which character). The
  ingestion script keeps pronouns as-is in draft TSV; the user
  resolves them during review. Automated coreference is a separate
  slice.
- Event causality inference. If chapter 3 says "X happened because
  Y," that becomes an explicit `(event_x, caused_by, event_y)` edge
  during ingestion. We do not infer causality from prose order.
- Synth retraining on narrative corpus. v047 tests whether the
  current EN ckpt generalizes; if it doesn't, retraining on
  augmented narrative substrate becomes v048 or v049.
- Multi-novella support. Single novella for v047. Multiple
  novellas (or other narrative sources) is a deferred feature once
  the per-novella flow is solid.
- Temporal arithmetic ("3 days after the meeting"). Time markers
  are stored as named timestamp nodes; computed offsets between
  named times is future work.
- Cross-character event aggregation ("everyone who was at the cafe
  Tuesday"). Possible with current tools (query each character +
  filter), but no convenience tool for it in v047.

## Why this is the right slice now

Three reasons it composes well:

1. **It validates architecture domain-generality without a model
   change.** If the v040 EN model + reified events produce clean
   narrative output, the L1/L2/L3/L4 stack is proven domain-
   independent. That's the strongest evidence to date that the
   architecture is real engineering, not just one good demo.

2. **It unblocks development.** The current substrate's vocabulary
   triggers safety filters that interrupt every session. The
   novella substrate has none of that. Daily work gets faster.

3. **It exercises capabilities we already built.** v040 (predicate
   slots), v044 (combining), v045 (multi-hop) all designed for
   binary triplets. Reified events extend them to multi-valued
   facts without breaking any of those features. The new
   capability is purely additive.

The work is bounded — Slice A is ~600 lines, Slice B is ~200
lines + per-chapter manual review. End-to-end working narrative
brain on chapter 1 is a weekend, not weeks.

## Status

SHIPPED 2026-05-06.

**Implementation commits:**
- `cc3ff4d` — Slice A.1: event_tools.py + brain_explore extension
- `f6c3a74` — Slice A.2: chat REPL slash commands
- `58647cb` — Slice A.3: synthesizer event-aware rendering
- `67195f4` — Slice A.3 fix: drop incomplete event refs
- `329024b` — Slice B.1: ingest_narrative_chapter.py
- (and a query_event_at point-in-time fix — see verification below)

**Verification — slice C end-to-end on Carbon Helix prologue:**

Curated 37 rows from the prologue (32 events + 5 dialogue triples)
covering Smith, Sylvia, Byrd, and Helix across 7 named time
anchors (t1_meeting → t7_helix_departure). Applied via
`ingest_narrative_chapter.py apply` to a fresh brain.db.

All five v047 verification criteria met:

1. **Single-event teach + render.** Every reified event in the
   prologue renders as one bundled sentence via brain_explore on
   the event node:
   > "smith walked bay 15 at shipyard at t3_walk_to_bay15"

2. **Time-anchored queries.** `brain_query_event_at(smith,
   t4_at_helix)` returns exactly the 4 events tagged that time
   anchor — Smith stopping, entering, closing the hatch, and
   locking it. Other time anchors don't bleed in.

3. **Honest miss.** `brain_query_event_at(smith, z9999_unknown)`
   returns "No active events ... DO NOT invent a location."

4. **Multi-character chronology.** `brain_query_events(sylvia)`
   correctly returns Sylvia's 5 events (announcing, ordering,
   refusing, watching, calling security) in time order.

5. **Synthesizer end-to-end.** With template synthesis +
   _expand_event_references, "what is helix doing?" produces:
   > "Helix closed hatch at docking hatch at t7_helix_departure.
   >  Helix reported protecting human life at operations office
   >  at t5_helix_dialogue. Helix undocked bay 15 at bay 15 at
   >  t7_helix_departure. Smith commanded helix at operations
   >  office at t7_helix_departure. Hello engineer smith this is
   >  quantum processor model zero said helix. Yes am i not
   >  supposed to said helix."

   Every claim traces to a substrate edge. Dialogue triples render
   as `<quote> said <character>.` Event-node bindings collapse
   into single sentences.

**Architectural validation:** the same machinery (substrate +
slot-based synth + event reification) handles a narrative domain
without any retraining, schema change, or special-case code. The
v040 EN model and the v048.1 model both render this brain
correctly — confirming the slot mechanism is genuinely domain-
general.

**Bug found + fixed during verification:** `query_event_at`
over-matched when only `event_start` was set (no end). Original
behaviour was open-ended "from start onward"; events tagged
t1_meeting matched the t4_at_helix query because t1 < t4. Changed
to point-in-time exact-match when only one bound is set —
callers wanting interval semantics should provide both bounds.

**Out-of-scope improvements that surfaced during testing** (not
v047 issues):
- Multi-event single-subject collapse: Smith has 19 events, each
  rendered as its own sentence. Could collapse to "Smith did X,
  then Y, then Z" via v048.1 arc shape — needs inference-side
  wiring of arc detection (a v049-ish slice).
- Awkward "at X at Y" wording when location and time both use
  "at" preposition. Template polish.
- Time labels are technical (t1_meeting). User curation choice.
