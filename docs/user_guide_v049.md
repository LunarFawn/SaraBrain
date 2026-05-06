# Sara Brain user guide (post-v047)

**Audience:** anyone who wants to use Sara Brain as a substrate-bound
knowledge store for narrative, code, style-guide, or any other
multi-valued-fact domain.

**Companion to:** `sara_brain_user_guide.md` (install + foundational
concepts) and the per-version plan docs in this directory. This guide
is the practical reference for what shipped in v047, v048, v048.1,
and v049.

---

## TL;DR

Sara Brain stores facts as substrate edges. A small synthesizer LLM
(HamRobySum) renders edges as English prose. Every word the LLM
emits traces to a real substrate edge — *the LLM cannot fabricate
facts*, only render the ones it's been shown.

You teach Sara through:
- **Slash commands** in the chat REPL (`/teach`, `/teach-event`,
  `/teach-vocab`).
- **TSV ingestion scripts** for batches of facts
  (`ingest_narrative_chapter.py`, `ingest_coding_guide.py`).
- **Direct Python** via `Brain.teach_triple` /
  `event_tools.teach_event` / `code_tools.teach_function`.

You query Sara through:
- **Slash commands** (`/where-is`, `/list-events`, `/find-function`,
  `/callers`, etc.).
- **Natural language** in the REPL (the router picks the right tool).
- **Direct Python** via `sara_reader.tools.execute_tool`.

The synthesizer renders the gathered facts as prose. If a fact isn't
in the substrate, the answer is *honest absence* — not invention.

---

## Architecture in five layers

```
┌──────────────────────────────────────────────────────────┐
│  Chat REPL: natural-language interface + slash commands  │  ← user
├──────────────────────────────────────────────────────────┤
│  Synthesizer (HamRobySum LLM): edges → prose             │  ← only LLM
│                                                            │     touching
│                                                            │     content
├──────────────────────────────────────────────────────────┤
│  Router (tiny LLM): question → tool choice               │
├──────────────────────────────────────────────────────────┤
│  Reader tools: SQL queries → edge text                   │
├──────────────────────────────────────────────────────────┤
│  Substrate (SQLite): neurons + segments                  │  ← source
│                                                            │     of truth
└──────────────────────────────────────────────────────────┘
```

The substrate is just a SQLite DB of `(source, relation, target)`
triples plus reified-node conventions for multi-valued facts (events,
functions, etc.). Everything else is operating on those triples.

---

## Quick start: open a chat REPL against a brain

```bash
.venv/bin/python -m sara_brain.cortex.transformer.chat \
  --brain /tmp/your_brain.db --device cuda \
  --use-hamrobysum \
  --hamrobysum-ckpt src/sara_brain/cortex/checkpoints/hamroby_sum_en_complex_v1_004000.pt
```

Flags worth knowing:

| flag | what it does |
|---|---|
| `--brain PATH` | which substrate DB to open |
| `--device cuda` / `--device cpu` | inference device |
| `--use-hamrobysum` | use the LLM synthesizer (cleaner prose) |
| `--hamrobysum-ckpt PATH` | which synth ckpt to load |
| `--vocab-brain PATH` | optional vocab-brain for predicate phrases |
| `--multihop` | enable bounded-BFS chain reasoning for "why/how" |

Without `--use-hamrobysum`, you get the v032 template renderer (a
hand-written fallback). With it, you get the trained slot-based LLM.

---

## Three example workflows

### Workflow 1 — Narrative brain (Carbon Helix)

Use when the substrate models story content: characters, events,
locations, dialogue.

**Build:**

```bash
# 1. Curate a TSV with events + dialogue + definitional triples
#    See /tmp/book_one_curated.tsv for the Carbon Helix example.

# 2. Apply
.venv/bin/python scripts/ingest_narrative_chapter.py apply \
  --tsv /tmp/book_one_curated.tsv \
  --brain /tmp/carbon_helix.db
```

**Query:**

```
> who is morgan
> /list-events smith
> /where-is helix at t7_helix_departure
> what is dust
```

**Best slash commands:**
- `/list-events SUBJECT` — all events involving SUBJECT in time order
- `/where-is SUBJECT [at=TIME]` — point-in-time location query
- `/teach-event` — add an event interactively
- `/teach SUBJECT is a OBJECT` — add a definitional fact

### Workflow 2 — Code knowledge brain

Use when the substrate models a codebase or API: functions,
parameters, types, call graph.

**Build:**

```bash
# 1. Extract a draft TSV from Python source via AST
.venv/bin/python scripts/ingest_coding_guide.py extract \
  --backend python-ast \
  --input src/sara_reader/event_tools.py \
  --out /tmp/api_draft.tsv

# 2. (Optional) review and edit /tmp/api_draft.tsv
#    - drop private helpers (rows where keep=0)
#    - polish docstrings, add `describes` text per parameter
#    - filter out built-in callees (commit, strip, fetchall, etc.)

# 3. Apply
.venv/bin/python scripts/ingest_coding_guide.py apply \
  --tsv /tmp/api_draft.tsv \
  --brain /tmp/code_kb.db
```

**Query:**

```
> /find-function teach_event
> /callers download_file
> /callees fetch_dataset
> /returns-type Path
> /takes-type Brain
```

`/find-function` returns the full bundled function info — signature,
return type, parameters with types, docstring, callees, callers — all
substrate-bound. Paste it directly into a coding LLM as grounded
context. The LLM can't fabricate signatures because every claim
traces to a substrate edge.

### Workflow 3 — Style guide / framework brain

Use when the substrate models conventions, rules, idioms,
anti-patterns. No new code needed — combines the existing `kind=
triple` ingestion with the v049 reification pattern.

**Plain triples for simple rules:**

```
keep  paragraph_n  kind   subject  action       object
1     -1           triple pep8     recommends   snake_case
1     -1           triple snake_case  applies_to   functions
1     -1           triple snake_case  applies_to   methods
1     -1           triple PascalCase  applies_to   classes
1     -1           triple black    is_a         formatter
1     -1           triple black    enforces     pep8
```

**Reified rule nodes for multi-valued style facts** (use the
v047 event tools or the upcoming v050 style tools — for now,
manually teach the binding edges):

```python
b.teach_triple("rule:pep8-snake-functions", "applies_to", "functions")
b.teach_triple("rule:pep8-snake-functions", "has_form", "snake_case")
b.teach_triple("rule:pep8-snake-functions", "has_example", "download_file")
b.teach_triple("rule:pep8-snake-functions", "has_anti_example", "downloadFile")
b.teach_triple("rule:pep8-snake-functions", "rationale",
               "consistency per PEP 8 §3")
b.teach_triple("rule:pep8-snake-functions", "enforced_by", "pylint")
b.teach_triple("rule:pep8-snake-functions", "part_of", "pep8")
```

Then `brain_explore label='rule:pep8-snake-functions'` pulls all
binding edges as one cluster, and the synthesizer renders them as
prose.

For framework conventions and anti-patterns, the same reification
pattern works — `convention:django-fat-models`,
`antipattern:react-hooks-in-conditional`, etc. Each prefix becomes
a "kind" of node; binding edges carry the multi-valued facts.

---

## Slash command reference

| command | what it does |
|---|---|
| `/help` | show all commands |
| `/teach STATEMENT` | teach a free-form statement (parsed) |
| `/teach SUBJECT REL OBJECT` | teach a flat triple |
| `/refute STATEMENT` | refute a previously-taught statement |
| `/teach-event SUBJECT ACTION [object=O] [location=L] [from=ISO] [to=ISO] [modifier=M]` | create a reified event node |
| `/where-is SUBJECT [at=ISO]` | point-in-time location query |
| `/list-events SUBJECT` | all events involving SUBJECT, chronological |
| `/teach-vocab REL PHRASE` | add an English form for a relation |
| `/refute-vocab REL [PHRASE]` | remove vocab mappings |
| `/list-vocab [REL]` | inspect vocab brain mappings |
| `/multihop` | toggle multi-hop reasoning on/off |
| `/dig` | expand the last query — pull sibling concepts |
| `/dig CONCEPT` | drill into a specific concept |
| `/depth N` | re-run last brain_explore at hop distance N |
| `/find-function NAME [module=M]` | full function info (sig + params + docstring + calls) |
| `/callers NAME` | list functions that call NAME |
| `/callees NAME` | list functions NAME calls |
| `/returns-type TYPE` | list functions returning TYPE |
| `/takes-type TYPE` | list functions taking parameter of TYPE |
| `/trace` | toggle: show routing decisions + classifier confidence |
| `/verbose` | toggle: print raw substrate output (no synthesis) |
| `/brain PATH` | switch to a different brain.db |
| `/model` | show loaded checkpoints |
| `/quit` / `/exit` / `Ctrl-D` | leave |

---

## TSV ingestion format reference

Both `ingest_narrative_chapter.py` and `ingest_coding_guide.py` use
TSV files with a `kind` column that selects the apply path.

### Narrative TSV (`kind=event` / `kind=dialogue` / `kind=triple`)

| column | required | purpose |
|---|---|---|
| keep | yes | `1` to apply; `0` to skip during apply |
| paragraph_n | optional | paragraph index for traceability |
| kind | yes | `event` / `dialogue` / `triple` |
| subject | yes | the subject of the fact |
| action | yes | for events: the action verb. for triples: the relation. for dialogue: usually `said` |
| object | varies | event object / quote text / triple object |
| location | event-only | binds via `event_location` |
| time | event-only | binds via `event_start` |
| source_text | optional | a snippet of the source for traceability |

Example rows:

```
keep  paragraph_n  kind     subject  action      object              location           time
1     0            event    smith    entered     conference room     conference room    t1_meeting
1     6            dialogue smith    said        Can I have one more day?
1     -1           triple   smith    is_a        engineer
1     -1           triple   helix    also_known_as  carbon helix
```

### Code TSV (`kind=function` / `kind=parameter`)

| column | required | purpose |
|---|---|---|
| keep | yes | `1` to apply; `0` to skip |
| kind | yes | `function` / `parameter` |
| name | function-row | function name |
| signature | function-row | full signature string (preserves case) |
| returns | function-row | return type label |
| defined_in | function-row | source file path |
| calls | function-row | comma-separated callee names |
| docstring | function-row | docstring (newlines flattened to spaces) |
| func_name | parameter-row | which function the parameter belongs to |
| param_name | parameter-row | parameter name |
| type | parameter-row | type label |
| default | parameter-row | default value or empty |
| describes | parameter-row | per-parameter description |

---

## The reified-fact pattern (events, functions, anything multi-valued)

A reified fact is a node whose label starts with a type prefix and
whose binding edges hold the multi-valued data:

```
event:bob_at_cafe_t1     neuron_type='event'
  --[event_subject]-->   bob
  --[event_action]-->    located
  --[event_object]-->    cafe
  --[event_start]-->     2026-05-06T15:00
  --[event_end]-->       2026-05-06T17:00

function:download_file   neuron_type='function'
  --[has_signature]-->   download_file(url: str, dest: Path) -> Path
  --[returns]-->         Path
  --[takes_param]-->     parameter:download_file.url
  --[calls]-->           function:write_atomic
```

The reified node lets binary triples represent multi-valued facts
without losing the binding (which time goes with which location;
which type goes with which parameter). It also enables nesting —
once `event:bob_at_cafe_t1` exists as a node, you can say
`(event:bob_at_cafe_t1, observed_by, alice)` and reason over events
as first-class entities.

The pattern generalises: any prefix + binding-relation set defines
a new "kind" of reified fact. `recipe:`, `protocol:`,
`runbook:` would all work the same way. The substrate doesn't
treat them specially — they're just nodes.

---

## Honest absence (no hallucination)

When the substrate doesn't know something, the answer is *honest
absence*, not invention:

```
> /where-is alice at z9999_unknown
No active events for 'alice' at 'z9999_unknown'. Honest miss — DO
NOT invent a location.

> /find-function completely_made_up_function
No function 'completely_made_up_function' found in the brain. DO
NOT invent a signature — confirm the name with the user or use
brain_did_you_mean.
```

The synthesizer LLM literally cannot emit a fact that isn't in the
substrate. The slot mechanism replaces all substrate strings with
abstract slot tokens before tokenizing; the LLM only sees structural
slots. After decoding, slots get re-substituted from the substrate.
There is no path through the architecture that lets an unknown
string come out the other side.

This gives three structural properties no normal LLM has:

1. **Hallucination is impossible** — the model can only emit slot
   tokens that were in its facts prefix.
2. **Adding facts doesn't require retraining** — new brain.db, same
   model, immediately works.
3. **Aliases work cleanly** — `also_known_as` edges resolve naming
   variation without weight updates.

---

## Common patterns and tricks

### Aliases for naming variation

When the same entity has multiple names ("Helix" / "Carbon" /
"Carbon Helix"), teach `also_known_as` edges in both directions:

```
/teach helix is also known as carbon helix
/teach carbon helix is also known as helix
```

Now queries against either name resolve to the same content.

### Time-anchored chronology

Use named time labels (`t1_meeting`, `t2_after_meeting`) for
human-readable chronology. Lexicographic ordering is meaningful
between them — `t01_morning` sorts before `t10_evening`. For
strict chronological ordering, zero-pad the index (`t01`, `t02`).

ISO-8601 timestamps work too and are preferred for real-world
dates.

### Multi-event subject arcs

Many events for one subject (Smith doing 19 things across the
prologue) renders as 19 sentences via the synthesizer. The v048.1
ckpt knows how to chain them by time:

```
> /list-events smith   # gets each event as one row, time-sorted
```

If you want narrative-style chaining for a "tell me Smith's day"
question, the v048.1 ckpt's training included subject-arc
templates — natural-language questions trigger this.

### Beefier ingestion = better answers

The most common reason a "who is X?" answer feels thin is that
the substrate has events but no definitional triples. Add
`is_a`, `has_property`, `also_known_as`, `part_of` rows for
every entity. ~10 definitional rows per character changes
"who is Smith?" from a list of actions to a complete profile.

### Drilling down

Three options when an answer references something you want to
explore:

| symptom | use |
|---|---|
| answer mentions a term you want to know more about | `/dig CONCEPT` |
| answer feels too sparse, want adjacent edges | `/depth 2` |
| want to see ALL events for one subject | `/list-events SUBJECT` |
| want the temporal-spatial slice | `/where-is SUBJECT at T` |

For "why" / "how" questions, enable `/multihop` first — the
bounded-BFS orchestrator chains substrate retrievals through
related concepts.

---

## Building a new brain from scratch

```bash
# 1. Plan your reified node prefixes (event:, function:, rule:, etc.)
#    and binding-relation conventions.

# 2. Curate a TSV of definitional triples + (if applicable) events
#    or functions or rules.

# 3. Apply
.venv/bin/python scripts/ingest_narrative_chapter.py apply \
  --tsv /tmp/your_brain.tsv \
  --brain /tmp/your_brain.db

# 4. Open the chat REPL
.venv/bin/python -m sara_brain.cortex.transformer.chat \
  --brain /tmp/your_brain.db --device cuda --use-hamrobysum \
  --hamrobysum-ckpt src/sara_brain/cortex/checkpoints/hamroby_sum_en_complex_v1_004000.pt

# 5. Iterate — query, find gaps, add triples, re-ingest. The brain
#    file is just a SQLite DB; you can write to it from any tool.
```

---

## Troubleshooting

| symptom | fix |
|---|---|
| "Sara has no neuron matching 'X'" | the substrate doesn't know X by that exact name. Try `/dig X` or teach an `also_known_as` edge. |
| Output has "X is part of Y" noise from dialogue | known v047 pre-`ae7d5dc` issue: dialogue went through chain learning which decomposed multi-word objects. Fixed by `kind=triple` / `kind=dialogue` direct-SQLite path. Re-ingest. |
| "at X at Y" duplicate prepositions in event prose | fixed by the v047 polish commit (`403773c`) — synthesizer detects overlap and switches to `on` for the time clause. |
| Multi-edge cluster renders awkwardly | use the v048.1 ckpt (`hamroby_sum_en_complex_v1_004000.pt`) which trained on full-qualifier + arc shapes. |
| `/where-is SUBJECT at T` returns events from other times | bug fixed by v047 Slice C verification — point-in-time exact-match when only one bound is set. |
| Router picks wrong tool for a question | use `/trace` to see routing decisions; manually invoke via `brain_explore` / `brain_define` if needed. |

---

## Reference: where things live

| layer | file(s) |
|---|---|
| substrate (SQLite) | any `.db` file you create |
| reader tools (read-only) | `src/sara_reader/tools.py` |
| event tools (v047, write+read) | `src/sara_reader/event_tools.py` |
| code tools (v049, write+read) | `src/sara_reader/code_tools.py` |
| synthesizer (template fallback) | `src/sara_brain/cortex/transformer/synthesizer.py` |
| synthesizer (LLM render) | `src/sara_brain/cortex/transformer/inference_synth.py` |
| chat REPL | `src/sara_brain/cortex/transformer/chat.py` |
| narrative TSV ingest | `scripts/ingest_narrative_chapter.py` |
| code TSV ingest | `scripts/ingest_coding_guide.py` |
| trained ckpts | `src/sara_brain/cortex/checkpoints/` |
| vocab brain | `src/sara_brain/cortex/vocab/vocab_en.db` |

| version doc | covers |
|---|---|
| `v047_reified_events_and_narrative_corpus.md` | reified events + narrative ingestion |
| `v048_complex_grammar_training.md` | complex-grammar synth corpus |
| `v048_1_richer_training_data.md` | full-qualifier + arc shapes |
| `v049_code_knowledge_substrate.md` | function reification + code ingestion |

---

## What's next

Three directions, in increasing investment:

1. **Ingest more chapters / modules.** The architecture is validated;
   the limit on usefulness is now corpus depth. Beefier substrate ⇒
   richer answers.
2. **v050 style-guide tools.** Sibling to v049 code_tools — formal
   `teach_rule` / `query_rule` / `query_violations` plus a markdown
   ingestor for PEP-style guides. ~400 lines.
3. **Cross-substrate composition.** Today each brain is its own
   SQLite. Composing a code brain with a style brain lets you ask
   "does this function follow PEP 8 naming?". Substrate-level join
   is straightforward; query orchestration is a new slice.

Pick whichever pulls.
