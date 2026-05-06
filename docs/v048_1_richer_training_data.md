# v048.1 — richer training data for complex grammar

**Date:** 2026-05-05
**Branch:** `feature/grammar-cortex`
**Builds on:** [v048_complex_grammar_training.md](v048_complex_grammar_training.md)
(complex-grammar slot extension + corpus generator).

## Context

v048 ships a working complex-grammar checkpoint (`hamroby_sum_en_complex_004000.pt`,
final dev_loss 0.19). Simple clusters render cleanly:

> Edges: `(alice, walked_to, cafe)`
> Output: "Alice walked to cafe."

And two-scene compounds work:

> Edges: `(alice, walked_to, cafe)` + `(bob, sat, chair)`
> Output: "Alice walked to cafe. however, bob sat chair."

But multi-edge single-subject clusters with qualifiers fall down:

> Edges: `(alice, walked_to, cafe)` + `(alice, at_location, downtown)`
>      + `(alice, at_time, tuesday)` + `(alice, in_manner, quickly)`
> Wanted: "On Tuesday, Alice quickly walked to the cafe at downtown."
> Got:    "Alice walked to cafe and in manner quickly."

Root cause: the v048 training corpus never showed the model what to
do with a 4-edge cluster where 3 of the edges are qualifier
relations on a single main verb. The closest pattern the model
knew was the compound template (`<C0> <P0> <C1> and <P> <C>`) so it
fell back to that — emitting the qualifier relation name verbatim.

This is purely a training-data gap, not an architecture failure.
v048's slot mechanism is content-agnostic; it'll learn whatever
patterns we show it.

## What ships in v048.1

Three deltas to `generate_complex_substrate.py`, in increasing
investment. (1) and (2) ship in v048.1; (3) is captured for
v048.2 if needed.

### Slice 1 — full-qualifier templates (~10 lines)

Add templates that render scenes with all three qualifiers in one
sentence:

```python
def t_temporal_located_modified(scene):
    if not (scene.time and scene.location and scene.modifier):
        return None
    s = _slot_for(scene, slot)
    return f"{s['time']} , {s['subj']} {s['mod']} {s['pred']} {s['obj']} at {s['loc']} ."
```

Plus the partial-qualifier combinations that currently aren't
covered (e.g. `t_modified_located` — modifier + location without
time). The goal: every (time, location, modifier) presence
combination has at least one template that uses ALL the present
qualifiers in one sentence.

Expected impact: 4-edge clusters render as coherent single
sentences instead of falling back to compound patterns.

### Slice 2 — multi-event-per-subject clusters (~50 lines)

Currently each scene generates one event with optional qualifiers.
Add a "multi-event scene group" generator that produces 2-4 events
sharing a single subject across different time anchors:

```python
def _generate_subject_arc(rng, n_events=3) -> list[Scene]:
    """Generate N scenes that share a subject and chain through
    distinct time anchors. Used for 'subject does many things'
    training patterns the model currently doesn't know."""
    subj = rng.choice(subjects)
    times = rng.sample(time_pool, n_events)
    times.sort()  # chronological so sequencing reads naturally
    out = []
    for t in times:
        out.append(Scene(
            subject=subj, action=rng.choice(verbs),
            object=rng.choice(objects), time=t,
            location=rng.choice(locations) if rng.random() < 0.5 else None,
            event_id=f"event:arc_{rng.randrange(10**9)}",
        ))
    return out
```

A new template renders the whole arc as a chain of sentences:

```python
def t_temporal_chain(scenes: list[Scene], slot: SlotFn) -> str:
    parts = []
    for i, sc in enumerate(scenes):
        s = _slot_for(sc, slot)
        prefix = f"{s['time']} , " if 'time' in s else ""
        parts.append(f"{prefix}{s['subj']} {s['pred']} {s['obj']} .")
    return " ".join(parts)
```

This teaches the model that one subject + N edges (where N > 4)
renders as N sentences chained by time, NOT as one compound
sentence.

### Slice 3 — role-tagged slot pools (DEFERRED to v048.2)

Currently the serializer slots time/location/modifier as `<C>`
content slots. The vocab declares `<T>`/`<L>`/`<M>`/`<E>` slot
pools that go unused.

Routing per-role substrate strings into per-role slot pools at
serialization time would give the model a semantic hint about each
slot's grammatical position. `<T0>` always means "time expression",
which biases the model toward sentence-initial position with
trailing comma.

Cost: refactor `_serialize_complex_pair` to be role-aware. Decide
the role from the relation (`at_time` -> T-pool target,
`at_location` -> L-pool target, etc.). Update inference to expand
role-typed slots back to their substrate values.

Defer to v048.2 because (1)+(2) may be sufficient and (3) is more
invasive.

## What stays unchanged

- The slot mechanism (`<C>`/`<P>`/`<R>` already exist).
- The model architecture.
- v048 ckpt remains as the simpler-prose fallback.
- Inference path. v048.1 generates new training data and produces
  a new ckpt; chat REPL just points at the new path.

## Files

**Modified:**
- `papers/instrument_validation/generate_complex_substrate.py` —
  new templates (Slice 1) + new arc generator (Slice 2). ~60 lines
  total of additions.

**New:**
- `docs/v048_1_richer_training_data.md` — this plan.

**Reused unchanged:**
- `scripts/build_complex_corpus.sh` — corpus build orchestrator.
- `scripts/train_hamrobysum_complex.sh` — training wrapper.
- `train_synth.py`, `vocab_synth.py`, slot expansion in `inference_synth.py`.

## Order of operations

1. Save plan + commit (this commit).
2. Slice 1: add full-qualifier templates. Smoke-test that the
   scene-template list now produces the 4-qualifier form.
3. Slice 2: add `_generate_subject_arc` + `t_temporal_chain`.
   Smoke-test the corpus has multi-sentence chained training rows.
4. Rebuild corpus: `./scripts/build_complex_corpus.sh`.
5. Train v048.1 ckpt resuming from v048:
   ```
   PAIRS=/tmp/synth_pairs_complex.jsonl \
   CKPT_NAME=hamroby_sum_en_complex_v1 \
   STEPS=4000 \
   SESSION=sara-synth-complex-v1 \
   RESUME_FROM=src/sara_brain/cortex/checkpoints/hamroby_sum_en_complex_004000.pt \
   ./scripts/train_hamrobysum.sh
   ```
6. Inference smoke-test the same problem cluster from v048:
   `(alice, walked_to, cafe) + (alice, at_location, downtown) +
    (alice, at_time, tuesday) + (alice, in_manner, quickly)`.
   Pass criterion: output mentions all 4 substrate values in one
   coherent sentence.
7. Verification on the Carbon Helix prologue brain. After
   ingestion completes (B.2 of v047), test that the v048.1 ckpt
   renders Smith's events in narrative-style prose.

## Verification

End-to-end pass criteria:

1. The new corpus has at least 500 examples of 4-qualifier
   single-sentence rendering (Slice 1 working).
2. The new corpus has at least 200 examples of subject-arc
   multi-sentence rendering (Slice 2 working).
3. Training completes (4000 steps, ~15 min, dev_loss < 0.25).
4. The 4-qualifier test cluster renders as a single coherent
   sentence — no relation-name leakage, no compound fallback.
5. v048's existing test cases (simple, two-scene compound) still
   render correctly. No regression.

## Out of scope for v048.1

- Slice 3 (role-tagged slot pools) — deferred to v048.2 if
  (1)+(2) prove insufficient.
- Quoted speech / dialogue rendering as an LLM output (vs slash-
  command dialogue triples). Worth its own slice.
- Relative clauses ("Alice, who saw Bob, smiled"). Recursive prose
  slot mechanism is bigger than v048.1.
- Tense / aspect / mood inflection across multiple clauses.
- Coreference resolution ("Alice ... she ..."). Pronouns are a
  separate substrate-aware concern.

## Status

PLANNED. Implementation begins after this plan commits.
