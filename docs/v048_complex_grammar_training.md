# v048 — complex grammar training (compound, temporal, modified)

**Date:** 2026-05-05
**Branch:** `feature/grammar-cortex`
**Builds on:** [v035_generic_slot_hamrobysum.md](v035_generic_slot_hamrobysum.md)
(slot architecture), [v040_predicate_slots_and_vocab_brain.md](v040_predicate_slots_and_vocab_brain.md)
(predicate slots + vocab brain), [v047_reified_events_and_narrative_corpus.md](v047_reified_events_and_narrative_corpus.md)
(events as substrate nodes — what this enables prose-side).

## Context

The v040 EN model was trained on triplet → simple-sentence pairs:

```
<C0> <P0> <C1> .
```

Output stays one-clause. Real prose — especially narrative prose
the v047 novella corpus will expose — needs more:

- **Compound:** "Alice walked in and sat down."
- **Complex:** "Alice left because Bob arrived."
- **Conditional:** "If Bob arrives, Alice will leave."
- **Temporal:** "On Tuesday, Alice walked to the cafe."
- **Located:** "At the cafe, Alice met Bob."
- **Modified:** "Alice quickly walked to the cafe."
- **Discourse-connected:** "Alice walked in. However, Bob was already gone."
- **Relative-clause:** "Alice, who saw Bob, smiled."
- **Quoted speech:** "Alice said, '...'."

The slot architecture is content-agnostic *by design* — extending
to complex grammar means extending the *patterns* the model trains
on, not changing the slot mechanism. The generator and trainer
already exist; we only add new slot types and richer templates.

## What ships in v048

A new training corpus + new ckpt. v040 EN stays as the simple-prose
fallback so existing chat sessions don't regress.

### Slice 1 — extend slot vocabulary

Append-only additions to [vocab_synth.py](../src/sara_brain/cortex/transformer/vocab_synth.py).
The 51-relation v040 EN ckpt's vocab IDs stay valid because every
existing token keeps its existing position; new slot types append
at the end.

New slot types:
- `<T0>...<T7>` — time expressions ("on Tuesday", "at dawn", "the
  next day"). Up to 8 distinct time references per scene.
- `<L0>...<L7>` — location expressions. ("the cafe", "the kitchen")
- `<M0>...<M7>` — modifiers (manner adverbs, intensifiers).
  ("quickly", "reluctantly", "very")
- `<E0>...<E3>` — event references for nesting. ("the meeting",
  "the call") — corresponds to v047 event nodes.
- `<R0>...<R3>` — discourse connectives. ("however", "therefore",
  "meanwhile", "furthermore")

The substrate-content-never-enters-vocab principle is preserved —
all slot values still get filled from substrate at inference.

### Slice 2 — random scene generator

A new generator script `papers/instrument_validation/generate_complex_substrate.py`.
Replaces the flat triplet-stream of `generate_synthetic_substrate.py`
with **scenes**: clusters of related triplets that share temporal
and spatial frame.

Schema of a scene:
```python
{
  "subject": "<random nonsense word>",
  "action": "<one of the verb pool>",
  "object": "<random nonsense word>",
  "location": "<random nonsense word>" | None,
  "time": "<random nonsense word>" | None,
  "modifier": "<one of the modifier pool>" | None,
  "discourse_link": "<one of the connectives>" | None,
}
```

Successive scenes can share entities and time (sequential narrative)
or be discourse-linked. The generator produces:
- `complex_brain.db` — scenes flattened into substrate edges
  (`subject --[action]--> object`, `event_X --[location]--> ...`,
   `event_X --[time]--> ...`)
- `complex_pairs.jsonl` — (facts, prose) training pairs

### Slice 3 — extended template renderer

New module `src/sara_brain/cortex/transformer/synth_templates_complex.py`.

Templates are functions that take a scene (or pair of scenes for
compound/discourse) and render slotted prose. Sample:

```python
def t_simple(scene):
    return f"{scene.C0} {scene.P0} {scene.C1} ."
def t_temporal(scene):
    return f"{scene.T0} , {scene.C0} {scene.P0} {scene.C1} ."
def t_located(scene):
    return f"{scene.C0} {scene.P0} {scene.C1} at {scene.L0} ."
def t_modified(scene):
    return f"{scene.C0} {scene.M0} {scene.P0} {scene.C1} ."
def t_compound(s1, s2):
    return f"{s1.C0} {s1.P0} {s1.C1} and {s2.P1} {s2.C2} ."
def t_complex_because(s1, s2):
    return f"{s1.C0} {s1.P0} {s1.C1} because {s2.C2} {s2.P1} {s2.C3} ."
def t_discourse(s1, s2):
    return f"{s1.C0} {s1.P0} {s1.C1} . {s2.R0} , {s2.C2} {s2.P1} {s2.C3} ."
```

For each scene/pair, the generator picks 1-3 templates uniformly so
the same substrate produces multiple legitimate prose forms (built-
in augmentation).

### Slice 4 — train the new ckpt

Reuse `train_synth.py` unchanged. The trainer is corpus-agnostic; it
consumes whatever tokenized JSONL `synth_data.write_serialized_jsonl`
produces. Build script:

```bash
./scripts/build_complex_corpus.sh   # generate substrate + pairs + serialize
PAIRS=/tmp/synth_pairs_complex.jsonl \
CKPT_NAME=hamroby_sum_en_complex \
STEPS=4000 \
SESSION=sara-synth-complex \
RESUME_FROM=src/sara_brain/cortex/checkpoints/hamroby_sum_en_002500.pt \
./scripts/train_hamrobysum.sh
```

Results in `hamroby_sum_en_complex_004000.pt`. Vocab grows from
v040's ~520 tokens to ~580 (new slot types only — no new content
words). The `--resume-from` path uses `project_base_into_synth`
(v040 fix) so the new tokens random-init while existing weights
copy through.

### Slice 5 — chat REPL integration

`chat.py` accepts both ckpts:
- `--hamrobysum-ckpt path/to/v040.pt` — simple prose, no complex grammar
- `--hamrobysum-ckpt path/to/v048.pt` — complex grammar

For v048 ckpts, `inference_synth.synthesize_cluster` learns to
expand the new slot types: `<T>` / `<L>` / `<M>` / `<E>` / `<R>` get
filled from the cluster's events / locations / modifiers if
present, otherwise fall back to omitting that clause.

If the substrate has no temporal/spatial annotations (the current
science demo brain), v048's output collapses gracefully to v040-
style simple prose. The complex grammar lights up only when the
substrate carries the structure to support it — narrative brains,
v047-event-rich substrates.

## What stays unchanged

- The synth model architecture (same GrammarModel).
- Training procedure (same `train_synth.py`, same trainer config).
- v040 EN ckpt — preserved as the small-prose fallback.
- The slot expansion mechanism — `<C>` / `<P>` work identically;
  `<T>` / `<L>` / `<M>` / `<E>` / `<R>` use the same expansion
  pattern.
- Honesty guarantees. New slot types still come from substrate;
  the model can't invent times, locations, or modifiers — it can
  only render what's there.

## Files

**New:**
- `docs/v048_complex_grammar_training.md` — this plan.
- `papers/instrument_validation/generate_complex_substrate.py` —
  scene-based random substrate generator. ~250 lines.
- `src/sara_brain/cortex/transformer/synth_templates_complex.py` —
  the extended template renderer. ~150 lines.
- `scripts/build_complex_corpus.sh` — orchestrates substrate
  generation, pair extraction, tokenized serialization. ~30 lines.

**Modified:**
- `src/sara_brain/cortex/transformer/vocab_synth.py` — append new
  slot type tuples. Existing tokens unchanged.
- `src/sara_brain/cortex/transformer/inference_synth.py` — extend
  `_expand_slots` to handle the new slot types alongside `<C>` /
  `<P>`. Backwards-compatible.

**Reused unchanged:**
- `train_synth.py` — corpus-agnostic.
- `synth_data.serialize_example` — already handles arbitrary slot
  tokens, only needs the per-example mapping to include the new
  slot types.
- `model.py` (GrammarModel) — vocabulary size grows but the model
  doesn't care about token semantics.

## Order of operations

1. Save plan + commit.
2. **Slice 1:** extend `vocab_synth.py` — append new slot types.
   Single commit. Verify import passes the dedup assertions.
3. **Slice 2:** write `generate_complex_substrate.py`. Run it once
   to confirm output shape. Commit.
4. **Slice 3:** write `synth_templates_complex.py`. Render a few
   scenes by hand to confirm template output looks right. Commit.
5. **Slice 4:** wire `build_complex_corpus.sh`. Run it locally
   (CPU, quick) to produce `/tmp/synth_pairs_complex.jsonl`. Commit.
6. **Slice 4 (training):** print the train command for the user
   to paste into tmux. ~15 min on the 3070.
7. **Slice 5:** extend inference to handle new slot types. Smoke
   test in chat REPL with the new ckpt. Commit.

## Verification (when implemented)

End-to-end pass criteria:

1. Vocab assertions still pass (no duplicate slot tokens; no overlap
   with vocab_en).
2. The complex substrate generator produces a brain.db + pairs.jsonl
   where pairs include compound, temporal, modified, and discourse-
   connected sentences — visible by sampling 50 random pairs.
3. The serialized JSONL has rows that include the new slot tokens
   (`<T0>`, `<L0>`, etc.). UNK count stays low.
4. Training completes in ~15 min, loss curve shows learning (drops
   below 1.0 by step 2000).
5. The new ckpt loads cleanly in `inference_synth.load_synth_checkpoint`.
6. Sampling on a complex-substrate cluster produces a multi-clause
   sentence with at least one temporal/spatial/modifier slot
   correctly expanded.
7. Sampling on the existing science demo brain (no complex slots)
   produces simple prose comparable to v040 — no regression on the
   simple case.

## Out of scope

- Quoted speech / dialogue rendering. Deferred to v049 because it
  needs a recursive prose slot (`<Q0>` containing another sentence)
  and changes how `synthesize_cluster` invokes itself. Worth a
  separate slice.
- Relative clauses. Same reason as above — the slot mechanism is
  recursive and benefits from its own slice.
- Tense / aspect / mood inflection. The model emits the lemma; the
  vocab brain provides the surface form. Tense agreement across
  clauses is a future slice.
- Coreference resolution in generation. ("Alice walked in. She sat
  down.") The model would emit `<C0>` twice rather than emitting a
  pronoun. Could become v050.
- Cross-lingual extension. v048 stays English-only; building a
  Spanish/Mandarin overlay against the same complex-grammar
  templates is a separate slice.
- Eval framework. Same gating as v039: defer until v048 is shipping
  visibly better prose than v040 on real substrates.

## Why the existing random word generator is the right tool

The slot architecture is content-agnostic — that property holds
regardless of whether the model emits one clause or five. The
random word generator already produces training-orthogonal
substrates that no LLM has ever seen. Extending it to scenes
(rather than independent triplets) lets us train the model on
complex-grammar PATTERNS without committing to any specific
real-language content.

The model never memorizes time labels or location labels — it
learns that `<T0>` can appear at the start of a sentence followed
by a comma, that `<L0>` can be the object of a prepositional
phrase, that `<R0>` introduces a contrasting clause. Inference
fills these slots from substrate.

This means v048 is **substrate-faithful by construction**, the
same property v040 already had — extended to richer surface forms.

## Status

PLANNED. Implementation begins after this plan commits.
