# v040 — Predicate slots + vocab brain: pushing verb choice to substrate

**Date:** 2026-05-03
**Branch:** `feature/grammar-cortex`
**Builds on:** [v035_generic_slot_hamrobysum.md](v035_generic_slot_hamrobysum.md),
[v037_layered_synth_architecture.md](v037_layered_synth_architecture.md),
[v039_finish_synthesis.md](v039_finish_synthesis.md)

## Context

v039 finished synthesis: Core (structural) + EN (English-verb
overlay) + article post-processor + chat REPL integration. v039
slice 1c surfaced an architectural ceiling, not a bug: extending
`_RELATIONS_POOL` from 12 to 61 verbs **regressed** verb correctness
on the most common relation (`is_a`) because the model now had 61
grammatically interchangeable options to choose from. More verbs in
the pool dilutes the signal on each.

The user's diagnosis (correct):

> "There are a lot of verbs that are grammatically the same, so what
> if we are just running into that — and this is where having access
> to the brain's vocabulary would be good. Maybe we have a brain
> substrate that is just vocab that the grammar and EN LLM can draw
> from as appropriate based on grammar rules."

The architectural fix is to do for verbs what v035 already did for
content: **stop asking the model to choose, push the choice to
substrate.** The verb is already in the substrate (the relation
name on the edge — `forms` for the `forms` relation). The model
shouldn't be picking; it should be slot-emitting and letting the
substrate fill.

## The framework refinement

The L1/L2/L3 layering from v028/v029 generalizes one notch:

| Layer | Sourced from | Per | Universal? |
|---|---|---|---|
| **L1** grammar | UD treebanks; structural priors (UPOS + UD deps), no words | all users | yes — ships once |
| **L2** function-word overlay | UD + closed-class English literals | language | per language |
| **L3** vocab substrate (**NEW**) | relation → English-phrase mappings | language + domain + user-editable | swappable, inspectable |
| **L4** content substrate | user's facts | user | per user |

Each layer is thinner and more user-substitutable as you go up.
L1 is shared infrastructure that ships once. L2 is shared per
language. L3 is per language AND per user (you can `/teach-vocab`
your own forms). L4 is entirely yours.

The interesting property: **anything the model would otherwise have
to learn gets pushed to the appropriate layer**.

- "What grammatical patterns exist in human language?" → L1
- "What's the function-word vocabulary of English?" → L2
- "What English verb does this substrate's `produces` relation render as?" → L3
- "What facts about the world?" → L4

The model itself only learns what's left after every other thing
has been pushed out: pure structural composition.

This is the v028 thesis generalized:

> Original (v028): knowledge in substrate, not weights.
>
> Refined (v040): any choice that depends on user / language /
> domain belongs in substrate at the appropriate generality. The
> model learns only the universal residual.

## The predicate slot mechanism

Mirrors the v035 content-slot mechanism exactly. Today's
serialization (per v035):

```
<facts>
  <subj> <C0> <pred> is <obj> <C1> <attr> <edge_sep>
  <subj> <C2> <pred> part of <obj> <C3> <edge_sep>
<prose>
<C0> is a <C1> . <C2> is part of <C3> .
</prose>
```

The model emits real English verbs (`is`, `part of`) in the predicate
position. v039 demonstrated this fails when the verb pool is
non-trivially sized.

v040 serialization:

```
<facts>
  <subj> <C0> <pred> <P0> <obj> <C1> <attr> <edge_sep>
  <subj> <C2> <pred> <P1> <obj> <C3> <edge_sep>
<prose>
<C0> <P0> <C1> . <C2> <P1> <C3> .
</prose>
```

Per-example mappings carried alongside (parallel to the existing
`slot_mapping` for content):

```
slot_mapping:    {<C0>: "multicellular organism", <C1>: "sea urchin",
                  <C2>: "ribosome",               <C3>: "cell"}
pred_mapping:    {<P0>: "is_a",  <P1>: "part_of"}      ← relation names
```

At inference, both maps drive expansion:

1. Decode prose → `<C0> <P0> <C1> . <C2> <P1> <C3> .`
2. Expand content slots (existing v035 path).
3. Expand predicate slots by querying the **vocab substrate** for
   each relation name → English phrase.
4. Detokenize + article post-processor (existing v039 path).

Final output:
```
Multicellular organism is a sea urchin. Ribosome is part of cell.
```

The model *cannot* substitute the wrong verb because it doesn't
choose verbs at all. The substrate is the source of truth for both
content AND verbs.

## The vocab substrate (L3)

Sara's existing `Brain` class is the right primitive — vocab is
just another substrate. Schema convention:

| Neuron | Segment relation | Neuron | Means |
|---|---|---|---|
| `produces` | `english_form` | `produces` | the relation's canonical English verb |
| `is_a` | `english_form` | `is a` | (note multi-word phrase) |
| `is_an_instance_of` | `english_form` | `is an instance of` | |
| `forms` | `english_form` | `forms` | |
| `is_a` | `english_form` | `is a kind of` | (multiple forms allowed) |

A vocab brain = a `brain.db` whose neurons are *relation names* and
whose segments map them to English phrases via the `english_form`
relation.

### Default vocab brain

The project ships `vocab_en.db` containing the relation→English
mappings derived from the existing v032 `_TEMPLATES` and
`_ATTR_TEMPLATES` tables. Same data, promoted from code to substrate.

Building it: a one-shot script (`scripts/build_vocab_brain_en.sh`)
walks the v032 templates, extracts the predicate verb portion of
each, teaches the mapping into a fresh brain.db. Reproducible.

### Per-user / per-domain customization

A user can:
- Edit the default `vocab_en.db` directly (same as editing any
  brain.db).
- Maintain a per-project vocab override (`~/myproject/vocab.db`)
  loaded alongside the default (later, overrides win).
- `/teach-vocab consumes "consumes"` from the chat REPL (future
  slice — same machinery as `/teach`, just targeting the vocab
  brain instead of the content brain).

### Per-language overlays

Spanish HamRoby:
1. Generate a Core (already exists, language-agnostic).
2. Generate an EN-equivalent for Spanish: train an `es` overlay
   (small adapter) on Spanish UD treebanks via the v037 recipe.
3. Build `vocab_es.db` with Spanish predicate forms:
   ```
   ("produces", "spanish_form", "produce")
   ("is_a",     "spanish_form", "es un")
   ("part_of",  "spanish_form", "es parte de")
   ```
4. Load `(Core, ES-overlay, vocab_es.db, your_substrate.db)` at
   inference.

The model never changes; only the substrates do.

## What this changes vs what stays

**Stays exactly the same:**
- L1 grammar checkpoint. Already structural-only; doesn't see verbs.
- HamRoby-Sum-Core checkpoint. Already verb-agnostic by construction
  (Core was trained with `--nonsense-relations` so it never had
  verb priors).
- v035 slot architecture, v032 article heuristic, v039 chat
  integration. All compose with v040 naturally.

**Changes:**
- `vocab_synth.py`: add `<P0>...<P31>` predicate slot tokens.
- `synth_data.py`: extend slot substitution to predicates (parallel
  helper to `build_slot_mapping`).
- `inference_synth.py`: load vocab brain at startup, query it during
  predicate expansion.
- HamRoby-Sum-EN: retrains on the new format. Cheap (~10 min on a
  3070).
- New: `scripts/build_vocab_brain_en.sh` to construct `vocab_en.db`
  from the v032 templates.

**Future (not v040):**
- `/teach-vocab` REPL command.
- `vocab_es.db`, `vocab_fr.db`, etc. (recipe documented).
- Multi-vocab loading (default + user override).

## Out of scope for v040 itself

- Actual implementation slicing (a separate plan picks up after
  this doc lands).
- Multi-language vocab brains. The recipe is documented above; one
  language ship is enough to validate.
- Per-domain customization tooling (e.g. swap vocab between
  `medical_en.db` and `legal_en.db` mid-session). The architecture
  supports it; the UX is a future slice.
- Stylistic variation. v040 makes the model emit the SAME English
  phrase for any given relation. Paraphrase variety would require
  vocab brain entries with multiple `english_form` segments and an
  inference-time choice — possible, deferred.

## Honest framing

v040 is the first **architectural** step beyond "v035 generic
slots." v035 said content lives in substrate; v040 says vocabulary
also lives in substrate. The pattern repeats. Each next slice that
moves another piece of model behavior to substrate further
validates the principle that the right architecture is "model =
universal structural composer; everything else = substrate."

Whether this approach scales to ALL of LLM capability remains
unproven (per v038's honest read). What v040 demonstrates is that
the principle generalizes one more notch beyond v035's content
slots — verbs, the next-most-fragile thing in the synth weights,
also belong in substrate. Each successive shed makes the broader
thesis more defensible.

## Why this doc, why now

v039 closed the synthesis layer for the existing architecture. v040
is the architectural decision to extend the architecture before
freezing it. Doing this BEFORE we ship to the chat REPL as a
default means we don't ship a flawed version users have to migrate
off later.

The choice point: v039 + article fix is "good enough to ship." v040
is "good enough to ship AND structurally honest about how synthesis
should be decomposed long-term." Picking v040 is choosing the
architecture over the speed.
