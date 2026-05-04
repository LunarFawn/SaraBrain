# v041 — predicate slots + vocab brain implementation plan

**Date:** 2026-05-04
**Branch:** `feature/grammar-cortex`
**Builds on:** [v040_predicate_slots_and_vocab_brain.md](v040_predicate_slots_and_vocab_brain.md)
(architecture); [v039_finish_synthesis.md](v039_finish_synthesis.md) (the
status this builds from)

## Context

v040 documented the architecture: predicate slots `<P0>`...`<P15>`
parallel to content slots `<C0>`...`<C31>`, with a vocab brain
(`vocab_en.db`) holding relation→English mappings. The model emits
`<C0> <P0> <C1>` instead of `<C0> is a <C1>`; the verb gets resolved
at inference time by a substrate query.

v041 implements that.

## Locked positions (from chat)

| # | Decision |
|---|---|
| 1 | Vocab seeding: **(a) v032 templates only**. Fallback for unknown relations: `relation_name.replace("_", " ")`. |
| 2 | Vocab brain location: `src/sara_brain/cortex/vocab/vocab_en.db`. |
| 3 | Predicate slot count: `N_PRED_SLOTS = 16`. |
| 4 | Slot substitution: at the `synth_data.py` serializer (via a new `render_edges_slotted()` helper in `synthesizer.py` that uses the existing template tables with slot substitution). |
| 5 | No Core retrain. Only EN retrains; Core's verb-agnostic property already covers `<Pn>` slots. |

## Slice breakdown

### A — Build `vocab_en.db` from v032 templates

**New file:** `scripts/build_vocab_brain_en.py`

Walks `synthesizer._TEMPLATES` and `synthesizer._ATTR_TEMPLATES`,
extracts the predicate phrase from each template (the part between
or surrounding `{src}` / `{tgt}`), teaches `(relation, english_form,
phrase)` triples into a fresh brain at
`src/sara_brain/cortex/vocab/vocab_en.db`.

For attribute templates (e.g. `"{tgt} is a {src}"`), encode the
predicate phrase plus an `arg_order` flag indicating tgt-first.
This information is technically already implicit in the prose order
the model learns from training data (via the existing `<attr>`
facts-prefix flag), so the vocab brain only NEEDS to provide the
phrase. Storing arg_order is for inspectability + future tooling.

Schema convention (per v040):

```
neuron:   relation_name    (e.g. "is_a")
neuron:   english_phrase   (e.g. "is a")
segment:  (relation_name, "english_form", english_phrase)
```

Multiple `english_form` segments per relation supported.

### B — Add predicate slot tokens to `vocab_synth.py`

**Modified:** `src/sara_brain/cortex/transformer/vocab_synth.py`

- New constant `N_PRED_SLOTS = 16`
- New tuple `SYNTH_PRED_SLOTS = ("<P0>", ..., "<P15>")`
- Append to `_SYNTH_ADDED` so they fold into `VOCAB_SYNTH`
- New helpers: `pred_slot_token(i)`, `SYNTH_PRED_SLOT_IDS`,
  `SYNTH_PRED_SLOT_ID_SET`
- VOCAB_SIZE_SYNTH grows from 425 → 441 (16 new tokens)

L1, L2-en, and Core checkpoints all stay loadable — slot tokens
just append above their existing IDs (same pattern as v035).

### C — Predicate-slot training data generation

**Modified:** `src/sara_brain/cortex/transformer/synthesizer.py`
and `src/sara_brain/cortex/transformer/synth_data.py`.

In `synthesizer.py`:
- New `render_edges_slotted(edges, content_mapping, pred_mapping,
  topic=None)`: identical structural logic to `render_edges`
  (filter, cluster, combine, capitalize, join), but uses the
  per-relation slot substitution at format time. Where the original
  template is `"{tgt} is a {src}"` and `pred_mapping["is_a"] =
  "<P0>"`, the function emits `"<C1> <P0> <C0>."` (with the slot
  substitutions already in place).

In `synth_data.py`:
- New `build_pred_mapping(ex)`: per-cluster dedup of relations.
  Returns `{relation_name: <Pn>_token}`. Caps at `N_PRED_SLOTS`.
- `serialize_example` builds BOTH `slot_mapping` (content) AND
  `pred_mapping` (predicate), passes both to
  `render_edges_slotted`, encodes the result.
- Each row's serialized output now carries both mappings:
  `{input_ids, loss_mask, slot_mapping, pred_mapping, n_facts, n_prose}`.
- The facts prefix also uses `<Pn>` tokens for predicates instead
  of the current relation-name tokenization.

### D — Inference: load vocab brain, expand `<Pn>` slots

**Modified:** `src/sara_brain/cortex/transformer/inference_synth.py`.

- New `load_vocab_brain(path)` helper: opens the vocab brain and
  builds a `{relation_name: english_phrase}` lookup dict (taking
  the first `english_form` segment per relation; multiple-form
  support is future).
- `synthesize_cluster` gains a `vocab_lookup` param (the dict from
  the helper). At cluster prep time, build `pred_mapping` from the
  cluster's relation names + `vocab_lookup`. Format facts prefix
  with `<Pn>` tokens (using the same substitution helper as
  training).
- After decode, expand `<Pn>` tokens via reverse lookup, then
  expand content slots (existing path), then detokenize, then
  article fix (existing v039 path).
- Fallback: if a relation isn't in `vocab_lookup`, use
  `relation.replace("_", " ")` directly.

CLI: add `--vocab-brain PATH` flag (default
`src/sara_brain/cortex/vocab/vocab_en.db`).

### E — Regenerate corpus + retrain EN (user, GPU)

After A-D land, the user runs:

```
./scripts/build_layered_corpus.sh
PAIRS=/tmp/synth_pairs_en.jsonl CKPT_NAME=hamroby_sum_en STEPS=2500 \
  SESSION=sara-synth-en \
  RESUME_FROM=src/sara_brain/cortex/checkpoints/hamroby_sum_core_002500.pt \
  ./scripts/train_hamrobysum.sh
```

Same command shape as v039. EN retrains on data with predicate
slots; Core stays.

### F — Inference comparison

After retrain, sample on demo + drifted_s1 + a small synthetic
substrate. Check:
- Demo brain: predicates that were UNK or wrong-substituted in v039
  now resolve correctly (the model emits `<P0>`, vocab brain maps it
  to the correct English).
- drifted_s1: cross-brain still works; `is_a` resolves correctly
  again (was regressed by v039's expanded pool).
- A synthetic substrate using a relation NOT in vocab_en.db: should
  fall back to `relation.replace("_", " ")` cleanly.

### G — Doc updates

- v028 status: mark slice-4 architecture as v040 (predicate-bound)
  vs v035 (content-bound only).
- v039 status: note that v039's predicate-pool approach is
  deprecated in favor of v040's slot approach.
- v040 status: implementation done.

## Files

**New:**
- `scripts/build_vocab_brain_en.py`
- `src/sara_brain/cortex/vocab/vocab_en.db` (artifact, gitignored)

**Modified:**
- `src/sara_brain/cortex/transformer/vocab_synth.py`
- `src/sara_brain/cortex/transformer/synthesizer.py`
- `src/sara_brain/cortex/transformer/synth_data.py`
- `src/sara_brain/cortex/transformer/inference_synth.py`

**Reused unchanged:**
- `train_synth.py` — already vocab-size-agnostic; consumes whatever
  VOCAB_SIZE_SYNTH it sees in vocab_synth.
- `train_hamrobysum.sh`, `build_layered_corpus.sh` — same flags.
- `model.py` — embedding grows by 16 rows transparently.

## Order of operations

1. Save v041 plan + commit (this commit).
2. **Slices A-D bundled in one commit** — wiring change. All four
   files at once because nothing works partial.
3. **Slice E** (user, GPU): rebuild corpus + retrain EN in tmux.
   Wait for ping.
4. **Slice F** (CPU): inference comparison; commit findings.
5. **Slice G**: status doc updates; commit.

## Verification

End-to-end pass criteria:

1. **A:** `vocab_en.db` exists; `Brain('vocab_en.db').list_paths()`
   returns at least the relations from v032 templates with their
   English forms.
2. **B:** `VOCAB_SIZE_SYNTH == 441`. v0 (Core) ckpt still loads
   despite the larger vocab (just the projection padding pattern
   from existing slots).
3. **C:** `serialize_example(ex)` returns rows containing
   `<P0>`...`<P15>` tokens in both facts and prose; `pred_mapping`
   field populated with relation→slot lookup.
4. **D:** `inference_synth --ckpt hamroby_sum_en_002500.pt --brain
   /tmp/sara_demo.db` (after E) produces prose where the predicate
   verbs match the substrate's relation names exactly via vocab
   brain expansion.
5. **F:** drifted_s1 cluster `multicellular organism` (4 edges,
   2 distinct relations: `produces` + `is_a`) renders as:
   `Multicellular organism produces cell division, is a organism, is
   an individual, and is a sea urchin.` — both verbs correct, all
   4 edges present, article fix applied.

## Out of scope

- `/teach-vocab` REPL command. Future slice; same machinery as
  `/teach`.
- Multi-language vocab brains. Recipe documented in v040; one
  language ship is enough to validate the architecture.
- Per-domain swap mid-session. Architecture supports it; UX is
  future.
- Stylistic variation via multiple english_forms per relation. Plumbing
  supported (multiple segments per relation), but inference picks the
  first; choosing among them is future.
