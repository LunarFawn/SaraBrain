# v043 — finish vocab management (`/list-vocab`, `/refute-vocab`, multi-form)

**Date:** 2026-05-04
**Branch:** `feature/grammar-cortex`
**Builds on:** [v040_predicate_slots_and_vocab_brain.md](v040_predicate_slots_and_vocab_brain.md),
[v042_teach_vocab_repl.md](v042_teach_vocab_repl.md)

## Context

v042 shipped `/teach-vocab` but the vocab management trio is still
incomplete: there's no way to *list* what's mapped or to *refute*
(remove) a mapping. v042 also chose REPLACE-on-existing for
simplicity; this slice adds proper multi-form support so a relation
can carry several English phrasings and inference rotates among them
for stylistic variety.

This closes out the v040 vocab story before we start the next
architecturally-significant slice.

## What ships

### `/list-vocab [RELATION]` — inspect

- No arg: print every relation → english_form mapping in the vocab
  brain, alphabetized.
- One arg: print all forms for that one relation.
- Format: `relation_name -> "phrase"` (one form per line; multiple
  forms per relation indented under the relation header).

### `/refute-vocab RELATION [PHRASE]` — remove

- One arg: remove ALL english_form segments for that relation. The
  inverse of having taught it.
- Two args: remove ONLY the specific form. Other forms for that
  relation stay.
- After refute, the relation falls back to
  `relation.replace("_", " ")` if no other forms remain.

### `/teach-vocab RELATION PHRASE...` — add (changed from v042)

v042 behavior was REPLACE: teaching a relation deleted any prior
form. v043 changes the default to ADD: teaching a relation keeps any
prior forms and adds a new alternate.

- If the relation already has the same phrase, no-op (no duplicate).
- To replace, `/refute-vocab RELATION` first then `/teach-vocab`.

This is a behavior change for `/teach-vocab`; v042 users who relied
on replace must add the explicit refute step.

### Inference: rotate forms across emissions

`inference_synth.load_vocab_brain` returns `dict[str, list[str]]`
instead of `dict[str, str]` — every relation may have multiple forms.

`_expand_pred_slots` tracks emission count per slot. The N-th time a
given `<Pn>` token appears in prose, we pick `forms[N % len(forms)]`.
Deterministic: same cluster + same checkpoint always produces the
same expansion. A cluster with 3 same-relation edges and 2 forms for
that relation produces:

- Edge 1: form 0
- Edge 2: form 1
- Edge 3: form 0 (wraps)

Lets the same substrate produce varied phrasing without any model
retraining or non-determinism.

## Files

**Modified:**
- `src/sara_brain/cortex/transformer/chat.py` — three slash commands
  (`/list-vocab`, `/refute-vocab`, modified `/teach-vocab`).
- `src/sara_brain/cortex/transformer/inference_synth.py` —
  `load_vocab_brain` return-type change, `_expand_pred_slots` rotation
  logic.

**Reused:**
- Same SQL pattern as `scripts/build_vocab_brain_en.py` and v042's
  `_do_teach_vocab` (direct SQLite, bypassing `Brain.teach_triple`'s
  chain-learning machinery).

## Verification

End-to-end in the chat REPL with `--use-hamrobysum`:

1. `/list-vocab` — shows the 51 default mappings from `vocab_en.db`.
2. `/teach-vocab is_a "is a kind of"` — adds alternate. `is_a` now
   has two forms.
3. `/list-vocab is_a` — shows both `"is a"` and `"is a kind of"`.
4. Ask "what is X" where X has 4 `is_a` edges. Output rotates:
   `... is a Y. ... is a kind of Z. ... is a W. ... is a kind of V.`
5. `/refute-vocab is_a "is a kind of"` — removes the alternate.
   `/list-vocab is_a` shows only `"is a"`.
6. `/refute-vocab is_a` (no phrase) — removes ALL `is_a` forms.
   Output for that relation falls back to `"is a"` (literal `_`->` `).
7. Restart chat. State persists.

Without `--use-hamrobysum`: all three commands error cleanly.

## Out of scope

- Per-form weighting (some forms preferred over others). For now,
  uniform round-robin.
- Form metadata (formal vs casual register, domain tags). Schema
  could carry it but inference doesn't use it.
- Bulk import / export of vocab mappings (`/dump-vocab`,
  `/load-vocab FILE`). Vocab brain is a SQLite file users can copy
  / share / version-control directly.
