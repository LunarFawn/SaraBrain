# v032 — synthesizer article-insertion (anchored on vocab_en)

**Date:** 2026-05-03
**Branch:** `feature/grammar-cortex`
**Builds on:** [v027_synthesizer_naturalness_plan.md](v027_synthesizer_naturalness_plan.md),
[v029_vocab_en_plan.md](v029_vocab_en_plan.md)

## Context

v027 Wave 1 (commit `9b267bf`) shipped sentence combining and decomp
filtering. Wave 2 (article heuristic) was punted in favor of L2-en
(v028+).

L2-en is now trained (v031, dev_ppl 4.13). The cleanest minimal
"synthesizer + L2-en" integration is to use **`vocab_en`'s
function-word allowlist as the single source of truth** for what
counts as a determiner / when an article is needed. The heuristic
itself stays simple; the determiner set comes from L2-en, so when
that allowlist evolves the synthesizer follows automatically.

This is sub-step 1 of the v028 synthesizer-pipeline integration:
**templates become richer using L2-en's vocabulary.** Sub-step 2 is
the full L2-driven scoring approach (call L2 forward at synthesis
time to score article variants) — bigger lift, deferred.

## What changes

In [src/sara_brain/cortex/transformer/synthesizer.py](../src/sara_brain/cortex/transformer/synthesizer.py):

1. **Pull determiner set from `vocab_en`** so any future allowlist
   change propagates to the renderer with no second edit.
2. **Add `_maybe_article(target: str) -> str`** that returns `"a "`,
   `"an "`, or `""` based on:
   - already starts with a determiner from `vocab_en` → no article
   - in a small mass-noun shortlist (`inertia`, `information`,
     `water`, `gravity`, ...) → no article
   - last word ends in plural `s` (with exceptions) → no article
   - first word is in a small common-adjective shortlist OR has an
     adjective suffix (`-ous`, `-ful`, `-able`, `-ible`) → no article
   - vowel onset → `"an "`
   - otherwise → `"a "`
3. **Apply in `_render_edge`** only for copula-shaped relations where
   the result is `{X} is {Y}` and `{Y}` is the slot that should take
   the article. Non-copula templates untouched.

## What does NOT change

- The sentence combiner, decomposition filter, and cluster ordering
  from v027 Wave 1 stay exactly as they are.
- Templates that already include an article (`is_a`,
  `is_an_instance_of`, `offers_metric`, `is_subsystem_of`) are
  unaffected — `_maybe_article` won't double-insert because the slot
  already starts with `a` / `an` / `the`.
- L2-en model is **not** loaded at synthesis time. We only consult
  the static `ENGLISH_FUNCTION_WORDS` tuple from `vocab_en`. Zero
  runtime dependency on the trained checkpoint.
- Labeler in `synth_data.py` automatically picks up the new behavior
  since it shares `render_edges`.

## Files

**Modified:**
- [src/sara_brain/cortex/transformer/synthesizer.py](../src/sara_brain/cortex/transformer/synthesizer.py)

**Read-only references:**
- [src/sara_brain/cortex/transformer/vocab_en.py](../src/sara_brain/cortex/transformer/vocab_en.py)
  — pulls `ENGLISH_FUNCTION_WORDS` for the determiner subset.

## Verification

Sanity cases:

| Input edge | Before | After |
|---|---|---|
| `(tendency..., is, inertia in rna[attr])` | `Inertia in rna is tendency to maintain current state.` | `Inertia in rna is a tendency to maintain current state.` |
| `(mammal, is, cat[attr])` (uses `is_a`) | `Cat is a mammal.` | `Cat is a mammal.` (unchanged — already had article) |
| `(soft, is, cat[attr])` | `Cat is soft.` | `Cat is soft.` (adjective, no article) |
| `(energy, is, gravity[attr])` | `Gravity is energy.` | `Gravity is energy.` (mass-noun shortlist) |
| `(apple, is, fruit[attr])` | `Fruit is apple.` | `Fruit is an apple.` (vowel-onset → `an`) |
| `(students, is, group[attr])` | `Group is students.` | `Group is students.` (plural-shape) |
| `(the king, is, cat[attr])` | `Cat is the king.` | `Cat is the king.` (already determined) |

Heuristic mis-fires are expected (English is irregular). Mitigation:
ship conservative defaults; expand mass-noun and adjective shortlists
when real usage surfaces an annoying mis-fire.

## Out of scope

- True L2-driven scoring of article variants (call L2 forward at
  synthesis time). Deferred — would require loading the L2 ckpt into
  the synthesizer pipeline + spaCy parse of each candidate. Real
  next-step if heuristic mis-fires too often.
- HamlinSum (path 2) training. Separate slice.
- Per-language synthesis. Spanish synthesizer would need its own
  determiner set (from `vocab_es` when it exists).
