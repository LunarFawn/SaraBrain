# v027 — Synthesizer naturalness plan (A0 → A1)

**Date:** 2026-05-03
**Branch:** `feature/grammar-cortex`
**Builds on:** [v025_hamlinllm_status.md](v025_hamlinllm_status.md),
[v026_hamroby_name.md](v026_hamroby_name.md)

## Problem

Template-rendered prose is choppy and weird. Failing example:

> Inertia in rna has is of tendency to maintain current state. Inertia
> is part of inertia in rna. In is part of inertia in rna. Rna is
> part of inertia in rna.

After the v025-followup fixes (commit `0d6bf8c`) the worst failures
("has is of", stop-word `in` as subject) are gone, but the residual
output is still:

> Inertia in rna is tendency to maintain current state. Inertia is
> part of inertia in rna. Rna is part of inertia in rna.

Three remaining issues:
1. **Decomposition leakage.** `Inertia is part of inertia in rna` and
   `Rna is part of inertia in rna` are tautological — substrate
   ingestion creates `is_part_of` from every constituent token of a
   multi-word label, and content-word constituents (`inertia`, `rna`)
   slip past the stop-word filter.
2. **Choppy single-edge sentences.** Multiple edges about the same
   subject render as separate one-clause sentences.
3. **Missing articles / lexical monotony.** "is tendency" not "is a
   tendency". Every assertion uses the same template phrasing —
   templates by construction cannot vary surface form.

Critically, **the labeler (`synth_data.py`) uses the same templates**
to generate the (edge_list, prose) pairs the future neural synthesizer
head will train on. Training a head before fixing templates bakes the
choppiness into the model.

## Plan

### Phase A0 — template & labeler hygiene (ships in hours)

Two waves so output can be reviewed mid-stream.

**Wave 1 — high-signal structural fixes.** Most of the perceived
"choppiness" is probably 1 + 2; fix those first and see what's left.

- **Decomposition filter.** Drop `part_of` edges where `src` is a
  single-token content word that appears as a token in `tgt`. Generalizes
  the stop-word filter to content-word decomposition.
- **Sentence combining.** Cluster rendered sentences by their leading
  subject (split off via a small whitelisted-verb tokenizer); within
  each cluster, join predicates with commas and Oxford "and". Sentences
  whose subject can't be cleanly extracted stay standalone.
- **Sync labeler noise filter.** `synth_data._NOISE_RELATIONS_FOR_LABELER`
  blanket-drops `part_of`; remove that since `render_edges` now drops
  only the decomposition cases. Useful `part_of` ("RNA is part of cell")
  flows through to both inference and training labels.

Expected effect on the running example:

> Inertia in rna is a tendency to maintain current state.

(decomposition edges vanish; only the copula remains → no combining
needed for this case).

For multi-edge clusters:

> Cat is mammal. Cat has whiskers.

becomes

> Cat is mammal and has whiskers.

**Wave 2 — article heuristic.** Insert "a" / "an" before singular
indefinite count nouns in `is` / `is_a` templates. Skip when target
already has a determiner, looks plural (heuristic), or is in a small
mass-noun shortlist (`inertia`, `information`, `water`, ...).
Vowel-onset → "an", otherwise "a". Will mis-fire on edge cases (proper
nouns indistinguishable from common in lowercase substrate); ship,
review on real questions, expand the shortlist.

### Phase A1 — neural synthesizer head

After A0 ships and the labeler emits cleaner pairs:

- Architecture: frozen grammar LM encoder + small generative head
  (same pattern as `router_head.py`).
- Data expansion to address 634-pair thinness:
  - vary edge-subset sub-sampling (labeler already partially does this)
  - run the labeler over additional brains beyond `aptamer_full`
  - per-cluster paraphrase prompts so each cluster contributes 2–3
    surface forms
- Training: bf16, cosine LR, dev-loss eval, resume — mirrors
  `train_router.py`.
- Replaces template renderer at runtime; templates stay as labeler.

The head adds **variety within structure** — same cluster sometimes
renders as `"A cat is a mammal and has whiskers."`, sometimes
`"Cats, which are mammals, have whiskers."`. Templates can't do this.

## Order of operations

1. Save this plan (this commit).
2. Implement Wave 1, commit, look at output on the failing query and
   a few others.
3. Decide whether Wave 2 is needed before A1, or whether A1's neural
   head subsumes the article problem.
4. Begin A1 only after A0 output is judged "good enough as training
   labels."
