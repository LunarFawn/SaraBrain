# v044 — long-cluster handling: same-subject combining at inference

**Date:** 2026-05-04
**Branch:** `feature/grammar-cortex`
**Builds on:** [v040_predicate_slots_and_vocab_brain.md](v040_predicate_slots_and_vocab_brain.md),
[v043_finish_vocab_management.md](v043_finish_vocab_management.md)

## Context

v040+v043 ship a working slot-based synthesizer, but long clusters
produce choppy output. The known case is the `more` cluster on
`/tmp/sara_demo.db` — 10 same-subject `part_of` edges that render
as 10 short, near-identical sentences:

> More is part of more stable configuration when cooled. More is part
> of more inertia. More is part of more force for same acceleration.
> ... (7 more)

The v035 design was "one sentence per edge, no combining." That made
training-data generation simple but produces low-quality output for
clusters of 5+ same-subject edges.

v044 restores the v027-era same-subject combining (Oxford-comma
clauses) but does it **at inference time on slot-format prose**, not
in the training-data generator. This means **no model retraining
required** — the existing v040 EN checkpoint (`hamroby_sum_en_002500.pt`)
keeps working unchanged; we just post-process its output.

## Why inference-time combining (not retraining)

Two reasons:

1. **No model change.** Retraining for combined prose would mean
   regenerating the corpus with combined prose targets, retraining
   EN, deprecating the current ckpt. Inference-time combining
   preserves the model + just adds a step.
2. **The model's per-edge sentence emission is fine.** It's
   semantically right — one edge, one sentence. The cosmetic
   problem is "10 short same-subject sentences read awful in human
   prose." That's a presentation problem, not a model problem. Solve
   it where the cosmetics matter: at the boundary between model
   output and human consumption.

## The combining algorithm

Operates on **decoded slot-format prose tokens BEFORE slot
expansion**, so the subject identification is unambiguous (subjects
are slot tokens like `<C0>`, not arbitrary substrate strings).

Input: prose tokens like
```
<C0> <P0> <C1> . <C0> <P1> <C2> . <C0> <P2> <C3> .
```

Output:
```
<C0> <P0> <C1> , <P1> <C2> , and <P2> <C3> .
```

After slot expansion + detokenize + article fix:
```
Multicellular organism produces cell division, is an organism, and is a sea urchin.
```

Steps:

1. Split prose tokens into sentences at `.` (period token).
2. For each sentence, identify the subject: the contiguous prefix of
   tokens up to but not including the first `<Pn>` slot token.
3. Group **adjacent** same-subject sentences (preserve overall
   order — never move sentences across non-adjacent positions; this
   protects the model's emitted ordering, which encodes attribute
   flags and topic priority).
4. For each group with ≥2 sentences:
   - Strip the subject from each sentence beyond the first.
   - Join the resulting predicates with `, ` separators and `, and `
     before the last (Oxford comma).
   - Emit `subject + " " + joined_predicates + "."`
5. 1-sentence groups: emit unchanged.

Edge cases:
- Sentence with no `<Pn>` (defensive — shouldn't happen): treat as
  standalone, no combining.
- 2 sentences in a group: `"A and B"` (no Oxford comma, just `and`).
- Empty subject: standalone (defensive).
- Sentence with `</prose>` or `<eos>` mid-decode: should already be
  filtered by the existing structural-id strip; if not, treat as
  end of decoding.

## What changes

**Modified:** `src/sara_brain/cortex/transformer/inference_synth.py`

Add `_combine_same_subject_slotted(prose_tokens)` that operates on
slot-format prose tokens and returns combined slot-format prose
tokens. Wire it into `synthesize_cluster` between the structural-id
strip and the predicate-slot expansion:

```
prose_tokens = strip structural ids
prose_tokens = _combine_same_subject_slotted(prose_tokens)   # NEW
expanded     = _expand_pred_slots(prose_tokens, ...)
expanded     = _expand_slots(expanded, slot_mapping)
text         = _detokenize(expanded)
text         = _fix_articles(text)
```

That's the only change. ~50 lines added to inference_synth.py.

## What stays

- v040 EN checkpoint (`hamroby_sum_en_002500.pt`) — no retraining.
- Training-data generator (`render_edges_slotted`) — still emits one
  sentence per edge. The combining is purely an inference cosmetic.
- v032 templates fallback path (in chat.py for degenerate clusters)
  — already uses the v027 combining via `render_edges`.
- Article post-processor (`_fix_articles`) — still applies as the
  last step.

## Verification

End-to-end:

1. **Demo `5'` cluster** (3 same-subject `part_of` edges):
   - before v044: `5' is part of 5' and 3' ends. 5' is part of 5' end of rna. 5' is part of sufficient axial forces joining 5' and 3' ends.`
   - after v044: `5' is part of 5' and 3' ends, is part of 5' end of rna, and is part of sufficient axial forces joining 5' and 3' ends.`

2. **Demo `more` cluster** (10 same-subject `part_of` edges):
   - before: 10 staccato sentences
   - after: one sentence with 10 Oxford-comma clauses

3. **Mixed-subject cluster** (different subjects across edges):
   - before: each edge as its own sentence
   - after: same-subject runs combine; different subjects stay separate. Ordering preserved.

4. **Single-edge cluster**: unchanged (1 sentence, no combining).

5. **drifted_s1 `multicellular organism`** (4 edges, 2 distinct
   subjects via attr-flag flipping):
   - before: 4 sentences
   - after: same-subject runs combine, different subjects stay
     separate, end-to-end output still all-correct verbs and
     articles.

Inline tests: I'll run the demo brain through inference both with
and without the combining, side-by-side compare. Commit the findings
inline.

## Out of scope

- Topic-based subcluster splitting (was option 3 in the chat). Not
  needed if combining alone produces clean output for long clusters.
- Truncation with footer (was option 1). The combining handles long
  clusters by clause-stacking; users who want compactness can
  truncate at substrate-query time, not at synthesis.
- Combining ACROSS non-adjacent same-subject sentences. We only
  combine adjacent runs to preserve the model's emitted order. If
  the model emits `A, B, A` we keep that order: `A`, `B`, `A`
  (three groups of 1) — not `A, A, B` (which would discard the
  ordering).
- Retraining EN with combined prose targets. v044 is purely an
  inference-time fix; the model stays as v040 trained it.
- Multi-hop reasoning (option B). Sequenced after this slice — once
  long clusters render naturally, multi-hop output (which produces
  long clusters) renders naturally too.
