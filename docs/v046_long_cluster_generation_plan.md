# v046 — long-cluster generation quality (planned, deferred)

**Date:** 2026-05-05
**Branch:** `feature/grammar-cortex`
**Builds on:** [v044_long_cluster_combining.md](v044_long_cluster_combining.md),
[v045_multihop_reasoning_plan.md](v045_multihop_reasoning_plan.md)
**Status:** PLANNED, not yet implemented. Captured here so it stays
in the queue.

## Context

v044 fixed long-cluster *presentation* (Oxford-comma combining) but
not the underlying *model emission quality*. On clusters with 20+
edges, the v040 EN model produces broken output:

> Rna stability role in mechanical forces, contribute to mechanical
> forces, and role in contribute to mechanical forces. optimized
> for contribute to cumulative negative axial forces. rna stability
> contribute to cumulative negative axial forces. cumulative
> negative axial forces. ...

The model is repeating predicates (`contribute to mechanical forces,
and role in contribute to mechanical forces`), emitting bare slot
expansions with no verb (`cumulative negative axial forces.`),
and looping on similar tokens. This is degeneration in the
generation itself, not the combiner.

v044's combiner faithfully Oxford-commas whatever the model emits —
when the model emits nonsense, the combined output is
nonsense-with-Oxford-commas.

## Why deferred (not done now)

v045 follow-up tightened multi-hop defaults (max_depth=1, max_extra_edges=15)
which reduces how often this case is hit in practice. And the
brain_value→brain_explore fallback added a separate path that also
exposes this limit. Both reduce blast radius; neither fixes the
root cause.

The fix is a real architectural slice (cluster-size-aware
synthesis), not a one-line tweak. Worth a focused plan, not an
inline patch. v046 is that plan.

## Why the model degenerates on big clusters

Two probable causes:

1. **Training-distribution mismatch.** The synth corpus had cluster
   sizes ranging mostly 1–8 edges. 20+ edge clusters are rare in
   training; the model generalizes badly to them.
2. **Repetition penalty + decoding interact poorly.** With ~58
   edges to render through a fixed `max_new_tokens=80`, the model
   runs out of budget AND the same content slots get hit
   repeatedly. The repetition penalty makes the model avoid recent
   tokens, which can flip it into emitting partial garbage instead
   of the right repeat.

## Three candidate fixes

### (1) Cluster-size-aware splitting at synthesis time (cheap, mechanical)

In `synthesize_cluster`, if the cluster has more than N edges (say
8), split it into chunks and render each chunk separately:

```
chunks = [cluster[i:i+8] for i in range(0, len(cluster), 8)]
chunk_proses = [synthesize_cluster(model, chunk, ...) for chunk in chunks]
return " ".join(chunk_proses)
```

The model never sees a 20+ edge cluster; each render gets a typical-
size cluster. Prose joins with a space (or "; furthermore," etc.).

**Pro:** ~30 min to implement; immediate fix; no retraining.
**Con:** Loses the combining benefit ACROSS chunks (subject combining
only happens within a chunk). Output reads as multiple shorter
sentences instead of one long one. Acceptable trade.

### (2) Targeted retraining on long-cluster-augmented corpus (medium)

Generate synthetic substrates that explicitly produce 15-30 edge
clusters more often, retrain EN on the augmented corpus. Model
learns to handle bigger clusters.

**Pro:** Solves the root cause.
**Con:** Few hours of training time, requires regenerating corpus,
deprecates current EN ckpt.

### (3) Better decoding for big clusters (medium)

For clusters above N edges, use a different decoding strategy:
- Higher `max_new_tokens` (proportional to cluster size)
- Disable repetition penalty (or lower it)
- Beam search instead of greedy
- Stop emitting once each edge has been "covered" once (track which
  slots have been emitted)

**Pro:** No retraining; addresses the decoding-side cause directly.
**Con:** "Coverage tracking" is non-trivial; slot tokens repeat
naturally in valid output too.

## Recommended slice composition

**v046 recommendation: do (1) first, fall back to (2) if needed.**

(1) is mechanical, fast, and likely sufficient. The cluster-size-aware
split is essentially a "synth-side equivalent" of v044's combining:
v044 cleaned presentation, v046 cleans generation by feeding the
model only sizes it can handle.

If after (1) the prose still feels disjointed (because each chunk
renders independently and they don't compose well), escalate to (2)
— retrain on long-cluster-augmented data so single-render works.

## Files (when implemented)

**Modified:** `src/sara_brain/cortex/transformer/inference_synth.py`

Changes to `synthesize_cluster`:
- New param `max_cluster_size: int = 8`. When `len(edges) >
  max_cluster_size`, split into chunks of that size, render each,
  join.
- Each chunk's prose still gets v044 same-subject combining within
  the chunk.
- Joining: `" ".join(chunks)` for v0; could become
  `" Furthermore, ".join(chunks)` if explicit hop-style connectors
  feel needed.

That's the only change for (1). ~50 lines.

## Verification (when implemented)

- Demo brain `rna stability` cluster (58 edges):
  - before v046: looping/broken ~50-line wall of text
  - after v046: 8 sub-renders, each ~one sentence with Oxford-comma
    clauses; total output ~8 sentences, all grammatical, all
    substrate-grounded
- Demo brain `5'3'` cluster (22 edges):
  - before: looping
  - after: 3 sub-renders, each clean
- Single-edge cluster (1 edge):
  - before: 1 sentence
  - after: 1 sentence (no chunking triggered)
- Architecture: every claim still traces to substrate; chunking is
  presentation-layer, doesn't change the substrate-bound guarantee.

## Out of scope for v046 itself

- (2) and (3) above. Captured here for completeness; only do if (1)
  proves insufficient.
- Cross-chunk same-subject combining (combine `<C0>` clauses across
  separate chunks). Adds complexity; defer until we see whether (1)
  alone is good enough.
- Substrate-side cluster pruning (drop edges before they reach the
  synth). Architecturally different layer; not v046's concern.

## Status

PLANNED. No code yet. Slice is queued for whenever the long-cluster
output quality becomes the visible blocker again. With v045's
tightened multi-hop defaults, that may not be soon.
