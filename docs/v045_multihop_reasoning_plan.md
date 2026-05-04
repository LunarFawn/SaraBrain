# v045 — multi-hop reasoning over substrate

**Date:** 2026-05-04
**Branch:** `feature/grammar-cortex`
**Builds on:** [v028_multi_layer_cortex_architecture.md](v028_multi_layer_cortex_architecture.md)
(architecture), [v040_predicate_slots_and_vocab_brain.md](v040_predicate_slots_and_vocab_brain.md)
(slot-bound synthesis), [v044_long_cluster_combining.md](v044_long_cluster_combining.md)
(presentation cleanup that unblocks this).

## Context

Today's flow handles **one-hop** questions cleanly:

> "What is multicellular organism?" → router fetches all edges where
> `multicellular organism` appears as subj/obj → synthesizer renders.

It fails silently on **multi-hop** questions where the answer requires
chaining substrate retrievals:

> "Why does chronic stress impair recovery?"
> Substrate has:
>   `recovery period --[required for]--> protein synthesis`
>   `protein synthesis --[disrupted by]--> chronic stress`
> Today: router picks one node, fetches its edges, misses the chain.
> What we want: traverse the chain, compose all retrieved facts into
> one coherent answer.

v045 extends the substrate-bound principle from "what does the
substrate say about X" to "what does the substrate say across X→Y→Z."
Every claim still traces to a substrate edge — no invented reasoning,
no hallucinated chain steps. The chain logic is deterministic
structural composition over substrate edges; the model's job stays
"render edges as prose."

## What ships in v045

A new module + small wiring change. v0 behavior stays default
(single-hop); `--multihop` opts in.

### `multihop.py` — planner + orchestrator

**`should_multihop(question: str) -> bool`** — heuristic planner.

Triggers multi-hop on question shapes that imply chaining:
- Starts with `why` / `how`
- Contains `because` / `due to` / `caused by` / `results in` /
  `leads to` / `connects to`
- Contains `what does X do` (functional implication)
- Otherwise: false (default to single-hop)

Heuristic for v0; could become a learned classifier later.

**`plan_chain(brain, initial_decision, max_depth=2,
max_edges=50) -> list[dict]`** — chain orchestrator.

1. Execute the initial decision; collect edges.
2. For each edge, identify "follow candidates" — concepts on the
   non-anchor side worth recursing into. Anchor side = whichever
   end matches the original question's topic; the other end is the
   candidate.
3. For each candidate: if depth < max_depth and concept not already
   visited, recursively query (`brain_explore label=concept`).
4. Cycle detection via a visited set (concept labels seen at any
   depth).
5. Edge cap: stop when total accumulated edges ≥ `max_edges`.
6. Return `gathered` in the same shape `_synthesize()` already
   consumes — list of `{call, result}` dicts.

The orchestrator is purely structural. No ML, no learned policies —
just a bounded BFS over substrate edges with a topic anchor.

### Chat REPL integration

In `chat.py`:

- Add `--multihop` CLI flag. Off by default for v0.
- In `_route_and_run`, after `router.route(question)`:
  - If `--multihop` AND `should_multihop(question)`: call
    `plan_chain(...)` and use its `gathered` list
  - Else: existing single-hop behavior
- The downstream `_synthesize(question, gathered)` works unchanged —
  it already iterates all edges in `gathered`. v044's combining
  handles the (now-larger) cluster cleanly.

### Multi-cluster connector

When gathered contains edges from multiple distinct queries (the
multi-hop case), the synthesizer's per-cluster output gets joined
with a structural connector between hops:

```
[hop 1 prose]. Additionally, [hop 2 prose]. Additionally, [hop 3 prose].
```

For v0, "Additionally" is fine (static, structural — not invented
reasoning). Future polish could pick connectors based on the relation
that bridged hops (`because of`, `which`, `further`, etc.).

The connector is added in `_synthesize()` when it detects multiple
gathered entries — only the multi-hop path produces this; single-hop
gathered has one entry and no connector is emitted.

## What stays unchanged

- The model. No retraining. v040 EN ckpt unchanged.
- The substrate. Single-brain at a time; cross-brain is out of scope.
- Single-hop behavior: default off for `--multihop`, all existing
  questions render as before.
- Honesty guarantees. Every claim traces to a substrate edge. The
  chain logic is deterministic; "Additionally" is a connector, not
  invented reasoning. The architecture's "structurally impossible
  to hallucinate" property holds.

## Files

**New:**
- `src/sara_brain/cortex/transformer/multihop.py` — planner +
  orchestrator. ~200 lines.

**Modified:**
- `src/sara_brain/cortex/transformer/chat.py` — `--multihop` flag,
  invocation in `_route_and_run`, connector logic in `_synthesize`.

**Reused:**
- `router.CortexRouter.route` — initial routing decision.
- `sara_reader.tools.execute_tool` — substrate query execution.
- `synthesizer.parse_edges_from_gathered` — already parses gathered
  output into Edges.
- `synth_data.cluster_by_subject` — already groups edges by subject.
- `inference_synth.synthesize_cluster` — renders one cluster.

## Order of operations

1. Save plan + commit (this commit).
2. Implement `multihop.py` (planner + orchestrator). Single commit.
3. Wire `--multihop` flag and connector logic in `chat.py`. Single
   commit.
4. Manual end-to-end test: pick a hand-crafted multi-hop question
   over the demo brain (which has a few "X part_of Y, Y has_property
   Z" chains) and verify the multi-hop output traces all edges
   correctly.
5. Eval slice (optional): formal Q&A pair set with expected
   substrate paths. Defer if time-constrained.

## Verification

End-to-end:

1. **Should-multihop heuristic**: questions starting with "why" /
   "how" / containing "because" trigger multi-hop. Simple "what is
   X" stays single-hop.
2. **Chain orchestrator over demo brain**:
   - Pick a 2-hop chain: e.g. find edge A→B, then B→C in the demo.
   - Question: "Why does A connect to C?" with `--multihop`.
   - Expected: gathered contains edges from BOTH A's and B's
     substrate queries; output reflects both hops.
3. **Cycle detection**: handcraft a query whose chain would cycle
   back to the original concept. Expected: orchestrator stops
   without revisiting.
4. **Edge cap**: handcraft a query where naive expansion would
   exceed 50 edges. Expected: orchestrator stops at 50, output
   reflects truncation.
5. **No-regression on single-hop**: with `--multihop` flag, but a
   "what is X" question, behavior is identical to today's single-hop
   (heuristic correctly routes to single-hop path).

## Out of scope

- Learned planner (replaces heuristic). The heuristic is good enough
  for v0; revisit if it mis-fires often on real usage.
- Reasoning over abstract claims (deduction, syllogisms). Multi-hop
  here is purely substrate-edge traversal — "what edges connect X
  to Y," not "X implies Y."
- Cross-substrate reasoning (multiple brains). One brain at a time.
- Optimal path selection (shortest, most-relevant). v0 does
  bounded BFS; ranking is future work.
- `--multihop` becoming default-on. v045 keeps it opt-in until the
  heuristic + orchestrator are validated against real questions.
- Multi-hop training data for the synthesizer. The model never sees
  multi-hop prose during training; we rely on per-cluster
  composition + the connector to handle the multi-hop case.
- Branching exploration UX (the chain has multiple candidate paths;
  v0 follows ALL viable candidates BFS-style; ranked or top-k
  exploration is future).

## Honest difficulty

This is genuinely a research-flavored slice, not pure engineering.
Subproblems with their own difficulty:

- **When to multi-hop**: heuristic mis-fires both ways. Some "what
  is" questions actually need multi-hop ("what is the cause of X");
  some "why" questions don't ("why is X spelled that way" — just
  needs definition).
- **How deep to go**: depth=2 is a safe default but loses long
  chains; depth=4 fetches noise. No general right answer.
- **Cluster coherence at long fetches**: even with v044 combining,
  multi-hop output of 30 edges across 5 hops becomes a paragraph,
  not a sentence. Readability vs completeness.
- **Eval is hard**: "is this answer correct" requires a held-out
  Q/A set with expected substrate paths. Building that is its own
  slice.

A weekend gets a v0 that works on hand-picked examples. A quality
version that's robust across question shapes is weeks. v045 ships
the v0 — the architectural commitment that multi-hop IS substrate-
bound and can be done substrate-faithfully — and we evolve from
there based on real usage.
