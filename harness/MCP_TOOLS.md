# Sara's MCP Tool Surface — When to Use Which

The sara-brain MCP server exposes multiple retrieval and teach tools. Reader tool-discipline determines measurement validity: the same substrate produces different answers depending on which tool the model reaches for. This document explains each tool's behavior and when to use it, so readers don't accidentally under-retrieve the substrate.

## The retrieval-tier finding (2026-04-24)

In live testing on the JKD substrate (106 triples covering Chapter 3 of paper2), the same question produced four different outcomes depending on reader discipline:

| Tier | Behavior | Content surfaced |
|---|---|---|
| 0 | No retrieval — answer from training | 0% of substrate |
| 1 | `brain_why` on exact seed label only | ~9% — direct attributes of seed |
| 2 | Multiple `brain_why` / `brain_trace` calls on adjacent neurons | ~70% — rich neighborhood, requires the reader to guess which adjacent labels matter |
| 3 | Single `brain_explore(label, depth=2)` call | ~70% — same rich neighborhood, in one call |

Tier 3 exists because of the tool added in response to this finding. Prior to `brain_explore`, reaching Tier 3 required the reader to do Tier 2's manual traversal. Most readers do not do this and default to Tier 0 or Tier 1, which silently under-measures substrate fidelity.

**For all substantive "what is X?" questions about a taught topic, start with `brain_explore`.**

## Tool reference

### Primary — use first

#### `brain_explore(label, depth=2)`

Breadth-first walk outward from `label` through `depth` semantic hops, returning a structured summary of every neuron and edge in the neighborhood. This is the broadest single-call tool and should be the first move on any substantive retrieval question.

**When to use:** any "what is X?", "what does Sara know about X?", "explain X in detail" question.

**How depth is counted:** one semantic hop corresponds to one taught triple's distance. Internally the BFS walks two segment-hops per unit of depth because each taught triple produces a two-segment chain through an internal attribute-bridge neuron. Users should think in semantic hops and not worry about the bridge structure.

**Recommended depth values:**
- `depth=1` — only concepts directly connected by a taught triple to the seed. Useful for narrow lookups.
- `depth=2` (default) — one more hop out; recovers most of what a substrate "says about" a topic.
- `depth=3` — two more hops out; expands into tangentially related content. Use sparingly because it can return hundreds of edges on hub-like seeds.

**Case handling:** labels are normalized to lowercase before resolution (matches the `teach_triple` convention). Prefix `CAPITAL:` only if the label was taught with that prefix.

**Output format:** a summary of neurons grouped by hop distance, then a list of edges organized by discovery depth. Internal `*_attribute` chain nodes are visible in the edge listing but should be ignored when reading semantic content; the `source --[relation]--> target` triple is the unit of meaning.

**Safety:** `max_edges=500` (default, non-tunable from MCP) prevents explosion on hub-like seeds. If the walk hits the cap, the response includes `TRUNCATED` and stops adding edges. Re-query with a more specific seed if you hit this.

#### `brain_query(label)`

One-hop in both directions. Equivalent to calling `brain_why` and `brain_trace` together. Use when `brain_explore` is too broad (e.g., the seed is a hub and depth=2 returns hundreds of edges you don't need).

### Single-direction — use when direction matters

#### `brain_why(label)`

Returns only paths **terminating at** `label`. Use when you want to know "what does Sara say about X as a target / property / object of claims?" Will return nothing for neurons that only appear as claim subjects — see `brain_trace` for those.

Known failure mode documented in `model_infections_draft_v1.md` Case 2.5: a reader who calls only `brain_why` on a label that's a path source gets a null result and may conclude Sara doesn't have the content. Use `brain_explore` or `brain_query` to avoid this.

#### `brain_trace(label)`

Returns only paths **originating from** `label`. Complement to `brain_why`. Use for subjects of taught claims ("life is_a circular staircase" shows up when you trace from `life` or from `circular staircase`; it doesn't show up as a brain_why on `circular staircase`).

### Multi-seed — use for recognition

#### `brain_recognize(inputs)`

Comma-separated list of seed labels. Launches parallel wavefronts from each and returns neurons where the wavefronts converge. Use for "given these properties, what concept is Sara recognizing?" not for "tell me about X."

Example: `brain_recognize("red, round, sweet")` — wavefronts from the three seeds converge on concepts they share, surfacing the intended object.

### Disambiguation — use when the label is uncertain

#### `brain_did_you_mean(term)`

Fuzzy-match term against all neuron labels. Returns nearest matches with descriptions. Use when a `brain_explore` / `brain_query` returned nothing and you suspect a typo or a slightly different phrasing of the concept.

### Similarity — use for sibling discovery

#### `brain_similar(label)`

Finds neurons that share downstream paths with the given label. Use for "what's similar to X in Sara's knowledge?" — e.g., finding other concepts with similar properties.

### Teach — use when directed by the user to add content

#### `brain_teach_triple(subject, relation, obj, source=None)`

Canonical teach path. Labels lowercased by default; prefix with `CAPITAL:` to preserve case. Use only when the user explicitly directs a teach — see `feedback_claude_as_teacher_surrogate.md` in the user's auto-memory for the authorization contract.

#### `brain_teach(statement)`

Parser-based teach. Fragile on technical prose — prefer `brain_teach_triple` unless the user specifically wants parsed ingestion.

#### `brain_refute(statement)`

Mark a claim as known-to-be-false. Sara never deletes — the original path is preserved with negative strength plus a refutation marker. Use when correcting a hallucination or recording a debunked claim.

### Inspection — use for meta-questions

#### `brain_stats`

Returns neuron/segment/path counts. Use to verify which brain is loaded.

#### `brain_ingest(source)`

Ingest a document via the agent bridge. Advanced. Prefer brain_teach_triple for substantive teaching; use brain_ingest only when the user explicitly asks for bulk ingestion.

## Suggested retrieval playbooks

### "What is X?" for a substantive topic

1. `brain_explore(X, depth=2)` — single call, broad context.
2. If truncated or too broad: `brain_explore(X, depth=1)` instead, then selectively `brain_explore` on interesting adjacent concepts.
3. Synthesize the answer using ONLY what was returned. Do not interpolate from training.

### "What is X?" when X looks like a concept but resolves to nothing

1. `brain_did_you_mean(X)` — fuzzy-match candidates.
2. If a candidate matches your intent: `brain_explore(candidate_label)` to retrieve.
3. If none match: tell the user Sara doesn't hold this concept, rather than inventing.

### "What is the relationship between X and Y?"

1. `brain_explore(X, depth=2)` — see X's neighborhood.
2. `brain_explore(Y, depth=2)` — see Y's neighborhood.
3. Identify overlap in neurons reached.
4. If no overlap at depth 2: try depth=3, or report honestly that Sara does not connect them directly.

### "Identify the concept from these properties"

1. `brain_recognize(p1, p2, p3)` — wavefront convergence.
2. If multiple candidates: `brain_explore` on each to compare coverage.

### "Tell me about the author's specific framing of X" (clarification-paper retrieval)

Per the Row 5 finding in `model_infections_draft_v1.md`: when the question is about a clarification the author has made on a received concept (e.g., Jennifer Pearl's Creed 2 is a clarification of Bruce Lee's Creed 1), training alone may produce the underlying philosophy. The author's specific refinement lives in Sara.

1. `brain_explore(X, depth=2)` — gets the author's framing, examples, attributions.
2. Cite the taught triples in the synthesis.
3. Explicitly distinguish the author's specifics from the underlying received concept, if that distinction appears in the substrate.

## Antipatterns to avoid

- **Calling only `brain_why` on the exact question phrase.** Misses 90% of substrate; hits direction asymmetry (Case 2.5).
- **Not calling any retrieval tool for a "what is X?" question.** Defaults to training-recall, potentially confident confabulation (Case 2.4 Row 2). The fact that you *could* retrieve doesn't help the measurement if you don't.
- **Using `brain_teach_triple` without explicit user direction.** The LLM is authorized as teacher-surrogate, but only per-fact with user approval — not unilaterally.
- **Inventing connective tissue between retrieved facts.** Inference grounded in the graph is fine; invention labeled as retrieval is contamination.

## What this document is not

- Not a user-facing tutorial. For human operators, see `PROTOCOL.md`.
- Not a spec for the MCP protocol itself. See [Model Context Protocol](https://modelcontextprotocol.io) for that.
- Not exhaustive — niche tools (`brain_scan_pollution`, `brain_list_*_candidates`, etc.) are documented in their own tool descriptions within the server code.
