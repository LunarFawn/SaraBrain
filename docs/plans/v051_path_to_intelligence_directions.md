# v051 — path-to-intelligence directions (architectural exploration, not committed)

**Date:** 2026-05-08
**Branch:** `feature/grammar-cortex`
**Builds on:** Pearl 2026a (foundational paper) — particularly §3.5
(parallel wavefront recognition) and §3.7 (innate primitive layer);
[v050_two_layer_realignment.md](../v050_two_layer_realignment.md)
which clarified the cortex/hippocampus split.
**Status:** EXPLORATORY — directions captured for later evaluation,
no slice committed yet.

## Context

The user named an architectural intuition worth recording: in their
original framing, Sara Brain was a *path to intelligence*. Recent
work (v047 reified events, v049 reified functions, planned v050 style
guides) has been substrate-extension — adding more domains for
**knowledge storage**. The codebase is starting to feel more like a
knowledge store than a cognitive architecture.

This document records the directions that would push Sara back
toward the path-to-intelligence framing — exercising primitives the
foundational paper claims as load-bearing but the codebase has
under-exercised. Captured here so we don't lose the thread; not
committed as a plan slice yet.

## The two halves of Pearl 2026a

The foundational paper makes two arguments interleaved:

**Argument 1 (knowledge store half):** "the cerebellum stores facts,
the cortex translates them into output." Substrate-bound retrieval.
This is what v047/v049 has been doing.

**Argument 2 (path-to-intelligence half):** "a thought is a path
through recorded knowledge, and recognition is the convergence of
independent paths from simultaneous observations." This is the
*cognitive* claim — Sara *thinks* by walking paths, not just stores
facts.

Recent work has barely exercised argument 2. Wavefront recognition
exists in `brain_recognize` but the user-facing workflow has been
"router → tool retrieval → output." That's a database with a tool
wrapper. The recognition primitive — multiple independent
observations converging at a concept = identification — is still
present in the substrate but not surfaced.

## What "path to intelligence" actually means in Sara's terms

The foundational paper names specific primitives that distinguish a
cognitive architecture from a knowledge graph:

1. **Recognition via wavefront convergence** — Sara identifies a
   concept when multiple independent paths from observations meet at
   it. Bidirectional retrieval composed into inference.

2. **Path-of-thought** — a thought is a *chain* of segments, recorded
   as a Path object with provenance.

3. **Hebbian strengthening on co-activation** — `strength = 1 + ln(1
   + traversals)` already exists. Frequently-walked paths become
   more salient.

4. **Concept-specific relation neurons** — structural innovation
   that prevents cross-concept contamination during wavefront
   propagation. Already shipped, mostly invisible to the query
   layer.

5. **Hardwired innate layer** (SENSORY / STRUCTURAL / RELATIONAL /
   ETHICAL primitives) — survives database reset, enforces
   behaviour at the API level. Currently enforcement-only.

6. **Refutation as counter-path, not deletion** — Sara never
   forgets. Refutations are first-class. Currently treated as
   scrubbing rather than preserved disagreement.

7. **Provenance traceability** — every path stores its source text.
   Currently used for debugging; could become a "show your work"
   surface.

## Concrete candidate directions (not committed; evaluate later)

Each is a real slice that exercises primitives the paper claims as
load-bearing but the codebase has under-exercised.

### A. Recognition-first query mode

Instead of "tell me about X," the user gives Sara properties:
"I have something with these properties — what is it?" Wavefront
propagation from each property; convergence count = confidence. The
paper has this; it's barely surfaced. A `brain_recognize_from_properties`
flow with structured input would expose Sara's actual recognition
primitive.

### B. Path composition reasoning

Given two concepts X and Y, what *paths* connect them through the
substrate? Multi-hop, but unlike v045's BFS-for-traversal, it's "show
me the chain of reasoning that connects these." Output is paths, not
edges. Different shape, different purpose.

### C. Conflict surfacing

When the substrate has contradictions (X is_a Y and X is_a not-Y),
surface them as first-class structures. Currently they sit as opposing
edges. A `brain_conflicts` tool returns "these claims about X
disagree, here are the counter-paths." This is reasoning Sara *could*
do but doesn't.

### D. Curiosity / gap detection

Sara has a graph. The graph has dangling edges (concepts referenced
as objects but never as subjects), one-shot concepts (referenced
once, never elaborated), high-degree nodes (well-developed) versus
low-degree (under-developed). A `brain_what_do_i_not_know(topic)`
tool surfaces these gaps. Meta-reasoning about the substrate's own
shape — "where are my blind spots."

### E. Strengthen-on-traversal as a reasoning signal

`strength` already updates on traversal. Could expose "what concepts
have been heavily traversed lately" as a "Sara's current attention"
surface. The substrate equivalent of working memory — what's hot.

### F. Recognition as a teaching primitive

Currently teaching is `(subject, relation, object)`. Sara could be
taught from *patterns*: "things with properties P1, P2, P3 are
usually concept C." Then recognition becomes inferential — Sara not
only recognizes but *generalizes*. Closer to the path-of-thought
paper's original vision than what we've built.

### G. Innate-layer reasoning surface

SENSORY / STRUCTURAL / RELATIONAL / ETHICAL primitives are hardwired
but invisible. Expose `brain_ethics_check(action)` — would teaching
X violate ETHICAL primitives? `brain_structural_check(claim)` — does
this claim violate STRUCTURAL ones? Sara becomes a *constrained
reasoner* rather than just storage.

## Recommended ordering (if we commit to this path)

For maximum return on the cognitive-architecture argument:

1. **A (recognition-first mode)** — exposes the primitive that's
   already there but hidden; pure UX win on existing machinery.
2. **D (gap detection)** — meta-reasoning about substrate shape;
   novel and easy. Demonstrates the substrate isn't just storage but
   *self-aware structure*.
3. **C (conflict surfacing)** — uses the refutation-as-counter-path
   machinery, makes disagreement first-class.
4. **B (path composition reasoning)** — explicit "show your work"
   surface; pairs naturally with the rev8 measurement protocol.
5. The rest later (E, F, G).

These all have the property that they exercise primitives the
foundational paper claims as load-bearing but the codebase has
under-exercised.

## What would need to be true to commit a slice

This document is exploratory. Before committing v051 as an
implementation slice, the following should be decided:

1. **Which direction (A–G)** is the first one to ship. The user's
   intuition / what feels most central comes first.
2. **What the success criterion is.** "Sara now exposes recognition"
   needs a measurable test — a question that's answerable via the
   recognition primitive but not via plain retrieval.
3. **Whether this is a v051 slice or a separate companion paper
   direction.** Some of these (especially E and F) start to look
   like research questions, not engineering slices.
4. **How it interacts with rev8 measurement protocol.** The new
   primitives should also be measurable via the audit log /
   3-session protocol; if they're not, they're invisible to the
   instrument.

## Notes for future-self

- The user said "i'm starting to see it more as a knowledge store"
  — that's the architectural drift this document addresses.
- The user said "i would like to approach this path a bit more"
  — that's a soft commitment to evaluate, not a hard commitment to
  implement.
- The papers' Argument 2 is about cognition, not storage. We've
  honored Argument 1 thoroughly; Argument 2 is the underbuilt half.
- Each of A–G is independent. Pick one without committing to
  the rest.
- §3.5 (parallel wavefront recognition) and §3.7 (innate primitive
  layer) of Pearl 2026a are the load-bearing references.

## Status

EXPLORATORY. Captured for evaluation. Not yet a plan to execute.
