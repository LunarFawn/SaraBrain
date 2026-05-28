# Sara Brain as Knowledge Weights: Architecture Direction

**Date:** 2026-05-26
**Status:** Design direction (not yet implemented)

---

## The Problem

Current LLMs fuse language competence and factual knowledge into one
set of weights. You can't swap the knowledge without retraining. You
can't correct a fact without fine-tuning. You can't inspect what the
model "knows." The datacenter and H100 GPUs exist to bake knowledge
into weights — an expensive, opaque, uncorrectable process.

## The Thesis

Sara Brain replaces the *factual content* in the weights while the
LLM keeps its *cognitive operations* (parsing, reasoning, inference,
analogy, synthesis). Sara is not a RAG database that informs the LLM —
it is the knowledge layer that **constrains** what the LLM can say.

The LLM contributes: language comprehension, reasoning patterns,
synthesis across concepts, natural language generation.

Sara contributes: the facts, the causal chains, the domain knowledge —
inspectable, correctable, swappable, no retraining.

## Architecture: Multiple Sets of Weights as One System

```
┌─────────────────────────────────────────────────────┐
│                  Unified Intelligence                │
├─────────────────────────────────────────────────────┤
│                                                     │
│  LLM-A (Language Cortex)                            │
│    - Parses questions into concept seeds            │
│    - Understands question structure                 │
│    - Trained for language competence ONLY           │
│         ↕                                           │
│  Sara Brain (Knowledge Weights)                     │
│    - Wavefront propagation returns the              │
│      knowledge neighborhood                         │
│    - The "noise IS the data" — full causal/         │
│      relational context, not just top-k results     │
│    - Multiple brains composable at query time       │
│         ↕                                           │
│  LLM-B (Synthesis Cortex)                           │
│    - Renders Sara's knowledge as natural language   │
│    - Reasons WITHIN the substrate (not beyond it)   │
│    - Constrained: can only say what traces back     │
│      to paths in the substrate                      │
│                                                     │
└─────────────────────────────────────────────────────┘
```

LLM-A and LLM-B may be the same model or different models. The key
insight: they contribute cognitive operations, not knowledge. Sara
contributes knowledge. Together they function as one intelligence
with separable, inspectable, correctable layers.

## What This Buys

- **Swap knowledge without retraining** — change the .db file
- **Swap reasoning style without losing knowledge** — change which LLM synthesizes
- **Run a 3B model with expert-level knowledge** — proven in Pearl 2026b (80% MMLU Bio)
- **Correct a fact in one place** — every LLM in the system immediately uses the correction
- **Multiple Sara brains for multiple domains** — composed at query time
- **Full auditability** — every claim traces to a path with provenance
- **No datacenter required** — laptop + SQLite + small local model

## How Sara Functions as Weights

In a traditional LLM, weights determine what the model "believes."
When you ask "what is X?", weights activate a path through the
network that produces the answer.

For Sara to function as weights:

1. The wavefront output defines the **boundary of what's sayable**.
   The LLM cannot generate claims outside the substrate.

2. The substrate must be **rich enough** that the LLM doesn't need
   to escape to its training weights. Not just relevant triples —
   the full reasoning scaffold: causal chains, part-of relationships,
   context that enables inference WITHIN the substrate.

3. The LLM's training knowledge becomes **inert** — the substrate
   fills the cognitive space so completely that training-derived
   facts are irrelevant. (This is what the 45-facts result proved:
   quality substrate overrides training.)

## Multi-Brain Composition

```
Question: "How does the molecular snare achieve mechanical equilibrium?"

Sara Brain A (RNA aptamer domain):
  → wavefront returns: molecular snare paths, static stem mechanics,
    conformational change, nucleotide forces

Sara Brain B (reasoning patterns):
  → wavefront returns: cause-effect templates, mechanical equilibrium
    conditions, force-balance reasoning patterns

LLM receives BOTH neighborhoods as its world:
  → reasons about the aptamer using the mechanical reasoning patterns
  → produces answer grounded in both substrates
  → no training-weight knowledge needed
```

## Implementation Path

1. ✅ Substrate-aware seed extraction (wavefront works on large brains)
2. **Next:** Make wavefront output richer — full causal neighborhood,
   not just intersection labels. Include the paths themselves (source
   text provenance) so the LLM has the reasoning chain.
3. **Next:** Constrained synthesis — the LLM receives the substrate
   as its ONLY knowledge source. Strict mode that rejects any claim
   not traceable to a path.
4. **Next:** Multi-brain composition — query multiple Sara brains,
   merge their wavefront outputs, hand the combined neighborhood to
   the synthesis LLM.
5. **Future:** Train a purpose-built cortex model — small, language-
   competent, designed to reason over structured substrate input
   rather than recall from weights. The "Sara-native LLM."

## The Sara-Native LLM

The end state: an LLM trained specifically to:
- Parse natural language into concept seeds (input processing)
- Reason over structured triple neighborhoods (not free-associate)
- Generate natural language constrained to substrate content (output)
- Never fall back to training-derived factual claims

This model would be small (language competence doesn't need 70B
parameters), cheap to train (no encyclopedic corpus needed), and
powerful when paired with Sara (any domain, any scale, correctable).

The training data for this model:
- Language competence corpus (grammar, syntax, instruction-following)
- Structured-reasoning examples (given these triples, produce this answer)
- Constraint-following examples (given this substrate, do NOT say X)

NOT in the training data:
- Encyclopedic facts (those go in Sara)
- Domain knowledge (that's the substrate's job)
- Anything that would create training-weight "beliefs"

## Relationship to the Papers

- Pearl 2026a: establishes the cortex-cerebellum architecture
- Pearl 2026b: proves 45 facts > 28k LLM-ingested facts (quality > quantity)
- Instrument paper: proves the substrate is measurable and the LLM's
  behavior against it is observable at per-triple granularity
- This document: the next step — making Sara not just a measurement
  instrument but the operational knowledge layer of a new kind of LLM
