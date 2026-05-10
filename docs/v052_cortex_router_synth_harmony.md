# v052 — cortex-router + LLM-synth: the working harmony

**Date:** 2026-05-08
**Branch:** `feature/v052-local-ollama-cortex`
**Builds on:** [v050_two_layer_realignment.md](v050_two_layer_realignment.md) (architectural reasoning), [local_cortex_setup.md](local_cortex_setup.md) (v052 user guide).

## What this documents

An empirical finding from running the v052 local-Ollama stack against
the aptamer substrate: **deterministic cortex routing + LLM synthesis
is the only configuration that holds substrate fidelity.** All-LLM
configurations drift, regardless of model size at the local-quant
tier we run on consumer hardware.

This is not a new design — it is the design the foundational papers
already specify, validated under load. Documenting it explicitly so
future work doesn't re-discover it the hard way.

## The two configurations tested

### Config A — works

```
--cortex-router  --synthesis-model qwen2.5:7b-instruct-q4_K_M
```

Routing is performed by [`src/sara_brain/cortex/router.py`](../src/sara_brain/cortex/router.py)
— a deterministic graph walker over Sara's neuron table, with fuzzy
match and safety-grounded label resolution. The 7B Ollama model
receives the retrieved triples and writes prose under strict-Sara
grounding.

Behavior on `what is the fulcrum` against a substrate that contains
fulcrum triples: returns the taught triples (`is_a`, `part_of`,
`described_by`, `provides`, `enables`, `function`) as fluent prose.
No drift, no training-derived speculation.

### Config B — does not work

```
--router-model qwen2.5:7b-instruct-q4_K_M  --synthesis-model qwen2.5:7b-instruct-q4_K_M
```

Routing is delegated to the same 7B model. The model emits JSON tool
calls (`brain_explore`, `brain_define`, `DONE`) by language-model
inference rather than structural lookup.

Observed failure modes on the same query:

- Mangled labels — emits `brain_explore(label="the fulcrum")` instead
  of `"fulcrum"`, returning no match.
- Wrong tool — picks `brain_define` when `brain_explore` is needed,
  or vice versa.
- Premature `DONE` — model "knows" what a fulcrum is from training
  and shortcuts before consulting the substrate.
- Drift to related concepts — after one empty result, re-routes to
  `"lever"`, `"pivot"`, `"mechanical advantage"` and pulls back
  unrelated substrate.
- Hallucinated synthesis — substrate fragments grafted onto generic
  lever-physics knowledge, producing *"if you're looking for X,
  consider Y"* speculation that reads grounded but isn't.

The Anthropic-API-routed configuration was not tested in this run
because the goal of v052 is local, key-free, stateless operation.
Earlier work suggests larger frontier models route more reliably,
but that's a separate viability question; the local stack must work
on its own.

## Why the working config works

The two halves of the architecture do *different* things, and each
half needs the right substrate to do its job:

| Half | Job | Right tool | Wrong tool |
|---|---|---|---|
| **Retrieval** | Find triples that match the question | Deterministic graph walker (cortex router) | Language model |
| **Synthesis** | Render triples as prose | Language model | Deterministic template |

Local 7B-quantized models cannot do retrieval reliably because
retrieval is a *structural* problem (which neurons exist, which
edges connect them, which labels match) and language models reason
*distributionally*. They fill in plausible labels and tool choices
the way they fill in plausible next tokens — and "plausible" is not
the same as "present in the graph."

Conversely, the cortex router cannot write prose. It returns
neighborhoods of triples; turning that into an answer requires
language production, which is what a frontier-trained transformer is
for.

The harmony is each half doing the job it can do correctly:

- Cortex router is **incapable of hallucinating** because it can only
  return neurons that exist.
- LLM synthesizer is **incapable of mis-routing** because it never
  sees the routing decision — it only sees the triples that came back.

Failure modes from one half cannot leak into the other. That is the
property that makes the system trustworthy.

## Connection to the foundational design

From the MCP server instructions for `sara-brain`:

> *Sara Brain — path-of-thought knowledge system. Persistent memory
> that never forgets. The LLM is the senses, Sara is the brain.*

From [v050_two_layer_realignment.md](v050_two_layer_realignment.md),
quoting Pearl 2026a §7.3:

> *transformers are the best sensory processing system ever
> engineered. They are not whole brains. They process; they do not
> store. They are stateless; they do not accumulate. They infer; they
> do not remember.*

The v052 finding is the same principle observed at routing time
specifically. Routing is not language production — it is a structural
lookup against a persistent store. Asking the LLM to do it is asking
the senses to do the brain's job.

## Practical guidance

For any v052+ deployment where substrate fidelity matters, use:

```
--cortex-router  --synthesis-model <ollama-model-that-fits-VRAM>
```

The synthesis model is the only quality knob that should vary with
hardware. Routing always goes through the cortex router.

The all-LLM router config (`--router-model X --synthesis-model X`)
remains in the codebase as a reference point and for experimentation
with stronger router models, but it is not the recommended path.
The CLI does not gate it because experimentation should remain easy;
this doc is the gate.

## Scope of this finding

- **Tested:** local Ollama, 7B-quant synthesis, the aptamer substrate.
- **Not tested:** stronger router models (e.g. Anthropic-API-routed),
  larger local models if hardware permits.
- **Expected to generalize:** the structural argument does not depend
  on model size. A larger LLM router would hallucinate labels less
  often, but "less often" is not the same as "structurally cannot."
  The cortex router's correctness is a property of its construction;
  any LLM router's correctness is a property of its training and
  prompt.
