# v017 — The Sensory Shell: Theory of Operation

**Date:** April 17, 2026
**Author:** Jennifer Pearl
**Branch:** signed_refutation_paths
**Module:** `src/sara_brain/sensory/`

---

## What It Is

The sensory shell is an empty processing engine that sits on top of Sara Brain. It has no knowledge, no weights, no training data, and no opinions. It takes text input, breaks it into words, feeds those words as wavefront seeds into Sara's graph, and renders whatever paths converge as output. Every answer traces to a specific taught fact. If Sara hasn't been taught something, the shell says "I don't know."

The shell is to Sara Brain what a CPU is to RAM. The CPU doesn't know anything. The memory holds the knowledge. The CPU just processes instructions.

---

## Why It Exists

The AI industry uses transformers as both the processing engine and the knowledge store. Facts are compressed into billions of floating-point weight parameters during training. This creates three problems:

1. **Opacity.** When the system produces a wrong answer, there is no traceable path from input to error. The wrong answer emerged from opaque weight interactions.

2. **Immutability.** Adding one fact requires retraining or fine-tuning the entire model. Cost: millions of dollars. Time: weeks to months.

3. **Hallucination.** The model cannot distinguish between what it knows and what it is inventing. There is no separable belief store inside a transformer.

The sensory shell solves all three by separating the processing engine from the knowledge store:

- **Processing engine:** The shell. Tokenization, wavefront propagation, convergence detection, rendering. Pure algorithm, no knowledge.
- **Knowledge store:** Sara Brain's path graph. Explicit, inspectable, provenance-traced neuron-segment chains stored in SQLite.

Teaching a fact is a single database INSERT. The shell immediately has access to it. No retraining. No fine-tuning. No GPU.

---

## Theory of Operation

### The Pipeline

```
Text Input
    |
    v
[1. Tokenizer]
    Split text into words. Strip stopwords (a, the, is, what, how).
    Greedy longest-match against Sara's neuron labels to detect
    multi-word phrases ("electron transport chain" = one token).
    Validate singularization against the graph — if the singular
    form isn't a known neuron, keep the original ("osmosis" stays
    "osmosis", not "osmosi").
    |
    v
[2. Wavefront Seeding]
    Each token becomes a seed. Sara Brain launches one independent
    wavefront per seed into the graph.
    |
    v
[3. Parallel Propagation]
    Wavefronts propagate through Sara's paths — following directed
    segments between neurons. Propagation runs at three inhibition
    thresholds (focused 0.5, relaxed 0.3, open 0.1) to capture
    both confident and speculative connections. Bidirectional echo
    lets thoughts bounce: concept <- property -> new concept.
    |
    v
[4. Convergence Detection]
    Where two or more independent wavefronts reach the same CONCEPT
    neuron, that neuron is a recognition candidate. Confidence =
    count of converging wavefronts. Three wavefronts converging on
    "apple" means three independent lines of evidence.
    |
    v
[5. Rendering with Provenance]
    Each converging path is rendered using its original source_text
    (the exact sentence the human used when teaching the fact).
    Every output line shows the path ID and source. Below each
    fact, the full trace shows every intermediate node the
    wavefront passed through.
    |
    v
Output (traceable, auditable, every word sourced)
```

### What Makes It "Empty"

The shell contains zero world knowledge. Compare:

| Component | Shell | Traditional LLM |
|-----------|-------|-----------------|
| World facts | None | Billions compressed in weights |
| Grammar | None (stopword list only) | Trained on trillions of tokens |
| Vocabulary | None (uses Sara's neurons) | BPE/tokenizer trained on corpus |
| How to parse a question | None | Instruction-tuning |
| How to generate text | None (renders source_text) | Autoregressive generation |

If Sara Brain is empty, the shell can do nothing. It cannot guess, speculate, or hallucinate. The only thing it can produce is "I don't know."

### The Wavefront Engine

The shell does not implement its own wavefront engine. It calls Sara Brain's existing API:

- `brain.propagate_echo(seeds, short_term, ...)` — bidirectional spreading activation through the graph. Read-only. No graph mutation.
- `brain.short_term(event_type)` — opens a session-scoped scratchpad. Convergence accumulates here. Discarded when the query ends.
- `brain.why(label)` — all paths leading TO a neuron (reverse lookup).
- `brain.trace(label)` — all paths leading FROM a neuron (forward lookup).

The shell is a thin layer: tokenize input, call the brain, render the output with provenance.

### Singularization Validation

The statement parser singularizes words at teach time ("apples" -> "apple"). This sometimes mangles words: "osmosis" -> "osmosi" (the parser treats "-ses" as a plural suffix).

The sensory tokenizer handles this at query time without modifying the parser:

1. Look up the word as-is in Sara's graph.
2. If not found, try the singularized form.
3. If the singular IS found, use it ("apples" -> "apple" if "apple" exists).
4. If the singular is NOT found, keep the original ("osmosis" stays "osmosis").

The parser is not modified. The sensory module handles its own lookups.

### Session Context

The `Session` tracks recent topics across turns. If the user asked about methane last turn and now says "how many hydrogen atoms," methane is still in context. Session state is not persisted — it lives only for the current conversation and never mutates the graph.

---

## What This Proves

### Transformers Are Being Used Wrong

Self-attention is mathematically equivalent to parallel wavefront convergence (Ramsauer et al., 2021). The mechanism is correct. The storage is wrong. Compressing facts into weight parameters loses traceability, correctability, and provenance.

The sensory shell demonstrates that the processing engine and the knowledge store can be separated. The engine walks paths. The paths are the knowledge. Every answer is traceable to specific taught facts.

### Teaching > Training

Training a model on trillions of tokens costs $10M-$100M and produces a system where no individual fact is inspectable. Teaching Sara Brain one fact at a time costs zero and produces a system where every fact has provenance. The April 2026 benchmarks showed that 45 hand-taught facts in Sara Brain raised a 3B model from 58.4% to 80% on MMLU Biology — exceeding GPT-3.5 (~70%).

The sensory shell takes this further: no LLM at all. Pure graph traversal. The "model" is the wavefront algorithm. The "weights" are Sara's paths.

### Auditability

Every output from the shell includes:

- The path ID of the fact that produced it
- The original source text (the exact sentence the human taught)
- The full trace showing every intermediate node the wavefront passed through

When the shell is wrong, the specific gap or error is identifiable. When it is right, the provenance is complete. No transformer-only system offers this guarantee.

---

## Relationship to Other Modules

| Module | Role | Uses LLM? |
|--------|------|-----------|
| `sensory/` | Empty processing shell over Sara Brain | No |
| `cortex/` | Rule-based NL parser + template generator | No (optional Ollama fallback) |
| `agent/` | ReAct loop with Sara Brain validation | Yes (Ollama) |
| `nlp/` | Vision observer, document reader, LLM providers | Yes |
| `core/` | Brain, recognizer, learner, digester, perceiver | No |

The sensory shell is the only module designed to operate with zero LLM dependency. It is the architectural proof that the knowledge lives in the graph, not in model weights.

---

## Limitations

1. **No language generation.** The shell renders stored source_text. It does not compose novel sentences. If Sara was taught "apples are red" and "apples are round," the shell returns both facts verbatim. It does not synthesize "apples are red and round."

2. **No causal reasoning.** The shell finds convergence — where paths from different inputs meet. It does not chain: "if X then Y, and if Y then Z, therefore if X then Z." Causal chains require a reasoning engine that does not yet exist.

3. **Parser-dependent teaching.** Facts enter Sara through the statement parser, which rejects ~50% of natural English sentences. The shell's output quality is limited by what could be taught.

4. **Singularization artifacts.** The parser's singularization creates some mangled neuron labels (e.g., "osmosi" from "osmosis"). The tokenizer works around this at query time but the underlying neuron labels remain wrong until the parser is improved.

5. **No disambiguation.** The shell does not handle homonyms. "Bank" (river) and "bank" (financial) resolve to the same neuron. The cortex module has disambiguation logic that the shell does not yet incorporate.
