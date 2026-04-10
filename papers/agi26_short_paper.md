# Path-of-Thought: A Persistent Memory Architecture for LLM Steering Through Neuron-Chain Knowledge Graphs

**Jennifer Pearl**
University of Houston, Center for Nuclear Receptors and Cell Signaling
ORCID: 0009-0006-6083-384X
jenpearl5@gmail.com

**Keywords:** cognitive architecture, knowledge representation, path-of-thought, parallel wavefront recognition, LLM steering, persistent memory

---

## Abstract

We present Sara Brain, a cognitive architecture in which knowledge is stored as directed neuron-segment chains with full source-text provenance, and concept recognition is performed through parallel wavefront convergence. Concept-specific relation neurons structurally prevent cross-concept contamination. Knowledge accumulates monotonically under a logarithmic strength formula modeling biological long-term potentiation. The system implements a two-layer architecture pairing a stateless LLM (sensory cortex) with a persistent path graph (hippocampus). In experiments, a 94KB path graph (77 neurons) reliably steered a billion-parameter LLM toward principled code, and a 500KB graph (793 neurons) transformed a 3B-parameter model into a system producing domain-expert-level output on planetary physics — a domain outside its training. The entire system runs on Python 3.11+ standard library with zero dependencies.

---

## 1. Introduction

The dominant paradigm in artificial intelligence encodes knowledge implicitly in billions of floating-point parameters trained on large corpora. This produces capable systems that cannot explain their reasoning. When output is incorrect, there is no traceable path from input to error.

This work starts from a different premise: a thought is a path through recorded knowledge, and recognition is the convergence of independent paths from simultaneous observations. This idea, originating in the 1990s, predates transformers and deep learning. We present Sara Brain, a working implementation that makes knowledge paths the primary data structure rather than an artifact to be recovered through interpretability techniques.

The central contribution is not the path-graph architecture alone but its pairing with large language models: the LLM functions as stateless sensory cortex (feature extraction), while the path graph functions as persistent hippocampus (memory formation and retrieval). This mirrors the biological division between sensory processing and memory that evolved precisely because stateless perception is insufficient for intelligence.

## 2. Architecture

### 2.1 Data Model

Knowledge is stored in three structures. **Neurons** are typed nodes: concept neurons (entities), property neurons (attributes), relation neurons (concept-specific connectors), and association neurons (inter-concept links). **Segments** are directed weighted edges between neurons, with strength computed as `1 + ln(1 + traversals)`, modeling biological long-term potentiation. Strength is strictly monotonically increasing — no decay exists. **Paths** are recorded neuron chains preserving the original natural-language source text.

### 2.2 Concept-Specific Relation Neurons

This is the central structural innovation. Teaching "apple is red" and "banana is yellow" creates distinct relation neurons: `red → apple_color → apple` and `yellow → banana_color → banana`. The relation label `{subject}_{property_type}` ensures no wavefront from `red` can propagate to `banana` through shared structure. Cross-concept contamination is structurally impossible, modeling hippocampal context-specific encoding that prevents interference between similar memories [1].

### 2.3 Parallel Wavefront Recognition

Given input labels, one independent wavefront is launched per input simultaneously. Each performs breadth-first traversal through reachable segments. Any concept neuron reached by two or more independent wavefronts is a recognition candidate, ranked by convergence count. Recognition confidence is the count of independent lines of evidence — deterministic, not statistical.

For inputs {red, round, crunchy}: the `red` wavefront reaches apple, ball, cherry; `round` reaches apple, ball; `crunchy` reaches only apple. Apple is recognized with confidence 3 (three converging wavefronts). Full path traces with source texts are returned for each.

### 2.4 The LLM-as-Sensory-Cortex Architecture

Transformer attention is parallel wavefront propagation encoded in weight matrices [2]. Each attention head independently searches the input; convergence determines output. Sara Brain makes this same mechanism explicit and persistent.

The architectural division maps to biology: biological sensory cortex is stateless (each fixation processed fresh), extracts features without persistent storage, and does not learn from processing alone. LLMs share all three properties. The biological solution is the hippocampus — a separate structure forming permanent memories through Hebbian co-activation. Sara Brain is the computational hippocampus.

```
Biological:  retina → visual cortex (V1→V4→IT) → hippocampus → memory
Sara Brain:  input  → LLM (feature extraction)   → path graph  → permanent paths
```

A hardwired innate layer (SENSORY, STRUCTURAL, RELATIONAL, ETHICAL primitives) provides behavioral constraints enforced at the API level, surviving database reset. Ethics is structural, not configured.

## 3. Experiments

### 3.1 Experiment 1: Code Quality Steering

A Sara Brain instance containing 77 neurons, 56 segments, and 31 paths encoding software engineering principles (94KB database) was connected to an Amazon Q Developer agent. The same agent without Sara Brain served as control. Both received an identical task: write a Python program adding animal group counts.

| Property | With Sara Brain | Without |
|---|---|---|
| Hardcoding | Parameterized with defaults | Values via `input()` |
| Architecture | Separated frontend/backend | Monolithic block |
| Testability | Directly unit-testable | Requires mocking `input()` |

Every difference traces to a specific stored path. The ratio of path-graph knowledge to model parameters is approximately 1:10,000,000.

### 3.2 Experiment 2: Minimal Model Augmentation

A 793-neuron, 500KB Sara Brain instance (knowledge from Wikipedia on Newton's laws and the solar system) was paired with `qwen2.5-coder:3b` — a 3B-parameter model running locally via Ollama. No fine-tuning, no RAG pipeline, no vector database.

The 3B model produced structured, categorized output on planetary motion: correct three-class planet taxonomy, all three Kepler's laws with formal names and definitions, and gravitational force relationships. A naive 3B coding model lacks the parameter capacity for this domain knowledge. The knowledge came from Sara's paths; the language came from the model's weights. Neither could produce this output alone.

When a user corrected an error in the model's dwarf planet definition, the correction was accepted and recordable as a new path — demonstrating the tribal trust model where user corrections override training data.

### 3.3 Resource Asymmetry

| Resource | Large Model | Sara Brain |
|---|---|---|
| Model size | 70B–400B+ params | 3B (smallest viable) |
| Training cost | $10M–$100M+ | $0 |
| Add one fact | Retrain/fine-tune | One SQLite INSERT |
| Knowledge persistence | Lost at session end | Permanent |
| Traceability | Not inspectable | Full path provenance |

## 4. Discussion and Conclusion

Sara Brain demonstrates that a small, persistent, fully traceable knowledge base can reliably steer large language model output — without retraining, fine-tuning, or GPU infrastructure. The architecture makes catastrophic forgetting structurally impossible: new learning creates new neurons and segments without modifying existing paths.

The system has been applied in a professional computational biology context to steer LLM-generated scientific code for RNA dynamics modeling, by the author — a peer-reviewed computational biologist [3,4,5].

We argue the AI industry is over-investing in cortex capacity (model size, training data) and under-investing in memory architecture. Biological brains did not evolve by scaling the visual cortex until it could remember. They evolved a hippocampus. Sara Brain is an implementation of that architectural insight: build the memory, not a bigger cortex.

The system is implemented in pure Python with zero dependencies beyond the standard library, is LLM-agnostic (supporting any MCP-compatible client), and is publicly available at https://github.com/LunarFawn/SaraBrain.

---

## References

[1] Yassa, M.A., Stark, C.E.L.: Pattern separation in the hippocampus. Trends in Neurosciences 34(10), 515–525 (2011)

[2] Ramsauer, H., et al.: Hopfield Networks is All You Need. In: ICLR 2021. https://arxiv.org/abs/2008.02217

[3] Pearl, J., et al.: Crowdsourced RNA design discovers diverse, reversible, efficient, self-contained molecular switches. PNAS 119(18) (2022). https://doi.org/10.1073/pnas.2112979119

[4] Pearl, J., et al.: Exploring the Accuracy of Ab Initio Prediction Methods for Viral Pseudoknotted RNA Structures: Retrospective Cohort Study. JMIRx Bio (2024). https://doi.org/10.2196/58899

[5] Tse, V., et al.: OpenASO: RNA Rescue — designing splice-modulating antisense oligonucleotides through community science. RNA 31(8), 1091–1102 (2025). https://doi.org/10.1261/rna.080288.124

[6] Vaswani, A., et al.: Attention Is All You Need. In: NeurIPS 2017. https://arxiv.org/abs/1706.03762

[7] Elhage, N., et al.: A Mathematical Framework for Transformer Circuits. Transformer Circuits Thread, Anthropic (2021)

[8] Olsson, C., et al.: In-context Learning and Induction Heads. Transformer Circuits Thread, Anthropic (2022)

[9] Quiroga, R.Q., et al.: Invariant visual representation by single neurons in the human brain. Nature 435(7045), 1102–1107 (2005)

[10] Hebb, D.O.: The Organization of Behavior. Wiley (1949)

[11] French, R.M.: Catastrophic forgetting in connectionist networks. Trends in Cognitive Sciences 3(4), 128–135 (1999)

[12] Templeton, A., et al.: Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet. Anthropic Research (2024)
