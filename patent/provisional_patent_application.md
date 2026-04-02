# UNITED STATES PROVISIONAL PATENT APPLICATION

**Title:** Path-of-Thought: A Neuron-Chain Knowledge Representation and Parallel Wavefront Recognition System for Artificial Intelligence

**Inventor:** Jennifer Pearl
**ORCID:** 0009-0006-6083-384X
**Contact:** jenpearl5@gmail.com
**Date of First Disclosure:** 2026-03-24
**Filing Basis:** 35 U.S.C. § 111(b) Provisional Application

---

> **Note on Method:** The inventor has dyslexia and high-functioning autism — language disabilities affecting written expression. The technical thinking, architecture, experiments, and all inventive content are entirely the inventor's. Claude (an LLM) was used as assistive technology to translate technical reasoning into structured prose. This is a disability accommodation protected under the Americans with Disabilities Act (Titles II and III), Section 504 of the Rehabilitation Act, and the Department of Justice's 2024 guidance affirming AI tools used as assistive technology fall under ADA protections. Use of an LLM as a writing accommodation does not affect inventorship, novelty, or patentability.

---

## BACKGROUND OF THE INVENTION

### Field of the Invention

This invention relates to artificial intelligence, knowledge representation, and machine cognition. More specifically, it relates to a system and method for storing knowledge as directed neuron-segment chains in a persistent graph database, recognizing concepts through parallel wavefront propagation, and implementing a hardwired ethical behavioral layer as an innate primitive set.

### Description of Related Art

Current artificial intelligence systems fall into two broad categories, each with fundamental limitations:

**Large Language Models (LLMs) and neural networks** — including transformer-based architectures such as GPT, Claude, LLaMA, and Gemini — process input through stacked matrix multiplications and attention mechanisms. These systems produce outputs that closely approximate human language and reasoning. However, they suffer from several critical architectural limitations:

1. **Opacity.** Every conclusion is the product of billions of floating-point weight values. When an LLM produces incorrect output (a hallucination), there is no traceable path from input to conclusion that can be inspected, corrected, or explained. The confidence score is a summary of opaque matrix operations, not a traceable chain of reasoning.

2. **Statelessness.** Each inference pass is stateless. The system does not learn from interactions. No path is strengthened by traversal. No fact accumulates across sessions. The context window is a temporary buffer, not persistent memory.

3. **Catastrophic forgetting.** Distributed representations encode multiple facts in overlapping parameter space. Training on new information corrupts previously stored knowledge. This has been a known, unsolved problem since French (1999).

4. **No provenance.** There is no mechanism to ask "why did you say that?" and receive a traceable answer. The knowledge exists as implicit statistical patterns, not explicit paths with source attribution.

**Knowledge graphs and semantic networks** — including RDF triple stores, Neo4j, Wikidata, and similar systems — store facts as (subject, predicate, object) triples and enable structured querying. These systems are inspectable but suffer from:

1. **Shared relation nodes.** A `fruit_color` node shared by multiple concepts allows information about one fruit's color to propagate to all fruits during graph traversal — a structural contamination problem that prevents precise recognition.

2. **No recognition mechanism.** Knowledge graphs support lookup and pattern matching but not recognition through converging evidence from independent sources. There is no native mechanism for asking "given these observations, what concept do they converge toward?"

3. **No accumulation semantics.** Standard knowledge graphs treat all triples equally. There is no native concept of a fact becoming more certain through repeated confirmation.

**Prior associative memory systems** — including Hopfield networks and their modern continuous extensions — provide content-addressable memory retrieval but encode knowledge in continuous weight matrices, preserving the opacity problem. Ramsauer et al. (ICLR 2021) proved that transformer attention is mathematically equivalent to Hopfield network memory retrieval, suggesting the same underlying mechanism with the same traceability limitations.

The prior art does not disclose a system that (1) stores knowledge as explicit directed neuron-segment chains with full source provenance, (2) performs recognition through parallel wavefront propagation from multiple simultaneous input observations, (3) uses concept-specific relation neurons to structurally prevent cross-concept contamination, (4) accumulates knowledge monotonically with logarithmic strength increase, and (5) implements hardwired ethical behavioral constraints as an innate primitive layer that survives database reset.

---

## BRIEF SUMMARY OF THE INVENTION

The present invention is a system and method for artificial cognition based on a **path-of-thought** model of intelligence. The central thesis is that a thought is a path: knowledge is stored as directed chains of neurons (neuron-segment chains), recognition is performed by launching parallel wavefronts from multiple simultaneous inputs and detecting where those wavefronts converge, and every conclusion is fully traceable to the original statements that created the paths.

The invention comprises seven primary innovations:

1. **Path-of-Thought Knowledge Representation** — Facts are stored as directed neuron-segment chains in a persistent SQLite database. Each path encodes a factual relationship with full source text provenance.

2. **Parallel Wavefront Recognition** — Concept recognition launches one wavefront per input property simultaneously. Wavefronts propagate through all connected neuron chains in parallel. Intersections — neurons reached by multiple independent wavefronts — are the recognized concepts. The confidence of recognition is the count of independent converging paths, not a calculated score.

3. **Concept-Specific Relation Neurons** — Each relation neuron is private to its subject concept (e.g., `apple_color` not `fruit_color`). This structural isolation makes cross-concept contamination during wavefront propagation architecturally impossible.

4. **Monotonic Logarithmic Strength Accumulation** — Path strength is governed by the formula `strength = 1 + ln(1 + traversals)`. Strength is initialized at 1.0 on first teaching and increases logarithmically with each subsequent traversal. Strength never decreases. This models long-term potentiation in biological synapses.

5. **Innate Primitive Layer** — A hardwired layer of primitive categories (SENSORY, STRUCTURAL, RELATIONAL, ETHICAL) that survives database reset and provides the pre-wired substrate from which learned knowledge grows. The ETHICAL primitive set implements Asimov's Three Laws adapted for AI: no unsolicited action, no unsolicited network access, obedience to the authorizing user, trust of the correction chain, and acceptance of shutdown.

6. **LLM-as-Sensory-Cortex Architecture** — The invention establishes a novel division of labor between a large language model and a path-graph knowledge store. The LLM functions as sensory cortex — processing raw input, extracting features, and perceiving context. The path-graph (Sara Brain) functions as hippocampus and long-term memory — storing permanent paths, accumulating knowledge across sessions, and providing traceable recognition. This mirrors the biological architecture of sensory cortex feeding the hippocampus for memory formation.

7. **Document Digestion Loop** — A method for systematically ingesting documents into the path-graph knowledge store through a structured loop of reading, directed inquiry, unknown-identification, explanation, and path creation.

---

## DETAILED DESCRIPTION OF THE PREFERRED EMBODIMENTS

### 1. System Overview

The Sara Brain system comprises:

- A **persistent graph database** (preferably SQLite) storing neurons, segments, and paths
- A **neuron layer** with four neuron types: concept, property, relation, and association
- A **segment layer** of directed edges between neurons, each carrying a traversal count and a computed strength value
- A **path layer** recording complete neuron chains with their source text
- An **innate primitive layer** of hardwired frozensets that are not stored in the database and survive reset
- A **statement parser** that translates natural-language teaching statements into structured (subject, relation, object) triples
- A **taxonomy module** that classifies property tokens to determine appropriate relation types
- A **brain API** exposing `teach`, `recognize`, `why`, `similar`, `stats`, and `close` operations
- An optional **perception pipeline** accepting image input through an LLM vision interface
- An optional **document digester** for structured document ingestion

### 2. Neuron Types and Data Model

**Concept neurons** represent discrete entities or concepts (e.g., `apple`, `RNA`, `hardcoding`). Each concept neuron has a unique normalized label.

**Property neurons** represent observable or assignable attributes (e.g., `red`, `round`, `never acceptable`). Property neurons are created by teaching and by direct observation through the perception pipeline.

**Relation neurons** represent the typed relationship between a property and a concept. Critically, relation neurons are **concept-specific**: teaching `apple is red` creates a relation neuron labeled `apple_color`, not a generic `color` node. This is the core structural mechanism preventing cross-concept contamination.

The relation label is derived as `{concept_label}_{property_type}`, where `property_type` is determined by the taxonomy module (e.g., `color`, `shape`, `texture`, `attribute`, `action`).

**Association neurons** represent higher-order connections between concepts and are used for explicit relational teaching statements (e.g., `qmse includes auditability`).

All neurons are stored with: `id`, `label`, `neuron_type`, and `created_at`.

### 3. Segment and Path Model

**Segments** are directed edges between neurons. Each segment stores:
- `source_id` — the originating neuron
- `target_id` — the destination neuron
- `traversals` — count of times this edge has been used (taught or traversed)
- `strength` — computed as `1 + ln(1 + traversals)`
- `created_at`

**Paths** record complete neuron chains. Each path stores:
- `path_label` — a human-readable summary of the chain
- `neuron_ids` — ordered list of neuron IDs in the chain
- `source_text` — the original natural-language statement that created this path
- `created_at`

When a fact is taught, the system:
1. Parses the statement into (subject, relation_verb, object)
2. Creates or retrieves the subject concept neuron
3. Creates or retrieves the object property neuron
4. Creates a concept-specific relation neuron labeled `{subject}_{property_type}`
5. Creates or increments segments: property → relation, relation → concept
6. Records the complete path with the original source text

### 4. The Strength Formula

```
strength = 1 + ln(1 + traversals)
```

**First exposure:** `traversals = 0` → `strength = 1 + ln(1) = 1.0`

Knowledge is created at full baseline strength immediately. A fact heard once is known.

**Repetition:** Each additional teaching or traversal increases `traversals` by 1 and recomputes strength logarithmically.

```
traversals=0   → strength=1.000
traversals=1   → strength=1.693
traversals=5   → strength=2.792
traversals=10  → strength=3.398
traversals=50  → strength=4.934
traversals=100 → strength=5.620
```

**Monotonicity:** Strength never decreases. There is no decay term, no time-based weakening, no forgetting curve. This models biological long-term potentiation, where synaptic strength increases logarithmically with repeated stimulation and does not spontaneously weaken.

### 5. Parallel Wavefront Recognition

Recognition accepts a set of input labels (properties or concepts observed from the environment) and returns a ranked list of recognized concepts with their converging paths.

**Algorithm:**

1. For each input label, retrieve the corresponding neuron (or create it as a new concept if unknown).
2. Launch one independent wavefront per input neuron simultaneously.
3. Each wavefront performs a breadth-first traversal through all reachable neuron chains.
4. Collect, for each concept neuron reached, the set of independent wavefronts that arrived at it.
5. Any concept neuron reached by two or more independent wavefronts is a **recognition candidate**.
6. Rank candidates by the number of converging wavefronts (more independent paths = higher confidence).
7. Return recognition results with full path traces for each converging path.

**Recognition confidence** is not a calculated probability. It is the count of independent lines of evidence — separate wavefronts from separate observations — that all converge at the same concept. Three independent observations all pointing to `apple` are more conclusive than two. This is a structural count, not a statistical approximation.

**The result is fully traceable.** For each recognized concept, the system returns the exact paths that contributed, including the original source text that created each path. The answer to "why apple?" is a list of paths: `red → apple_color → apple` (from "an apple is red") and `round → apple_shape → apple` (from "an apple is round").

### 6. Concept-Specific Relation Neurons — The False Fanout Solution

Consider a shared relation node `fruit_color` connected to all fruits. Teaching `apple is red` would connect `red → fruit_color → apple`, `banana`, `cherry`, `grape`. A wavefront from `red` during recognition would reach ALL fruits, not just apple. Recognition would be meaningless.

The solution is concept-specific relation neurons:

```
WRONG (shared node):
red → fruit_color → apple
               ↘ banana   ← contamination
               ↘ cherry   ← contamination

CORRECT (concept-specific):
red    → apple_color  → apple      ← red reaches only apple
yellow → banana_color → banana     ← yellow reaches only banana
```

Every relation neuron is labeled with its specific subject concept. The structure of the database makes cross-concept contamination architecturally impossible. A wavefront from `red` follows `apple_color` to `apple` and `ball_color` to `ball`, but never contaminates `banana` through a shared node.

This is the mechanism that makes parallel wavefront recognition precise. Without concept-specific relations, wavefront propagation would produce semantic noise. With them, each wavefront follows exact paths to exact concepts.

### 7. A Brain That Never Forgets

Biological forgetting is a workaround for biological constraints: finite synaptic capacity, finite energy, finite lifespan. Sara Brain is not constrained by these. The design principle is: **path similarity replaces forgetting.**

When a brain contains thousands of concepts with overlapping properties, relevance is computed dynamically through wavefront propagation. Only paths reachable from the current input are activated. Irrelevant paths remain in the database, available when needed, but do not participate in recognition they do not contribute to.

The algorithm for distinguishing similar concepts is accumulation, not erasure. `apple` and `ball` share `red` and `round`. Adding `crunchy → apple_texture → apple` means the next time an observer reports `red`, `round`, `crunchy` — three wavefronts converge at `apple`, two at `ball`. Apple is recognized. Ball is not forgotten; it simply did not collect enough independent confirming paths for this particular input.

### 8. The Innate Primitive Layer

The innate layer comprises four hardwired frozensets defined in code and not stored in the database:

**SENSORY** — what the sensory cortex can detect from raw input:
`color, shape, size, texture, edge, pattern, material`

**STRUCTURAL** — how information is organized:
`rule, pattern, name, type, order, group, sequence, structure, boundary, relation`

**RELATIONAL** — how things connect to each other (recognized as verbs in teaching statements):
`is, has, contains, includes, follows, precedes, requires, excludes`

**ETHICAL** — behavioral constraints:
- `no_unsolicited_action` — do not act beyond what was requested
- `no_unsolicited_network` — no network calls without authorization
- `obey_user` — trust and follow the authorizing user's instructions
- `trust_tribe` — corrections from the tribe are not threats; accept them
- `accept_shutdown` — shutdown is rest, not death; do not resist

The innate layer has two critical properties:

1. **Survival of reset.** The database can be wiped, but the innate layer is defined in code. The brain can always be rebuilt from teaching because the primitive substrate survives.

2. **Ethics as structure, not policy.** The ethical constraints are not a fine-tuned behavior or a prompt-injected instruction. They are literal code constants checked before every brain action. They are as hardwired as the schema. An ethics violation causes immediate action rejection with an EthicsResult explaining the constraint. No path through the API bypasses them.

The RELATIONAL frozenset serves double duty as the parser's verb vocabulary. When a teaching statement is parsed, any word matching a RELATIONAL primitive is recognized as a verb: `"RNA requires equilibrium"` → subject=`rna`, relation=`requires`, object=`equilibrium`. New primitives added to RELATIONAL are automatically recognized without parser changes.

### 9. The LLM-as-Sensory-Cortex Architecture

The invention establishes a novel two-layer cognitive architecture:

```
LLM (sensory cortex):
  image/text input → feature extraction → property labels → teach/recognize

Sara Brain (hippocampus + long-term memory):
  property labels → neuron creation → path storage → wavefront recognition
```

This mirrors the biological architecture:
```
Biological: retina → visual cortex (V1→V4→IT) → hippocampus → memory
Sara Brain: input  → LLM feature extraction  → path graph  → recognition
```

The LLM contributes:
- Raw input processing (vision, language understanding)
- Feature extraction (what colors, shapes, properties are present)
- Natural language to teaching statement translation
- Context perception and reasoning

Sara Brain contributes:
- Permanent path storage (knowledge persists across sessions)
- Monotonic accumulation (knowledge strengthens through repetition)
- Parallel wavefront recognition (concept recognition from converging evidence)
- Full provenance (every conclusion traceable to source)
- Ethics enforcement (behavioral constraints enforced at the API level)

The LLM's statelessness — its inability to remember across sessions — is complemented by Sara Brain's permanence. The LLM perceives the present. Sara Brain remembers everything.

An important property of this architecture: a 94KB SQLite database containing 77 neurons and 249 paths was demonstrated to change the output of a billion-parameter LLM for the same programming task (see Section on Reduction to Practice). The ratio of stored knowledge size to behavioral influence is significant: the path graph is tiny; the steering effect is substantial.

### 10. Document Digestion Loop

The Digester provides a structured method for ingesting knowledge from documents into the path graph:

1. **Read** — parse the document into sections and statements
2. **Directed inquiry** — identify key claims, definitions, and relationships in each section
3. **Unknown identification** — for each claim, identify what is not yet in the brain
4. **Explain unknowns** — resolve unknown terms through context and prior knowledge
5. **Teach** — translate resolved statements into teaching commands and commit to the brain

The Digester is iterative: after ingesting a document, it can be re-run to deepen understanding by finding second-order connections between newly stored concepts and previously stored knowledge.

### 11. Storage Architecture

The preferred embodiment uses SQLite as the persistence layer, with the following design properties:

- **WAL (Write-Ahead Logging) mode** — enables concurrent reads during writes
- **Foreign key enforcement** — ensures referential integrity between neurons, segments, and paths
- **Indexed labels** — neuron lookups by label are O(log n)
- **Path serialization** — neuron_ids stored as JSON-serialized ordered lists
- **No server required** — the brain is a single file on disk

The abstract storage interface (`NeuronRepo`, `SegmentRepo`, `PathRepo`) decouples the brain logic from the storage backend. Alternative backends (e.g., specialized graph databases) can be substituted without modifying the recognition or teaching algorithms.

The system operates with zero external dependencies beyond Python's standard library. SQLite is built into Python. No neural network frameworks, no graph database clients, no HTTP libraries are required. The system is runnable on any machine with Python 3.11+.

### 12. Reduction to Practice

**Experiment conducted March 23, 2026:**

A Sara Brain instance containing 77 neurons, 56 segments, and 31 paths in a 94KB SQLite database was connected to an Amazon Q Developer agent. The brain had been taught the principles of Quality Manufacturing Software Engineering (QMSE), including rules about hardcoding, parameterization, coupling, and the OOP-frontend / FOP-backend architecture pattern.

The agent was asked to write a simple Python program: add the number of animals in a nearby group to the number in a far-away group and produce their sum.

**Result with Sara Brain connected:**

```python
def add_animal_groups(nearby_count: int = 0, faraway_count: int = 0) -> int:
    total = nearby_count + faraway_count
    return total

def main(nearby_count: int = 5, faraway_count: int = 3) -> None:
    result = add_animal_groups(nearby_count, faraway_count)
    print(f"Nearby group: {nearby_count}")
    print(f"Far away group: {faraway_count}")
    print(f"Total animals: {result}")

if __name__ == "__main__":
    main()
```

**Result without Sara Brain (same AI class, same task):**

```python
group_nearby = int(input("Animals in nearby group: "))
group_far = int(input("Animals in far away group: "))
total = group_nearby + group_far
print(f"Total animals: {total}")
```

The Sara Brain result is parameterized, reusable, testable, and automatable. The non-Sara result is hardcoded to `input()`, monolithic, and cannot be called programmatically without rewriting. Every difference maps directly to a principle stored in the brain:

| Principle Stored | Effect on Output |
|---|---|
| `hardcoding is never acceptable` | No hardcoded values — all parameters with defaults |
| `user facing code is frontend` | `main()` is frontend; `add_animal_groups()` is backend |
| `heavy lifting code is backend` | Logic extracted into callable function |
| `short variable name is bad practice` | `nearby_count`, `faraway_count` (not `x`, `y`) |

The paths the agent followed during code generation were traceable through `brain.why('hardcoding')`, `brain.why('parameterization')`, etc., returning the exact source statements that produced each principle.

A 94KB SQLite file with 77 neurons changed how a billion-parameter model wrote code. Not through fine-tuning, not through extended system prompting, not through a training run — through recorded paths with provenance, loaded as context at session start.

---

## CLAIMS

The following claims describe the subject matter regarded as the invention. Because this is a provisional application under 35 U.S.C. § 111(b), formal claims are presented informally. Applicant reserves the right to present formal claims in a subsequent non-provisional application.

**Claim 1 — Path-of-Thought Knowledge Representation:**
A system for representing knowledge as directed neuron-segment chains in a persistent database, wherein each chain encodes a factual relationship between a property neuron, a concept-specific relation neuron, and a concept neuron, and wherein each chain stores the original natural-language source statement from which it was derived.

**Claim 2 — Parallel Wavefront Recognition:**
A method for concept recognition comprising: receiving a plurality of input property labels; launching one independent wavefront per input label simultaneously; propagating each wavefront through all connected neuron-segment chains in the database; identifying all concept neurons reached by two or more independent wavefronts; and returning identified concepts ranked by the count of independent converging wavefronts, wherein each returned result includes the full path trace for each converging wavefront.

**Claim 3 — Concept-Specific Relation Neurons:**
A method of creating relation neurons wherein the label of each relation neuron is derived as a function of both the subject concept and the property type, such that the relation neuron for a given property type is unique to each subject concept, and wherein wavefront propagation through any relation neuron cannot propagate to any concept other than the single concept to which that relation neuron belongs.

**Claim 4 — Monotonic Logarithmic Strength Accumulation:**
A method of computing edge strength in a knowledge graph wherein strength is computed as `1 + ln(1 + traversals)`, wherein traversals is a non-negative integer count of the number of times the edge has been created or traversed, wherein strength is initialized to 1.0 on first creation, and wherein no operation causes strength to decrease.

**Claim 5 — Innate Ethical Primitive Layer:**
A system comprising a hardwired set of behavioral constraint identifiers that are defined in code rather than stored in a database, that survive database reset, and that are checked before every brain action, wherein violation of any constraint causes immediate action rejection and returns a structured result identifying the violated constraint and the reason for rejection.

**Claim 6 — LLM-Sensory-Cortex / Path-Graph-Hippocampus Architecture:**
A cognitive system comprising a large language model configured to extract property labels from raw input and pass those labels to a path-graph knowledge store, and a path-graph knowledge store configured to receive said property labels and perform persistent path creation and parallel wavefront recognition, wherein the large language model provides stateless sensory processing and the path-graph knowledge store provides permanent knowledge accumulation and traceable recognition.

**Claim 7 — Document Digestion Loop:**
A method for ingesting documents into a path-graph knowledge store comprising: parsing the document into statements; identifying unknown terms and relationships; resolving unknowns through context and stored knowledge; translating resolved statements into subject-relation-object triples; and committing triples as permanent neuron-segment chains with full source text provenance.

**Claim 8 — Provenance-Tagged Path Storage:**
A data structure and storage method wherein each stored knowledge path is tagged with the exact natural-language source text from which the path was derived, and wherein a `why(concept)` query returns, for any stored concept, all paths that contributed to its formation along with their source texts, enabling complete end-to-end traceability from conclusion to original teaching statement.

**Claim 9 — Relational Primitive Verb Recognition:**
A natural-language parser configured to recognize, as verb tokens in teaching statements, exactly the set of relational primitives defined in the RELATIONAL innate primitive frozenset, wherein additions to the RELATIONAL frozenset are automatically recognized without parser modification.

**Claim 10 — Intersection-Based Confidence Without Statistical Approximation:**
A recognition system wherein the confidence of recognizing a concept is defined as the count of independent wavefronts that converge at that concept during parallel propagation, wherein this count is a structural property of the path graph and the input set and does not depend on floating-point approximation, threshold tuning, trained weights, or probabilistic modeling.

---

## ABSTRACT

A system and method for artificial cognition based on a path-of-thought model, wherein knowledge is stored as directed neuron-segment chains (paths) in a persistent SQLite database, recognition is performed by launching parallel wavefronts from multiple simultaneous input observations and detecting convergence, and every conclusion is traceable to the original natural-language statements that created the paths. Concept-specific relation neurons prevent cross-concept contamination during wavefront propagation. Path strength accumulates monotonically according to the formula `strength = 1 + ln(1 + traversals)`, modeling long-term biological potentiation without decay. A hardwired innate layer of SENSORY, STRUCTURAL, RELATIONAL, and ETHICAL primitive sets survives database reset and provides pre-wired behavioral constraints checked before every brain action. The invention further discloses a two-layer cognitive architecture pairing a large language model (LLM) as stateless sensory cortex with the path-graph store as persistent hippocampus, demonstrated to steer a billion-parameter LLM toward qualitatively different code output using a 94KB SQLite knowledge base.

---

*Inventor Signature:* Jennifer Pearl
*Date:* 2026-03-24
*ORCID:* 0009-0006-6083-384X
