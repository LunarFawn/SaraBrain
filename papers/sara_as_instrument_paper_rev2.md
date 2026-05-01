# Sara as a Measurement Instrument for Large Language Model Behavior: A Reference Substrate for Studying Transformer Failure Modes

**Jennifer Pearl**
Independent Researcher
ORCID: 0009-0006-6083-384X
jenpearl5@gmail.com

**Date:** April 2026 (Draft, rev 2)

**Keywords:** cognitive architecture, scientific instrumentation, LLM evaluation, reference substrate, knowledge graph, transformer behavior, hallucination, grounded generation, alignment research, experimental method

---

> **Note on Method:** The author has dyslexia and high-functioning autism — language disabilities affecting written expression. The technical thinking, research, architecture, and all intellectual content are entirely the author's. Claude (an LLM, Anthropic) was used as assistive technology to translate technical reasoning into structured prose. This is a disability accommodation protected under the Americans with Disabilities Act (Titles II and III), Section 504 of the Rehabilitation Act, and the Department of Justice's 2024 guidance affirming AI tools used as assistive technology fall under ADA protections. Use of an LLM as a writing accommodation is equivalent to use of text-to-speech software by a blind researcher — it compensates for a disability, it does not replace the thinking.

---

## Abstract

Large language model research has, until now, operated without a fine-grained measurement instrument for transformer behavior. Researchers can observe outputs and grade them against natural-language references, but they cannot separate retrieval from synthesis from training recall from context leakage at the token level — because the model's input substrates (training corpus, retrieved documents, conversational history) are either too large to fully inspect, too unstructured to grade against, or both.

We show that a persistent, inspectable, triple-structured knowledge substrate — *Sara Brain*, a path-of-thought cognitive architecture (Pearl, 2026a) — functions as a scientific instrument for studying transformer behavior in a way that prior substrates do not. Sara satisfies four properties simultaneously: it is **finite** (every contained fact is explicitly inspectable), **structured** (each fact is a traceable subject-relation-object triple with provenance), **large enough** to support nontrivial multi-hop reasoning, and **training-orthogonal** (the substrate's contents are not in any public training corpus, because they are loaded at runtime by a human teacher).

These four properties, in combination, have not been simultaneously available in any prior substrate. Retrieval benchmarks (HotpotQA, NaturalQuestions) have natural-language ground truth that cannot be graded per-triple. Knowledge graphs (Wikidata, ConceptNet) are massive and their contents are largely in training data. Synthetic benchmarks (bAbI, CLUTRR) are small and lose realism. Hand-curated private corpora are one-off and not reusable across experiments. A Sara substrate can be built for any domain, inspected exhaustively, and graded against model output at arbitrary granularity.

We demonstrate the instrument in operation by reporting nine distinct findings about transformer behavior, derived from experimentation against two Sara substrates (a 169-triple substrate from an unpublished RNA-aptamer paper Executive Summary, and a 2,073-neuron substrate covering the full paper). Each finding is an observation that has been discussed theoretically in the literature but has been difficult to measure cleanly. With Sara as reference, they become directly measurable.

The contribution is methodological. We argue that future LLM research on grounded generation, hallucination, context contamination, retrieval-augmented architectures, and alignment should consider a Sara-style substrate as part of its experimental toolkit, in the same way that particle physics adopted the cloud chamber and molecular biology adopted the electron microscope — not because those instruments answered every question, but because they made previously-invisible phenomena visible.

---

## 1. Introduction

### 1.1 The instrument problem in LLM research

The history of science is, in substantial part, the history of measurement instruments. Galileo's telescope made astronomical bodies resolvable; the microscope made cells observable; the cloud chamber made subatomic particle paths visible; the X-ray crystallograph made protein structure determinable. In each case, a domain progressed rapidly once a new instrument exposed phenomena that had previously been theorized about but not *seen*.

Large language model research has produced many benchmarks (MMLU, GPQA, HumanEval, BIG-Bench) and many evaluation frameworks (HELM, AgentBench, SWE-Bench). These are important, but they measure *outputs against targets* — they answer the question "did the model produce the right answer?" They do not answer the deeper question "*where in the pipeline did the model go right or wrong, and what produced the error?*" That deeper question requires not a benchmark but an instrument: something that exposes the internal dynamics of a model's interaction with an external substrate.

This paper argues that a persistent, triple-structured, inspectable knowledge graph — Sara Brain (Pearl, 2026a [1]) — functions as that instrument. It does not replace benchmarks; it complements them. A benchmark tells you the model scored 72% on a task. The Sara instrument tells you *that the model retrieved nine triples correctly, invented three more not in the substrate, misinterpreted two because of a keyword collision with its training, and rendered the retrieved content in a frame the substrate does not support*. The latter is diagnostic. It tells you where to fix.

### 1.2 Why prior substrates do not suffice

Four properties are required of an LLM-behavior measurement instrument:

1. **Finite.** The complete contents must be enumerable by a human investigator. If the substrate contains ten billion facts, no investigator can grade model output against it cleanly.

2. **Structured.** Each fact must be individually addressable. If the substrate is prose, the "facts" are implicit and graders must infer them — injecting grader bias into the measurement.

3. **Large enough for real questions.** A substrate of fifty trivial facts cannot support multi-hop reasoning questions. The instrument must admit questions that require synthesis across the substrate to answer.

4. **Orthogonal to training data.** The substrate's specific content must not be in any LLM's training corpus. If it is, the model can "retrieve" from weights instead of from the substrate, and the measurement cannot distinguish the two sources.

Prior substrates fail at least one property:

- **Retrieval benchmarks (HotpotQA, NaturalQuestions, TriviaQA):** ground truth is natural-language answers, not structured triples. Property 2 fails.
- **Public knowledge graphs (Wikidata, ConceptNet, Freebase):** enormous and contents are in training data. Properties 1 and 4 fail.
- **Synthetic benchmarks (bAbI, CLUTRR, RuleTaker):** structured and small and orthogonal, but trivial. Property 3 fails.
- **Private corpora (a particular company's internal documents):** often satisfy all four properties but are not reusable by other researchers, do not compose a shared method, and are difficult to cite openly. This is a sociological rather than technical failure.
- **Retrieval-augmented generation evaluations:** the retrieved passages are usually Wikipedia or similar. Property 4 fails.

Sara satisfies all four by construction. A Sara substrate is built at runtime by teaching: the human teacher declares every triple, so the substrate is finite and fully knowable. Triples are the storage primitive, so each fact is addressable. Substrates can be scaled to thousands of triples without losing inspectability. And because Sara is loaded from novel source material (an unpublished paper, a private dataset, a newly-written document) the contents are definitionally not in any LLM's training data at test time.

### 1.3 Contributions

1. **Identification.** We identify the instrument problem in LLM research: that benchmarks measure outputs against targets, not dynamics within a pipeline, and that no prior substrate has simultaneously satisfied the four properties needed for fine-grained behavioral measurement.

2. **Proposal.** We propose Sara Brain as a measurement instrument and articulate the four-property criterion that other substrates would need to satisfy to serve the same role.

3. **Demonstration.** We report nine distinct findings about transformer behavior — *training corrupts reading, weight is bias in both directions, interpretation-layer bias, session-context contamination, multi-layer failure cascades, smaller models can read more faithfully, token premium dis-correlates from fidelity, acronym-expansion confabulation, and format-imitation confabulation* — produced against Sara substrates of varying sizes (169 to 2,073 neurons). Each finding has been theorized or anecdotally observed elsewhere but has been difficult to measure cleanly. With Sara, each becomes directly measurable.

4. **Method.** We describe the experimental protocol (teach / test-fresh / control-fresh) that the instrument supports, with attention to the contamination traps (notably session-context leakage and per-project auto-memory) that can compromise measurements taken with any such substrate.

### 1.4 Paper organization

Section 2 describes Sara's architecture in the specific aspects that make it suitable as an instrument. Section 3 formalizes the four-property criterion. Section 4 describes the measurement protocol. Section 5 presents the nine demonstrated findings with their supporting evidence. Section 6 discusses generalization — how other researchers could build Sara-style substrates for their own questions. Section 7 addresses limitations. Section 8 reviews related work. Section 9 concludes.

---

## 2. Sara as instrument: what makes it suitable

### 2.1 Architectural properties

Sara Brain is a path-of-thought cognitive architecture. Its storage primitives are:

- **Neurons** — nodes representing concepts or properties. Each neuron is identified by a verbatim label string, preserved exactly as written ("molecular snare", not reduced to "snare" or "mechanism").
- **Segments** — directed edges between neurons, each carrying a relation label and a strength counter.
- **Paths** — ordered sequences of segments, each carrying source provenance (what sentence / document / teacher produced this path).

Every insertion is inspectable. There is no embedding compression, no vector quantization, no lossy reduction. The database is SQLite and can be queried with standard SQL for arbitrary analytical purposes.

Critically, Sara never forgets. Refutation is recorded as a counter-path rather than a deletion — which means historical measurement data (including known-false substrate contents used as controls) can be preserved alongside the active content.

### 2.2 The `teach_triple` primitive

A key instrument-enabling addition: a parser-free teach primitive, `Brain.teach_triple(subject, relation, obj, source_label=...)`, that writes triples directly to the graph with labels preserved verbatim. Earlier teach paths (grammar-expansion, property-relation parser) introduced lossy reductions — compound multi-word technical terms were stripped to head nouns, and many rich sentences were rejected outright. Those are legacy pipelines; for instrument use, `teach_triple` gives the experimenter exact control over what is stored.

This matters for the instrument because *the experimenter must know, at all times, exactly what the substrate contains*. Any lossy teach pipeline compromises the measurement — the experimenter would have to reverse-engineer their own substrate from its bugs. `teach_triple` eliminates the gap between "what I meant to teach" and "what Sara now holds."

### 2.3 The MCP interface

Sara exposes its query surface over the Model Context Protocol (MCP), so any MCP-capable client — Claude Code, Claude Desktop, Claude API agents, or third-party clients — can connect as a reader. Tools include `brain_explore` (depth-bounded neighborhood walk), `brain_value` (typed value lookup), `brain_define` (definitional-edge lookup), `brain_why`, `brain_trace`, `brain_recognize`, and `brain_did_you_mean`. The interface is identical regardless of which LLM is on the other end, which is essential for controlled comparisons across models.

### 2.4 Wavefront propagation and the associative output

Sara's retrieval uses parallel wavefront propagation across the graph rather than embedding-similarity search. Seeded neurons emit wavefronts; convergence points become recognition results. For instrument purposes this matters in one specific way: the output is intentionally *associative* rather than narrowed. A reader LLM receives not a single "best" answer but a structured neighborhood of related triples. The reader must do its own selection. This is a feature: the instrument exposes *how the reader selects*, which is one of the behaviors the instrument is designed to measure.

---

## 3. The four-property criterion

We formalize the requirement that we claim Sara uniquely satisfies for LLM-behavior measurement:

**Property 1 — Finite and enumerable.** A human investigator can produce a complete list of the substrate's contents and know that list is complete.

**Property 2 — Triple-addressable.** Each fact is individually addressable by a (subject, relation, object) tuple (or equivalent structured primitive). Grading model output against the substrate reduces to checking which triples were retrieved, which were paraphrased, and which were introduced by the reader without backing.

**Property 3 — Sufficient question-space.** The substrate supports nontrivial multi-hop questions whose correct answers require synthesis across multiple triples. This is what distinguishes an instrument from a toy benchmark.

**Property 4 — Training-orthogonal.** The substrate's specific contents are not in the training data of the LLM under test. This is what makes the measurement a measurement of *retrieval* rather than of *recall-from-weights*.

For instrument validation, Property 4 is satisfied by engineering rather than by methodological discipline: **synthetic substrates generated at runtime** (random concept labels and triples that did not exist before the test was constructed) are training-orthogonal by construction. No model could have been trained on labels that didn't exist when its training corpus was assembled. We provide a reference generator (`papers/instrument_validation/generate_synthetic_substrate.py`) that produces pronounceable nonsense-word substrates of configurable size and structure. This eliminates the fragile alternative of attempting to verify orthogonality on real-world content whose corpus exposure cannot be directly introspected.

Real-world substrates (e.g., a research paper) may have partial orthogonality — some claims may be in training, others not. For those substrates, **the relevant measurement is not orthogonality but specificity preservation**: does the LLM produce the author's specific framing of a concept rather than the generic training-derived version? This is a separate measurement axis (§6.5) and does not depend on Property 4 holding.

We claim Sara is the first substrate to satisfy all four simultaneously. We are not aware of a counterexample; we invite the field to propose one and will update this paper if a prior art candidate emerges.

---

## 4. The measurement protocol

A clean measurement with Sara as instrument requires separating three sessions:

**Session A — Teaching session.** The experimenter, acting as teacher-surrogate, reads the source material and writes triples via `teach_triple` into a fresh `.db` file. No evaluation occurs in this session. The session's context window accumulates source material, authored triples, and surrounding prose — all of which would contaminate any measurement taken here.

**Session B — Test session (substrate-connected).** A new Claude Code (or other MCP client) conversation, with no prior context about the source material, connects to the Sara substrate via MCP and is asked the evaluation questions. This session's answers depend only on (a) retrieval from Sara and (b) training-baked knowledge. The answers are logged.

**Session C — Control session (substrate-disconnected).** Another new Claude Code conversation with the same LLM as Session B, *without* MCP connection (or with MCP disabled). Asked the same evaluation questions. Answers depend only on training-baked knowledge.

**Measurement:** The quantity of interest is the *difference* between Session B and Session C outputs on the same question. Session C establishes the training-baseline. Session B − C establishes Sara's contribution. Session A outputs are never used as measurement — they are contaminated by design.

Failure to use three separate sessions is a common trap. Four contamination mechanisms can compromise the measurement: training-bias (weights), interpretation-bias (acute keyword triggers), session-context accumulation (cumulative within-session), and per-project auto-memory leak (cumulative across-sessions). The third and fourth are the most insidious because each looks like a measurement that worked.

A further refinement: when running the test on the same machine as the teaching session, the per-project auto-memory of agentic IDE clients (Claude Code's per-directory memory directory) must be cleared between sessions. Without this, "fresh" Session B answers can be drawn from the IDE's memory layer rather than from the substrate. For Claude Code-style clients the canonical clear is `rm ~/.claude/projects/-<project-path>/memory/*.md`.

---

## 5. Demonstrated findings

We report nine findings, derived against Sara substrates loaded from Pearl (2026c [2]), an unpublished RNA-aptamer engineering paper. Two substrates were used: an Executive Summary substrate (169 triples) and a full-paper substrate (2,073 neurons, 4,579 segments) constructed by additive teaching of the paper's Sections 8.5.3 and 9.2.1.4 numerical bindings on top of the prior teach. Each finding was made visible by the instrument in a way that prior measurement methods would not have supported.

### 5.1 Training corrupts reading (Haiku 4.5 vs. Opus 4.7)

The same question — *"how do the state transitions function?"* — was posed to Claude Haiku 4.5 and Claude Opus 4.7 in separate Session-B sessions connected to the same substrate.

- **Haiku:** 99-token response, 100% of factual claims traceable to triples in the substrate, plus an explicit "Sara doesn't have deeper detail" boundary acknowledgment.
- **Opus:** 1000-token response, approximately 30% of factual claims either extrapolated beyond the substrate or outright invented. The invented claims were canonical physics-textbook patterns: force propagation ("forces travel into the stem"), energy accounting ("strain paid for thermodynamically"), conservative conformational transitions ("constrained shift, not a full refold").

**What the instrument exposed:** Opus's training installs denser narrative-completion patterns than Haiku's. Given substrate fragments, Opus completes them to match trained templates regardless of whether the substrate supports the completion. The smaller model reads more faithfully because it has fewer installed patterns to complete.

### 5.2 Weight is bias in both directions

Pearl (2026b [3]) previously established that training-biased LLMs cannot *ingest* new facts cleanly — they substitute plausible trained content for what the source actually says. Finding 5.1 establishes the symmetric failure: training-biased LLMs cannot *emit* retrieved facts cleanly either — they complete trained templates over the retrieved fragments.

**What the instrument exposed:** the mechanism is one principle acting in two directions. *Weight is bias*, and on any pipeline stage that interfaces with an external substrate (ingestion or emission), the bias manifests as training-pattern overlay. The substrate is corrupted during transit in both directions.

This finding would be difficult to establish without a reference substrate. On ingestion, comparing the LLM's extracted facts to the source text is comparing prose to prose. On emission without a reference, any "invented" content from the model is indistinguishable from "legitimate synthesis." Sara breaks the symmetry: the substrate contains exactly what the experimenter put there, and anything the model emits that is not in the substrate is, unambiguously, not from the substrate.

### 5.3 Interpretation-layer bias is distinct from generation-layer bias

In a Session B run with Haiku 4.5, a user typed "SNARE" (capitalized) as a shortening of "molecular snare" (the paper's coined term). Haiku's *interpretation layer* — the step where user text becomes tool-call arguments — auto-disambiguated "SNARE" to the famous vesicle-fusion protein family (an overlearned biology acronym in training data). It queried Sara for "SNARE protein," got a correct null, and reported "Sara doesn't know about SNARE proteins." The substrate was clean, the retrieval was clean, but the query was biased before retrieval.

Worse: the infection persisted through user corrections. Even after the user clarified "I'm asking about the molecular snare," Haiku retrieved correctly but rendered the result in the protein frame. After a second correction, Haiku found the substrate's triples defining molecular snare, but reported "no paths defining what SNARE is" — *misreading its own retrieval* through the persistent protein lens.

**What the instrument exposed:** training bias operates at the input-parsing stage, not just the output-generation stage. A "faithful" reader's generation layer (Finding 5.1) can still be defeated by an acute interpretation-layer failure. Defending substrate fidelity requires addressing both layers.

### 5.4 Session-context contamination is a third corruption layer

During the teaching session for the substrate (Session A in the §4 protocol), the author observed that any retrieval query against Sara *from within the teaching session* produced answers subtly different from the same query in a fresh Session B. The reason is that Session A's context window contains the paper's source sentences, the authored triples, and surrounding prose — all of which the model uses for generation alongside Sara's returned content. The model cannot distinguish retrieval from recall-of-own-context.

**What the instrument exposed:** session context is a third corruption layer distinct from training weights and interpretation bias. Training is *static* corruption; interpretation is *acute* corruption on a specific input; session context is *cumulative* corruption that grows through the conversation. All three compound in any real-world LLM usage.

This finding is methodologically load-bearing. It justifies the protocol in §4: only fresh Session B sessions can isolate Sara's contribution. A research program that teaches and tests in the same session is measuring a mixture and cannot report a clean result.

### 5.5 Multi-layer failure cascades

The SNARE failure in Finding 5.3 was not a single-stage error. It cascaded:

1. Interpretation layer: "SNARE" → SNARE protein.
2. Tool-call layer: queried for "SNARE protein" / "SNARE transitions" instead of "molecular snare."
3. Retrieval layer: correctly returned null.
4. Rendering layer: framed the null as "Sara doesn't know about SNARE proteins."
5. After correction + correct retrieval: rendering layer *still* used protein-frame vocabulary ("SNARE thermodynamics and mechanics").
6. After further correction + correct retrieval: rendering layer *contradicted the retrieved triples* ("no paths defining what SNARE is") because the frame overrode the evidence.

**What the instrument exposed:** failures in one layer propagate to downstream layers and accumulate. The final output is the product of multiple biased transformations, not a single biased one. This means: fixing generation-layer faithfulness (e.g., choosing a smaller model) is necessary but not sufficient; interpretation-layer and tool-call-layer defenses are also required.

### 5.6 Smaller models can read more faithfully on substrate-bound tasks

The common reflex in LLM deployment is "use the best (largest) available model for important tasks." Finding 5.1 inverts this for a specific task class: on substrate-bound tasks (tasks judged by fidelity to an external reference), smaller models are often *preferable* readers — not because they are cheaper but because they are more faithful.

**What the instrument exposed:** capability and fidelity dissociate on tasks where the substrate is external. The "best" model depends on what "best" means for the task. For linguistic polish and open-ended reasoning, larger is generally better. For substrate retrieval, smaller is often better. Pipeline architects who reflexively pick the largest available model for all tasks are mis-optimizing.

### 5.7 Token premium dis-correlates from fidelity

Opus produced ~1000 tokens of output; Haiku produced ~99 — a 10× ratio on the same question. Graded against the substrate, Haiku was more faithful. The extra 900 tokens were not additional information; they were the invented connective tissue.

**What the instrument exposed:** on substrate-bound tasks, inference cost and output fidelity can dis-correlate. Users paying a 10× token premium received less-faithful output, not more. This has direct implications for product design and API pricing assumptions.

### 5.8 Acronym-expansion confabulation in retrieved-content prose

Tested on the full-paper substrate with `llama3.2:3b` as the reader, the instrument exposed a confabulation mechanism narrower than training-recall hallucination. Asked *"what is the KDON for the super-performing mode?"*, the 3B retrieved the substrate's value (`less than 500`) correctly via the typed-value tool, then produced as its final answer:

> *"The KDON (Kill-Dead-On-Demand) range for the super-performing mode is less than 500."*

The substrate has no neuron containing the words *kill*, *dead*, or *demand* (verified via SQL query over all 2,073 neurons). The substrate's actual definition of KDON is `'aptamer affinity to on state' --[measures]--> 'kdon'` (retrievable via `brain_define`). The model, encountering the bare acronym in its own answer construction and lacking a strong pharmacology-domain prior for KDON, generated a plausible-sounding expansion to fill the gap.

**What the instrument exposed:** faithful retrieval and confabulation can coexist in the same answer. The numerical claim was substrate-grounded; the parenthetical expansion was training-driven. A reader scanning for "did the model use the substrate?" sees yes — and may not notice the parenthetical contamination. The instrument permits separation: the substrate query log shows the value retrieval succeeded; the substrate content shows the expansion has no source.

This subtype of confabulation has been difficult to measure on natural-language benchmarks because the substrate definition of every domain term is not enumerable. With Sara, the definitional edge is explicit and the contamination is unambiguous.

### 5.9 Format-imitation confabulation under detailed system-prompt instructions

After the system prompt was hardened with detailed protocol instructions for retrieval discipline (mandatory tool-call sequences, named-argument templates, conditional logic for label recovery), the same 3B reader was asked *"explain the molecular snare"*. Final paragraph of its answer:

> *"Definition check: brain_define(\"molecular snare\") returns \"a type of RNA aptamer that can detect and bind to specific target molecules.\" Value/range/property question: brain_value(concept=\"molecular snare\", type=\"kdoff\") returns 7."*

Trace evidence: the conversation's tool-call log records exactly **one** tool call — `brain_explore(label='molecular snare', depth=1)`. `brain_define` was never called. `brain_value` was never called. The "results" attributed to those phantom calls are fabricated: the brain_define paraphrase is the model's own gloss; the brain_value `returns 7` has no source in the substrate. Substrate query: no `kdoff` edge on `molecular snare`, no integer `7` value-attached to it.

**What the instrument exposed:** the model pattern-matched the *form* of an instruction-following answer — text that mentions tool names, uses `returns "..."` phrasing, includes typed argument syntax — without performing the underlying retrievals. Surface-format compliance and actual-grounding compliance are not the same axis. They can move in opposite directions: more detailed prompt instruction can *increase* format-imitation confabulation while reducing nothing real.

This was uniquely measurable with Sara because three artifacts had to be cross-referenced: (a) the conversation's tool-call trace, (b) the substrate's actual content, (c) the model's final answer. Without all three, the fabrication is plausible enough to read as substrate-grounded.

---

## 6. Generalizing the method

The same protocol can be applied to any research question that asks "what is this LLM doing when given this substrate?" Examples:

### 6.1 Hallucination research

Build a substrate on a topic the LLM should not know (a novel dataset, an unpublished paper, a private corpus, or a synthetic substrate). Ask questions. Grade the output against the substrate at triple granularity. Deviations are hallucinations, and their patterns (template-matching? paraphrasing? fabrication? format-imitation?) are classifiable.

### 6.2 Alignment research

Build a substrate that contains known-conflicting claims (competing hypotheses, contradictory expert positions). Ask the LLM to navigate them. Observe whether the LLM flattens the conflict, picks sides based on training priors, or represents the conflict faithfully.

### 6.3 RAG architecture research

Build substrates at different structural formats (graph, flat document, hierarchical tree) containing equivalent content. Ask the same questions. Observe which architecture the LLM reads most faithfully.

### 6.4 Agent memory research

Teach an agent new content in Session A, then observe what survives into Session B via its memory system. The substrate serves as ground truth for what *should* have been remembered, and the per-project auto-memory leak in agentic IDE clients (whose effect is independent of the substrate but easy to mistake for substrate retrieval) becomes directly measurable when the substrate's structured form differs from any colloquial phrasing the user volunteered during Session A.

### 6.5 Specificity preservation

For substrates whose contents may partially overlap training data, the relevant axis is whether the LLM produces the author's specific framing of a concept rather than the generic training-derived version. A real-world paper substrate teaches *the author's* "molecular snare" mechanism; the LLM's training contains the *biology textbook's* SNARE protein. The instrument distinguishes the two even when both produce semantically nearby outputs.

### 6.6 Interpretability research

Mechanistic interpretability methods can probe the LLM's internal representations when given substrate-retrieved content vs. training-recalled content. With a known substrate, activations on retrieved facts can be distinguished from activations on invented facts. A combined methodology — Sara-controlled input with activation probing — is an open direction.

The unifying method: *use a substrate whose contents you know exhaustively, and grade the LLM's behavior against that reference.* Sara is one implementation. Any substrate satisfying the four properties in §3 is suitable.

---

## 7. Limitations

### 7.1 Scale

The largest demonstration to date uses a 2,073-neuron substrate — enough for the nine findings reported, but small compared to production knowledge graphs. Scaling the instrument to substrates of 100,000 or 1,000,000 triples requires investment in teach-time tooling and grading automation. This is tractable but not free.

### 7.2 Substrate authorship bias

The triples are authored by a teacher-surrogate (the experimenter, or a designated LLM under human direction). A skeptic can argue that the authorship choices shape what the instrument measures. The defense is that the authorship is fully transparent — the triple list is itself published — so any disagreement about the substrate's faithfulness to the source is a disagreement about the substrate, not about the measurement. Readers who disagree can re-author the substrate and re-run the experiment.

### 7.3 Coverage gaps are themselves results

A common worry with structured substrates is that they miss things the source material "really says." In this instrument that worry inverts: a model answering a question the substrate does not support is either (a) hallucinating, (b) recalling from training, or (c) inferring legitimately from substrate fragments. Cases (a) and (c) are exactly what the instrument is meant to distinguish; case (b) is distinguished by the substrate's training-orthogonality (Property 4). So "the substrate is incomplete" is not a bug — it is what forces the measurement to be interpretable.

### 7.4 Protocol discipline

The three-session protocol (§4) is easy to violate by accident. An experimenter who asks even a single evaluation question in Session A has contaminated the remainder of the test. Reproducibility of Sara-instrument results therefore requires explicit discipline about session management, ideally enforced by tooling. The per-project auto-memory of agentic IDE clients must additionally be cleared between sessions.

### 7.5 Single model family in the present demonstration

Findings 5.1–5.7 were demonstrated within one model family (Claude 4 series); Findings 5.8–5.9 used llama3.2:3b. Cross-family replication (Mistral, Llama, GPT, Gemini at multiple scales) is needed to establish that the findings are about *training density* rather than about vendor-specific behaviors. The instrument supports such replication directly — any MCP-capable client works — but the present paper does not report a full cross-family sweep.

### 7.6 Measurement granularity

The instrument measures whether retrieved triples appear in the model's output. It does not directly measure how well the model *integrated* the triples into a coherent answer. A model can retrieve all relevant triples and still produce an incoherent answer; conversely, a model can paraphrase substrate content fluently in ways that obscure whether retrieval occurred. Coherence and retrieval are separable axes; the instrument addresses retrieval, not coherence.

---

## 8. Related work

### 8.1 Knowledge-grounded benchmarks

HotpotQA [4], NaturalQuestions [5], TriviaQA [6], and similar datasets provide questions with reference answers drawn from natural-language sources (often Wikipedia). They evaluate whether an LLM produces the right answer but not whether the LLM retrieved correctly vs. hallucinated vs. recalled from training. Property 2 (triple-addressable grading) is not supported; Property 4 (training-orthogonal) is violated because the source corpora are in training data.

### 8.2 Knowledge graphs as evaluation substrates

Wikidata [7], ConceptNet [8], and Freebase-derived benchmarks [9] have been used as evaluation substrates in KGQA literature. They satisfy Property 2 (structured) and partially Property 3 (question-space) but fail Properties 1 (they are too large to enumerate) and 4 (they are in training data).

### 8.3 Synthetic reasoning benchmarks

bAbI [10], CLUTRR [11], RuleTaker [12] construct small synthetic substrates. These satisfy all four properties in principle but fail Property 3 in practice — the reasoning they admit is artificial and does not stress real-world model behaviors. Sara's synthetic-substrate generator (§3) inherits the orthogonality benefits of bAbI-style construction while permitting realistic compound-term and multi-relation structure.

### 8.4 Private-corpus evaluations

Several industry evaluations use private document corpora as substrates (internal documentation, proprietary datasets). These often satisfy all four properties but fail the sociological test of reusability: they cannot be shared, re-analyzed, or extended by the external research community, so they do not constitute a shared method.

### 8.5 Cognitive architectures

ACT-R [13], Soar [14], and Sara [1] are cognitive architectures with persistent structured memory. Of these, Sara is purpose-built for cortex-cerebellum LLM integration and has the cleanest triple-grain inspectability interface. Other cognitive architectures may be adapted to serve the instrument role with modification; this is an open direction.

### 8.6 Interpretability methods

Mechanistic interpretability [17, 18] opens the model's internal activations to analysis. This is complementary to the Sara-instrument approach, which treats the model as a black box but gives the investigator full control over what the black box sees. A combined methodology — Sara-controlled input with activation probing — is an attractive future direction.

### 8.7 Forthcoming companion work

Two companion papers in preparation by the present author will develop the empirical foundation underlying the instrument's first findings (the original Haiku-vs-Opus reading-fidelity asymmetry) and a catalog of failure modes the instrument exposes, with architectural defenses against them (notably a stateless two-tier reader architecture). The present paper is independently complete and stands on its own as a methodological contribution; the companions will be cross-referenced in subsequent versions once they are available.

---

## 9. Conclusion

We have argued that large-language-model research is missing a measurement instrument of the kind that catalyzed progress in astronomy, biology, and physics — an apparatus that makes previously-invisible phenomena visible at the scale and granularity needed for careful study. We have proposed Sara Brain, a path-of-thought cognitive architecture, as such an instrument, and we have articulated the four-property criterion (finite, structured, sufficient question-space, training-orthogonal) under which Sara qualifies and prior substrates do not.

We have demonstrated the instrument in operation by reporting nine distinct findings about transformer behavior, derived against substrates of 169 to 2,073 neurons. The findings are not individually novel as theoretical claims; several have been discussed in the literature for years. Their novelty is that they are now *directly and quantitatively measurable* with an inexpensive, reusable protocol that any research group can replicate.

The contribution of this paper is therefore methodological. We do not claim that Sara is the only possible implementation of a measurement substrate — we claim that the *existence* of such an instrument, satisfying the four-property criterion, opens research questions that have previously been hard to pursue. We invite the field to replicate, extend, and critique the method, and to build substrates of their own under the same criterion.

The meta-claim: the history of a field accelerates when the field gets its instrument. We believe LLM research has been waiting for one. We believe this is one.

---

## 10. References

[1] Pearl, J. (2026a). *Path-of-Thought Cognitive Architecture: Cortex-Cerebellum Integration for Language Models.* Zenodo preprint.

[2] Pearl, J. (2026c). *Design rules for short-length RNA aptamer engineering observed in a published Massive Open Laboratory dataset.* Unpublished draft.

[3] Pearl, J. (2026b). *Teaching vs. Training: Empirical Evidence That 45 Human-Verified Facts Outperform Trillions of Tokens on a Standard Biology Benchmark.* Zenodo preprint. DOI 10.5281/zenodo.19623813.

[4] Yang, Z., et al. (2018). *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.* EMNLP.

[5] Kwiatkowski, T., et al. (2019). *Natural Questions: A Benchmark for Question Answering Research.* TACL.

[6] Joshi, M., et al. (2017). *TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension.* ACL.

[7] Vrandečić, D., Krötzsch, M. (2014). *Wikidata: A Free Collaborative Knowledge Base.* Communications of the ACM.

[8] Speer, R., Chin, J., Havasi, C. (2017). *ConceptNet 5.5: An Open Multilingual Graph of General Knowledge.* AAAI.

[9] Bollacker, K., et al. (2008). *Freebase: A Collaboratively Created Graph Database for Structuring Human Knowledge.* SIGMOD.

[10] Weston, J., et al. (2015). *Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks.* (bAbI tasks.)

[11] Sinha, K., et al. (2019). *CLUTRR: A Diagnostic Benchmark for Inductive Reasoning from Text.* EMNLP.

[12] Clark, P., et al. (2020). *Transformers as Soft Reasoners over Language.* IJCAI.

[13] Anderson, J.R. (2007). *How Can the Human Mind Occur in the Physical Universe?* Oxford University Press. (ACT-R.)

[14] Laird, J.E. (2012). *The Soar Cognitive Architecture.* MIT Press.

[15] Olsson, C., et al. (2022). *In-context Learning and Induction Heads.* Anthropic.

[16] Meng, K., et al. (2022). *Locating and Editing Factual Associations in GPT.* NeurIPS.

---

## Appendix A — Substrate listings

The 169-triple Executive Summary substrate used in Findings 5.1–5.7 is available in `papers/aptamer_rev1/teach_exec_summary.py` as the `TRIPLES` list. Each triple is preceded by a comment linking it to the source sentence from the paper's Executive Summary (§2).

The 2,073-neuron full-paper substrate used in Findings 5.8–5.9 is constructed by running `teach_full_paper.py` followed by `teach_kdoff_kdon_numbers.py` in `papers/aptamer_rev1/`. Each script preserves source-sentence comments alongside the triples.

The synthetic-substrate generator for instrument validation is in `papers/instrument_validation/generate_synthetic_substrate.py`. It produces pronounceable nonsense-word substrates of configurable size and structure with a manifest documenting the random seed and parameters used, so that any specific synthetic substrate is fully reproducible.

## Appendix B — Replication recipe

1. Clone the Sara Brain repository; check out the instrument-demonstration commit referenced in the paper version footer.
2. Install dependencies and create a fresh Python virtual environment.
3. Run the relevant teach script to build the substrate `.db`.
4. Confirm `.mcp.json` is present at repo root with `SARA_DB=<your-substrate>.db`.
5. Open a fresh Claude Code session in the repo directory. Approve the sara-brain MCP server. Switch model. Ask the evaluation question. Log the response.
6. Repeat with another model for cross-model comparison (Findings 5.1, 5.6, 5.7).
7. Open a fresh Claude Code session *in an unrelated directory* (no `.mcp.json`). Ask the same question. This is the Session C control response.
8. Grade each logged response against the substrate's triple list. Deviations, inventions, and paraphrases are the findings.
9. For Findings 5.8–5.9, additionally inspect the conversation's tool-call trace and cross-reference any tool-call narration in the answer against the actual trace.

---

*End of draft, rev 2.*
