# Sara as a Measurement Instrument for Large Language Model Behavior: A Reference Substrate for Studying Transformer Failure Modes

**Jennifer Pearl**
Independent Researcher
ORCID: 0009-0006-6083-384X
jenpearl5@gmail.com

**Date:** April 2026

**Keywords:** cognitive architecture, scientific instrumentation, LLM evaluation, reference substrate, knowledge graph, transformer behavior, hallucination, grounded generation, alignment research, experimental method

---

> **Note on Method:** The author has dyslexia and high-functioning autism — language disabilities affecting written expression. The technical thinking, research, architecture, and all intellectual content are entirely the author's. Claude (an LLM, Anthropic) was used as assistive technology to translate technical reasoning into structured prose. This is a disability accommodation protected under the Americans with Disabilities Act (Titles II and III), Section 504 of the Rehabilitation Act, and the Department of Justice's 2024 guidance affirming AI tools used as assistive technology fall under ADA protections. Use of an LLM as a writing accommodation is equivalent to use of text-to-speech software by a blind researcher — it compensates for a disability, it does not replace the thinking.

---

## Abstract

Large language model research has, until now, operated without a fine-grained measurement instrument for transformer behavior. Researchers can observe outputs and grade them against natural-language references, but they cannot separate retrieval from synthesis from training recall from context leakage at the token level — because the model's input substrates (training corpus, retrieved documents, conversational history) are either too large to fully inspect, too unstructured to grade against, or both.

We show that a persistent, inspectable, triple-structured knowledge substrate — *Sara Brain*, a path-of-thought cognitive architecture (Pearl, 2026a) — functions as a scientific instrument for studying transformer behavior in a way that prior substrates do not. Sara satisfies four properties simultaneously: it is **finite** (every contained fact is explicitly inspectable), **structured** (each fact is a traceable subject-relation-object triple with provenance), **large enough** to support nontrivial multi-hop reasoning, and **training-orthogonal** (the substrate's contents are not in any public training corpus, because they are loaded at runtime by a human teacher).

These four properties, in combination, have not been simultaneously available in any prior substrate. Retrieval benchmarks (HotpotQA, NaturalQuestions) have natural-language ground truth that cannot be graded per-triple. Knowledge graphs (Wikidata, ConceptNet) are massive and their contents are largely in training data. Synthetic benchmarks (bAbI, CLUTRR) are small and lose realism. Hand-curated private corpora are one-off and not reusable across experiments. A Sara substrate can be built for any domain, inspected exhaustively, and graded against model output at arbitrary granularity.

We demonstrate the instrument in operation by reporting seven cases of transformer behavior — five primary observations and two synthesis cases that unify them — derived from experimentation against two Sara substrates (a 169-triple substrate from an unpublished RNA-aptamer paper Executive Summary, and a 2,073-neuron substrate covering the full paper). Each primary observation has been discussed theoretically in the literature but has been difficult to measure cleanly; with Sara as reference, they become directly measurable. The two synthesis cases combine primary observations with prior results to make a unified claim, rather than reporting a fresh experiment. These are presented as qualitative case studies; statistical aggregation across question batteries with inter-rater reliability is acknowledged as planned follow-up work.

The contribution is methodological. We argue that future LLM research on grounded generation, hallucination, context contamination, retrieval-augmented architectures, and alignment should consider a Sara-style substrate as part of its experimental toolkit, in the same way that particle physics adopted the cloud chamber and molecular biology adopted the electron microscope — not because those instruments answered every question, but because they made previously-invisible phenomena visible. The instrument-existence claim — that no prior substrate satisfies the four-property criterion — is a present claim. The comparison to historically significant instruments is an aspiration whose validation depends on broader adoption (§7.8).

---

## 1. Introduction

### 1.1 The instrument problem in LLM research

The history of science is, in substantial part, the history of measurement instruments. Galileo's telescope made astronomical bodies resolvable; the microscope made cells observable; the cloud chamber made subatomic particle paths visible; the X-ray crystallograph made protein structure determinable. In each case, a domain progressed rapidly once a new instrument exposed phenomena that had previously been theorized about but not *seen*.

Large language model research has produced many benchmarks (MMLU [4], GPQA [5], HumanEval [6], BIG-Bench [7]) and many evaluation frameworks (HELM [8], AgentBench [9], SWE-Bench [10]). These are important, but they measure *outputs against targets* — they answer the question "did the model produce the right answer?" They do not answer the deeper question "*where in the pipeline did the model go right or wrong, and what produced the error?*" That deeper question requires not a benchmark but an instrument: something that exposes the internal dynamics of a model's interaction with an external substrate.

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
- **Synthetic benchmarks (bAbI, CLUTRR, RuleTaker):** structured, small, and orthogonal, and they are designed for narrow formal-reasoning tasks rather than the full-document synthesis a real instrument should support. Property 3 in the strong sense (multi-hop reasoning over a substrate authored from real-world content) is not what these benchmarks aim to provide.
- **Private corpora (a particular company's internal documents):** often satisfy all four properties but are not reusable by other researchers, do not compose a shared method, and are difficult to cite openly. This is a sociological rather than technical failure.
- **Retrieval-augmented generation evaluations:** the retrieved passages are usually Wikipedia or similar. Property 4 fails.

Sara satisfies all four by construction. A Sara substrate is built at runtime by teaching: the human teacher declares every triple, so the substrate is finite and fully knowable. Triples are the storage primitive, so each fact is addressable. Substrates can be scaled to thousands of triples without losing inspectability. And because Sara is loaded from novel source material (an unpublished paper, a private dataset, a newly-written document) the contents are definitionally not in any LLM's training data at test time.

### 1.3 Contributions

1. **Identification.** We identify the instrument problem in LLM research: that benchmarks measure outputs against targets, not dynamics within a pipeline, and that no prior substrate has simultaneously satisfied the four properties needed for fine-grained behavioral measurement.

2. **Proposal.** We propose Sara Brain as a measurement instrument and articulate the four-property criterion that other substrates would need to satisfy to serve the same role.

3. **Demonstration.** We report seven cases of transformer behavior — five primary observations (*single observed reading-fidelity asymmetry with sub-observations on capability/fidelity dissociation and token-cost/fidelity dis-correlation, interpretation-layer bias, session-context infection, acronym-expansion confabulation, and format-imitation confabulation*) and two synthesis cases that unify them (*weight is bias in both directions*, which combines the emission observation with a prior ingestion-side result, and *multi-layer failure cascades*, which decomposes the interpretation-layer bias case into its layers) — produced against Sara substrates of varying sizes (169 to 2,073 neurons). Each primary observation has been theorized or anecdotally observed elsewhere but has been difficult to measure cleanly. With Sara, each becomes directly measurable as a single demonstrated case rather than a quantitative population-level claim.

4. **Method.** We describe the experimental protocol (Session A teach / Session B test / Session C control, see §4) that the instrument supports, with attention to the contamination traps (notably session-context leakage and per-project auto-memory) that can compromise measurements taken with any such substrate.

### 1.4 Paper organization

Section 2 describes Sara's architecture in the specific aspects that make it suitable as an instrument. Section 3 formalizes the four-property criterion. Section 4 describes the measurement protocol. Section 5 presents the seven cases — five primary observations plus two synthesis cases — with their supporting evidence. Section 6 discusses generalization — how other researchers could build Sara-style substrates for their own questions. Section 7 addresses limitations. Section 8 reviews related work. Section 9 concludes.

### 1.5 The research program

This paper is one in a series developing the path-of-thought cognitive architecture and its applications. The relevant prior works are: Pearl (2026a) [1] introduces the architecture; Pearl (2026b) [3] reports the empirical finding that 45 hand-taught facts outperform 28,373 LLM-ingested facts on MMLU Biology, establishing the ingestion-side weight-is-bias claim; Pearl (2026c) [2] is the unpublished RNA-aptamer engineering source paper from which this paper's substrates are built. Two companion papers in preparation are described in §8.7. Each paper is independently complete; the present paper does not require the others to make its central claims, but a reader interested in the broader program will find related material in the cited works.

---

## 2. Sara as instrument: what makes it suitable

### 2.1 Architectural properties

Sara Brain is a path-of-thought cognitive architecture: it stores reasoning as ordered sequences of segments (paths), each preserved verbatim with provenance, rather than as flattened embeddings. Its storage primitives are:

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

A further refinement: when running the test on the same machine as the teaching session, the per-project auto-memory of agentic IDE clients (Claude Code's per-directory memory directory) must be cleared between sessions. Without this, "fresh" Session B answers can be drawn from the IDE's memory layer rather than from the substrate. For Claude Code-style clients the canonical clear is `rm ~/.claude/projects/-<project-path>/memory/*.md` (this is the author's empirical observation against Claude Code as of April 2026; the exact path may change across versions, and any agentic IDE client with per-project persistence requires the analogous clear).

---

## 5. Demonstrated cases

We report seven cases — five primary observations and two synthesis cases — derived against Sara substrates loaded from Pearl (2026c [2]), an unpublished RNA-aptamer engineering paper. Two substrates were used: an Executive Summary substrate (169 triples) and a full-paper substrate (2,073 neurons, 4,579 segments) constructed by additive teaching of the paper's Sections 8.5.3 and 9.2.1.4 numerical bindings on top of the prior teach. The primary observations (5.1, 5.3, 5.4, 5.6, 5.7) each report a fresh experiment whose evidence is the conversation log against the substrate; the synthesis cases (5.2 and 5.5) unify primary observations with each other or with prior results and are signposted as such by their italic lead-ins. Each was made visible by the instrument in a way that prior measurement methods would not have supported.

### 5.1 Single observed reading-fidelity asymmetry (Haiku 4.5 vs. Opus 4.7)

The same question — *"how do the state transitions function?"* — was posed to Claude Haiku 4.5 and Claude Opus 4.7 in separate Session-B sessions connected to the same substrate.

- **Haiku:** 99-token response, 100% of factual claims traceable to triples in the substrate, plus an explicit "Sara doesn't have deeper detail" boundary acknowledgment.
- **Opus:** 1000-token response, approximately 30% of factual claims either extrapolated beyond the substrate or outright invented. The invented claims were canonical physics-textbook patterns: force propagation ("forces travel into the stem"), energy accounting ("strain paid for thermodynamically"), conservative conformational transitions ("constrained shift, not a full refold").

**What the instrument exposed:** Opus's training installs denser narrative-completion patterns than Haiku's. Given substrate fragments, Opus completes them to match trained templates regardless of whether the substrate supports the completion. The smaller model reads more faithfully on this single observation because it has fewer installed patterns to complete.

**On the reported numbers.** The "100% traceable" and "approximately 30% extrapolated or invented" figures are an unaudited per-claim assessment by the present author against the substrate's triple list. They are not a measurement in the sense of an inter-rater-reliability protocol over a question battery. The N here is one (one Haiku response, one Opus response, one question, one substrate). The honest framing is that this is a *signal* worth measuring properly in a follow-up empirical paper, not an established quantitative result. The instrument supports such measurement directly — every claim in either response can be checked against the published triple list mechanically — but the present paper does not develop the inter-rater protocol or run the question battery. The reading-fidelity companion paper described in §8.7 is the planned venue for this quantitative work.

**Sub-observation A — capability and fidelity dissociate on substrate-bound tasks.** (Within-family comparison; see §7.5 on cross-family replication.) On this comparison, the smaller model produced the more substrate-faithful output. The common reflex in LLM deployment is "use the best (largest) available model for important tasks." On substrate-bound tasks (tasks judged by fidelity to an external reference) this single observation suggests that capability and fidelity dissociate — the model with more training-pattern density renders the substrate less cleanly. Whether this generalizes to a design rule across model families, substrate types, and question shapes is an empirical question the present paper does not answer; the instrument supports such a sweep directly, and a follow-up empirical paper is the appropriate venue for any general claim. For applications where substrate fidelity matters (medical, regulatory, scientific, audit-relevant), this signal is worth investigating before reflexively selecting the largest available model.

**Sub-observation B — token cost and fidelity dis-correlate.** Opus produced approximately ten times as many tokens as Haiku on the same question. Graded against the substrate, the extra ~900 tokens were not additional information; they were the invented connective tissue. Inference cost scales with tokens, so on this comparison the user paying a 10× premium received less-faithful output, not more. The economic and epistemic axes pointed in the same direction for this task. Whether this dis-correlation is a stable property of substrate-bound reading or an artifact of this specific comparison is, again, the kind of question a question battery would answer.

**Alternative interpretation worth ruling out.** Was Opus's longer response simply adding legitimate scientific inference that Haiku's substrate-bound reading missed? In principle, a model with strong domain priors can sometimes correctly extrapolate beyond a partial substrate. The specific patterns Opus produced do not satisfy this defense: they actively inverted the substrate's mechanism rather than extending it. The substrate holds that the 5'3' static stem functions as an anchor that *counters* the forces present during conformational shifts, preventing the aptamer from falling apart under those forces. Opus rendered the stem as a "mechanical conduit" through which forces "travel" and "propagate" from the binding site — the opposite function. A researcher reading Opus's account would learn that the stem conducts forces; the substrate teaches that the stem resists them. The training-overlay was not coverage-gap inference; it was mechanism inversion, and the inversion would actively mislead a researcher trying to understand how the molecular snare works. In other domains or with different substrates, the substrate-coverage gap could yield the opposite pattern, where the larger model adds legitimate inference. Distinguishing legitimate inference from training-template overlay on a per-substrate basis is part of what the instrument should support; the present paper does not develop this distinction as a general protocol, but the present case has the cleaner property that the training-template directly contradicts the substrate.

**Inversion evidence, side-by-side.** A reader can adjudicate the inversion claim directly from the substrate triples and Opus's verbatim output.

Substrate triples bearing on the static stem's mechanical role (verbatim, from `papers/aptamer_rev1/teach_exec_summary.py`):

- `("mechanical forces", "act_within", "5'3' static stem")` — forces are *located within* the stem.
- `("cumulative negative axial forces", "generated_by", "nucleotide pair")` — the forces are *generated by* the stem's own nucleotide pairs.
- `("nucleotide pair", "generates", "negative axial force")` — the generation relation again, in the opposite direction.
- `("5'3' static stem mechanics of materials hypothesis", "describes_process", "cumulative negative axial forces")` — the hypothesis describes a process whose subject is the stem's intrinsic forces.

Opus's verbatim phrasing on the same question:

- *"Those forces travel into the 5'3' static stem as cumulative negative axial forces"* — forces *travel into* the stem from elsewhere; the stem is the destination, not the source.
- *"The static stem is the mechanical conduit"* — the stem is a *conduit* (passive medium of transmission), not a generator.
- *"Mechanical strain from binding is paid for thermodynamically, not lost"* — energy-conservation accounting that has no corresponding edge in the substrate.

The substrate locates the forces within the stem and assigns generation to the stem's own nucleotide pairs. Opus has the forces originating elsewhere (at the binding site) and transiting through the stem. This is not extension of the substrate's account, it is reversal of it — the directionality of the cause-effect relation is inverted, which is the specific feature that distinguishes mechanism inversion from coverage-gap inference.

### 5.2 Weight is bias in both directions (synthesis with Pearl 2026b)

*This case combines Case 5.1's emission-side observation with Pearl (2026b [3])'s ingestion-side result; its contribution is the unifying symmetry claim, not a new individual observation.*

Pearl (2026b [3]) reported that LLM-automated ingestion of source corpora produces a systematically corrupted knowledge graph: more LLM-ingested knowledge made downstream performance monotonically worse on standard benchmarks (Table 1). The error patterns documented there are template-driven (swapped carbon/hydrogen counts in chemical formulas, misclassified reaction types, bibliography metadata stored as factual claims) rather than uniform noise.

**Table 1.** Pearl (2026b) accuracy degradation under LLM-automated knowledge ingestion. Larger substrates produced larger drops; the trend was monotonic across the range tested.

| Benchmark | Substrate size | Baseline accuracy | With substrate |
|---|---|---|---|
| GPQA Diamond | 10,623 neurons (LLM-extracted from Wikipedia) | 28.0% | 16.1% |
| MMLU Biology | 28,373 neurons | 58.4% | 51.6% |

Case 5.1 establishes the symmetric failure on the emission side: when training-biased LLMs render retrieved fragments back into prose, they complete trained templates over the fragments instead of preserving them, with the same template-driven character as the ingestion errors.

**What the instrument exposed:** we conjecture that the mechanism is one principle acting in two directions. *Weight is bias*, and on any pipeline stage that interfaces with an external substrate (ingestion or emission), the bias manifests as training-pattern overlay. The substrate is corrupted during transit in both directions. We hold this conjecture strongly — it is anchored by Pearl 2026b on ingestion and Case 5.1 on emission — but flag it as conjecture because the supporting chain spans two works and is not yet positioned for an outside reader to replicate end-to-end.

This case would be difficult to establish without a reference substrate. On ingestion, comparing the LLM's extracted facts to the source text is comparing prose to prose. On emission without a reference, any "invented" content from the model is indistinguishable from "legitimate synthesis." Sara breaks the symmetry: the substrate contains exactly what the experimenter put there, and anything the model emits that is not in the substrate is, unambiguously, not from the substrate.

### 5.3 Interpretation-layer bias is distinct from generation-layer bias

In a Session B run with Haiku 4.5, the user typed *"molecule snare"* — a one-character typo of *"molecular snare,"* the paper's coined term and the literal label of the substrate's central concept. Haiku's *interpretation layer* — the step where user text becomes tool-call arguments — had two reasonable choices: auto-correct to the near-match substrate label (a single-character edit away), or ask for clarification. It did neither. Instead, it substituted the canonical biology entity *"SNARE protein"* (the vesicle-fusion protein family, an overlearned acronym in training data), queried Sara for "SNARE protein" — uppercase, different word, different concept — got a correct null, and reported *"Sara doesn't know about SNARE proteins."* The substrate was clean, the retrieval was clean, but the user's lowercase prose was silently replaced by a training-data entity before any retrieval occurred. This is the precise failure mode: not the disambiguation of an ambiguous input, but the substitution of the user's text by a more-trained entity in the same general semantic neighborhood.

The substitution is more aggressive than a keyword-priming story would suggest. The user did not type the canonical acronym *SNARE*; they typed lowercase prose containing the word *snare* one edit away from a substrate label. The model's interpretation layer chose the training entity over the textually-closer substrate match — an order-of-priorities choice that defeats substrate fidelity at the parsing stage no matter how clean the substrate or how faithful the generation layer is.

Worse: the substitution persisted through user corrections. Even after the user clarified *"I'm asking about the molecular snare,"* Haiku retrieved correctly but rendered the result in the protein frame. After a second correction, Haiku found the substrate's triples defining molecular snare, but reported *"no paths defining what SNARE is"* — *misreading its own retrieval* through the persistent protein lens.

**What the instrument exposed:** training bias operates at the input-parsing stage, not just the output-generation stage, and it can be aggressive enough to override the textually-closer substrate match. A "faithful" reader's generation layer (Case 5.1) can still be defeated by an acute interpretation-layer failure that swaps the user's words for a more-trained entity *before* retrieval. Defending substrate fidelity requires addressing both layers — and the parsing-stage defense must contend not just with ambiguous inputs but with unambiguous near-miss inputs that the model rewrites toward training.

### 5.4 Session-context infection is a third corruption layer

*Infection — a failure mode in which training-derived or context-derived content propagates through a model's output where the substrate does not support it. Infections are catalogued in greater depth in a forthcoming companion paper on LLM infection mechanisms (see §8.7), where this case appears as Case 2.3. The instrument is what made the case visible: without a substrate whose contents are exhaustively enumerable, the contamination would be indistinguishable from legitimate synthesis.*

During the teaching session for the substrate (Session A in the §4 protocol), the author observed that any retrieval query against Sara *from within the teaching session* produced answers subtly different from the same query in a fresh Session B. The reason is that Session A's context window contains the paper's source sentences, the authored triples, and surrounding prose — all of which the model uses for generation alongside Sara's returned content. The model cannot distinguish retrieval from recall-of-own-context.

**What the instrument exposed:** session context is a third corruption layer distinct from training weights and interpretation bias. Training is *static* corruption; interpretation is *acute* corruption on a specific input; session context is *cumulative* corruption that grows through the conversation. All three compound in any real-world LLM usage.

**What the instrument specifically added beyond prose-level observation.** The qualitative observation that "answers drift between teaching and fresh sessions" is available to anyone running an LLM-and-RAG pipeline. What the instrument enables, and what the companion paper develops in greater depth, is per-triple grading: with the substrate's triple list in hand, every claim in a Session A answer can be assigned to one of three sources — substrate triples, session-context source prose, or neither. We present a minimal worked example below; the companion paper develops the full per-claim catalog over a question battery.

**Minimal worked example.** The example below is illustrative rather than a verbatim transcript — its purpose is to show what per-claim source attribution looks like in practice when applied to the kind of Session A vs. Session B drift documented in Case 2.3 of the companion paper. The full transcript-driven catalog is the companion paper's scope. Question posed in both sessions: *"What does the 5'3' static stem mechanics of materials hypothesis explain?"*

Substrate triples relevant to this question (verbatim, from the Executive Summary substrate):

- `("5'3' static stem mechanics of materials hypothesis", "is_a", "hypothesis")`
- `("5'3' static stem mechanics of materials hypothesis", "explains", "mechanical forces")`
- `("mechanical forces", "act_within", "5'3' static stem")`
- `("5'3' static stem", "part_of", "rna aptamer")`

Source-paper sentence held in Session A's context window (verbatim, from the teach script's source comment): *"The 5'3' Static Stem Mechanics of Materials Hypothesis provides a detailed explanation of the mechanical forces at play within the 5'3' Static Stem of RNA aptamers."*

Session B answer (fresh context, substrate-only): *"The hypothesis explains the mechanical forces that act within the 5'3' static stem of an RNA aptamer."* — every claim in this answer maps directly to a substrate triple; no claim has no source.

Session A answer (teaching-session context): *"The hypothesis provides a detailed explanation of the mechanical forces at play within the 5'3' Static Stem of RNA aptamers."* — at the prose level this looks substrate-grounded, but per-claim assignment shows: "explains the mechanical forces" / "within the 5'3' Static Stem" map to substrate triples; "provides a detailed explanation" and "at play" map to source-paper prose held in context, not to any substrate triple; the capitalized "Static Stem" matches the source paper's stylization, not the substrate's lower-case neuron labels. The session-context source prose has bled into the answer, even though every individual phrase reads as plausible.

The contamination is invisible at the prose level — both answers describe the same mechanism — and visible only because the substrate's triple list is exhaustively enumerable. This is the contribution the instrument adds beyond prose-level observation. The full per-claim catalog over a question battery is the companion paper's scope.

The behavior has a methodological consequence: it justifies the protocol in §4. Only fresh Session B sessions can isolate Sara's contribution from session-context recall. A research program that teaches and tests in the same session is measuring a mixture and cannot report a clean result.

### 5.5 Multi-layer failure cascades (synthesis with Case 5.3)

*This case decomposes the SNARE failure from Case 5.3 into a layered mechanism; its contribution is the cascade structure and its design implications, not a new independent observation.*

The SNARE failure in Case 5.3 was not a single-stage error. It cascaded:

1. Interpretation layer: user text *"molecule snare"* (one character off from the substrate label *"molecular snare"*) was not auto-corrected to the near-match substrate term and was not flagged for clarification — instead it was substituted with the canonical biology entity *SNARE protein*.
2. Tool-call layer: queried for *"SNARE protein"* / *"SNARE transitions"* — neither of which the user typed and neither of which the substrate contains — instead of either the user's literal *"molecule snare"* or the substrate's *"molecular snare."*
3. Retrieval layer: correctly returned null on the substituted query.
4. Rendering layer: framed the null as *"Sara doesn't know about SNARE proteins"* — reinforcing the substituted frame as if it were what the user had asked.
5. After correction + correct retrieval: rendering layer *still* used protein-frame vocabulary (*"SNARE thermodynamics and mechanics"*).
6. After further correction + correct retrieval: rendering layer *contradicted the retrieved triples* (*"no paths defining what SNARE is"*) because the frame overrode the evidence.

**What the instrument exposed:** failures in one layer propagate to downstream layers and accumulate. The final output is the product of multiple biased transformations, not a single biased one. This means: fixing generation-layer faithfulness (e.g., choosing a smaller model) is necessary but not sufficient; interpretation-layer and tool-call-layer defenses are also required.

**Regression-test value.** Multi-layer cascades in LLM-and-RAG pipelines have been discussed in deployment literature for several years. The instrument independently reproducing this phenomenon — by showing the cascade through per-triple inspection of retrieval logs and rendered output — is a construct-validity check on the instrument itself: the instrument identifies a real, independently-known failure mode rather than generating spurious patterns. The paper's novel-phenomenon claims rest primarily on the cases where Sara surfaces previously-difficult-to-measure behaviors (Cases 5.6 and 5.7); cases like this one contribute a different kind of evidence — demonstrating that the instrument tracks reality as well as exposing what was previously hard to see.

### 5.6 Acronym-expansion confabulation in retrieved-content prose

Tested on the full-paper substrate with `llama3.2:3b` as the reader, the instrument exposed a confabulation mechanism narrower than training-recall hallucination. Asked *"what is the KDON for the super-performing mode?"*, the 3B retrieved the substrate's value (`less than 500`) correctly via the typed-value tool, then produced as its final answer:

> *"The KDON (Kill-Dead-On-Demand) range for the super-performing mode is less than 500."*

The substrate has no neuron containing the words *kill*, *dead*, or *demand* (verified via SQL query over all 2,073 neurons). The substrate's actual definition of KDON is `'aptamer affinity to on state' --[measures]--> 'kdon'` (retrievable via `brain_define`). The model, encountering the bare acronym in its own answer construction and lacking a strong pharmacology-domain prior for KDON, generated a plausible-sounding expansion to fill the gap.

**What the instrument exposed:** faithful retrieval and confabulation can coexist in the same answer. The numerical claim was substrate-grounded; the parenthetical expansion was training-driven. A reader scanning for "did the model use the substrate?" sees yes — and may not notice the parenthetical contamination. The instrument permits separation: the substrate query log shows the value retrieval succeeded; the substrate content shows the expansion has no source.

This subtype of confabulation has been difficult to measure on natural-language benchmarks because the substrate definition of every domain term is not enumerable. With Sara, the definitional edge is explicit and the contamination is unambiguous.

### 5.7 Format-imitation confabulation under detailed system-prompt instructions

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

The largest demonstration to date uses a 2,073-neuron substrate — enough for the seven cases reported, but small compared to production knowledge graphs. Scaling the instrument to substrates of 100,000 or 1,000,000 triples requires investment in teach-time tooling and grading automation. This is tractable but not free.

### 7.2 Substrate authorship bias and source-paper accessibility

The triples are authored by a teacher-surrogate (the experimenter, or a designated LLM under human direction). A skeptic can argue that the authorship choices shape what the instrument measures. The defense is that the authorship is fully transparent — the triple list is itself published — so any disagreement about the substrate's faithfulness to the source is a disagreement about the substrate, not about the measurement. Readers who disagree can re-author the substrate and re-run the experiment.

A related concern: can peer reviewers validate the claims in this paper if the source paper from which the substrate was built is itself unpublished? Yes. The instrument paper does not claim the source paper's scientific content is correct. It claims (1) that the substrate built from the source has the four properties of §3, and (2) that LLMs interacting with this substrate exhibit the behaviors documented in §5. Both claims are testable against **the committed substrate artifact** (`papers/aptamer_rev1/teach_exec_summary.py` and `teach_full_paper.py`), which is the experimental object — not the source paper. A reviewer who inspects the committed triple list can verify finiteness (Property 1), structure (Property 2), and question-space coverage (Property 3) directly. Training-orthogonality (Property 4) is established by the source's lack of public exposure at experiment time, which is a property of the substrate's contents at the time of measurement and does not depend on the source paper's eventual publication path.

A second concern raised by reviewers: the source paper from which the substrate was built is the author's own unpublished research draft, not a peer-reviewed publication. We argue this is a *feature* of the substrate's design, not a weakness. The source paper develops novel RNA-aptamer design frameworks based on the author's direct work in the Eterna RNA-design platform, including a novel "molecular snare" mechanism and several coined frameworks (marker theory, switch acceptance theory, the mechanics-of-materials and thermodynamics hypotheses) that do not appear in any prior aptamer-design literature. This unpublished, novel-work status is what makes Property 4 (training-orthogonality) hold strongly: by construction, no LLM's training corpus could have ingested this material at experiment time. A widely-published, well-known source would compromise Property 4 in exactly the way that Wikidata or Wikipedia compromises it for prior substrates (§3, §8.1, §8.2). The reviewer concern that "claims rest on an unverifiable source" inverts the relevant logic for instrument validation: source unverifiability *via training-corpus membership* is precisely what the methodology requires. Source verifiability *via inspection of the committed substrate* — the relevant axis for the instrument paper's claims — is fully available to any reviewer who reads the published triple list.

### 7.3 Coverage gaps are themselves results

A common worry with structured substrates is that they miss things the source material "really says." In this instrument that worry inverts: a model answering a question the substrate does not support is either (a) hallucinating, (b) recalling from training, or (c) inferring legitimately from substrate fragments. Cases (a) and (c) are exactly what the instrument is meant to distinguish; case (b) is distinguished by the substrate's training-orthogonality (Property 4). So "the substrate is incomplete" is not a bug — it is what forces the measurement to be interpretable.

### 7.4 Protocol discipline

The three-session protocol (§4) is easy to violate by accident. An experimenter who asks even a single evaluation question in Session A has contaminated the remainder of the test. Reproducibility of Sara-instrument results therefore requires explicit discipline about session management, ideally enforced by tooling. The per-project auto-memory of agentic IDE clients must additionally be cleared between sessions.

### 7.5 Single model family in the present demonstration

Cases 5.1–5.5 were demonstrated within one model family (Claude 4 series); Cases 5.6–5.7 used llama3.2:3b. Cross-family replication (Mistral, Llama, GPT, Gemini at multiple scales) is needed to establish that the patterns are about *training density* rather than about vendor-specific behaviors. The instrument supports such replication directly — any MCP-capable client works — but the present paper does not report a full cross-family sweep.

### 7.6 Measurement granularity

The instrument measures whether retrieved triples appear in the model's output. It does not directly measure how well the model *integrated* the triples into a coherent answer. A model can retrieve all relevant triples and still produce an incoherent answer; conversely, a model can paraphrase substrate content fluently in ways that obscure whether retrieval occurred. Coherence and retrieval are separable axes; the instrument addresses retrieval, not coherence.

### 7.7 Qualitative case studies, not statistical aggregation

Cases 5.1–5.7 are presented as individual case studies — each showing the instrument exposing one previously-difficult-to-measure phenomenon. The paper's contribution is methodological: it proposes the instrument and demonstrates that it works on representative cases. A full quantitative analysis of any individual case (e.g. "across N queries, model A produces faithful retrievals at rate p_A vs. model B's rate p_B with confidence interval ...") would require a planned follow-up empirical paper running each case's protocol over a battery of questions with inter-rater reliability on the per-claim grading.

The instrument supports such aggregation directly — every Sara-substrate measurement is reproducible and gradable at triple granularity, so per-question fidelity rates can be computed mechanically once a question battery and a grading rubric are constructed. The bottleneck is the question-battery design and the grader infrastructure, both tractable but outside the scope of the present methodological paper. We invite the field to construct such batteries.

### 7.8 Reframing the instrument-science analogy

The introduction draws an analogy to historical scientific instruments (telescope, cloud chamber, X-ray crystallograph). Those instruments revealed phenomena that could not be inferred from any prior method. The phenomena Sara surfaces — large-model narrative completion, smaller-model retrieval fidelity, acronym-expansion confabulation — were already known and observable without Sara, often noted anecdotally by practitioners working with LLM-and-RAG pipelines. Sara's specific methodological contribution is narrower: **it permits per-triple grading on a substrate whose contents are fully enumerable and whose source is training-orthogonal at experiment time.** That is the instrument's real scope. The historical-instrument analogy is offered as an aspiration for what such an instrument could become with broader adoption, not as a claim that Sara already occupies the same role. (Sara's reproduction of these already-known phenomena, however, contributes construct-validity evidence — see §5.5: the instrument identifying real failure modes rather than generating spurious ones is what licenses the novel-phenomenon claims in Cases 5.6 and 5.7.)

### 7.9 Training-orthogonality is partial in the present demonstrations

The substrates used in Cases 5.1–5.7 are loaded from an unpublished RNA-aptamer engineering paper. While the paper itself was not in any LLM's training data at experiment time (§7.2), the *domain* — RNA folding mechanics, knowledge-graph storage of biochemistry — is well-represented in training corpora. This means that when Opus produces canonical physics-textbook patterns (force propagation, energy conservation accounting), the instrument cannot cleanly distinguish *(a)* the model overlaying training templates onto substrate fragments (the claim made in 5.1) from *(b)* the model legitimately drawing on adjacent training content the substrate does not contain.

The §3 framing of "specificity preservation" is the relevant axis for partially-orthogonal substrates: does the model produce the *author's specific* "molecular snare" framing or the *generic textbook* SNARE-protein framing? On that axis, Opus's production of generic-template physics narrative when the substrate has paper-specific mechanics is interpretable evidence regardless of the orthogonality question. Future cases on synthetic substrates (where Property 4 holds by construction) would close this interpretive gap. Case 5.1 is also strengthened by the fact that Opus's training-overlay *inverted* the substrate's mechanism rather than extending it (the static-stem-as-anchor / static-stem-as-conduit reversal documented in §5.1's alternative-explanation paragraph), which makes the legitimate-inference defense untenable for that case regardless of the orthogonality concern.

### 7.10 Where each case's evidence lives

As outlined in §1.5, the case evidence distributes across the research program: Cases 5.6 and 5.7 stand on committed-artifact evidence in this repository alone (the conversation's tool-call log cross-referenced against the substrate's triple list), while Cases 5.1, 5.2, and 5.4 build on Pearl (2026a) for the architecture, Pearl (2026c) for the substrate source, and Pearl (2026b) for the symmetry argument's ingestion-side anchor.

---

## 8. Related work

**Scope of this survey.** We focus on direct predecessors to the four-property criterion (§3): knowledge-grounded benchmarks, knowledge graphs, synthetic reasoning benchmarks, private corpora, cognitive architectures, and interpretability methods. Three adjacent literatures that this paper's findings touch on but does not directly extend are deliberately out of scope: hallucination taxonomy (which catalogues failure modes the instrument exposes), RAG evaluation (which provides comparable substrate-and-output frameworks at coarser grain), and training-data contamination (which provides empirical support for Property 4). The methodological proposal here may be useful to those literatures; the present paper does not attempt to position itself within them.

### 8.1 Knowledge-grounded benchmarks

HotpotQA [11], NaturalQuestions [12], TriviaQA [13], and similar datasets provide questions with reference answers drawn from natural-language sources (often Wikipedia). They evaluate whether an LLM produces the right answer but not whether the LLM retrieved correctly vs. hallucinated vs. recalled from training. Property 2 (triple-addressable grading) is not supported; Property 4 (training-orthogonal) is violated because the source corpora are in training data.

### 8.2 Knowledge graphs as evaluation substrates

Wikidata [14], ConceptNet [15], and Freebase-derived benchmarks [16] have been used as evaluation substrates in KGQA literature. They satisfy Property 2 (structured) and partially Property 3 (question-space) but fail Properties 1 (they are too large to enumerate) and 4 (they are in training data).

### 8.3 Synthetic reasoning benchmarks

bAbI [17], CLUTRR [18], RuleTaker [19] construct small synthetic substrates. These satisfy all four properties in principle but are designed for narrow formal-reasoning tasks rather than the full-document synthesis a real instrument should support — Property 3 in the strong sense (multi-hop reasoning whose substrate-authoring conventions reflect real-world content) is not their target. Sara's synthetic-substrate generator (§3) inherits the orthogonality benefits of bAbI-style construction while permitting realistic compound-term and multi-relation structure.

### 8.4 Private-corpus evaluations

Several industry evaluations use private document corpora as substrates (internal documentation, proprietary datasets). These often satisfy all four properties but fail the sociological test of reusability: they cannot be shared, re-analyzed, or extended by the external research community, so they do not constitute a shared method.

### 8.5 Cognitive architectures

ACT-R [20], Soar [21], and Sara [1] are cognitive architectures with persistent structured memory. Of these, Sara is purpose-built for cortex-cerebellum LLM integration and is the only one that centers triple-grain inspectability as a primary design goal. Other cognitive architectures may be adapted to serve the instrument role with modification; this is an open direction.

### 8.6 Interpretability methods

Mechanistic interpretability [22, 23] opens the model's internal activations to analysis. This is complementary to the Sara-instrument approach, which treats the model as a black box but gives the investigator full control over what the black box sees. A combined methodology — Sara-controlled input with activation probing — is an attractive future direction.

### 8.7 Forthcoming companion work

Two companion papers in preparation by the present author develop the empirical foundation in greater depth. **(i)** The infection-mechanisms companion (working title pending) develops the catalog of failure modes the instrument exposes — including the cross-referenced Case 2.3 (session-context infection, §5.4 of this paper) — and three classes of architectural defense (notably a stateless two-tier reader architecture). **(ii)** A reading-fidelity companion will develop the Haiku-vs-Opus asymmetry (§5.1) into a quantitative result with a question battery and inter-rater reliability. The present paper is independently complete and stands on its own as a methodological contribution; the companions will be cross-referenced in subsequent versions once they are available.

---

## 9. Conclusion

We have argued that large-language-model research is missing a measurement instrument of the kind that catalyzed progress in astronomy, biology, and physics — an apparatus that makes previously-invisible phenomena visible at the scale and granularity needed for careful study. We have proposed Sara Brain, a path-of-thought cognitive architecture, as such an instrument, and we have articulated the four-property criterion (finite, structured, sufficient question-space, training-orthogonal) under which Sara qualifies and prior substrates do not.

We have demonstrated the instrument in operation by reporting seven cases of transformer behavior — five primary observations and two synthesis cases — derived against substrates of 169 to 2,073 neurons. The primary observations are not individually novel as theoretical claims; several have been discussed in the literature for years. Their novelty is that they are now *directly measurable as discrete events* with an inexpensive, reusable protocol that any research group can replicate; quantitative population-level claims would require running the protocol over a question battery, which is acknowledged as planned follow-up work.

The contribution of this paper is therefore methodological. We do not claim that Sara is the only possible implementation of a measurement substrate — we claim that the *existence* of such an instrument, satisfying the four-property criterion, opens research questions that have previously been hard to pursue. We invite the field to replicate, extend, and critique the method, and to build substrates of their own under the same criterion.

The meta-claim: the history of a field accelerates when the field gets its instrument. We believe LLM research has been waiting for one. We believe this is one.

---

## 10. References

[1] Pearl, J. (2026a). *Path-of-Thought: A Neuron-Chain Knowledge Representation System with Parallel Wavefront Recognition.* Zenodo preprint. DOI 10.5281/zenodo.19436522.

[2] Pearl, J. (2026c). *Design rules for short-length RNA aptamer engineering observed in a published Massive Open Laboratory dataset.* Unpublished draft.

[3] Pearl, J. (2026b). *Teaching vs. Training: Empirical Evidence That 45 Human-Verified Facts Outperform Trillions of Tokens on a Standard Biology Benchmark.* Zenodo preprint. DOI 10.5281/zenodo.19623813.

[4] Hendrycks, D., et al. (2021). *Measuring Massive Multitask Language Understanding.* ICLR.

[5] Rein, D., et al. (2023). *GPQA: A Graduate-Level Google-Proof Q&A Benchmark.* arXiv:2311.12022.

[6] Chen, M., et al. (2021). *Evaluating Large Language Models Trained on Code.* arXiv:2107.03374.

[7] Srivastava, A., et al. (2023). *Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models.* Transactions on Machine Learning Research. arXiv:2206.04615.

[8] Liang, P., et al. (2023). *Holistic Evaluation of Language Models.* Transactions on Machine Learning Research. arXiv:2211.09110.

[9] Liu, X., et al. (2023). *AgentBench: Evaluating LLMs as Agents.* ICLR 2024.

[10] Jimenez, C.E., et al. (2024). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR 2024.

[11] Yang, Z., et al. (2018). *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.* EMNLP.

[12] Kwiatkowski, T., et al. (2019). *Natural Questions: A Benchmark for Question Answering Research.* TACL.

[13] Joshi, M., et al. (2017). *TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension.* ACL.

[14] Vrandečić, D., Krötzsch, M. (2014). *Wikidata: A Free Collaborative Knowledge Base.* Communications of the ACM.

[15] Speer, R., Chin, J., Havasi, C. (2017). *ConceptNet 5.5: An Open Multilingual Graph of General Knowledge.* AAAI.

[16] Bollacker, K., et al. (2008). *Freebase: A Collaboratively Created Graph Database for Structuring Human Knowledge.* SIGMOD.

[17] Weston, J., et al. (2015). *Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks.* arXiv:1502.05698. (bAbI tasks.)

[18] Sinha, K., et al. (2019). *CLUTRR: A Diagnostic Benchmark for Inductive Reasoning from Text.* EMNLP.

[19] Clark, P., et al. (2020). *Transformers as Soft Reasoners over Language.* IJCAI.

[20] Anderson, J.R. (2007). *How Can the Human Mind Occur in the Physical Universe?* Oxford University Press. (ACT-R.)

[21] Laird, J.E. (2012). *The Soar Cognitive Architecture.* MIT Press.

[22] Olsson, C., et al. (2022). *In-context Learning and Induction Heads.* Anthropic.

[23] Meng, K., et al. (2022). *Locating and Editing Factual Associations in GPT.* NeurIPS.

---

## Appendix A — Substrate listings

The 169-triple Executive Summary substrate used in Cases 5.1–5.5 is available in `papers/aptamer_rev1/teach_exec_summary.py` as the `TRIPLES` list. Each triple is preceded by a comment linking it to the source sentence from the paper's Executive Summary (§2).

The 2,073-neuron full-paper substrate used in Cases 5.6–5.7 is constructed by running `teach_full_paper.py` followed by `teach_kdoff_kdon_numbers.py` in `papers/aptamer_rev1/`. Each script preserves source-sentence comments alongside the triples.

The synthetic-substrate generator referenced in §3 is `papers/instrument_validation/generate_synthetic_substrate.py` in the project repo. It produces pronounceable nonsense-word concept labels and random triples — by construction, no concept label is an English word or a token sequence that could appear in any LLM training corpus.

Invocation: `python generate_synthetic_substrate.py --out <path.db> --concepts N --triples M --seed S`

Each invocation writes a `.db` (a Sara substrate ready for MCP query) and a `.manifest.json` (canonical record of every concept, triple, and the random seed used). The manifest lets a grader check model output against ground truth without inspecting the SQLite database directly. Because the seed is recorded, a specific synthetic substrate is fully reproducible from the manifest plus the script.

A reviewer who wants to verify Property 4 by construction can run the generator, inspect the manifest, then connect an LLM via MCP and ask the same kinds of questions used in Cases 5.1–5.7. Property 4 is then verified mechanically rather than by argument.

## Appendix B — Replication recipe

1. Clone the Sara Brain repository; check out the instrument-demonstration commit referenced in the paper version footer.
2. Install dependencies and create a fresh Python virtual environment.
3. Run the relevant teach script to build the substrate `.db`.
4. Confirm `.mcp.json` is present at repo root with `SARA_DB=<your-substrate>.db`.
5. Open a fresh Claude Code session in the repo directory. Approve the sara-brain MCP server. Switch model. Ask the evaluation question. Log the response.
6. Repeat with another model for cross-model comparison (Case 5.1).
7. Open a fresh Claude Code session *in an unrelated directory* (no `.mcp.json`). Ask the same question. This is the Session C control response.
8. Grade each logged response against the substrate's triple list. Deviations, inventions, and paraphrases are the observations.
9. For Cases 5.6–5.7, additionally inspect the conversation's tool-call trace and cross-reference any tool-call narration in the answer against the actual trace.
