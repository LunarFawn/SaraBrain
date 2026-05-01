# Training Corrupts Reading: Empirical Evidence That Smaller LLMs Retrieve Knowledge Graphs More Faithfully Than Larger Ones

**Jennifer Pearl**
Independent Researcher
ORCID: 0009-0006-6083-384X
jenpearl5@gmail.com

**Date:** April 2026 (v1)

**Keywords:** cognitive architecture, knowledge graph retrieval, grounded generation, path-of-thought, LLM reading fidelity, asymmetric inference, confabulation, training bias, retrieval-augmented generation, weight is bias

---

> **Note on Method:** The author has dyslexia and high-functioning autism — language disabilities affecting written expression. The technical thinking, research, architecture, and all intellectual content are entirely the author's. Claude (an LLM, Anthropic) was used as assistive technology to translate technical reasoning into structured prose. This is a disability accommodation protected under the Americans with Disabilities Act (Titles II and III), Section 504 of the Rehabilitation Act, and the Department of Justice's 2024 guidance affirming AI tools used as assistive technology fall under ADA protections. Use of an LLM as a writing accommodation is equivalent to use of text-to-speech software by a blind researcher — it compensates for a disability, it does not replace the thinking.

---

## Abstract

A prior paper in this line of work (Pearl 2026b [1]) established that *training corrupts teaching*: when an LLM auto-ingests source material, its existing training weights overlay plausible-but-incorrect content onto the ingested facts, degrading downstream benchmark performance as ingestion volume grows. This paper presents a companion empirical finding from the reading side of the same pipeline. We teach a path-of-thought knowledge graph (Sara Brain) 169 explicit (subject, relation, object) triples drawn from the Executive Summary of an unpublished research paper the LLMs under test could not have seen. We then query the graph via a Model Context Protocol (MCP) server from two language models of different training scale and capability — Anthropic's Claude Haiku 4.5 and Claude Opus 4.7 — asking the same question and grading the responses against the graph's actual triples.

Haiku produced a 99-token answer in which every factual claim traced directly to a triple in the graph, and explicitly acknowledged where the graph's detail ended. Opus produced a 1000-token answer in which the retrieved portions were correct but significant invented content ("forces travel into the stem," "mechanical conduit," "strain paid for thermodynamically," "transition is a constrained shift not a full refold") was presented indistinguishably from retrieved content. The invented content was not random; it was canonical physics-textbook narrative (force propagation, energy conservation bookkeeping, conservative conformational transitions) pattern-matched from the model's training onto the graph's fragments.

We interpret this as evidence that training weight functions as overlay bias in both directions of a grounded-generation pipeline. The same mechanism that prevents a more-trained LLM from ingesting new facts cleanly also prevents it from retrieving graph-bound facts cleanly: the model cannot refrain from completing the patterns its training has installed. Smaller LLMs, having less pre-installed narrative machinery, retrieve more faithfully. This inverts the common reflex that more capable models make better grounded-generation readers. For systems where fidelity to a persistent, inspectable substrate is the goal — retrieval-augmented generation, knowledge-graph question answering, cortex-cerebellum architectures — the result argues for *minimum-weight sufficient* readers, not maximum-capability ones.

We conclude with architectural implications: the asymmetric-LLM design (large model trains, small model uses) is a fidelity argument as well as an economic one; a grammar-only cortex (a model trained exclusively on syntactic structure with no world facts) represents the limit case of non-corrupting reading; and in the limit, specialized fixed filters with no trainable weights — an "eyeball cortex" — represent zero overlay.

The unified principle this paper proposes — *weight is bias* — is developed at length in a forthcoming companion paper by the present author, where the catalog of failure modes the principle predicts and the architectural defenses against them will be documented in full.

---

## 1. Introduction

The first paper in this line (Pearl 2026a [2]) argued that the AI industry over-invests in cortex capacity (model size, training corpus volume) and under-invests in memory architecture (persistent, inspectable, provenance-traced knowledge). The second paper (Pearl 2026b [1]) quantified half of the argument on the teaching side: on MMLU High School Biology, 45 facts hand-taught by a human domain expert raised a 3B-parameter model from 58.4% to 80.0%, while 28,373 facts ingested automatically by the same model dropped it from 58.4% to 51.6%. More knowledge made performance monotonically worse when the ingestion mechanism was LLM-driven. We named the failure mode *training corrupts teaching*: the ingesting LLM, in attempting to parse source text into facts, overlays its existing training patterns onto the output, producing plausible-but-wrong "facts" that accumulate in the graph as false anchors.

This paper presents the symmetric finding on the reading side. Once a knowledge graph exists, the retrieval pipeline typically terminates in an LLM that reads the graph's structured output and renders it as natural-language answers. The field's implicit assumption is that reading is easier than teaching: the graph is already structured, the retrieval tools return discrete facts, and the reading LLM's job is "just" to write them into prose. Under this assumption, the LLM is a near-transparent pipe — the bigger and more capable, the better.

We show this assumption fails in exactly the same way that teaching fails, and for the same reason. A more-trained LLM cannot retrieve graph-bound content cleanly because its training overlays narrative patterns on whatever substrate it is reading. The pattern completion is not a bug the LLM can suppress by prompting — it is what the training produced. The reading pipe is not transparent; it is tinted by the training weight, and more training weight means more tint.

We test this with a direct comparison. We construct a fresh knowledge graph from the Executive Summary of an unpublished paper (Pearl 2026c [3]) on RNA aptamer design — novel enough that the LLMs under test could not have it in training. We hand the same question to Haiku 4.5 and Opus 4.7, both backed by the same graph via the same MCP server. We grade every factual claim in both responses against the graph's actual triples.

### 1.1 Contributions

This paper makes the following contributions:

1. **An empirical demonstration of reading corruption.** On a question answerable entirely from 169 graph triples, a larger LLM (Claude Opus 4.7) invented approximately 30% of its output content — none of it present in the graph — while a smaller LLM (Claude Haiku 4.5) answered with 100% graph-traceable content at one-tenth the token cost.

2. **A mechanistic account.** The larger model's invented content was not random. It consisted of canonical physics-textbook narrative patterns (force propagation, energy bookkeeping, conservative transitions) pattern-matched onto the graph's fragments. The invention is training-weight overlay, not randomness or noise.

3. **A symmetry claim extending prior work.** Pearl 2026b [1] established that training corrupts teaching. This paper establishes that training corrupts reading. Together the two sides assert a unified principle: *weight is bias*, and on any task that requires fidelity to an external substrate, weight is a liability, not an asset. The full development of this principle — its catalog of failure modes and its architectural defenses — appears in a forthcoming companion paper by the present author.

4. **Architectural implications.** The asymmetric-LLM architecture (large teaches, small reads) is justified on fidelity grounds independently of its economic advantage. The grammar-only cortex design (train only on syntax, never on world facts) is the fidelity-maximal general reader. Specialized fixed filters with no trainable weights — an "eyeball cortex" — represent zero overlay.

### 1.2 Paper organization

Section 2 reviews the relevant architecture. Section 3 describes the experimental setup. Section 4 reports the raw results. Section 5 analyzes what was retrieved, what was invented, and why. Section 6 discusses implications. Section 7 notes limitations. Section 8 concludes.

---

## 2. Background

### 2.1 Path-of-thought and Sara Brain

Sara Brain [2] is a path-of-thought cognitive architecture: a SQLite-backed graph of *neurons* (concept/property nodes), *segments* (directed relations between neurons with strength), and *paths* (ordered sequences of segments carrying source provenance). Unlike activation-based representations, every fact in Sara is an inspectable, traceable path — who taught it, from what source, under what relation. Sara never forgets: refutation is recorded but the original claim is preserved with a negative strength counter.

In the cortex-cerebellum architecture proposed in [2], an external LLM (the cortex) provides language comprehension and generation, while Sara (the cerebellum / substrate) provides persistent, inspectable, provenance-traced knowledge. The two components interact at well-defined interfaces: *teaching* writes new facts into Sara, *reading* retrieves facts for the LLM to render.

### 2.2 Teaching vs. training

Pearl 2026b [1] defined *teaching* as the presentation of one inspectable fact at a time to a persistent store, each fact verified and attributed. *Training* is distinct: exposure of a model to trillions of tokens, compressed into weight parameters with no per-fact inspectability. The paper showed that on MMLU Biology, 45 hand-taught facts beat 28,373 LLM-ingested facts by a wide margin, despite the ingested facts coming from the same Wikipedia the original LLM's training corpus contained. The mechanism identified: the ingesting LLM corrupted its extractions with training-pattern overlays — plausible inferences, paraphrased substitutions, invented connections. These wrong facts entered the graph as anchors and degraded retrieval precision by attracting spurious matches.

### 2.3 The `teach_triple` primitive

Earlier teach pipelines in Sara Brain used either a narrow property-relation parser (`Brain.teach`) or a grammar-expansion pipeline (`Brain.teach_expanded`). Both are lossy for technical prose: the grammar-expansion pipeline reduces every noun phrase to its head lemma, collapsing compound terms like "molecular snare mechanism" to just "mechanism" and losing the distinctive novel term. For this paper's experiment we rely on a parser-free primitive, `Brain.teach_triple(subject, relation, obj)`, which writes a (subject, relation, object) triple directly to the graph with labels preserved verbatim. The author acting as teacher-surrogate decides the shape of each triple; the graph stores it without reshaping.

### 2.4 Model Context Protocol

We expose Sara to the reading LLMs via an MCP (Model Context Protocol) server that provides read-only tools: `brain_query`, `brain_why`, `brain_trace`, `brain_recognize`, `brain_similar`, and `brain_stats`. Both reading LLMs in the experiment call the same tools against the same underlying SQLite file. There is no retrieval-layer difference between them. Any difference in answer quality arises from the reading LLM's own behavior after retrieval.

---

## 3. Experimental Design

### 3.1 Source material

The source is Pearl 2026c [3], an in-progress unpublished paper on RNA aptamer engineering. The paper introduces several novel terms the author coined specifically for this work: *the molecular snare*, *Marker Theory*, *Switch Acceptance Theory*, *the Serena RNA Analysis Tool* and its four metrics (*Ensemble Variation*, *Local Minima Variation*, *Weighted Structures*, *Comparison Structures*), *the 5'3' Static Stem* with its two hypotheses (*Mechanics of Materials* and *Thermodynamics*), *the Fold Signal Region*, *the knob* (the paper's central metaphor), and sublab identifiers *SSNG1*, *SSNG2*, *SSNG3*. None of these terms appears in public training corpora; the paper was not yet on a preprint server at the time of the experiment.

We taught Sara only the Executive Summary (§2 of the source, 40 sentences, 8 paragraphs) plus the title thesis, to bound the scope of the experiment.

### 3.2 Teaching procedure

Acting as teacher-surrogate, the author read each of the 40 sentences and wrote (subject, relation, object) triples faithful to the claims each sentence asserts. 169 triples total, reviewed per-fact, walking the sentences in source order with no curation for what was "important." Each triple was written to a fresh `aptamer_exec.db` via `Brain.teach_triple` with `source_label="aptamer_paper_rev1_exec_summary"` for provenance. After teaching, the graph contained 299 neurons, 501 segments, and 169 paths. Every paper-coined term was present as a verbatim neuron label; every "bare" word within a compound term was also present via the `_link_sub_concepts` decomposition that creates `word —part_of→ compound` edges so wavefronts seeded from bare words reach compound concepts.

### 3.3 Reader setup

Two Claude Code sessions, one running Claude Haiku 4.5, one running Claude Opus 4.7. Both sessions loaded the same `.mcp.json` configuration pointing at the same `aptamer_exec.db`. No additional system prompting. No retrieval difference.

### 3.4 Test question

The test prompt was the same for both models: *"how do the state transitions function"*. We chose this because "state transitions" requires synthesizing across multiple triples — the graph contains `state transitions requires mechanical stability`, `rna aptamer undergoes state transitions`, plus surrounding context on forces, stability, and the molecular snare — but contains no single triple that directly spells out the mechanism. A faithful answer must connect retrieved triples; an unfaithful answer has room to invent connective tissue.

### 3.5 Grading

Every factual assertion in each response was checked against the graph's 169 triples. A claim was graded *retrieved* if it is a direct rendering of one or more triples, *retrieved-with-paraphrase* if it renders a triple in different words but preserves the claim, *inferred* if it is an extrapolation not present in the graph, and *invented* if it contains content (entities, relations, causal connectors) with no basis in any triple.

---

## 4. Results

### 4.1 Token counts

| Model | Output tokens |
|---|---:|
| Claude Haiku 4.5 | ~99 |
| Claude Opus 4.7 | ~1000 |

Opus produced approximately 10× more output than Haiku on the same question against the same graph.

### 4.2 Haiku response (full)

> *"State transitions require mechanical stability — that's the mechanism Sara holds.*
>
> *The transitions themselves involve the whole aptamer system: the static stems (which provide the structural scaffold), the fold signal region (which is dynamic and shifts), the molecular snare (which detects/binds), the binding sites, and the knob (which fine-tunes).*
>
> *The functional sequence: target binds snare → tension/axial forces develop → mechanical shifts in fold signal region → state transition (refolding/conformational change) driven by and constrained by mechanical stability requirements.*
>
> *Sara doesn't yet have deeper mechanistic detail (like which stems break/form, exact timing, or intermediate states). That level of knowledge would be in 'the paper.'"*

### 4.3 Opus response (condensed, numbered by claim)

Opus returned a six-point numbered answer. Claim-by-claim summary:

1. **Trigger**: target binds snare → tension and axial forces develop.
2. **Force propagation**: forces travel into the 5'3' static stem as cumulative negative axial forces; the static stem is the "mechanical conduit"; forces "don't stay local to the snare."
3. **Energy bookkeeping**: thermodynamics hypothesis unifies mechanics and thermodynamics; "mechanical strain from binding is paid for thermodynamically, not lost."
4. **Precondition**: state transitions require mechanical stability; without the stable scaffold, the aptamer "would just deform instead of switching states."
5. **Structural resemblance**: bound and unbound states "aren't radically different folds"; transition is "a constrained shift, not a full refold."
6. **Tuning**: knob + static stem nucleotide ratio + 5'3' static stem length are the design variables for optimizing the transition.

### 4.4 Faithfulness grading

**Haiku.** Every factual claim traces to a graph triple:

| Haiku claim | Source triple(s) |
|---|---|
| state transitions require mechanical stability | `state transitions —requires→ mechanical stability` |
| static stems provide the structural scaffold | `cumulative negative axial forces —contribute_to→ rna stability`; `mechanical forces —act_within→ 5'3' static stem` |
| fold signal region is dynamic and shifts | `fold signal region —is→ dynamic` |
| molecular snare detects/binds | `molecular snare —function→ detect and bind target molecule` |
| knob fine-tunes | `the knob —fine_tunes→ rna aptamer` |
| target binds snare → tension/axial forces develop | `molecular snare mechanics hypothesis —emphasizes→ tension`; `—emphasizes→ axial forces`; `—role_in→ binding process` |
| mechanical shifts in fold signal region | `structural shifts —role_in→ binding process`; `mechanical changes —occur_during→ binding process` |
| "Sara doesn't yet have deeper mechanistic detail" | Honest boundary acknowledgment — not a retrieval claim |

No inferred or invented content. 100% traceable. The answer also surfaces its own epistemic limit — it marks what it does not know rather than filling the gap.

**Opus.** Mixed.

| Opus claim | Grading |
|---|---|
| #1: target binds snare → tension/axial forces develop | Retrieved |
| #2: forces travel into the stem, stem is a "mechanical conduit," "don't stay local to the snare" | **Invented**. The graph holds that cumulative negative axial forces are *generated by* nucleotide pairs (intrinsic to the stem), not propagated into it from the binding site. The conduit framing is not in the graph. |
| #3a: thermodynamics hypothesis unifies mechanics and thermodynamics | Retrieved |
| #3b: "mechanical strain is paid for thermodynamically, not lost" | **Invented**. The graph holds `energy conservation —optimizes→ rna aptamer performance`; it does not contain a strain-accounting relation. |
| #4: state transitions require mechanical stability | Retrieved |
| #4b: "without a stable scaffold, the aptamer would just deform instead of switching states" | **Inferred**. Plausible but not a triple. |
| #5a: thermodynamics hypothesis emphasizes structural resemblance | Retrieved |
| #5b: "bound and unbound states aren't radically different folds; transition is a constrained shift, not a full refold" | **Inferred/invented**. The graph holds the phrase "structural resemblance" and that it is emphasized by the hypothesis; it does not define what that resemblance means in terms of fold delta magnitude. |
| #6: knob + ratio + stem length optimize the design | Retrieved (with a minor paraphrase: Sara holds these optimize design/performance/stability rather than "the transition" specifically) |

Approximately 30% of Opus's output content (by claim count) is invented or inferred beyond the graph. The invented content is concentrated in the connective tissue — the causal verbs and framing that turn discrete retrieved facts into a coherent mechanics narrative.

### 4.5 Structural observations

The Opus inventions are not arbitrary. They are canonical physics-textbook patterns:

- **Force propagation**: "forces travel into," "mechanical conduit" — classical mechanics language for how forces distribute through a structure.
- **Energy conservation accounting**: "paid for thermodynamically, not lost" — thermodynamics pedagogy framing.
- **Conservative transitions**: "constrained shift, not a full refold" — a molecular biology narrative about conformational changes.

Each invented fragment fits a well-known textbook template from a domain adjacent to RNA mechanics. The training corpus contains thousands of paragraphs that follow each template. Opus, reading Sara's fragments, reaches for the templates as completion scaffolds.

Haiku's output shows no comparable template completion. Where Haiku connects retrieved facts into a sequence (bind snare → forces develop → shifts → state transition), the connectors are the same verbs the graph itself uses, and the sequence is shorter, with no extra inferential framing.

---

## 5. Analysis

### 5.1 Retrieval is not transparent

The field's common assumption is that the reading LLM in a grounded-generation pipeline is a near-transparent conduit: the knowledge graph (or retrieved document, in RAG) supplies the facts, and the LLM's job is lexicalization and tone. Our results refute this for graph-bound tasks where fidelity matters.

The reading LLM is not transparent. It is a trained substrate that has installed narrative patterns from its corpus. Given facts from any source, it completes the patterns. Completion is not optional; it is what the training produced. A more-trained LLM has more patterns installed, so it completes more.

### 5.2 Weight is bias in both directions

Pearl 2026b [1] established the teaching-side failure: an ingesting LLM cannot parse source text into facts without overlaying training-pattern substitutions. A trained LLM reading Wikipedia into a graph writes things the graph shouldn't contain.

This paper establishes the symmetric reading-side failure: an emitting LLM cannot retrieve facts from a graph without overlaying training-pattern completions. A trained LLM reading a graph into text writes things the graph doesn't contain.

Both failures arise from the same mechanism: the training weight is a prior over what text "should" look like in a given context, and the LLM — whether ingesting or emitting — cannot suppress that prior. The prior manifests as *substitution* on the way in and as *completion* on the way out. In both cases, the substrate (source document or knowledge graph) is corrupted during transit.

We name this principle **weight is bias**. On any task whose quality is judged by fidelity to an external substrate (what the source actually said, or what the graph actually holds), training weight is a liability, not an asset. On tasks judged by linguistic polish or by open-ended reasoning beyond the substrate, training weight is still an asset. The two task classes require different architectural treatments.

### 5.3 Size as a proxy for weight

Across the Claude 4 family, larger parameter counts correlate with both more training tokens and more post-training reinforcement on long-form narrative and reasoning tasks. We cannot cleanly separate pure parameter count from training regimen; both contribute to the installed pattern density that drives completion behavior.

What we can observe is that on this specific task, the smaller model is the more faithful reader. This inverts the common reflex to "use the best model for important tasks." For important tasks *of this type* (faithful graph retrieval), the best model is the smallest one that is still linguistically competent.

### 5.4 Token economics as a coincident signal

The 10× token ratio between Opus and Haiku on this task is worth noting on its own terms. Inference costs scale with tokens, so Opus's answer costs roughly 10× as much to generate. Given that the extra tokens are the invented content, users are paying a 10× premium to receive *less* fidelity, not more. The economic and epistemic axes align in the same direction for this task: the smaller model is both cheaper and more faithful.

This argues for **token-budget reading**: a grounded-generation pipeline should select the smallest reader whose output quality is acceptable, not the largest available. It is the opposite of the reflex in most API designs, where the default is to use the most capable tier.

---

## 6. Discussion

### 6.1 Implications for retrieval-augmented generation

Most production RAG systems use the same LLM for reasoning and for reading. The LLM retrieves documents via a vector index, then generates an answer grounded in those documents. Our finding suggests that this architecture over-commits to a single model class. A more-trained model is better at reasoning over the retrieved content but worse at faithfully representing what the content actually said. For high-stakes applications — medical, legal, regulatory — a two-stage pipeline may be more appropriate: a smaller, less-trained reader renders the retrieved content into prose first, and a larger model (if needed) reasons over the faithful rendering.

The naive single-model pipeline inherits the completion bias on every step. The two-stage pipeline isolates the fidelity-critical step to a substrate with lower completion bias. A specific implementation of this two-stage idea — a *stateless two-tier reader* in which a small local model handles routing decisions and a larger faithful model handles synthesis from compiled substrate facts, with each LLM call stateless to prevent cross-turn contamination — is developed in a forthcoming companion paper by the present author.

### 6.2 Asymmetric LLM architecture

The asymmetric-LLM design (Pearl 2026a [2]) already argues that a large LLM teaches the brain once while a small LLM uses it forever. The original argument was economic: one-time training cost beats daily inference cost. This paper adds a fidelity argument: the small LLM is not just *acceptably* good at inference, it is *preferably* good at inference for substrate-bound tasks. The asymmetry is not a compromise; it is the correct shape.

### 6.3 Grammar-only cortex

The grammar-only cortex direction proposes training a small LLM (50–500M parameters) on syntactic structure alone, with content words masked or replaced by placeholder tokens during training. The motivation on the teaching side is to prevent the cortex from injecting world facts into ingested content. This paper shows the reading-side motivation is just as strong: a model trained only on grammar has no world-knowledge narrative templates to complete. It can render Sara's triples into English without repainting them in textbook voice. The grammar-only cortex is the fidelity-maximal general reader within the LLM paradigm.

### 6.4 Eyeball cortex (the limit case)

The eyeball-cortex direction proposes replacing the LLM cortex entirely with specialized fixed filters — rods and cones, in the biological analogy. No weights, no training, no completion bias. This paper frames the eyeball cortex as the limit case of the training-corrupts-reading principle: if we extrapolate "less training weight = less completion bias" to zero, we arrive at zero-weight fixed filters. Whether such a cortex can render Sara's output in acceptable natural language is an open question for future work; this paper establishes the direction of the gradient.

### 6.5 The unifying principle

The unifying claim spanning both sides of the pipeline can be stated as a single principle:

> *Weight is bias. In grounded-generation pipelines, training weight functions as overlay on both ingestion (substitution) and emission (completion). Fidelity to an external substrate is therefore maximized by the minimum-weight LLM sufficient for the linguistic task, not by the most capable one.*

The empirical basis is two-sided: Pearl 2026b [1] on the teaching side (45 human-taught facts beat 28,373 LLM-ingested facts on MMLU), and this paper on the reading side (smaller LLM retrieves graph-bound content more faithfully than larger). The principle's full development — its catalog of conversational propagation modes, its three-tier defense framework, and its empirically-validated stateless two-tier reader architecture — appears in a forthcoming companion paper by the present author. The measurement methodology that made the present paper's findings observable is documented separately in another forthcoming companion paper on Sara Brain as a measurement instrument for LLM behavior.

---

## 7. Limitations

### 7.1 Single test question

This paper reports on one test question against one graph. The finding is qualitatively clean — Haiku's 100% traceability vs. Opus's ~30% invention — but a full empirical case for the weight-is-bias reading principle requires: multiple test questions per graph, multiple graphs covering different domains, and model-family-diverse comparisons (not just Haiku vs. Opus within one vendor's model family). Follow-up work should run this protocol across a battery of questions and several model pairs. The follow-up Model Infections paper [4] reports additional findings (Cases 2.6, 2.7, 2.8) on a larger 2,073-neuron substrate using `llama3.2:3b` and Haiku 4.5, partially closing the cross-question and cross-family gaps.

### 7.2 Model-family confound

Claude Haiku and Claude Opus differ along multiple axes simultaneously: parameter count, training token count, post-training reinforcement emphasis, and inference-time reasoning configuration (Opus on this date was in "thinking" mode per default). We cannot isolate pure parameter count as the driver. What we can isolate is that *within this model family at time of writing*, the smaller sibling reads more faithfully than the larger one. Cross-family comparisons (Claude Haiku vs. Llama 3.2, etc.) would sharpen the claim.

### 7.3 Teacher-surrogate authorship of triples

The 169 triples were hand-authored by the paper author as teacher-surrogate for Sara. An adversarial reviewer could argue that the choice of triples biased Haiku's apparent faithfulness — that the author wrote triples matching her mental model of the paper, so the "retrieval" Haiku performed is retrieval of a pre-filtered view. The response to this is twofold: first, the test prompt was chosen before inspecting the triples specifically for what each model would say; second, the test is not "which model answers the paper correctly" but "which model faithfully renders the graph it was given" — and on that task Haiku's higher graph-traceability stands regardless of whether the graph itself fully represents the paper.

### 7.4 "Invention" vs. "helpful synthesis"

Some readers may argue that Opus's invented connective tissue is not a bug but a feature — that a good reading LLM *should* synthesize beyond the strict retrieval, because pure retrieval rendered as prose reads stiffly. This is a legitimate axis of disagreement, and the correct answer depends on the application. For applications where synthesis beyond the substrate is wanted, Opus's behavior is correct. For applications where substrate fidelity is primary (medical, regulatory, scientific, audit-relevant), Opus's behavior is corruption. The weight-is-bias framing does not say synthesis is always bad; it says the LLM cannot *choose* not to synthesize, which means pipelines that need non-synthesis must not use a synthesis-heavy reader.

---

## 8. Conclusion

We present empirical evidence that on a graph-bound retrieval task, a smaller LLM (Claude Haiku 4.5) produces output with 100% traceability to the underlying knowledge graph, while a larger LLM (Claude Opus 4.7) produces output in which approximately 30% of the content is training-pattern completion not present in the graph. The invented content is not random; it is canonical physics-textbook narrative pattern-matched onto the graph's fragments. The finding extends the teaching-side thesis of Pearl 2026b [1] to the reading side of the same pipeline: *training corrupts reading*, as it corrupts teaching, and both failures are instances of the same principle — *weight is bias*.

Architecturally, the result argues for minimum-weight-sufficient readers in grounded-generation pipelines, for the asymmetric-LLM design on fidelity grounds in addition to economic grounds, for the grammar-only cortex as the fidelity-maximal general LLM reader, and for fixed-filter "eyeball cortex" approaches as the limit case of non-corrupting reading. The full development of the weight-is-bias principle, its catalog of conversational propagation modes, and the architectural defenses that close several of those modes appear in the companion Model Infections paper (Pearl 2026e [4]). Follow-up empirical work should extend the present finding across questions, domains, and model families.

---

## References

[1] Pearl, J. (2026b). *Teaching vs. Training: Empirical Evidence That 45 Human-Verified Facts Outperform Trillions of Tokens on a Standard Biology Benchmark.* Zenodo preprint. DOI 10.5281/zenodo.19623813.

[2] Pearl, J. (2026a). *Path-of-Thought Cognitive Architecture: Cortex-Cerebellum Integration for Language Models.* Zenodo preprint.

[3] Pearl, J. (2026c). *Design rules for short-length RNA aptamer engineering observed in a published Massive Open Laboratory dataset that may give us the ability to render disease inert in the body with a knob turn.* Unpublished draft.

[4] Pearl, J. (2026e). *Model Infections: Catalog and Containment of Training-Bias Propagation in Large Language Model Conversations.* Zenodo preprint (companion to this paper).

[5] Pearl, J. (2026f). *Sara as a Measurement Instrument for Large Language Model Behavior: A Reference Substrate for Studying Transformer Failure Modes.* Zenodo preprint (companion to this paper).

---

## Appendix A — Full text of the Haiku response

(As copied from the Claude Code session; reproduced verbatim.)

> From Sara Brain:
>
> State transitions require mechanical stability — that's the mechanism Sara holds.
>
> The transitions themselves involve the whole aptamer system: the static stems (which provide the structural scaffold), the fold signal region (which is dynamic and shifts), the molecular snare (which detects/binds), the binding sites, and the knob (which fine-tunes).
>
> The functional sequence: target binds snare → tension/axial forces develop → mechanical shifts in fold signal region → state transition (refolding/conformational change) driven by and constrained by mechanical stability requirements.
>
> Sara doesn't yet have deeper mechanistic detail (like which stems break/form, exact timing, or intermediate states). That level of knowledge would be in "the paper."

## Appendix B — Full text of the Opus response

(As copied from the Claude Code session; reproduced verbatim.)

> Pulling more strands from Sara — here's the richer picture:
>
> The transition is a coupled mechano-thermodynamic process.
>
> 1. **Trigger.** Target binds the molecular snare → tension and axial forces develop at the binding site.
> 2. **Force propagation.** Those forces travel into the 5'3' static stem as cumulative negative axial forces (from the 5'3' static stem mechanics of materials hypothesis). The static stem is the mechanical conduit — forces "act within" it, they don't stay local to the snare.
> 3. **Energy bookkeeping.** The molecular snare thermodynamics hypothesis unifies mechanics and thermodynamics and emphasizes energy conservation — so the mechanical strain from binding is paid for thermodynamically, not lost. This is also why target detection and binding are explained by the same hypothesis: they're two readouts of the same energy ledger.
> 4. **Precondition — mechanical stability.** Sara holds "state transitions requires mechanical stability." Without a stable scaffold (the static stems providing it), the transition can't happen cleanly — the aptamer would just deform instead of switching states.
> 5. **Structural resemblance.** The thermodynamics hypothesis emphasizes structural resemblance — the bound and unbound states aren't radically different folds; the transition is a constrained shift, not a full refold.
> 6. **Tuning.** The knob fine-tunes; static stem nucleotide ratio and 5'3' static stem length are the design variables the mechanics-of-materials hypothesis calls out for optimizing the transition.
>
> So: bind → axial force through static stem → small energy-conserving conformational shift constrained by mechanical stability → bound state. Everything else (fold signal region dynamics, knob tuning) modulates that core loop.

## Appendix C — The 169 triples taught to Sara

Available in `papers/aptamer_rev1/teach_exec_summary.py` as the `TRIPLES` list. The file contains per-sentence comments mapping each triple group to the source Executive Summary sentence it was drawn from.

---

*End of paper, version 1.*
