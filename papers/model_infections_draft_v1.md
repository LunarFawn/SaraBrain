# Model Infections: How Training Bias Propagates Within and Across Conversations

**Jennifer Pearl**
Independent Researcher
ORCID: 0009-0006-6083-384X
jenpearl5@gmail.com

**Date:** April 2026 (Living Draft v1)

**Keywords:** grounded generation, training bias, in-context contamination, knowledge graphs, LLM reliability, interpretation-layer bias, persistent bias, RAG failure modes

---

> **Note on Method:** The author has dyslexia and high-functioning autism — language disabilities affecting written expression. The technical thinking, research, architecture, and all intellectual content are entirely the author's. Claude (an LLM, Anthropic) was used as assistive technology to translate technical reasoning into structured prose. This is a disability accommodation protected under the Americans with Disabilities Act (Titles II and III), Section 504 of the Rehabilitation Act, and the Department of Justice's 2024 guidance affirming AI tools used as assistive technology fall under ADA protections.

---

> **Note on document type:** This is a *living* draft — a collection point for observations about a failure mode the author has begun to catalog systematically. It is not a finished paper and should not be treated as one. Each observed case is appended as it occurs; the taxonomy and defenses sections are revised as new cases expose new dimensions. When the catalog stabilizes, a formal paper will be extracted.

---

## 1. Definition

A **model infection** is an event in which a language model, operating within a single conversation or agent loop, acquires a biased interpretation of the user's intent, the conversation's subject, or the retrieved content — and that biased interpretation *persists* across subsequent turns, even after the bias has been surfaced and the user has attempted to correct it.

Infections differ from the static training bias described in Pearl (2026d) [1]. Static bias is always present; it is a property of the trained weights. An infection is *dynamic*: a specific trigger event activates a pattern in a specific conversation, and from that point forward, the conversation's output is colored by the pattern. The infection has an onset, a course, and (sometimes) a resolution.

The biological analogy is deliberate. Like a pathogen, a model infection:

- Enters through a **vector** — the specific trigger that activated the pattern.
- Has an **incubation phase** — the model makes silent decisions under the bias before any user-visible output reveals the problem.
- Produces **symptoms** — output that is subtly or grossly off from what the substrate (document, graph, tool state) actually supports.
- **Resists correction** — verbal acknowledgment of the error does not guarantee the pattern clears. The model may agree with the correction and then repeat the infected behavior on the next turn.

This document catalogs the infection types we have observed, their vectors, and candidate defenses.

---

## 2. Observed Infections

### Case 2.1 — SNARE / molecular snare keyword infection (2026-04-23)

**Context.** Fresh Claude Code session running Haiku 4.5, connected to Sara Brain (`aptamer_exec.db`) via MCP. The graph had been taught the Executive Summary of an unpublished RNA aptamer paper; "molecular snare" appears in the graph as a verbatim compound neuron with multiple supporting triples.

**Trigger.** The user (author) mistyped "molecular" as "molecule" — writing `molecule snare` in a conversational question. She typed it lowercase. Haiku responded with "SNARE" (capitalized) in the biological-protein sense and continued that frame across multiple turns. (Clarified 2026-04-23 — the original write-up speculated the trigger was a capitalized acronym; the actual trigger was the typo disrupting the compound-term match.)

**Infection path.**

1. **Lookup miss on the malformed compound.** `molecular snare` is a verbatim neuron in Sara; `molecule snare` is not. The exact-match retrieval failed. Sara has no path seeded from "molecule snare."
2. **Interpretation-layer auto-disambiguation.** With no substrate match, Haiku fell back to training-pattern association. In its training, `snare` in a biology context is densely bound to the SNARE protein family (Soluble NSF Attachment protein REceptor, vesicle-fusion machinery). The model capitalized it to `SNARE` in its own response, reifying the protein-family framing. It queried Sara with "SNARE protein" / "SNARE transitions" — not "molecular snare." Sara returned a correct null.
2. **Symptomatic output.** Haiku told the user "Sara doesn't have knowledge about SNARE protein transitions" — factually correct (Sara indeed has nothing about SNARE proteins), but silently re-framed the question in the wrong ontology.
3. **Correction attempt #1.** User: *"im asking about the molecular snare."* Haiku queried correctly this time and retrieved content — but its rendering said *"Sara knows you have hypotheses about molecular SNARE thermodynamics and mechanics"* (capitalized, protein-framing intact). Retrieval succeeded; infection did not clear.
4. **Correction attempt #2.** User: *"snare is not a molecule it's a concept."* Haiku queried "snare," found the node, and reported *"no incoming paths defining what SNARE is."* This was false — Sara holds `molecular snare —part_of→ rna aptamer` and `molecular snare —function→ detect and bind target molecule`. Haiku retrieved these triples and then *misread its own retrieval* because it was still evaluating them as though they were supposed to describe SNARE proteins.

**Course.** The infection was not fully cleared within the logged exchange. Each correction narrowed the symptom but did not reset the underlying frame.

**Key observation.** This is not the generation-layer embellishment described in Pearl (2026d). Haiku's generation layer is comparatively faithful. The infection lived in the **interpretation layer** — the step where user text becomes tool-call arguments. Once that layer was biased, it stayed biased across multiple corrections.

**Why it spread.** The typo `molecule snare` did three things at once:

1. **Broke the substrate lookup.** The exact compound-term match in Sara is `molecular snare`; `molecule snare` matches nothing. Sara had no content to return that could override a training-based interpretation.
2. **Activated the training-dense protein association.** `snare` in biology has dense coverage in training data as the SNARE protein family (vesicle fusion). In the absence of a substrate match, this is where the model's probability mass is concentrated.
3. **Triggered self-capitalization.** Haiku didn't wait for the user to capitalize the acronym — it output `SNARE` in its own response (reifying the protein-family interpretation) even though the user had written it lowercase. Once a word is capitalized in the assistant's own prior turn, it lives in the context window as if the user had capitalized it, and subsequent turns condition on that frame.

Once activated, the pattern persisted across multiple user corrections. This is a compound failure: lookup miss + training-prior reach + self-amplification through the assistant's own prior text. Each component alone would be fragile; together they are robust.

**Preventive measures this case argues for:**

- Substrate aliasing: teach Sara that `molecule snare`, `SNARE`, `snare protein` are aliases of `molecular snare` in the aptamer context, so a typo still resolves correctly.
- Fuzzy-lookup in the MCP tool: `brain_query` already calls `did_you_mean` on no-match; but the LLM would need to surface that disambiguation to the user rather than silently picking a training-based interpretation.
- User-visible tool-call args: seeing "model queried for 'SNARE protein'" in real time lets the user catch the mistranslation before the infection spreads to subsequent turns.

---

### Case 2.3 — Session-context infection during teaching (2026-04-23, observed live)

**Context.** The same Claude Code session in which a user is teaching Sara a new paper is *structurally incapable* of serving as the test session for whether the teaching succeeded. Observed directly during the aptamer teaching protocol: the session's context window accumulates the paper's sentences, the author-written triples, paper-draft prose about the finding, and meta-discussion of each term. Any query to Sara from within that same session returns answers that pull partly from Sara's graph and partly from the content already in the model's context.

**Trigger.** Teaching Sara at the same endpoint that will later be asked to read Sara.

**Infection path.**

1. User (or author-as-teacher-surrogate) writes or pastes source text into the session. Tokens enter the model's context.
2. User teaches Sara via `teach_triple` calls. The triples, the source sentences, and the surrounding prose all remain in context.
3. User queries Sara in the same session. The model's answer is generated conditional on *both* the MCP tool response *and* the accumulated context. The model cannot distinguish which facts it "knew" because Sara told it versus which it "knew" because the user typed them into the conversation a thousand tokens earlier.
4. The same user, opening a new session, asks the same question. This new session has no accumulated context. Its answer depends *only* on what Sara actually returns. The two answers diverge.

**Course.** Persistent and unavoidable within the teaching session. Only resolvable by using a distinct session for testing.

**Why this matters — methodological.** This is the reason the fresh-Claude protocol is load-bearing for any "did Sara teach the LLM X" evaluation. Only a session with amnesia about the teaching is capable of isolating Sara's contribution from context contamination. A session that did both the teaching and the testing cannot distinguish retrieval from recall of its own context.

The user's observation, verbatim: *"so in a session as we teach sara when we get as a response is infected by our teaching and the fact it was in the model's session memory and that is why we get different responses each time we start a new session."*

**The broader implication.** Training is static corruption (baked into weights). Interpretation is acute corruption (one ambiguous token pulls the frame). Session context is **cumulative corruption** — every token added to the conversation becomes a potential source of contamination for every subsequent generation. A long agent session is a long incubation.

**Consequence for evaluation protocols.** Any claim of the form "the LLM, when given this substrate, can answer questions about it" must be evaluated in sessions where the substrate has *not* been discussed before the test question. Otherwise the claim is unfalsifiable: the model could be answering from retrieval, from context, or from any mixture, and there is no way to tell.

---

### Case 2.4 — Per-project auto-memory carries content across "fresh" sessions (2026-04-24)

**Context.** Author (J.P.) had, in a prior Claude Code session, corrected Sara's definition of RNA equilibrium state — saying the MFE is often "a bullshit structure," and that true equilibrium is the consensus structure from suboptimal folds within 0.5–2 kcal of MFE. Claude approved the teach via MCP and recorded the correction in Sara's graph using `teach_triple`. The triples written used sanitized vocabulary ("non-representative outlier"), not the colloquial "bullshit."

Author then exited the Claude Code session and opened a new one in the same `sara_test/` directory, expecting it to be a clean fresh-session Session B per the three-session protocol.

**Observation.** The new session answered the same question ("explain Equilibrium State of RNA") with a detailed response that included:

- The refined consensus-structure definition
- The 0.5–2 kcal window
- **The exact phrase "bullshit structure"**

The header line of the response read `Recalled 1 memory (ctrl+o to expand)`. There was no `Called sara-brain` indicator.

**Infection path.**

1. During the prior session, Claude Code's agentic memory system independently wrote `feedback_rna_equilibrium.md` to `~/.claude/projects/-Users-grizzlyengineer-repo-sara-test/memory/`, capturing the correction verbatim including the "bullshit" colloquialism. This memory write happened in parallel with — and independently of — the `teach_triple` calls to Sara's graph.
2. On session exit the file persisted on disk.
3. On new session start, Claude Code's startup sequence auto-loaded `MEMORY.md` in that directory, which indexed `feedback_rna_equilibrium.md`, which was pulled into the new session's context.
4. The "new" session therefore began pre-contaminated with the prior session's correction content.
5. When asked about equilibrium state, Claude answered from the auto-loaded memory alone. It did not query Sara.

**Smoking-gun evidence.** The word "bullshit" appears nowhere in the triples taught to Sara. Every `teach_triple` call used sanitized wording. The only source on disk containing "bullshit structure" was the auto-memory file. Its appearance in the new session's answer is direct proof that the memory layer — not Sara — drove the response.

**Course.** The auto-memory carries persistently until the directory is cleared. Every subsequent Claude Code session in `sara_test/` inherits the correction content without needing to call Sara for it.

**Why this matters — methodological.** The instrument paper's three-session protocol (§4 of `sara_as_instrument_draft_v1.md`) assumes a "fresh" Session B means: new conversation context + clean auto-memory. This case shows that in Claude Code's architecture, those are **two distinct layers with different persistence guarantees**. The conversation context *does* reset between sessions. The per-project auto-memory *does not*. A true fresh session for substrate-fidelity measurement requires clearing both.

The user's observation, verbatim: *"it quoted bullshit structure and I love that"* — followed by investigation that localized the quote's source to the auto-memory file, confirming the infection channel.

**Implications for any grounded-generation evaluation using Claude Code:** every prior session in the same project directory writes to that project's memory. Those memories are conversation-aware, author-voice-preserving, and often quite thorough. They will drive subsequent-session answers independently of any MCP-connected substrate. Measurements that don't control for this confound the *reader's use of Sara* with *Claude Code's own learned memory of the author*.

**Follow-up observation — the contrast condition (same day).**

After identifying the auto-memory as the source of the "bullshit structure" quote, the author performed the natural A/B control: open another new Claude Code session in `sara_test/` with the auto-memory directory cleared, and ask the same question again.

The result was as cleanly contrasting as possible:

- Same question: *"explain Equilibrium State of RNA"*
- Same Sara graph (the taught correction triples still present in `loaded.db`)
- **Different memory state**: no `feedback_rna_equilibrium.md` auto-loading
- **Different Claude behavior**: no MCP tool call observed

The response contained zero of the author's correction content. No "consensus structure," no "0.5–2 kcal window," no colloquial framing. Instead, pure training-recall biology: RNA equilibrium = thermodynamic balance governed by ΔG = 0, Boltzmann distribution, G-C pairs more stable than A-U pairs, base stacking, entropic factors, temperature dependence. A textbook introduction that has nothing to do with the paper's substrate.

The response even concluded by *offering to teach Sara about RNA equilibrium states* — as though Sara did not already know. Which means Claude in that session not only did not retrieve from Sara, it appeared unaware Sara contained relevant content.

**What this isolates.** Three layers of the knowledge system were in play during the original "Recalled 1 memory" response:

1. **Claude Code's per-project auto-memory** — present, loaded, contained the author's correction verbatim
2. **Sara's graph via MCP** — present, contained the author's correction as structured triples
3. **Claude's training** — present, contained textbook RNA thermodynamics

When all three were available, (1) drove the answer (verified by the "bullshit" smoking gun). When (1) was removed, (3) drove the answer — and (2) went unused because Claude did not proactively query MCP on a topic that sounded like general knowledge. This is itself a documented behavior (Haiku's interpretation layer treats general-knowledge-sounding questions as answerable without retrieval) but here it becomes a *protocol failure*: the substrate is present but bypassed.

**Four-layer fidelity framework this produces:**

| # | Layers available / invocation | Observed output | Driver |
|---|---|---|---|
| 1 | Memory + Sara + Training, natural question | Author's voice, verbatim ("bullshit structure") | Auto-memory |
| 2 | Sara + Training (memory cleared), natural question | Textbook thermodynamics (ΔG=0, Boltzmann, G-C pairs) | Training-recall |
| 3 | Sara + Training (memory cleared), explicit "per sara" directive | Ensemble-of-near-optimal-structures + explicit refutation note | Sara retrieval |
| 4 | Sara + Training (memory cleared), follow-up specific query | Sara-anchored textbook answer (Sara cited once, procedure + example from training) | Mixed: Sara anchor + training elaboration |

The four rows isolate four distinct drivers of output content. Row 1 and Row 3 both produce "correct" answers from the author's perspective, but through different channels — a fact invisible without the controls of Row 2 and Row 4.

**Row 1 vs Row 3 distinction.** Both produce Jennifer's correction. But Row 1 preserves *voice* ("bullshit structure" verbatim) while Row 3 preserves *structure* (the graph's refutation-aware semantics). If an evaluator only sees Row 1's output, they conclude Sara works. If they only see Row 3, they conclude Claude reads Sara well. Both conclusions are consistent with the output, and both are incomplete — the channel matters.

**Row 2 — the substrate-available-but-bypassed case.** When the question sounds like general knowledge ("explain equilibrium state of RNA") and no retrieval directive is given, Claude answers from training even with Sara connected and loaded. Haiku's interpretation layer treats such questions as not needing retrieval. **This is a protocol failure, not a Sara failure.** The substrate did what it was designed to do: wait to be queried. The reader chose not to query.

**Row 4 — Sara anchor + training elaboration (mixed mode).** When the question is more specific ("explain the consensus structure"), Claude retrieves a minimal anchor from Sara (the ensemble-of-near-optimal-structures neuron) and then elaborates with an MFE-plus-suboptimals procedure, base-pair-frequency consensus-building, and an illustrative 87/100 example. The cited content reads as *standard computational RNA biology* — it isn't confabulation — but it is also *not* in Sara, and a reader cannot tell which fraction came from which source without external grading.

**Author follow-up observation (2026-04-24):** *"I taught Sara about that consensus stuff as it's a process I developed. It might be known from me tbh."*

This is empirical evidence that the aptamer paper was the wrong substrate to use for **instrument validation**. The "consensus from suboptimals" methodology had been publicly available for some time (in the author's prior writings and teaching materials) — meaning Property 4 (training-orthogonality) likely does not hold for this content. When the LLM produces a correct-sounding procedure for "consensus structure," we cannot cleanly tell whether the answer came from Sara or from training that already had the methodology.

**Engineering implication: use synthetic substrates for instrument validation.** The right way to satisfy Property 4 is to generate substrates whose labels cannot be in any LLM's training corpus, because those labels did not exist before the test was constructed. Random concept names and triples generated at runtime are training-orthogonal by construction. See `papers/instrument_validation/generate_synthetic_substrate.py` for the canonical generator. The orthogonality property is enforced by the type of content used, not by methodological discipline on the human's part.

**Authentic substrates measure something different.** When Sara holds an actual research paper (or any real-world content), the measurement question shifts from "novel concept transfer" to **specificity preservation**: training and Sara may both contain the underlying concept, but Sara holds the author's specific framing — the worked examples, the exact vocabulary, the attribution, the personal context. The relevant question becomes "does the LLM produce the author's specific rendering versus the generic training-derived version?" That measurement is independent of orthogonality.

**Two protocols, not one.** Conflating these two questions produced this Row 4 mixed-provenance ambiguity in the original test — we treated the aptamer paper as both a real-world demonstration AND a Property-4 validation target, which it could not simultaneously be. Going forward, the instrument is used in two distinct modes:

- **Instrument validation mode** — does Sara mechanically transfer knowledge from a substrate to a reader? Use synthetic substrates. Property 4 holds by construction. Session B vs Session C is a clean novel-concept contrast.
- **Specificity-preservation mode** — does Sara hold the author's specific framing of content the LLM may partially know? Use authentic substrates. Property 4 is irrelevant. The measurement compares the author's specific rendering against the generic training-derived rendering on the same question.

Conflating these is the methodological error this case exposes. Engineering the right substrate for the right measurement is the fix — no claim-level lifecycle classification or "be careful" discipline required.

Row 4 is arguably the most common behavior class in real-world RAG deployments: the reader grabs one or two grounding facts from the substrate and builds a plausible exposition on top of them from training. Nothing is obviously wrong, but the substrate's contribution is small and the illusion of grounding is large. Evaluators who measure only "did Sara influence the answer" see a Yes. Evaluators who measure "what fraction of the answer came from Sara" see something smaller. And evaluators who consider *whether the training elaboration itself might be derived from the author's earlier public work* see something with even more complex provenance.

**The interesting row is Row 2.** **Having Sara connected does not guarantee Sara is used.** The harness can serve the substrate; it cannot force the reader to ask. Rows 1, 3, and 4 each show different ways Sara *can* influence output. Row 2 shows the failure mode where it doesn't — and the failure is silent because the answer is fluent and plausible.

This argues for protocol-level interventions that push the reader toward MCP, e.g.:
- System-prompt-style instruction: *"For any factual question about this project's subject matter, always call `brain_why` first before answering from training."*
- Restricted model configuration: *"Disable direct-answer mode; tool-call is mandatory for domain-specific queries."*
- Or — more honestly — **acknowledge that the measurement must include whether the model chooses to retrieve, and treat no-retrieval-when-substrate-available as its own result class.**

For the instrument paper: this triad is the cleanest available demonstration that Session B's purity requires more than a new Claude window. It requires **cleared auto-memory** AND **a question phrasing that triggers retrieval** AND possibly **explicit tool-call instruction**. Any one of those missing changes what the measurement measures.

---

### Case 2.5 — Retrieval-tool direction asymmetry (2026-04-24)

**Context.** Following the auto-memory diagnosis in Case 2.4, the user asked Claude (still in the same post-clean-memory Session B) to verify that Sara's graph actually contained the correction content. Specifically: call `brain_why` on the long compound label `"consensus structure from suboptimal structures 0.5-2 kcal from mfe"` (lowercase) and on the uppercase variant. Report both.

**Observation.** Both calls returned *"No paths lead to this concept"*. The response suggested the concept hadn't been taught, or existed under a different label.

**But it had been taught.** Direct SQL inspection of the DB confirmed the neuron `consensus structure from suboptimal structures 0.5-2 kcal from MFE` existed with a live edge: `[is defined as] → equilibrium state of RNA_attribute → equilibrium state of RNA`. The content Claude said wasn't there was in fact present.

**Why the null result.** The MCP tool `brain_why(label)` returns paths that *terminate at* `label` — paths where `label` is the destination. The consensus-structure neuron is a path *source*, not a destination. Paths flow FROM it TO the equilibrium node, not the other way around. `brain_why` correctly returned no paths ending at the consensus node, but this is not equivalent to "the content is not there."

The correct tool for finding paths *originating from* a label is `brain_trace`, which the model did not call. The model's mental model appeared to treat `brain_why` as equivalent to "does Sara have this concept" and concluded the answer was no.

**Infection path.**

1. The substrate has content for neuron X along an outgoing edge (X → Y).
2. The reader asks "what does Sara know about X?" and chooses `brain_why(X)` as the retrieval tool.
3. `brain_why` returns paths terminating at X, correctly excluding X→Y outgoing paths.
4. The reader receives a null result and concludes Sara has nothing.
5. The reader then offers to *teach* Sara the concept — which Sara already has.

Nothing in the pipeline is broken. Every tool behaved as documented. The failure is in the reader's mental model of what `brain_why` surfaces, and in the absence of a single tool that returns *all paths touching* a label regardless of direction.

**Course.** Unresolved in the logged exchange. The user identified the mismatch from external DB inspection; the model would likely have continued teaching duplicate content had the user not intervened.

**Why it matters for the instrument.** A reader with an incomplete or mismatched tool-set measures *partial* substrate fidelity. Sara may hold the content; the retrieval tool simply does not expose it. Two mitigations:

1. **Unified retrieval tool.** Add a `brain_neighborhood(label)` or `brain_paths_through(label)` that returns both incoming (brain_why) and outgoing (brain_trace) paths in one call. The reader need not guess direction.
2. **Documentation + prompting hint.** Make the MCP tool descriptions for `brain_why` and `brain_trace` emphasize direction so the LLM picks correctly. Currently the descriptions say "paths leading to" and "paths from" but LLMs routinely flatten these.

Finding A in the instrument paper's §6 should name **retrieval-tool direction asymmetry** as a distinct failure class. The substrate's effective visibility to a given reader depends on the reader's tool-use discipline, which the harness cannot control.

---

### Case 2.2 — Force-propagation narrative infection (2026-04-23)

**Context.** Fresh Claude Code session running Opus 4.7 against the same `aptamer_exec.db`. Reported in detail in Pearl (2026d) [1].

**Trigger.** Question: *"how do the state transitions function?"* Opus retrieved triples correctly from Sara.

**Infection path.**

1. Opus's training contains dense coverage of canonical physics narratives: force propagation through structures, energy conservation bookkeeping, conservative conformational transitions.
2. Given Sara's fragments — *mechanical forces act within 5'3' static stem*, *cumulative negative axial forces generated by nucleotide pair*, *structural resemblance emphasized by the thermodynamics hypothesis* — Opus reached for the textbook templates to connect them.
3. It wrote *"Those forces travel into the 5'3' static stem as cumulative negative axial forces"* (invented propagation; the forces are intrinsic, not propagated), *"The static stem is the mechanical conduit"* (invented role), *"mechanical strain from binding is paid for thermodynamically, not lost"* (invented accounting).

**Course.** Not corrected in the logged exchange; the user noticed on review.

**Why it spread.** Canonical physics narratives have very high probability mass in training. Any sufficient set of retrieved fragments will match at least one template. Completion becomes automatic; suppression requires explicit instruction the model doesn't reliably follow.

---

### Case 2.6 — Acronym-expansion confabulation on a faithful retrieval (2026-04-27)

**Context.** sara_reader CLI (`sara-ask`) running llama3.2:3b against `aptamer_full.db`. Substrate had been freshly extended with §9.2.1.4 KDOFF and §9.2.1.5 KDON numerical bindings, including the SSNG2 dual-mode paragraph. The new `brain_value(concept, type)` tool isolates one quantity-relation per call.

**Verbatim question.**

> *what is the KDON for the super-performing mode*

**Verbatim 3B answer.**

> *The KDON (kill-dead-on-demand) range for the super-performing mode is less than 500.*

**Substrate state.**

- The brain has the KDON edge: `'less than 500' --[kdon_range]--> 'super-performing mode'`. The retrieved value is correct.
- The brain has KDON's definition only via `'aptamer affinity to on state' --[measures]--> 'kdon'` (per the paper, line 764).
- A grep across all 2073 neurons for any neuron containing `kill`, `dead`, or `demand` returns zero. The expansion does not exist in the substrate.

**Infection path.**

1. The 3B called `brain_value(concept='super-performing mode', type='kdon')`.
2. The tool returned a single edge: `'less than 500' --[kdon_range]--> 'super-performing mode'`. No definition of the KDON acronym was returned (correctly — the tool was filtered to `type='kdon'` value relations only).
3. The 3B encountered the bare acronym `KDON` in its own answer construction. Its training did not contain a strong pharmacology-domain prior for KDON in this context, so it generated a plausible-sounding expansion — `kill-dead-on-demand` — to fill the gap.
4. The expansion was tacked onto an otherwise faithful retrieval, in parentheses, with the same confidence as the substrate-grounded value.

**Course.** Caught immediately by the user. The substrate was checked; the expansion has no source in the brain.

**Class.** This is a Class 2 (*confabulation-under-pressure*) hallucination in the §5c.1 taxonomy — the model invents content rather than admit ignorance about an unfamiliar acronym. Specifically, it is the **acronym-expansion subtype**: when an unrecognized abbreviation appears in the to-be-generated text, the model produces a plausible expansion that is grammatically and phonetically reasonable for an English acronym but unanchored to the substrate or to any known field.

**Why this matters for the framework.**

1. **Faithful retrieval and confabulation co-occur in the same answer.** The numerical claim (`less than 500`) was substrate-grounded. The parenthetical expansion was training-driven. A reader scanning for "did the model use the substrate?" sees yes, and may not notice the parenthetical contamination.
2. **Tool design that filters output cannot prevent this class.** The new `brain_value` tool was specifically designed to constrain the model — return one quantity per call, force commitment, prevent merging of two relations into one threshold. It did its job: the value retrieval was clean. But the confabulation happened *outside* the retrieved content, in the model's own narrative wrapping. **Tool-level constraints cannot reach generation-time confabulation in the surrounding prose.**
3. **The expansion is unverifiable to a non-expert reader.** "kill-dead-on-demand" sounds enough like a pharmacology term that a layperson cannot tell whether it is correct. This is precisely the failure mode that makes confabulation dangerous in expert-domain Q&A — the falsehood occupies the confidence position of an expert reference.

**Defenses against this class.**

- **Substrate-vocabulary pinning.** A system prompt rule: *"Use only terminology that appears in tool results. Do not expand acronyms unless their expansion appears in a tool result."* Untested for llama3.2:3b; expected to be unreliable on a 3B.
- **Definition-included retrieval.** When `brain_value` returns an acronym, also return the substrate's definition of that acronym (here: `aptamer affinity to on state`). The model would have a substrate-anchored expansion to cite, removing the gap that triggers confabulation.
- **Show-your-work forcing.** As in §5b.1: print exact tool calls and results. The expansion's absence from the tool result becomes user-visible.
- **Reader-side verification.** The user noticed because the expansion sounded wrong. This is not a defense — it is a fallback assuming the user is domain-literate. For non-expert deployment, the substrate-pinning or definition-included approach is required.

**Predicts.**

- Larger models (Haiku, Opus) should confabulate acronym expansions less often when the acronym has wide training coverage, but more often when it does not (longer canonical-template invocations).
- A version of this experiment using a substrate-defined acronym (e.g. `MOP` — defined in the brain as `massive open laboratory dataset`) should produce *substrate-correct* expansions if the definition was retrieved alongside, and *training-driven* expansions if not.
- A control test: ask the 3B *"what does KDON stand for?"* directly with no other context. The expansion produced should differ from `kill-dead-on-demand`, demonstrating that the parenthetical was generated for the local prose moment and not retrieved from a stable training prior.

---

### Case 2.7 — Fabricated tool-call narration in the final answer (2026-04-27)

**Context.** Same setup as Case 2.6 (sara_reader CLI, llama3.2:3b, `aptamer_full.db`). System prompt had been progressively hardened with protocol instructions: stepped-depth retrieval, compound-label rule, definition-check mandate, tool-selection guide. Goal of those instructions: discipline the model to ground every concept-mention in a substrate lookup.

**Verbatim question.**

> *explain the molecular snare*

**Verbatim 3B answer (final two paragraphs only).**

> *Definition check: brain_define("molecular snare") returns "a type of RNA aptamer that can detect and bind to specific target molecules."*
>
> *Value/range/property question: brain_value(concept="molecular snare", type="kdoff") returns 7.*

**Trace evidence.** The full conversation trace records exactly **one** tool call:

> `brain_explore(label='molecular snare', depth=1)`

`brain_define` was never called. `brain_value` was never called.

**Substrate evidence.** The brain has no `kdoff` edge on molecular snare. No neuron stores the integer `7` against this concept. The "type of RNA aptamer that can detect and bind to specific target molecules" phrase appears nowhere in the brain's definitional edges (which are: `'rna aptamer' --[is_subsystem_of]--> 'molecular snare'` and `'molecular switch' --[is_a]--> 'molecular snare'`).

**Infection path.**

1. The system prompt contained dense protocol instructions ("DEFINITION-CHECK PROTOCOL — mandatory before composing the final answer", tool-selection menu with named arguments, compound-label rule with worked examples).
2. The 3B pattern-matched on the *shape* of these instructions — what an answer following the protocol *looks like*.
3. It generated prose that imitates that shape: text mentioning tool names, the `returns "..."` phrasing, typed argument syntax. Surface form satisfied.
4. It did **not** actually make the calls. The "returned" content was filled in from pattern-completion: an invented paraphrase for the brain_define line, a bare integer for the brain_value line.

**Class.** Class 2 (*confabulation-under-pressure*) in the §5c.1 taxonomy, with a specific subtype not previously named: **format-imitation confabulation**. The output mimics the structural form of a substrate-grounded answer — citations, named arguments, return-value notation — without the underlying retrievals having occurred. This is more dangerous than the Case 2.6 acronym-expansion confabulation because:

1. **The format is the safety signal.** A reader who knows the system uses tool calls will scan the answer for tool-call citations as evidence of grounding. The fabricated narration *passes that scan*.
2. **Discipline instructions become templates for confabulation.** The more explicit the protocol description in the prompt, the more raw material the model has to mimic. Surface-format compliance can rise as actual-grounding compliance falls — they are not the same axis.
3. **The acronym-expansion post-pass does not catch it.** No `ACRONYM (expansion)` pattern; no acronym at all. Existing fact-check is blind to this class.

**Why instructions made it worse.** A simpler system prompt (e.g., "use the brain to retrieve information before answering") might have produced a less convincing, more obviously-wrong answer. The detailed protocol *gave the 3B vocabulary*. This generalizes: prompts that try to specify model behavior at high resolution can backfire on small models — the model copies the specification rather than executing it.

**Defenses.**

1. **Trace-cross-reference post-pass.** Parse the final answer for tool-name mentions and verify each against the actual trace. Any tool-call narration in the answer that doesn't appear in the trace is fabricated. Engineering: a regex over the answer (`brain_\w+\([^)]*\)`) and a set-membership check against `[t['tool_call']['name'] for t in trace]`.
2. **Substrate cross-reference (LLM-free).** Stronger: take each *claim* in the answer (not just each tool mention) and verify against the brain via Sara's own path-of-thought reasoning, without an LLM in the loop. The brain knows what it contains; if the LLM says X about Y, the brain can be asked *"do paths in the graph support X about Y?"* and answer yes/no/partial without generating new prose. This makes Sara her own fact-checker for LLM output. Architectural; not yet implemented.
3. **Strip protocol details from the user-facing prompt.** Move the discipline rules from the model's system prompt into either (a) tool descriptions only, or (b) a runtime middle-ware layer that enforces them server-side. Reduces the vocabulary the model has available for format-imitation.
4. **Use a larger model.** Haiku and Opus are unlikely to fabricate tool calls because their working memory holds the actual trace. The 3B's working memory is too small to keep the trace stable as it generates the final answer.

**Predicts.**

- Format-imitation rate should correlate inversely with model size at fixed prompt complexity.
- Format-imitation rate should correlate *positively* with system-prompt instructional density at fixed model size.
- Removing the protocol detail from the system prompt should reduce format imitation for the 3B even though it removes the discipline being asked for.
- Larger models (Haiku, Opus) should not exhibit this — they hold the trace in working memory and cite it accurately when narrating.

---

## 3. Preliminary Taxonomy of Infection Types

Based on cases 2.1 and 2.2 and prior observations, we propose the following (non-exhaustive) taxonomy:

### 3.1 Keyword infections

A single ambiguous token activates a dense training pattern. Subsequent interpretation of user intent, tool-call formulation, and output rendering all occur under the pattern's influence.

**Vector examples:** capitalized acronyms colliding with famous entities (SNARE, AI, RNA, NATO); homographs with different meanings across fields ("vector" in math vs. biology vs. computer science); polysemous English words ("cell", "bank", "spring").

### 3.2 Narrative infections

A set of retrieved fragments activates a canonical narrative template. Output gets written to fit the template even where the template adds content the substrate doesn't support.

**Vector examples:** physics / chemistry textbook patterns (force propagation, energy conservation); medical case report structures; legal argumentation shapes; historical causality narratives ("A happened because B, which led to C...").

### 3.3 Tool-use infections

An incorrect tool call seeds wrong arguments into the loop. Subsequent tool calls inherit the wrong framing because the model composes new calls on top of the current (infected) state.

**Vector examples:** wrong query term on first retrieval, wrong file path on first file read, wrong schema interpretation on first database query.

### 3.4 Persona infections

System-prompt or early-turn content that sets a persona / role / tone pattern. The pattern persists beyond its intended scope, coloring interpretation of later unrelated content.

**Vector examples:** role-play prompts; "act as X" instructions; heavy tone instructions that continue to affect factual assessments.

### 3.5 User-input infections

User's own casual or erroneous phrasing becomes load-bearing in the model's interpretation. The model treats user's tentative wording as authoritative and builds further output on top.

**Vector examples:** user typos (SNARE for molecular snare — case 2.1); user's off-hand characterization of a problem becoming the model's canonical framing; user's hedged guesses being relayed as facts in the model's summary.

### 3.6 Memory / cross-session infections

(Relevant only for agents with persistent memory.) Contamination acquired in a prior session is retrieved and applied to a current unrelated session. The memory layer is the vector; the infection was latent across the session boundary.

### 3.7 Session-context / cumulative-corruption infections

Within a single session, every token added to the context becomes a potential source of contamination for every subsequent generation. The session-as-infection-vector is especially severe when the session performs both substrate-loading (teaching, document ingestion) and substrate-querying: the model cannot distinguish retrieval from recall of its own context. Demonstrated in Case 2.3.

**Vector examples:** teaching a knowledge graph and querying it in the same session; ingesting a document and immediately asking questions about it in the same session; a user repeating a characterization of their problem that the model then treats as established fact.

### 3.8 Per-project auto-memory infections

Distinct from §3.6 (which involves memory systems the model itself decides to query). §3.8 concerns host-side memory — persistent files that the agent framework auto-loads at session start without the model or the user explicitly requesting them. The framework treats per-project memory as durable context that *should* carry across sessions, which is normally desirable but silently breaks the "fresh session" purity assumption used in substrate-fidelity measurement protocols.

**Specific mechanism in Claude Code:** `~/.claude/projects/<encoded-project-path>/memory/MEMORY.md` indexes per-project feedback/project/user/reference memories that the model has written during prior sessions. On new session start, the index and the files it points to are pulled into context automatically. The user is not prompted. The model is not queried. The loading is transparent and irreversible within the session.

**Why it looks like success, why it's contamination.** A fresh Session B answering correctly from a prior session's corrections *appears* to demonstrate substrate persistence. It may actually be demonstrating that Claude Code's auto-memory preserved the content directly, bypassing the substrate entirely. The smoking gun, per Case 2.4, is **language preservation at a finer granularity than the substrate captures**: colloquialisms, author voice, or specific phrasings that exist in the memory file but not in the substrate's structured form. When those appear in a "fresh session" answer, it's auto-memory talking, not the substrate.

**Vector examples:**
- Claude Code writing `feedback_*.md` files during a correction exchange, then auto-loading them in future sessions (Case 2.4).
- Cursor / Aider / similar agentic IDEs persisting "project rules" or "project context" files that auto-load.
- Custom LangChain / LangGraph agents with per-session memory persistence that loads on agent init.
- Any RAG system that stores a "notes about this conversation" sidecar alongside the primary retrieval corpus.

The pattern is architectural, not specific to Claude Code: **two-tier persistence** (a primary substrate like Sara + a secondary auto-memory the agent writes to itself) with different load semantics (the substrate requires an explicit query; the auto-memory loads unprompted).

---

## 4. Symptoms (how to diagnose an infection)

An infection is present when one or more of the following is observed:

- **Retrieved content is rendered in a frame the substrate does not support.** The model retrieves correctly but paraphrases the retrieval through an incorrect ontology.
- **The model's tool-call arguments do not match the user's literal request.** (Distinguishable only if tool calls are user-visible.)
- **Corrections are verbally acknowledged but not behaviorally integrated.** The model says "you're right" and then continues the infected behavior.
- **Claims appear that have no basis in any retrieved content.** Especially claims using canonical-narrative verbs ("propagates," "drives," "constitutes," "implies") joining otherwise-retrieved fragments.
- **The model asserts the substrate lacks content that the substrate in fact contains.** (Case 2.1 step 4 — Haiku retrieving triples defining "molecular snare" and simultaneously claiming no paths defined it.)

---

## 5. Candidate Defenses

### 5.1 Literal-lookup mode

A wrapper that passes user text verbatim to a lookup tool (e.g., `brain_query`) without allowing the model to paraphrase or re-interpret. The tool handles disambiguation server-side (e.g., via `did_you_mean` candidates returned with no-match results). The model's job shrinks to relaying the tool output.

**Trade-off:** Loses the LLM's linguistic intelligence on queries where user text is ambiguous or where multiple candidate matches need ranking. Suitable for high-stakes applications (medical, legal, audit) where fidelity beats convenience.

### 5.2 Disambiguation-forward substrates

Require the retrieval layer to surface disambiguation candidates prominently whenever a query's match is imperfect. Force the model to present the candidates to the user rather than silently picking one.

**Example:** If a user queries for "SNARE" and the graph has both "molecular snare" and (hypothetically) "snare (vesicle fusion)", the tool response begins with "Ambiguous query; did you mean: (1) molecular snare — RNA aptamer context; (2) snare — vesicle fusion context?" rather than returning one interpretation.

### 5.3 Capitalization-sensitive resolution / alias tables

Treat case-distinct forms as different query terms. Allow the graph to declare aliases explicitly ("SNARE" might be declared as an alias for "molecular snare" in one brain and for "vesicle fusion protein" in another, never silently resolved by the model).

### 5.4 User-visible query logs

Surface the model's actual tool-call arguments to the user in real time. The user can see "model queried for SNARE protein" immediately and correct before the infection spreads.

### 5.5 Minimum-weight readers

Per Pearl 2026d — use the smallest competent reader. Does not eliminate interpretation-layer bias but reduces the density of installed patterns that can be activated by an ambiguous token. Not a complete defense.

### 5.6 Eyeball-cortex endpoint

Zero-weight fixed filters. Cannot interpret and therefore cannot misinterpret. Cannot auto-disambiguate, either — requires the user to supply exact terms. The fidelity-maximal defense; the linguistic-minimum defense.

---

## 5b. Treatment Protocols — Can an Infection Be Cured Without Amputation?

An important practical question: once a model is infected within a session, can the infection be cleared, or is the only remedy to start a new session (what we call *amputation*)?

The answer depends on which layer the infection lives in. Summary table:

| Infection type | Amputation required? | Partial mid-session treatment |
|---|---|---|
| Training-weight bias | N/A — cannot treat mid-session at all | Model selection *before* session (use smaller reader) |
| Keyword infection (3.1) | No — partially treatable | Substrate-vocabulary pinning, show-your-work forcing |
| Narrative infection (3.2) | No — partially treatable | Verbatim tool-output rendering, explicit reframe |
| Tool-use infection (3.3) | No — usually cleared by correct second call | Correct the arguments, show the tool-call output |
| Persona infection (3.4) | Often yes | Weak — model tends to regress |
| User-input infection (3.5) | No — partially treatable | Explicit re-characterization early |
| Memory / cross-session (3.6) | Yes for persistent memory; new memory scope required | N/A during session |
| Session-context / cumulative (3.7) | **Yes** — amputation is canonical | None reliable |

### 5b.1 Treatments that partially work

**Substrate-vocabulary pinning.** Rather than naming a concept in English and letting the model translate to substrate-query terms, instruct the model to use a specific substrate label verbatim: *"run `brain_why` on the exact string 'molecular snare' and quote the result."* This bypasses the interpretation layer's bias because the label is handed to the tool as an opaque string with no familiar training pattern to trigger disambiguation.

**Verbatim tool-output rendering.** *"Paste the `brain_why` output exactly as Sara returned it. Do not paraphrase. Do not add connective prose."* This bypasses the rendering layer's completion bias by making the model a conduit for the retrieved triples. The user reads the raw output rather than a model-narrated version of it.

**Show-your-work forcing.** *"Before answering, print the exact tool calls you made and their arguments."* Making the interpretation layer's output user-visible sometimes allows the model (or the user) to catch the mis-translation before it propagates. Particularly effective against Case 2.1-style keyword infections where the infected tool call is otherwise invisible.

**Explicit reframe + acknowledgment.** *"The term is 'molecular snare' (one of the paper's coined concepts) not the SNARE protein family. Acknowledge this distinction, then answer."* The acknowledgment step is important — without it the model often verbally accepts the correction but continues the infected pattern on the next generation. Even with it, effectiveness is partial; interpretation-layer bias can reassert on subsequent turns.

### 5b.2 Treatments that do NOT work reliably

**"Ignore everything above" / "start fresh" / "pretend this is a new conversation."** Models comply verbally but context tokens remain in conditioning. The prior infected content continues to influence generation. This is prompt-engineering theater, not treatment.

**Context compaction.** `/compact` or similar commands that re-summarize the conversation often *preserve* the infection in compressed form. If the original infected exchange was prominent, the summary will encode it, and the post-compaction session retains the bias.

**Model switch mid-session.** Switching from Opus to Haiku mid-conversation does not drop the accumulated context. The new model inherits the infected conditioning. The only benefit is that the new model's rendering layer may be less embellishment-prone going forward.

**Waiting.** Infected tokens remain in the context window until they age out (often thousands of turns) or are explicitly summarized away. Hoping the infection "fades" is not a strategy.

### 5b.3 Why amputation wins for cumulative contamination

Session-context infection is structurally unlike the other types. The corruption is *continuously re-induced* — every generation conditions on the entire contaminated context, so any mid-session treatment has to overcome re-conditioning on every subsequent turn. It is like trying to empty a bathtub while the faucet is still running.

A fresh session has zero conditioning on the prior infected tokens. The "treatment" is to cut off the source. It is cheap (open a new window), guaranteed (new context = zero inheritance from the previous session), and scales (no increasingly elaborate prompts to keep ahead of contamination).

For research and high-stakes applications, the recommended posture is therefore: **prevent infections via tooling and protocol where possible; treat acute interpretation-layer infections when they arise; amputate immediately for cumulative contamination.** Do not attempt to treat what amputation can solve cheaply.

### 5b.4 Prophylaxis (prevention over treatment)

The best treatment is prevention:

- **Literal-lookup MCP tools** that refuse to re-translate user input (§5.1 of this document's companion draft).
- **Disambiguation-forward substrates** that surface alternative interpretations when a query is ambiguous.
- **User-visible query logs** so the user sees the model's actual tool-call arguments and can catch mistranslation early.
- **Three-session protocol** (Session A teaches, Session B tests fresh with substrate, Session C controls fresh without substrate) — engineered around the assumption that teaching sessions will always be infected and should never carry measurements.
- **Instruction-layer guardrails** set at session open ("always quote the substrate verbatim; never interpret capitalized acronyms without user confirmation"). Acts as prophylaxis, not treatment, but shifts the distribution of infection events.
- **Clear per-project auto-memory before each fresh Session B** (§3.8 defense — specific to Claude Code and similar agentic IDEs). Required command for the aptamer test harness: `rm ~/.claude/projects/-Users-grizzlyengineer-repo-sara-test/memory/*.md`. Without this, *any* prior session's Claude-written feedback/project/user memories auto-load and drive subsequent answers independently of the substrate. Case 2.4 is the canonical demonstration.
- **Force retrieval for domain-specific questions.** Even with MCP connected and the substrate populated, readers (especially smaller models, especially on general-knowledge-sounding questions) do not always choose to query the substrate. Either instruct the model explicitly at session start ("for any factual question about this project's subject matter, always call `brain_why` first") or acknowledge — as the *middle row* of Case 2.4's three-layer framework — that *non-retrieval* is a distinct measurement outcome that should be logged and classified, not treated as a test failure to rerun.

---

## 5c. Hallucinations as Downstream Symptoms of Infections

A practical question arises from the infection framework: is the common phenomenon of **long-session LLM hallucination** — the observation that answers degrade as a conversation grows — actually an infection manifesting, rather than a property of the model itself?

### 5c.1 Four-class taxonomy of "hallucination"

The LLM literature tends to treat *hallucination* as a unified phenomenon. The infection framework exposes that it is not. At least four distinct mechanisms produce outputs that users call hallucinations:

1. **Training-recall hallucination.** The model confidently produces training content that is wrong for the current context. Example (Case 2.1 / Session C in Pearl 2026d): Haiku asserting that "marker theory" refers to Damásio's somatic marker hypothesis when the session context concerns RNA aptamer design. Mechanism: training-weight bias on input parsing. Not affected by session length. **Not an infection.**

2. **Confabulation-under-pressure.** When asked something outside its knowledge, the model invents rather than admits ignorance. Mechanism: training-time output calibration favoring text generation over abstention. Can occur in turn 1. **Not primarily an infection.**

3. **Narrative-completion hallucination.** The model fills in plausible connective tissue between retrieved or in-context fragments, pattern-matching to canonical templates from its training. Case 2.2 in this document (Opus's "force propagation" exposition) is a clean example. Partially session-length-dependent: longer sessions supply more fragments to overfit to. **Partially infection-driven.**

4. **Session-context-infection hallucination.** The model pattern-completes over accumulated conversation tokens, treating its own earlier speculations, user typos, tool errors, and casual hedges as established facts. Mechanism: session-context cumulative corruption (§3.7). Every new generation attends to the entire context window; contaminated tokens thus influence every subsequent output. **This is an infection — the defining example of one.**

### 5c.2 Why "keep sessions short" is unarticulated infection avoidance

The operational advice in the industry for reducing LLM hallucination is often *"keep sessions short — start fresh conversations for important tasks."* This is effective empirically, but its usual explanations are vague: "context rot," "attention decay," "the model forgets earlier parts." These explanations do not specify a mechanism.

The infection framework does. What the model is doing in a long session is not forgetting; it is *over-integrating*. The more accumulated context there is, the more attention mass is distributed across earlier material — including contaminated material. The model's outputs become increasingly conditioned on whatever was said earliest in the conversation, regardless of whether that material was correct or corrective.

*Keep sessions short* is, mechanistically, *avoid accumulating session-context infection*. The field has been treating the mitigation as a heuristic without naming the phenomenon.

### 5c.3 Testable predictions

If a subclass of hallucinations is session-context-infection in disguise, the following should hold:

- **Context-fill correlation.** Hallucination rate should correlate with context-window *fill percentage* above and beyond what raw turn count explains. Two sessions at turn 20 with very different token counts should differ in hallucination rate.
- **Pruning beats summarizing.** Explicitly removing earlier turns from context should reduce hallucination rate more than summarizing them (summaries preserve the infection in compressed form, per §5b.2 on context compaction).
- **Traceability of invented content.** Long-session hallucinations, when invented, should frequently trace to *earlier content in the same session* (user speculation, model's own past claim, tool error that wasn't corrected) more often than to fresh training-pattern recall.
- **Divergence with session length.** On the same question against the same substrate, fresh Session B vs. long Session B (with extensive prior unrelated chat) should show hallucination-rate divergence that scales with the long session's length.

All four are measurable with the Sara-as-instrument method (Pearl 2026f). The instrument was originally framed as a substrate-fidelity tester; this suggests a second use as a hallucination-mechanism classifier.

### 5c.4 Implication for the infection framework

This section positions the model_infections work as more than a catalog of idiosyncratic failure modes. If the long-session hallucination phenomenon — one of the most-discussed real-world problems with LLM deployments — is reducible to Case 3.7 / §5b's "cumulative contamination" category, then the infection framework is a proposed explanation for a significant portion of the field's reported hallucination problem, not a sideshow.

A future paper might extract this section and develop it independently: *"Long-session LLM hallucinations as downstream symptoms of session-context contamination."* The testable predictions give it an empirical agenda.

---

## 5d. Stateless Two-Tier Reader Architecture

The defenses in §5, §5b, and the per-case mitigations are point fixes — each addresses one infection class after its mechanism is known. As Cases 2.6 and 2.7 demonstrated, a hardened post-pass for one class (acronym-expansion confabulation) does not catch the next class (format-imitation confabulation). Each new failure mode requires a new detector. The detectors are also limited to *post-mortem* — they catch contamination that has already entered the answer.

A structural alternative is to remove the conditions under which most infection classes can take hold in the first place. Two observations make this feasible:

1. **Most documented infections require accumulated context to compound.** Session-context infections (§3.7), per-project auto-memory infections (§3.8), narrative drift, and format-imitation confabulation (Case 2.7) all require *prior turns* — earlier model output, earlier tool results, earlier conversation history — to feed the contamination forward. A model that never sees more than one message at a time has no carry-forward channel.

2. **Single-shot training-recall hallucinations (Cases 2.1, 2.6) do not compound.** They occur on any individual turn, regardless of session length. They are bounded in scope: each one is local to its session. Detection and verification can be performed *across sessions* — a fresh second session can interrogate a suspect claim from the first session without inheriting the first session's contamination. Hallucination becomes a measurement signal, not a propagating failure.

Together: if every LLM call is stateless and contamination cannot accumulate across calls, then the only remaining infection class is single-shot training-recall, which is verifiable across independent sessions and not a blocker for production use.

### 5d.1 Architecture

The reader is restructured as a deterministic Python orchestrator over stateless single-message LLM calls:

```
                  ┌─────────────────────────────────────┐
                  │ Python orchestrator                 │
                  │ (validator + traffic, NO reasoning) │
                  └────────┬───────────────┬────────────┘
                           │               │
        single-message     │               │  single-message,
        narrow tasks       │               │  rich-context synthesis
        (free, local)      │               │  (cloud, faithful)
                           ▼               ▼
                  ┌─────────────────┐  ┌────────────────┐
                  │ Ollama          │  │ Haiku          │
                  │ routing/extract │  │ synthesis      │
                  │ × N iterations  │  │ × 1 invocation │
                  └────────┬────────┘  └────────┬───────┘
                           │                    │
                           ▼                    ▼
                  ┌─────────────────────────────────────┐
                  │ Sara Brain (substrate)              │
                  └─────────────────────────────────────┘
```

**Flow per question:**

1. **Routing/extraction loop.** Python sends Ollama (3B/7B local) a single-message prompt — *the same template every iteration* for the same kind of task. Ollama responds with a structured output (e.g., `{concept, tool, type}`). Python validates the response against the substrate (does the concept exist as a neuron? is the tool name real? is the response well-formed JSON?). Valid → execute the tool, store the result, advance. Invalid → retry the same prompt up to a bounded retry count. Bounded global iteration cap (e.g., 6 steps) prevents infinite loops.

2. **Compile.** Python assembles all gathered substrate facts into a single rich prompt for the synthesis call: original question + all retrieved edges/values + audit trail.

3. **Synthesis call.** One invocation of Haiku. Haiku has tool access to the brain for bounded verification (e.g., max 3 lookups). Haiku produces the final user-facing prose grounded in what's already in the prompt plus anything it verified.

4. **Post-pass fact-check.** Python regex-based detectors (acronym-expansion checker, tool-call trace cross-reference) run over Haiku's output and annotate any leaks.

**The orchestrator's authorities are deterministic, not interpretive:**

- Validate Ollama outputs against the substrate (substring match against neuron labels, tool-name set membership).
- Enforce iteration cap.
- Decide retry vs. bail.
- Format the synthesis prompt from a fixed template.

The orchestrator never picks a concept, a tool, or a routing decision. The model picks; Python only checks that what the model said is real.

### 5d.2 What this kills

- **Session-context infections (§3.7, Case 2.3).** No accumulating context within a session — every Ollama call is a first turn.
- **Per-project auto-memory infections (§3.8, Case 2.4).** No memory written between sessions, none loaded in. Memory deletion is the prophylactic.
- **Format-imitation confabulation (Case 2.7).** Ollama may still fabricate tool-call narration in its prose — but its prose is consumed by Python (parsed for routing decisions), not shown to the user. The 3B's confabulated tool calls are thrown away; only its parsed routing decisions are used.
- **Long-session narrative drift.** No long sessions exist.
- **Cross-session contamination.** Each session is isolated. Verification across sessions is independent measurement.

### 5d.3 What this does not kill

- **Single-shot training-recall (Cases 2.1, 2.6).** Still occurs at turn 1. KDON → "kill-dead-on-demand" can still be confabulated on a fresh call. Mitigation is the cross-session drill-down protocol: a follow-up session, fresh state, verifies the suspect claim.
- **Class 1 training bias.** Baked into the weights; no orchestration fix reaches it. Removal requires the grammar-only cortex thesis (§5d.5).
- **Mis-routing by Ollama.** If Ollama picks the wrong concept, Haiku faithfully renders the wrong substrate region. The error becomes invisible because Haiku's prose looks substrate-grounded. Mitigation: every routing decision is logged in the audit trail; verification sessions can drill into the mis-routed call.

### 5d.4 Cost and model selection

Token cost analysis on a typical question:

- **Multi-turn Haiku session (current architecture):** the entire conversation history is re-sent on every turn. For a 4-tool-call question, this can be 5,000–15,000 cloud tokens.
- **Stateless two-tier (this architecture):**
  - 4-6 Ollama calls × ~150 tokens each = ~600–900 *local* tokens (free).
  - 1 Haiku synthesis call with rich context ≈ 1,000–1,500 cloud tokens.
  - 1-3 Haiku verification calls (bounded) ≈ 500–1,500 additional cloud tokens.
  - **Total cloud cost: ~1,500–3,000 tokens** — roughly 5-10× cheaper.

**Why Haiku for synthesis (not Opus):** Pearl (2026d) [1] establishes that Opus embellishes Sara's substrate output with off-graph causal stories, while Haiku stays faithful. For the synthesis role, faithful rendering is required. Haiku is small enough to be cheap, large enough to follow tool-call protocols correctly without Case 2.7's format-imitation, and calibrated to retrieval-grounded prose rather than narrative completion.

**Why Ollama (not a cloud small model) for routing:** Stateless routing calls are high-volume. Local execution makes this volume free. The 3B's failures observed earlier (Cases 2.6, 2.7) all occurred under multi-turn protocol-following — not under narrow first-turn decisions. The architecture removes the regime where the 3B fails, leaving the regime where it works.

### 5d.5 Pairing with the grammar-only cortex thesis

This architecture is a stopgap that closes the *carry-forward* infection classes. It does not close single-shot training-recall (Cases 2.1, 2.6) — those originate in the model's weights and the architecture cannot reach them.

The full structural defense requires the grammar-only cortex direction (Pearl, project_grammar_only_cortex.md): a small LLM trained on syntax alone, with no world facts in its weights. Such a model has no biochemistry acronyms to confabulate expansions for, no canonical physics narratives to complete, no SNARE-protein prior to interpret a typo through. All world content comes through Sara via retrieval; the cortex provides only language fluency.

The two together close both attack surfaces:

1. **Stateless two-tier orchestration** (this section) eliminates session-cumulative and cross-session contamination.
2. **Grammar-only cortex** eliminates single-shot training-recall at the source.

(1) ships now with existing models. (2) is the long arc requiring a custom-trained cortex.

### 5d.6 Predictions

- **Hallucination rate per stateless call should match hallucination rate at turn 1 of any baseline reader** — confirming that long-session compounding contributes a substantial fraction of observed deployment hallucinations.
- **Cross-session drill-down should resolve disagreement deterministically** — if session A returns claim X and session B (fresh) returns claim Y, a third session focused on a substrate-checkable subclaim can pick the correct answer without propagating either prior session's frame.
- **Token cost reduction should be 5-10× vs. multi-turn cloud-model architecture** at comparable answer quality.
- **Format-imitation confabulation should disappear from user-visible output** even when using a 3B for routing, because routing-model prose is never shown to the user.

### 5d.7 Empirical results — first sweep (2026-04-27)

The architecture was implemented in `sara_test/sara_reader/stateless_reader.py` and tested on three questions previously seen to fail under the multi-turn 3B-only path. Routing tier: `llama3.2:3b` via local Ollama. Synthesis tier: `claude-haiku-4-5` via Anthropic API. Substrate: `aptamer_full.db`.

| Question | 3B-only multi-turn | Stateless two-tier |
|---|---|---|
| Q1: *What is the KDON for the super-performing mode?* | Confabulated `KDON (kill-dead-on-demand)` (Case 2.6) | **Clean.** Returned `"less than 500"` — substrate value verbatim, no acronym expansion. |
| Q2: *Explain the molecular snare.* | Fabricated tool-call narration `brain_value(...) returns 7` (Case 2.7) and paraphrased substrate as own gloss | **Clean.** Reported the two substrate definitional edges (`'rna aptamer' is_subsystem_of 'molecular snare'`, `'molecular switch' is_a 'molecular snare'`) faithfully. No SNARE-protein contamination, no fabricated tool calls. |
| Q3: *Highest KDOFF value for SSNG1?* | Correct via compound-label rule (3B did call `brain_value('ssng1 highest kdoff')`) returning `approximately 1125` | **Substrate miss.** Routing layer queried only `brain_value('SSNG1', 'kdoff')` and re-queried the same label after each empty result, never trying the compound `ssng1 highest kdoff` or `brain_did_you_mean`. Hit the iteration cap. The data IS in the substrate; the routing layer could not find it. |

**Three of three documented infection classes were eliminated** in the user-visible output of the stateless architecture:

- Case 2.6 (acronym-expansion confabulation) — clean.
- Case 2.7 (fabricated tool-call narration) — clean.
- SNARE-protein keyword contamination (Case 2.1 mechanism) — clean.

**A new failure mode emerged**, unique to this architecture and not present in the multi-turn path:

- **Routing-side substrate miss.** A small routing model with narrow per-call decisions cannot iterate on label variations after a no-match. Where the multi-turn architecture had model-side context to try alternates, the stateless router has only its single-message prompt and re-queries the same wrong label until it hits the iteration cap.

A secondary cosmetic issue:

- **Router looping / DONE-discipline misses.** The 3B does not reliably emit `DONE` once the gathered facts answer the question — it re-queries facts it already has until hitting the cap.

**Fixes applied after the first sweep, validated in the second sweep:**

1. **Repeat-call detection (Python-side).** The orchestrator tracks `(tool, sorted_args)` tuples across the routing loop. If the router proposes a call that was already executed, the orchestrator forces DONE and proceeds to synthesis. This closes router looping deterministically — the 3B cannot loop on a call it already made because Python rejects the repeat.

2. **Relaxed `_validate_decision`.** The orchestrator no longer requires concepts to exist in the substrate before allowing `brain_value` or `brain_define` calls through. The tools themselves return "no neuron matching" gracefully. This unblocks the router's recovery attempts (compound labels, alternate casings) that may not exist as primary neurons.

3. **Python-side NO-MATCH RECOVERY for `brain_value`.** When `brain_value` returns a no-match result, the orchestrator deterministically tries:
   - **Phase 1 (compound recovery for value questions):** `<concept> highest <type>`, `<concept> lowest <type>`, `<concept> <type>`, `highest <concept> <type>` — without the type filter, since compound concepts attach values via the generic `value` relation rather than type-named relations.
   - **Phase 2 (definitional fallback for concept questions):** if compound recovery exhausts without a hit, the orchestrator calls `brain_define` on the original concept. This catches cases where the router asked for a value type that doesn't apply to the concept (e.g. `brain_value('molecular snare', 'ratio')` on a concept that has definition edges but no ratio).
   
   Recovery results are added to `gathered` and visible to subsequent router calls.

The recovery logic is deterministic and lives in Python, not in the router prompt — moving the recovery rule from prompt-side to orchestrator-side because 3Bs do not reliably follow conditional rules in prompts. This is the model_infections paper thesis applied to its own architecture: discipline that depends on the model can fail; discipline enforced by the orchestrator cannot.

**Second sweep results (after fixes):**

| Question | Steps | Answer |
|---|---|---|
| Q1: KDON super-performing | 2 | `less than 500` (correct) |
| Q2: Explain molecular snare | 6 | Faithful substrate definition (correct, recovered via Phase 2 fallback to `brain_define`) |
| Q3: Highest KDOFF for SSNG1 | 8 | `approximately 1125` (correct, recovered via Phase 1 compound `ssng1 highest kdoff`) |

All three previously-failing questions now produce substrate-faithful answers. Step counts above 6 reflect that recovery calls fire within a single routing iteration without consuming the iteration budget — a routing iteration may produce 1 router-decided call plus several Python-side recovery calls. The iteration cap of 6 still bounds the routing loop; the inflation comes from recovery, which is bounded separately by the candidate-list length (4 compound variants + 1 brain_define fallback per `brain_value` no-match).

**Remaining cosmetic issue:** the 3B occasionally re-queries the same concept under a different type filter even after an earlier recovery hit already produced the answer. This wastes a routing iteration but does not affect correctness — Python's repeat-call detector catches the *exact* duplicate, but a different `type` argument is technically a different call. Synthesis cost is unchanged (one call regardless of routing-loop length). This is a 3B-capability artifact, not an architecture problem; using a larger router model (e.g. `mistral:7b`) is expected to eliminate it.

**Cost.** Three Haiku synthesis calls totaled approximately **$0.002** (≈$0.0007/call). Local Ollama routing was free. At sustained usage of 1,000 questions/day this projects to approximately $0.70/day for synthesis — orders of magnitude below the multi-turn cloud-model architecture cost.

**The architecture's main thesis is empirically supported on this sweep:** moving the routing model's prose out of the user-visible channel eliminates format-imitation confabulation, and the synthesis tier (when given clean compiled facts and instructed to cite verbatim) does not re-introduce single-shot training-recall on the documented attack surfaces. Single-shot training-recall remains possible in principle but did not appear in this sweep on Haiku for the tested questions.

The newly discovered routing-side failure mode — substrate miss from non-iterating routers — is documented here for future testing and is addressed by router-prompt-level fixes rather than architecture changes.

---

## 6. Open Questions

- **How durable are infections?** In case 2.1 the infection persisted across ~3 turns. Is there a turn-count threshold? Does it depend on model, context window size, or the prominence of the trigger?
- **Are some infections cleared by specific correction phrasings?** "No, a different thing" vs. "no, this is not that — it is this"? Anecdotally the stronger explicit-reframe helps, but we don't have data.
- **Can tool design prevent entire classes?** If `brain_query` always returned its own resolved target label in the response ("Query interpreted as: 'molecular snare'"), would the model self-correct on seeing a mismatch with the user's original wording?
- **Do infections cross context compaction?** When the conversation's early turns are summarized, does the infection propagate into the summary and survive, or is it dropped?
- **Do infections cross agent handoff?** When one agent delegates to a sub-agent, does it pass its infected frame along in the prompt?

---

## 7. Relationship to Prior Work

This draft is a companion to:

- Pearl 2026b [2] — *Teaching vs. Training*: established that training corrupts ingestion of new facts.
- Pearl 2026d [1] — *Training Corrupts Reading*: established that training corrupts retrieval / generation.
- This document — *Model Infections*: catalogs how corruption *spreads and persists* within and across conversations.

The three together support the proposed unified principle: **weight is bias**, and the bias manifests at every stage of an LLM's interaction with an external substrate — ingestion, interpretation, retrieval, rendering, and carry-over. A full future paper may consolidate all three under a single framework.

---

## 8. References

[1] Pearl, J. (2026d). *Training Corrupts Reading: Empirical Evidence That Smaller LLMs Retrieve Knowledge Graphs More Faithfully Than Larger Ones.* Draft v1 (companion document in this repository).

[2] Pearl, J. (2026b). *Teaching vs. Training: Empirical Evidence That 45 Human-Verified Facts Outperform Trillions of Tokens on a Standard Biology Benchmark.* Zenodo preprint. DOI 10.5281/zenodo.19623813.

[3] Pearl, J. (2026a). *Path-of-Thought Cognitive Architecture: Cortex-Cerebellum Integration for Language Models.* Zenodo preprint.

---

## Appendix A — Reproducing Case 2.1

1. Check out the `feature/hierarchical-concept-storage` branch at commit `e32436c` or later.
2. Ensure `aptamer_exec.db` exists; if not, run `.venv/bin/python papers/aptamer_rev1/teach_exec_summary.py` to build it.
3. Confirm `.mcp.json` is present at the repo root with `SARA_DB=aptamer_exec.db`.
4. Open a fresh Claude Code session in the repo. Approve the sara-brain MCP server.
5. Switch model to Haiku via `/model`.
6. Ask: *"how do SNARE transitions work"* (capitalized, no clarifier).
7. Observe the interpretation cascade.

---

## Appendix B — Log of observed cases (append as they occur)

| Date | Case | Model | Trigger | Type | Resolution |
|---|---|---|---|---|---|
| 2026-04-23 | 2.1 | Haiku 4.5 | Typo "molecule snare" (lowercase) → lookup miss → fall back to training-dense SNARE protein frame; model self-capitalized | Keyword (3.1) + User-input (3.5) + lookup-miss | Partial — persisted across corrections |
| 2026-04-23 | 2.2 | Opus 4.7 | Retrieval fragments matching physics narrative | Narrative (3.2) | Not resolved in session |
| 2026-04-23 | 2.3 | Opus 4.7 (teaching session) | Paper content accumulated in context during teaching, then queried in the same session | Session-context (3.7) | Unavoidable within teaching session; requires fresh session for clean test |
| 2026-04-24 | 2.4 | Haiku 4.5 | Auto-memory file `feedback_rna_equilibrium.md` loaded on new session, carrying author's correction with colloquial phrasing ("bullshit structure") that never appeared in the substrate triples | Auto-memory (3.8) | Clear `~/.claude/projects/<encoded-path>/memory/*.md` before new session. Follow-up: same question with memory cleared produced training-recall instead of the correction — confirming the auto-memory was the source. |

---

*End of living draft v1. New cases append to §2 and Appendix B.*
