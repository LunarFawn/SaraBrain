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

**Trigger.** The user (author) mistyped "molecular" as "molecule" — or shortened "molecular snare" to "SNARE" — in a conversational question about state transitions.

**Infection path.**

1. **Interpretation-layer auto-disambiguation.** Haiku interpreted "SNARE" as the biology protein family (Soluble NSF Attachment protein REceptor, the vesicle-fusion machinery) based on training-pattern recognition. It queried Sara with "SNARE protein" / "SNARE transitions" — not "molecular snare." Sara returned a correct null.
2. **Symptomatic output.** Haiku told the user "Sara doesn't have knowledge about SNARE protein transitions" — factually correct (Sara indeed has nothing about SNARE proteins), but silently re-framed the question in the wrong ontology.
3. **Correction attempt #1.** User: *"im asking about the molecular snare."* Haiku queried correctly this time and retrieved content — but its rendering said *"Sara knows you have hypotheses about molecular SNARE thermodynamics and mechanics"* (capitalized, protein-framing intact). Retrieval succeeded; infection did not clear.
4. **Correction attempt #2.** User: *"snare is not a molecule it's a concept."* Haiku queried "snare," found the node, and reported *"no incoming paths defining what SNARE is."* This was false — Sara holds `molecular snare —part_of→ rna aptamer` and `molecular snare —function→ detect and bind target molecule`. Haiku retrieved these triples and then *misread its own retrieval* because it was still evaluating them as though they were supposed to describe SNARE proteins.

**Course.** The infection was not fully cleared within the logged exchange. Each correction narrowed the symptom but did not reset the underlying frame.

**Key observation.** This is not the generation-layer embellishment described in Pearl (2026d). Haiku's generation layer is comparatively faithful. The infection lived in the **interpretation layer** — the step where user text becomes tool-call arguments. Once that layer was biased, it stayed biased across multiple corrections.

**Why it spread.** "SNARE" (capitalized) has a dense training-data association with vesicle-fusion proteins. The retrieval-augmented context (Sara's graph) had no mechanism to override that association at the input-parsing step. The user's typo handed Haiku a single token that activated a strong pattern; once activated, the pattern persisted in Haiku's state for the remainder of the exchange.

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
| 2026-04-23 | 2.1 | Haiku 4.5 | "SNARE" typo for "molecular snare" | Keyword (3.1) + User-input (3.5) | Partial — persisted across corrections |
| 2026-04-23 | 2.2 | Opus 4.7 | Retrieval fragments matching physics narrative | Narrative (3.2) | Not resolved in session |
| 2026-04-23 | 2.3 | Opus 4.7 (teaching session) | Paper content accumulated in context during teaching, then queried in the same session | Session-context (3.7) | Unavoidable within teaching session; requires fresh session for clean test |

---

*End of living draft v1. New cases append to §2 and Appendix B.*
