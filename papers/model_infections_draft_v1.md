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

**Three-layer fidelity framework this produces:**

| Layers available | Observed output |
|---|---|
| Memory + Sara + Training | Author's voice (memory drives) |
| Sara + Training (memory cleared) | Training-recall (Sara bypassed because model doesn't reach for tool unprompted) |
| Training only (no memory, no MCP) | Training-recall (Sara not present) |

The interesting row is the middle one. **Having Sara connected does not guarantee Sara is used.** The harness can serve the substrate; it cannot force the reader to ask.

This argues for protocol-level interventions that push the reader toward MCP, e.g.:
- System-prompt-style instruction: *"For any factual question about this project's subject matter, always call `brain_why` first before answering from training."*
- Restricted model configuration: *"Disable direct-answer mode; tool-call is mandatory for domain-specific queries."*
- Or — more honestly — **acknowledge that the measurement must include whether the model chooses to retrieve, and treat no-retrieval-when-substrate-available as its own result class.**

For the instrument paper: this triad is the cleanest available demonstration that Session B's purity requires more than a new Claude window. It requires **cleared auto-memory** AND **a question phrasing that triggers retrieval** AND possibly **explicit tool-call instruction**. Any one of those missing changes what the measurement measures.

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
