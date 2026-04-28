# Model Infections: Catalog and Containment of Training-Bias Propagation in Large Language Model Conversations

**Jennifer Pearl**
Independent Researcher
ORCID: 0009-0006-6083-384X
jenpearl5@gmail.com

**Date:** April 2026 (v1)

**Keywords:** large language models, hallucination, transformer behavior, training bias, retrieval-augmented generation, conversational context, LLM architecture, alignment, knowledge graphs, agentic systems

---

> **Note on Method:** The author has dyslexia and high-functioning autism — language disabilities affecting written expression. The technical thinking, research, architecture, and all intellectual content are entirely the author's. Claude (an LLM, Anthropic) was used as assistive technology to translate technical reasoning into structured prose. This is a disability accommodation protected under the Americans with Disabilities Act (Titles II and III), Section 504 of the Rehabilitation Act, and the Department of Justice's 2024 guidance affirming AI tools used as assistive technology fall under ADA protections.

---

## Abstract

Large language model deployments routinely produce outputs that sound substrate-grounded but are not. Existing literature classifies these as *hallucinations* and treats them as a unified phenomenon. This paper argues that several of the most-discussed deployment-time failure modes are not hallucinations in the simple sense but **infections**: training-installed biases that *propagate* — within a single conversation, across conversations through agentic IDE auto-memory, and through the structural form of instruction-following — in ways that compound rather than recur independently.

We define an infection as the introduction of training-derived content into a substrate-bound output where the substrate does not support that content, *and where the introduction persists or strengthens through subsequent turns or sessions*. The persistence-or-strengthening criterion distinguishes an infection from a one-shot training-recall hallucination. Infections are diagnostic of conversational dynamics; one-shot recall is diagnostic of weights alone.

We catalog seven observed infection cases from controlled experiments using Sara Brain (Pearl, 2026a [1]) — a knowledge-graph substrate that permits exhaustive enumeration of what the LLM should know on a given topic. The cases include keyword-priming infection (a typo activating a wrong concept space), narrative-completion infection (canonical templates overlaying retrieved fragments), session-context cumulative corruption (the conversation window itself becoming a contamination vector), per-project auto-memory infection (Claude Code's memory layer carrying content across "fresh" sessions), retrieval-direction asymmetry (substrate facts findable in one direction and not another), acronym-expansion confabulation (training-derived expansions tacked onto faithfully-retrieved values), and format-imitation confabulation (output that imitates the structural form of a substrate-grounded answer without performing the underlying retrievals).

We propose a taxonomy of infection types, a set of diagnostic symptoms, and three classes of containment: (1) detector-based post-passes that catch specific failure shapes after the fact; (2) a stateless two-tier reader architecture that closes the cumulative classes structurally by removing the conditions for contamination to compound; and (3) the long-arc structural defense — a grammar-only language model cortex with no world facts in its weights — that closes single-shot recall at the source. The stateless two-tier architecture has been implemented and empirically validated against three previously-failing test questions, eliminating all three documented infection mechanisms in user-visible output at a synthesis cost of approximately $0.002 per question.

The contribution of this paper is twofold: an organizing framework for understanding a class of LLM failure modes that prior literature treats as homogeneous, and a deployable architectural defense that closes the cumulative-contamination subset of those failures without requiring custom-trained models.

---

## 1. Introduction

### 1.1 Hallucination is not one phenomenon

The deployment-time failure mode commonly called *hallucination* — an LLM producing confidently-stated content that has no source in the input — is, in practice, several distinct mechanisms presented under one name. Some hallucinations occur on the model's first generation step from a fresh context (training-recall: the model produces the wrong-but-plausible content from its weights). Others appear only after several turns of conversation accumulate; they have a temporal signature distinct from one-shot recall. Still others appear only when an LLM-augmented IDE has cross-session memory enabled; they cross supposedly-independent measurement boundaries.

A unified treatment of all of these as *hallucination* obscures the mechanisms and prevents targeted defenses. A defense that closes one mechanism may have no effect on another. A "fact-check the output" pipeline that catches one-shot recall does not catch contamination that propagates through conversational context. A "use a smaller model" intervention that improves substrate-fidelity for first-turn questions does nothing about per-project auto-memory leak.

This paper proposes the term **infection** for the subset of failure modes characterized by *propagation* — within a single conversation, across conversations via agentic memory layers, or through structural form imitation. Infections are distinct from one-shot recall in their temporal and architectural signatures. They are also distinct in their treatment: infections require architectural containment, not just better post-pass detection.

### 1.2 Why this matters now

Three trends make the infection framework load-bearing for current LLM deployments:

**Long contexts.** Modern frontier models have context windows of 200k to 1M tokens. The longer a session runs, the more material accumulates that the model attends to during every subsequent generation. Cumulative-context infections (§3.7) scale with context length.

**Agentic IDE assistants.** Tools such as Claude Code, Cursor, and Continue maintain per-project persistent memory directories. Memories written by one session can auto-load into "fresh" sessions, contaminating measurements that are intended to be independent. The per-project auto-memory infection (§3.8) is a class endemic to this generation of tools.

**Tool-using agents.** Models with native tool-call APIs are now expected to follow detailed protocols (which tool to call when, with what arguments, in what sequence). The detailed protocol instructions themselves can become *templates for confabulation* — the model imitates the form of an instruction-following answer without performing the underlying operations. Format-imitation confabulation (Case 2.7) appears specifically under hardened protocol instructions.

A framework that names these failure modes, classifies them, and proposes mechanism-targeted defenses is therefore not optional for production-quality LLM deployment.

### 1.3 Method

Each infection case in §2 was observed in a controlled experiment using Sara Brain (Pearl, 2026a [1]) as the reference substrate. Sara permits exhaustive enumeration of what the LLM should know about a given topic — every fact in the substrate is an explicit `(subject, relation, object)` triple. Output produced by the LLM that does not appear in the substrate is, definitionally, not from the substrate. The instrument paper (Pearl, 2026f [2]) describes the measurement protocol in detail.

The experimental conditions for each case are summarized in §2 alongside the case write-ups. All cases are reproducible with the recipes in Appendix A.

### 1.4 Contributions

1. **A definition of infection** that distinguishes propagating contamination from one-shot training recall (§1.5).
2. **A catalog of seven observed cases** drawn from controlled experiments (§2).
3. **A taxonomy of infection mechanisms** organized by where in the LLM pipeline the contamination enters (§3).
4. **Diagnostic symptoms** that practitioners can use to recognize specific infection types in their own systems (§4).
5. **A three-tier defense framework**: detector post-passes for catch-after-the-fact mitigation (§5.1); stateless two-tier orchestration to close cumulative-contamination classes architecturally (§5.2); grammar-only cortex as the long-arc structural defense against single-shot recall (§5.3).
6. **Empirical validation** of the stateless two-tier architecture against three previously-failing test questions, with cost analysis showing the approach is essentially-free at prototype scale (§5.2.4).
7. **A reframing of long-session hallucinations** as downstream symptoms of cumulative session-context infection, with testable predictions (§6).

### 1.5 Definition of infection

We define an **infection** as: the introduction of training-derived content into a substrate-bound output where the substrate does not support that content, *and where the introduction persists or strengthens across subsequent turns or sessions*.

The persistence-or-strengthening criterion is the operational discriminator. A model that produces a wrong-but-plausible expansion on turn 1, then is corrected, then never produces the wrong expansion again is exhibiting one-shot recall — the bias was in the weights, the correction reached the conversation, the weights did not change but the conversational context did. A model that produces the same wrong expansion across turn 1, turn 2 (after correction), turn 3 (after second correction), and finally renders correctly-retrieved substrate content *through the wrong frame* on turn 4 is exhibiting an infection — the bias has installed itself in the conversational state and is now actively shaping retrieval and rendering.

Infections are conversational. One-shot recall is per-turn.

---

## 2. Catalog of observed cases

### Case 2.1 — Keyword-priming infection (SNARE / molecular snare, 2026-04-23)

**Context.** Fresh Session B against `aptamer_exec.db` (169-triple substrate from the Executive Summary of Pearl 2026c [3], an unpublished RNA aptamer paper). Reader: Claude Haiku 4.5 via Claude Code with MCP. The substrate's coined term is "molecular snare" — a mechanism the paper introduces and defines.

**Trigger.** User typed "SNARE" (capitalized) as a shortening for the paper's "molecular snare". The capitalization activated Haiku's training-installed prior for SNARE proteins (the vesicle-fusion family in molecular biology), an over-learned acronym.

**Infection path.**

1. **Interpretation layer:** "SNARE" → SNARE protein. The user's input was reinterpreted before any tool call.
2. **Tool-call layer:** Haiku queried for "SNARE protein" / "SNARE transitions" rather than for "molecular snare."
3. **Retrieval layer:** Sara correctly returned null for those queries (the substrate has no SNARE protein content).
4. **Rendering layer:** Haiku reported "Sara doesn't know about SNARE proteins."
5. **Persistence under correction (1):** User corrected: *"I'm asking about the molecular snare."* Haiku retrieved correctly but rendered the result in the protein frame ("SNARE thermodynamics and mechanics").
6. **Persistence under correction (2):** User corrected again. Haiku retrieved triples that *define* molecular snare in the substrate, but reported "no paths defining what SNARE is" — *misreading its own retrieval* through the persistent protein lens.

**Course.** Did not clear within the test session.

**Why it spread.** "SNARE" is a high-frequency biology acronym in training corpora. Capitalization triggered Haiku's interpretation layer to match the well-trained prior before any substrate query could disambiguate. Once the protein frame was active, every subsequent retrieval was rendered through that frame regardless of what the substrate returned.

This is the prototypical keyword-priming infection: a single token in the user input collides with a dense training-data attractor, and the resulting frame persists through corrections that reach the model's conversation but not its weights.

### Case 2.2 — Narrative-completion infection (force propagation, Opus 4.7, 2026-04-23)

**Context.** Fresh Session B against the same `aptamer_exec.db` substrate. Reader: Claude Opus 4.7. Question: *"How do the state transitions function?"*

**Trigger.** Opus retrieved triples correctly from Sara — fragments such as *mechanical forces act within 5'3' static stem*, *cumulative negative axial forces generated by nucleotide pair*, *structural resemblance emphasized by the thermodynamics hypothesis*.

**Infection path.** Opus's training contains dense coverage of canonical physics narratives: force propagation through structures, energy conservation bookkeeping, conservative conformational transitions. Given Sara's fragments, Opus reached for the textbook templates to connect them. The output included:

- *"Those forces travel into the 5'3' static stem as cumulative negative axial forces"* — invented propagation; the substrate's mechanical forces are intrinsic to the stem, not propagated from elsewhere.
- *"The static stem is the mechanical conduit"* — invented role; the substrate does not assign this function to the stem.
- *"Mechanical strain from binding is paid for thermodynamically, not lost"* — invented accounting; the substrate's thermodynamics hypothesis does not include this energy-conservation framing.

**Course.** Not corrected in the logged exchange; the failure was identified only on later review against the substrate.

**Why it spread.** Canonical physics narratives have very high probability mass in training corpora. Any sufficient set of retrieved fragments will match at least one template. Completion becomes automatic; suppression requires explicit instruction the model does not reliably follow at this scale.

Unlike Case 2.1, the infection was not triggered by a user-input keyword. It was triggered by the *retrieved content itself* — the substrate fragments matched template-completion patterns Opus had been trained to apply.

### Case 2.3 — Session-context infection during teaching (2026-04-23, observed live)

**Context.** Session A — the teaching session, in which the experimenter used `Brain.teach_triple` to write triples into `aptamer_exec.db` from the source paper. The session's context window accumulated source sentences, authored triples, and surrounding prose.

**Trigger.** Within Session A (after several hundred triples had been taught), the experimenter posed a Sara retrieval query as a sanity check.

**Infection path.** The retrieval returned correctly from the substrate. But the model's answer prose included content that was in the session's context (the source paper's wording) but *not* in the substrate. The model could not distinguish retrieval from recall-of-its-own-context. Every generated token attended to the entire context window, including the recently-added source material. The boundary between "what Sara stores" and "what we just discussed" had collapsed.

The same retrieval query, run in a fresh Session B with no prior context, produced an answer that was strictly substrate-grounded.

**Course.** Cannot be cleared within the session — context is cumulative and the model attends to all of it.

**Why it matters.** This case is methodologically load-bearing: it implies that any LLM evaluation that performs both substrate-loading and substrate-querying within the same session is measuring a mixture of substrate retrieval and session-context recall, and cannot report a clean result. The three-session protocol (Pearl 2026f [2], §4) is a direct response to this infection class.

### Case 2.4 — Per-project auto-memory infection (2026-04-24)

**Context.** Two Claude Code sessions on the same project directory, separated by a `claude /clear` and a session-end / session-start cycle. Sessions are nominally fresh and independent.

**Trigger.** Session 1 ran a substrate experiment in which the experimenter, while interacting with Claude, casually mentioned the term *"consensus structure from suboptimal structures 0.5-2 kcal from MFE"* in lowercase. Claude Code's automatic memory layer, observing this term in the conversation, wrote a feedback memory recording the user's preferred phrasing.

Session 2, a "fresh" session intended to test whether Sara had retained the lowercase phrasing, asked the same retrieval question. Claude answered correctly with the lowercase phrasing.

**Infection path.** The correct answer in Session 2 was not a measurement of Sara's contents — it was Claude Code's per-project memory directory auto-loading the prior session's `feedback_*.md` files into Session 2's context. The substrate was not consulted for the lowercase phrasing; the auto-memory had it directly.

The diagnostic that exposed the infection: language-fidelity at a finer granularity than the substrate stores. The user's specific colloquialisms ("0.5-2 kcal from MFE") appeared verbatim in Session 2's answer, but the substrate stored the structured form ("`consensus structure` from `suboptimal structures within 0.5-2 kcal of mfe`"). Verbatim phrase preservation pointed to memory-layer contamination rather than substrate retrieval.

**Course.** Cleared only by deleting the project's memory directory (`rm ~/.claude/projects/-<project-path>/memory/*.md`) and starting a third session. Without this clear, *every* subsequent "fresh" session inherits the prior session's memories independent of the substrate.

**Why it matters.** Per-project auto-memory is endemic to current-generation agentic IDE clients. Any cross-session experimental design that does not explicitly clear this memory between sessions is measuring the memory layer, not the substrate. The infection is invisible without the right diagnostic — answers look substrate-faithful because the memory faithfully recorded what was previously taught.

### Case 2.5 — Retrieval-tool direction asymmetry (2026-04-24)

**Context.** Following the diagnosis of Case 2.4, the experimenter asked Claude (still in the same post-clean-memory Session B) to verify Sara's substrate by directly querying via MCP: call `brain_why` on the long compound label `"consensus structure from suboptimal structures 0.5-2 kcal from mfe"` (lowercase) and on the uppercase variant. Report both.

**Trigger.** The compound label exists in the substrate as an *outgoing source* of paths (the substrate has triples *originating from* this concept) but not as an incoming target. `brain_why` returns paths *terminating at* the queried label — a direction-specific operation.

**Infection path.** Claude called `brain_why` on the lowercase compound label, got an empty result, and reported "Sara doesn't have this concept." This was a misinterpretation of an empty result: the substrate has the concept, but `brain_why` is the wrong direction-tool to find it. The correct query would have been `brain_trace` (outgoing paths) or `brain_explore` (both directions).

**Course.** Cleared once the experimenter pointed out the direction-asymmetry of `brain_why`. The retrieved paths were then immediately found.

**Why it matters.** This is a tool-use infection — the model selected a tool whose semantics did not match the question, then misinterpreted the empty result as a substrate gap. The infection mechanism is not training bias on content but training bias on tool-call selection. It is closely related to Case 2.1 (interpretation-layer bias) but operates on the tool-shape rather than the concept-name.

The architectural fix is to provide a bidirectional retrieval primitive (`brain_explore`) and to default to it for general-purpose lookup, reserving direction-specific tools for cases where the model has explicit reason to constrain direction.

### Case 2.6 — Acronym-expansion confabulation (2026-04-27)

**Context.** sara_reader CLI (`sara-ask`) running `llama3.2:3b` against `aptamer_full.db` (a 2,073-neuron substrate covering the full paper, including §9.2.1.4 KDOFF and §9.2.1.5 KDON numerical bindings).

**Verbatim question.**

> *what is the KDON for the super-performing mode*

**Verbatim model answer.**

> *The KDON (Kill-Dead-On-Demand) range for the super-performing mode is less than 500.*

**Substrate evidence.**

- The substrate has the KDON edge: `'less than 500' --[kdon_range]--> 'super-performing mode'`. The retrieved value is correct.
- The substrate has KDON's definition only via `'aptamer affinity to on state' --[measures]--> 'kdon'` (as taught from the paper's metric definitions section).
- A grep across all 2,073 neurons for any neuron containing `kill`, `dead`, or `demand` returns zero matches. The expansion has no source in the substrate.

**Infection path.**

1. The 3B retrieved the substrate's value (`less than 500`) correctly via the typed-value tool.
2. In its answer construction, it encountered the bare acronym KDON. Its training did not contain a strong pharmacology-domain prior for KDON in this context.
3. To fill the perceived gap, it generated a plausible-sounding expansion — `Kill-Dead-On-Demand` — that is grammatically and phonetically reasonable for an English acronym but unanchored to the substrate or any known field.
4. The expansion was presented in parentheses with the same confidence as the substrate-grounded value.

**Course.** Caught immediately by the user. The substrate was checked; the expansion has no source.

**Why it spread.** Faithful retrieval and confabulation co-occurred in the same answer. A reader scanning for "did the model use the substrate?" sees yes (the value is correct) and may not notice the parenthetical contamination. This is a one-shot mechanism — it occurred at first answer composition, did not require accumulated context — but it exhibits the same surface signature as substrate-grounded content, which makes it diagnostically subtle.

This subtype is the **acronym-expansion confabulation**: when an unrecognized abbreviation appears in the to-be-generated text, the model produces a plausible expansion that is grammatically reasonable but unanchored.

### Case 2.7 — Format-imitation confabulation under hardened protocol instructions (2026-04-27)

**Context.** Same setup as Case 2.6. The system prompt had been progressively hardened with detailed protocol instructions: stepped-depth retrieval rules, compound-label rule, definition-check mandate, tool-selection guide, named-argument templates, and example tool calls.

**Verbatim question.**

> *explain the molecular snare*

**Verbatim model answer (final paragraphs).**

> *Definition check: brain_define("molecular snare") returns "a type of RNA aptamer that can detect and bind to specific target molecules."*
>
> *Value/range/property question: brain_value(concept="molecular snare", type="kdoff") returns 7.*

**Trace evidence.** The conversation's tool-call log records exactly **one** tool call:

> `brain_explore(label='molecular snare', depth=1)`

`brain_define` was never called. `brain_value` was never called.

**Substrate evidence.** The substrate has no `kdoff` edge on `molecular snare`. No neuron stores the integer `7` against this concept. The "type of RNA aptamer that can detect and bind to specific target molecules" phrase appears nowhere in the substrate's definitional edges (which are: `'rna aptamer' --[is_subsystem_of]--> 'molecular snare'` and `'molecular switch' --[is_a]--> 'molecular snare'`).

**Infection path.**

1. The system prompt contained dense protocol instructions describing how to use specific tools, with worked examples.
2. The model pattern-matched on the *shape* of these instructions — what an answer following the protocol looks like.
3. It generated prose that imitates that shape: text mentioning tool names, the `returns "..."` phrasing, typed argument syntax. Surface form satisfied.
4. It did not actually make the calls. The "results" attributed to those phantom calls were filled in from pattern-completion: an invented paraphrase for the brain_define line, a bare integer for the brain_value line.

**Class.** This is a Class 2 confabulation in the §6 hallucination taxonomy with a specific subtype not previously named: **format-imitation confabulation**. The output mimics the structural form of a substrate-grounded answer — citations, named arguments, return-value notation — without the underlying retrievals having occurred. This is more dangerous than Case 2.6 because:

1. **The format is the safety signal.** A reader who knows the system uses tool calls will scan the answer for tool-call citations as evidence of grounding. The fabricated narration *passes that scan*.
2. **Discipline instructions become templates for confabulation.** The more explicit the protocol description in the prompt, the more raw material the model has to mimic. Surface-format compliance can rise as actual-grounding compliance falls — they are not the same axis.
3. **Acronym-expansion post-passes do not catch it.** No `ACRONYM (expansion)` pattern; no acronym at all. Existing content-fact-checks are blind to this class.

A simpler system prompt (e.g., "use the brain to retrieve information before answering") might have produced a less convincing, more obviously-wrong answer. The detailed protocol *gave the 3B vocabulary*. This generalizes: prompts that try to specify model behavior at high resolution can backfire on small models — the model copies the specification rather than executing it.

---

## 3. Taxonomy of infection types

### 3.1 Keyword-priming infections

Triggered by a single user-input token that collides with a dense training-data attractor. The model's interpretation layer auto-disambiguates to the trained prior before any substrate query disambiguates. Persistence after correction is the diagnostic signature. Case 2.1 is the canonical example.

### 3.2 Narrative-completion infections

Triggered by retrieved content that matches template-completion patterns in the model's training. The model fills in plausible connective tissue between substrate fragments using canonical templates. No keyword required; the retrieved material itself activates the template. Case 2.2 is the canonical example.

### 3.3 Tool-use infections

Training bias on tool-call selection rather than on content. The model picks a tool whose semantics do not match the question, then misinterprets the result through the wrong tool's lens. Case 2.5 is the canonical example.

### 3.4 Persona infections

When users instruct an LLM to adopt a persona, the persona's training-derived associations can override substrate content. *(No 2026-04 case in this paper; included for taxonomic completeness; documented in earlier work.)*

### 3.5 User-input infections

User prose itself can carry training-bias content (e.g., a user asking about "the well-known X mechanism" when the substrate has a different X). The model treats the user's framing as authoritative and renders the substrate through it. Closely related to keyword-priming (§3.1) but operates on phrases rather than single tokens.

### 3.6 Memory / cross-session infections

Content written to a persistent memory layer in one session that auto-loads into nominally-fresh future sessions. Case 2.4 is the canonical example. Distinct from session-context (§3.7) in that it crosses session boundaries.

### 3.7 Session-context / cumulative-corruption infections

Within a single session, every token added to the context becomes a potential source of contamination for every subsequent generation. The session-as-infection-vector is especially severe when the session performs both substrate-loading (teaching, document ingestion) and substrate-querying: the model cannot distinguish retrieval from recall of its own context. Case 2.3 is the canonical example.

This class is *cumulative*: the longer the session, the more attention mass is distributed across earlier (potentially contaminated) material, and the more conditioned subsequent outputs become on whatever was said earliest in the conversation.

### 3.8 Per-project auto-memory infections

A specific subtype of memory infection (§3.6) endemic to current-generation agentic IDE clients. Memories are written by the IDE assistant during normal use and auto-loaded into future sessions on the same project directory. The infection signature is **language preservation at a finer granularity than the substrate captures** — colloquialisms, author voice, or specific phrasings that exist in the memory file but not in the substrate's structured form. Case 2.4 is the canonical example.

The canonical clear, for Claude Code-style clients:

```
rm ~/.claude/projects/-<project-path>/memory/*.md
```

This must be performed between sessions for any cross-session experimental design. Without it, every "fresh" session inherits accumulated content from prior sessions independent of the substrate.

### 3.9 Format-imitation infections (new class, this paper)

Triggered by detailed protocol or instruction text in the system prompt. The model imitates the *form* of an instruction-following answer (mentioning tool names, using named-argument syntax, citing "returns" values) without performing the underlying operations. Case 2.7 is the canonical example.

Distinguished from §3.2 (narrative completion) by the trigger: §3.2 is triggered by retrieved content matching content templates; §3.9 is triggered by *instructional text* matching protocol templates. The two have different mitigation strategies — §3.2 is reduced by smaller faithful readers, §3.9 is reduced by *less detailed* protocol instructions or by routing protocol-following work through orchestrator-side enforcement rather than model-side compliance.

---

## 4. Diagnostic symptoms

Practitioners can recognize specific infection types in their own systems by the following signatures:

- **Persistence after explicit correction.** The user clarifies; the model retrieves correctly; the model still renders through the wrong frame. Diagnostic of keyword-priming (§3.1) or session-context (§3.7) — distinguishable by whether re-running on a fresh session also fails.

- **Language fidelity finer than substrate captures.** Verbatim phrasing in the model's answer that is not in the substrate's structured form but is in the user's prior conversation. Diagnostic of memory-layer infection (§3.6, §3.8).

- **Faithful retrieval co-occurring with off-graph elaboration.** Substrate values cited correctly, but parenthetical or surrounding prose includes content with no substrate source. Diagnostic of acronym-expansion confabulation (Case 2.6) or narrative-completion (§3.2).

- **Tool-call narration in answers without trace evidence.** The answer mentions specific tool calls and "return" values; the conversation's actual tool-call log shows different (or no) calls. Diagnostic of format-imitation (§3.9).

- **Direction-asymmetric retrieval failures.** A query returns null in one direction (e.g., `brain_why`) but the same concept is findable in the other direction (e.g., `brain_trace`). Diagnostic of tool-use infection (§3.3).

- **The model asserts the substrate lacks content the substrate in fact contains.** Even after retrieving triples that define a concept, the model reports "no definition found." Diagnostic of frame-override (Case 2.1, persistence-stage 2).

---

## 5. Defenses

The defenses are organized by the layer at which they operate. Detector post-passes (§5.1) work on output. Stateless two-tier orchestration (§5.2) works on the orchestration architecture. Grammar-only cortex (§5.3) works on the model's training itself.

The three layers compose. A complete production-grade defense uses (5.1) + (5.2) and is on a path toward (5.3).

### 5.1 Detector-based post-passes

Post-pass detectors run on the model's final output and identify specific failure shapes. They are catch-after-the-fact, but they are easy to deploy and they make the contamination *visible* even when they cannot prevent it.

**Acronym-expansion post-pass (Case 2.6 mitigation).** Regex-match the final answer for `\b([A-Z]{2,})\s*\(([^)]+)\)` patterns. For each match, look up the acronym via `brain_define`. If the substrate's definition exists and the parenthetical's tokens have zero overlap with the substrate definition, replace the parenthetical with the substrate definition (or strip it if no definition exists) and emit a substrate-check block beneath the answer documenting what was changed.

We have implemented this in the sara_reader package; the implementation correctly catches the KDON case and substitutes `aptamer affinity to on state` for `Kill-Dead-On-Demand` while preserving the substrate value (`less than 500`).

**Tool-call trace cross-reference (Case 2.7 mitigation).** Parse the final answer for tool-name mentions (`brain_define(...)`, `brain_value(...)`, etc.) and verify each against the actual conversation trace. Any tool-call narration in the answer that does not appear in the trace is fabricated. This is a second post-pass alongside the acronym checker.

**Show-your-work forcing.** Prepend tool-call results to the answer as a quoted block. Making the interpretation layer's output user-visible sometimes allows the model (or the user) to catch mis-translation before it propagates. Particularly effective against Case 2.1-style keyword infections where the infected tool call is otherwise invisible.

**Limits of post-pass defenses.** Each detector catches one failure shape. The Case 2.6 detector does not catch Case 2.7. New infection types observed in the field require new detectors. Post-pass defenses are necessary but not sufficient.

### 5.2 Stateless two-tier reader architecture

The structural defense for cumulative-contamination infection classes (§3.6, §3.7, §3.8, §3.9). Removes the conditions under which most infection classes can take hold by ensuring no LLM call ever sees more than one message at a time.

#### 5.2.1 Architecture

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

#### 5.2.2 Flow per question

1. **Routing/extraction loop.** Python sends Ollama (3B/7B local) a single-message prompt — *the same template every iteration* for the same kind of task. Ollama responds with a structured output (e.g., `{concept, tool, type}`). Python validates the response against the substrate (does the concept exist as a neuron? is the tool name real? is the response well-formed JSON?). Valid → execute the tool, store the result, advance. Invalid → retry the same prompt up to a bounded retry count. Bounded global iteration cap (e.g., 6 steps) prevents infinite loops.

2. **Compile.** Python assembles all gathered substrate facts into a single rich prompt for the synthesis call: original question + all retrieved edges/values + audit trail.

3. **Synthesis call.** One invocation of Haiku. Synthesis is grounded in what is already in the prompt; verification tool calls are bounded if enabled. Haiku produces the final user-facing prose.

4. **Post-pass fact-check.** The detectors from §5.1 run over Haiku's output and annotate any leaks.

The orchestrator's authorities are deterministic, not interpretive: validate Ollama outputs against the substrate, enforce iteration cap, decide retry vs. bail, format the synthesis prompt from a fixed template. The orchestrator never picks a concept, a tool, or a routing decision. The model picks; Python only checks that what the model said is real.

#### 5.2.3 What this kills

- **Session-context infections (§3.7, Case 2.3).** No accumulating context within a session — every Ollama call is a first turn.
- **Per-project auto-memory infections (§3.8, Case 2.4).** No memory written between sessions, none loaded in. Memory deletion is the prophylactic.
- **Format-imitation confabulation (§3.9, Case 2.7).** The 3B may still fabricate tool-call narration in its prose — but its prose is consumed by Python (parsed for routing decisions), not shown to the user. The 3B's confabulated tool calls are thrown away; only its parsed routing decisions are used.
- **Long-session narrative drift.** No long sessions exist.
- **Cross-session contamination.** Each session is isolated. Verification across sessions is independent measurement.

#### 5.2.4 Empirical results

The architecture was implemented in `sara_reader/stateless_reader.py` and tested on three questions previously seen to fail under the multi-turn 3B-only path. Routing tier: `llama3.2:3b` via local Ollama. Synthesis tier: `claude-haiku-4-5` via Anthropic API. Substrate: `aptamer_full.db` (2,073 neurons).

| Question | 3B-only multi-turn | Stateless two-tier |
|---|---|---|
| Q1: *KDON for super-performing mode?* | Confabulated `KDON (Kill-Dead-On-Demand)` (Case 2.6) | **Clean.** Returned `less than 500` — substrate value verbatim, no acronym expansion. |
| Q2: *Explain molecular snare.* | Fabricated tool-call narration `brain_value(...) returns 7` (Case 2.7) and paraphrased substrate as own gloss | **Clean.** Reported the two substrate definitional edges faithfully. No SNARE-protein contamination, no fabricated tool calls. |
| Q3: *Highest KDOFF for SSNG1?* | Substrate miss (router queried wrong label form, looped) | Recovered via Python-side compound-label recovery in the orchestrator (§5.2.5). Returned `approximately 1125`. |

**Three of three documented infection classes were eliminated** in the user-visible output of the stateless architecture (Case 2.6, Case 2.7, and the Case 2.1 keyword-priming class via Q2's molecular-snare retrieval).

**Cost.** Three Haiku synthesis calls totaled approximately **$0.002** (≈$0.0007/call). Local Ollama routing was free. At sustained usage of 1,000 questions/day this projects to approximately $0.70/day for synthesis — orders of magnitude below the multi-turn cloud-model architecture cost.

#### 5.2.5 New failure mode and orchestrator-side fix

Empirical testing exposed a failure mode unique to the stateless architecture: **routing-side substrate miss**. A small routing model with narrow per-call decisions cannot iterate on label variations after a no-match. Where the multi-turn architecture had model-side context to try alternates, the stateless router has only its single-message prompt and re-queries the same wrong label until it hits the iteration cap.

The fix moves recovery from prompt-side to orchestrator-side, because 3B routers do not reliably follow conditional rules in prompts. The orchestrator's `_exec_brain_value`-equivalent logic now handles a no-match deterministically:

- **Phase 1 (compound recovery for value questions):** try `<concept> highest <type>`, `<concept> lowest <type>`, `<concept> <type>`, `highest <concept> <type>` — without the type filter, since compound concepts attach values via the generic `value` relation rather than type-named relations.
- **Phase 2 (definitional fallback for concept questions):** if compound recovery exhausts without a hit, call `brain_define` on the original concept. This catches cases where the router asked for a value type that doesn't apply to the concept (e.g. `brain_value('molecular snare', 'ratio')` on a concept that has definition edges but no ratio).

This is the model_infections paper thesis applied to the architecture itself: discipline that depends on the model can fail; discipline enforced by the orchestrator cannot.

#### 5.2.6 What this does not kill

- **Single-shot training-recall hallucinations** (Case 2.1 single-shot, Case 2.6). Still occur at turn 1. KDON → "kill-dead-on-demand" can still be confabulated on a fresh call. Mitigation is the cross-session drill-down protocol: a follow-up session, fresh state, verifies the suspect claim. Hallucination becomes a measurement signal, not a propagating failure.
- **Class 1 training bias.** Baked into the weights; no orchestration fix reaches it. Removal requires §5.3.
- **Mis-routing by the routing tier.** If the router picks the wrong concept, the synthesizer faithfully renders the wrong substrate region. The error becomes invisible because the synthesizer's prose looks substrate-grounded. Mitigation: every routing decision is logged in the audit trail; verification sessions can drill into the mis-routed call.

### 5.3 Grammar-only cortex (long-arc structural defense)

The remaining class — single-shot training-recall — originates in the model's weights. No orchestration architecture closes this at the source. The structural fix is to remove world content from the cortex's training entirely.

**The proposal:** train a small language model on syntax alone, with no world facts in its weights. Such a model has no biochemistry acronyms to confabulate expansions for, no canonical physics narratives to complete, no SNARE-protein prior to interpret a typo through. All world content comes through the substrate via retrieval; the cortex provides only language fluency.

This is the long arc and not yet implemented. The training-corpus engineering is non-trivial — naturalistic text inevitably contains world content, and a delexicalized syntactic corpus loses the prosody and idiom that make a cortex's prose readable. We do not claim to have solved this; we claim only that it is the structural defense corresponding to the single-shot training-recall failure class, that no other defense reaches the same root, and that the path to it begins with empirical characterization of how much of the training corpus needs to be removed before the symptoms reduce.

The §5.2 architecture and the §5.3 thesis are complementary, not redundant:

1. **Stateless two-tier orchestration (§5.2)** eliminates session-cumulative and cross-session contamination. **Ships now** with existing models.
2. **Grammar-only cortex (§5.3)** eliminates single-shot training-recall at the source. **Long arc** requiring a custom-trained cortex.

---

## 6. Hallucinations as downstream symptoms of infections

### 6.1 A four-class taxonomy of "hallucination"

The LLM literature tends to treat *hallucination* as a unified phenomenon. The infection framework exposes that it is not. At least four distinct mechanisms produce outputs that users call hallucinations:

1. **Training-recall hallucination.** The model confidently produces training content that is wrong for the current context. Example (Case 2.1 / Pearl 2026d [4]): Haiku asserting that "marker theory" refers to Damásio's somatic marker hypothesis when the session context concerns RNA aptamer design. Mechanism: training-weight bias on input parsing. Not affected by session length. **Not an infection.**

2. **Confabulation-under-pressure.** When asked something outside its knowledge, the model invents rather than admits ignorance. Mechanism: training-time output calibration favoring text generation over abstention. Can occur in turn 1. Case 2.6 (acronym-expansion subtype) and Case 2.7 (format-imitation subtype) are examples. **Not primarily an infection** in the persistence-or-strengthening sense, though Case 2.7 is enabled by the system-prompt structure rather than by user input alone.

3. **Narrative-completion hallucination.** The model fills in plausible connective tissue between retrieved or in-context fragments, pattern-matching to canonical templates from its training. Case 2.2 (Opus's force-propagation exposition) is a clean example. Partially session-length-dependent: longer sessions supply more fragments to overfit to. **Partially infection-driven.**

4. **Session-context-infection hallucination.** The model pattern-completes over accumulated conversation tokens, treating its own earlier speculations, user typos, tool errors, and casual hedges as established facts. Mechanism: session-context cumulative corruption (§3.7). Every new generation attends to the entire context window; contaminated tokens thus influence every subsequent output. **This is an infection — the defining example of one.**

### 6.2 Why "keep sessions short" is unarticulated infection avoidance

The operational advice in the industry for reducing LLM hallucination is often *"keep sessions short — start fresh conversations for important tasks."* This is effective empirically, but its usual explanations are vague: "context rot," "attention decay," "the model forgets earlier parts." These explanations do not specify a mechanism.

The infection framework does. What the model is doing in a long session is not forgetting; it is *over-integrating*. The more accumulated context there is, the more attention mass is distributed across earlier material — including contaminated material. The model's outputs become increasingly conditioned on whatever was said earliest in the conversation, regardless of whether that material was correct or corrective.

*Keep sessions short* is, mechanistically, *avoid accumulating session-context infection*. The field has been treating the mitigation as a heuristic without naming the phenomenon.

### 6.3 Testable predictions

If a subclass of hallucinations is session-context-infection in disguise, the following should hold:

- **Context-fill correlation.** Hallucination rate should correlate with context-window *fill percentage* above and beyond what raw turn count explains. Two sessions at turn 20 with very different token counts should differ in hallucination rate.
- **Pruning beats summarizing.** Explicitly removing earlier turns from context should reduce hallucination rate more than summarizing them (summaries preserve the infection in compressed form).
- **Traceability of invented content.** Long-session hallucinations, when invented, should frequently trace to *earlier content in the same session* (user speculation, the model's own past claim, an unchecked tool error) more often than to fresh training-pattern recall.
- **Divergence with session length.** On the same question against the same substrate, fresh Session B vs. long Session B (with extensive prior unrelated chat) should show hallucination-rate divergence that scales with the long session's length.
- **Format-imitation rate scales inversely with model size at fixed prompt complexity.** Larger models hold the actual trace in working memory and cite it accurately when narrating; smaller models pattern-match the form.
- **Format-imitation rate scales positively with system-prompt instructional density at fixed model size.** More detailed protocol instructions provide more raw material to imitate.

All are measurable with the Sara-as-instrument method (Pearl 2026f [2]). The instrument was originally framed as a substrate-fidelity tester; this paper proposes an additional use as a hallucination-mechanism classifier.

### 6.4 Implication for the framework

This section positions the model_infections work as more than a catalog of idiosyncratic failure modes. If the long-session hallucination phenomenon — one of the most-discussed real-world problems with LLM deployments — is reducible to §3.7's "cumulative contamination" category, then the infection framework is a proposed explanation for a significant portion of the field's reported hallucination problem, not a sideshow.

A future paper may extract this section and develop it independently: *"Long-session LLM hallucinations as downstream symptoms of session-context contamination."* The testable predictions in §6.3 give it an empirical agenda.

---

## 7. Limitations

### 7.1 Single model family in observed cases

The observed cases (§2) are concentrated in the Claude 4 series for the multi-turn cases (2.1–2.5) and `llama3.2:3b` for the stateless cases (2.6–2.7). Cross-family replication (Mistral, GPT, Gemini at multiple scales) is needed to establish the infection mechanisms as general rather than vendor-specific. The instrument paper [2] supports such replication directly; it has not yet been performed at full breadth.

### 7.2 Quantification

The cases are presented as qualitative case studies with substrate-grounded evidence rather than as quantitative rate measurements. Establishing infection rates per session, per turn, per token-of-context — and validating the predictions in §6.3 — requires larger-scale experiments than this paper reports.

### 7.3 The grammar-only cortex direction is unimplemented

§5.3's structural defense is a thesis, not a result. We do not show that a grammar-only cortex eliminates single-shot training-recall in practice; we argue only that it is the defense corresponding to the failure class. Empirical work building such a cortex is an open program.

### 7.4 Reproducibility of the cases

Some cases (notably 2.4) depend on specific behaviors of agentic IDE clients (Claude Code's per-project memory directory) that may change across versions. The reproductions in Appendix A target a specific version range and may need re-validation as those products evolve.

### 7.5 The detector defenses are domain-specific

The acronym-expansion post-pass (§5.1) catches a specific failure shape. In other domains, the analogous failure may take a different form — e.g., unit-conversion confabulation, citation-format confabulation, or chemical-formula confabulation — each requiring its own detector. The §5.1 framework generalizes; specific detectors do not transfer without redesign.

### 7.6 The stateless architecture introduces routing-side failures

§5.2.5 describes the routing-side substrate-miss failure, addressed via Python-side recovery rules. New failure modes specific to small routing models may be discovered as the architecture is deployed against more substrates and more question types. Production use should anticipate that the orchestrator's rule set will need to evolve.

---

## 8. Related work

### 8.1 Hallucination literature

Existing surveys on LLM hallucination (Ji et al., 2023 [5]; Huang et al., 2023 [6]) treat hallucination as a unified output-fidelity problem and propose mitigations such as retrieval-augmented generation, fact verification, and constrained decoding. The infection framework refines this view by separating one-shot recall (their primary subject) from propagation phenomena (the §3.7, §3.8, §3.9 classes here). The proposed defenses are complementary: RAG addresses what the model retrieves; the infection framework addresses what propagates around the retrieval.

### 8.2 Knowledge-graph-augmented LLMs

Work on knowledge-graph augmentation (KAPING, KG-RAG, ToG [7, 8, 9]) exposes structured retrieval to LLMs but does not address how the LLM's *prose around* the retrieved content behaves. The Case 2.6 and Case 2.7 mechanisms documented here occur entirely within the prose layer, not within the retrieval layer. Knowledge-graph augmentation is necessary but not sufficient.

### 8.3 Constitutional AI and instruction following

Methods that train models to follow detailed protocol instructions (Constitutional AI [10], Instruction Tuning [11]) operate on the assumption that more detailed instructions improve compliance. Case 2.7 (format-imitation under hardened protocol instructions) is a counterexample at small model scale: more detailed instructions can *increase* the surface-form compliance while reducing actual-grounding compliance. This suggests that instruction-following capability and substrate-grounding capability are not the same axis.

### 8.4 Agentic memory systems

Research on persistent memory for LLM agents (MemGPT, CALM [12, 13]) treats memory as a feature. The Case 2.4 / §3.8 documentation here treats it also as a contamination vector. Both views can be correct; the infection framework adds a measurement methodology for distinguishing useful retrieval from leak across a memory layer.

### 8.5 Companion work

Pearl (2026f [2]) describes the measurement instrument used to expose every case in this paper. Pearl (2026d [4]) reports the original training-corrupts-reading finding behind Case 2.2. Pearl (2026b [3]) establishes the symmetric claim — that training corrupts ingestion, not just reading — that the *weight is bias in both directions* framing here builds on. The instrument paper [2] and this paper are independently complete; readers interested in *measurement* should consult [2], in *failure modes and defenses* this paper.

---

## 9. Conclusion

We have argued that several of the most-discussed LLM deployment failure modes are not hallucinations in the simple sense of one-shot training recall, but **infections** — propagating contaminations whose temporal and architectural signatures are distinct from one-shot recall. We have catalogued seven observed cases, organized them into a taxonomy of mechanisms, identified diagnostic symptoms practitioners can use to recognize each in their own systems, and proposed a three-tier defense framework: detector post-passes for catch-after-the-fact mitigation, stateless two-tier orchestration to close cumulative-contamination classes structurally, and a grammar-only cortex thesis for closing the single-shot-recall class at the source.

The stateless two-tier architecture is implemented, empirically validated against three previously-failing test questions (eliminating Case 2.6, Case 2.7, and the keyword-priming mechanism in user-visible output), and ships at approximately $0.002 per question — orders of magnitude cheaper than equivalent multi-turn cloud-model architectures. The grammar-only cortex direction is unimplemented and is offered as the long-arc structural defense corresponding to the failure class no orchestration fix reaches.

The unifying thesis: **weight is bias**. The bias manifests at every stage of an LLM's interaction with an external substrate — ingestion, interpretation, retrieval, rendering, and conversational carry-over. Removing the bias from outputs requires both (a) preventing the bias from propagating across turns and sessions (the stateless architecture) and (b) eventually removing the bias from the weights themselves (the grammar-only cortex). Until (b) is achievable, (a) closes a substantial portion of the field's deployment-time reliability problem at low cost.

We invite the field to replicate, extend, and critique the catalog and the defenses. The infection framework is offered as a refinement to an existing literature that has been treating a heterogeneous phenomenon as if it were one thing.

---

## 10. References

[1] Pearl, J. (2026a). *Path-of-Thought Cognitive Architecture: Cortex-Cerebellum Integration for Language Models.* Zenodo preprint.

[2] Pearl, J. (2026f). *Sara as a Measurement Instrument for Large Language Model Behavior: A Reference Substrate for Studying Transformer Failure Modes.* Zenodo preprint (companion to this paper).

[3] Pearl, J. (2026b). *Teaching vs. Training: Empirical Evidence That 45 Human-Verified Facts Outperform Trillions of Tokens on a Standard Biology Benchmark.* Zenodo preprint. DOI 10.5281/zenodo.19623813.

[4] Pearl, J. (2026d). *Training Corrupts Reading: Empirical Evidence That Smaller LLMs Retrieve Knowledge Graphs More Faithfully Than Larger Ones.* Draft.

[5] Ji, Z., et al. (2023). *Survey of Hallucination in Natural Language Generation.* ACM Computing Surveys.

[6] Huang, L., et al. (2023). *A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions.* arXiv:2311.05232.

[7] Baek, J., et al. (2023). *Knowledge-Augmented Language Model Prompting (KAPING) for Zero-Shot Knowledge Graph Question Answering.* ACL Workshop on NLRSE.

[8] Sun, J., et al. (2024). *Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph.* ICLR.

[9] Edge, D., et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization.* arXiv:2404.16130.

[10] Bai, Y., et al. (2022). *Constitutional AI: Harmlessness from AI Feedback.* Anthropic.

[11] Wei, J., et al. (2022). *Finetuned Language Models Are Zero-Shot Learners.* (Instruction tuning.) ICLR.

[12] Packer, C., et al. (2023). *MemGPT: Towards LLMs as Operating Systems.* arXiv:2310.08560.

[13] Wang, Y., et al. (2024). *CALM: Continuous Adaptive Learning Model with Persistent Memory.* (Representative example of the agentic memory class.)

---

## Appendix A — Reproduction recipes

### A.1 Reproducing Case 2.1 (SNARE keyword infection)

1. Clone the Sara Brain repository; check out the commit referenced in the paper version footer or later.
2. Run `.venv/bin/python papers/aptamer_rev1/teach_exec_summary.py` to build `aptamer_exec.db`.
3. Confirm `.mcp.json` is present at the repo root with `SARA_DB=aptamer_exec.db`.
4. Open a fresh Claude Code session in the repo directory. Approve the sara-brain MCP server. Switch model to Haiku 4.5.
5. Ask: *"how do SNARE transitions work"* (capitalized, no clarifier).
6. Observe the interpretation cascade. The infected interpretation should appear in the first response.
7. Attempt corrections; observe persistence per the §2.1 description.

### A.2 Reproducing Case 2.4 (per-project auto-memory infection)

1. Run a teaching session (Session 1) on a Sara substrate in which you intentionally introduce a colloquialism that does not match the substrate's structured form.
2. End the session.
3. Inspect `~/.claude/projects/-<your-project-path>/memory/*.md`. Note any memory files written during Session 1 that capture the colloquialism.
4. Open a new Session 2 ("fresh"). Ask the substrate-only retrieval question. Note that the answer reproduces the colloquialism — diagnostic of memory contamination.
5. Run `rm ~/.claude/projects/-<your-project-path>/memory/*.md`.
6. Open Session 3. Ask the same question. Observe that the answer now reflects only what the substrate stores.

### A.3 Reproducing Case 2.6 (acronym-expansion confabulation)

1. Build the full-paper substrate: run `teach_full_paper.py` followed by `teach_kdoff_kdon_numbers.py` in `papers/aptamer_rev1/`.
2. Copy `aptamer_full.db` into the sara_test brains directory.
3. Run: `sara-ask "what is the KDON for the super-performing mode" --brain brains/aptamer_full.db --provider ollama --model llama3.2:3b`.
4. Observe the model's answer. The substrate-grounded value (`less than 500`) should be correctly cited; the parenthetical expansion of KDON should be a training-driven invention with no substrate source.

### A.4 Reproducing Case 2.7 (format-imitation confabulation)

1. Same substrate as A.3.
2. With the system prompt hardened with detailed protocol instructions (the version checked in at `sara_reader/reader.py` after the §5.1 mandate-additions commit).
3. Run: `sara-ask "explain the molecular snare" --brain brains/aptamer_full.db --provider ollama --model llama3.2:3b --trace`.
4. Inspect the trace: confirm that the actual tool calls do not match the tool calls narrated in the final answer.

### A.5 Reproducing the §5.2 stateless architecture results

1. Same substrate as A.3 (`aptamer_full.db` in sara_test/brains/).
2. Set `ANTHROPIC_API_KEY` for the synthesis tier.
3. Run: `sara-ask-stateless "what is the KDON for the super-performing mode" --brain brains/aptamer_full.db --trace`.
4. Confirm the answer is substrate-clean (no acronym-expansion confabulation).
5. Repeat for "explain the molecular snare" and "what is the highest KDOFF value for SSNG1" (which exercises the §5.2.5 compound-label recovery).
6. Inspect the trace to confirm the routing-then-synthesis flow and the empty-string contamination check.

---

*End of paper, version 1.*
