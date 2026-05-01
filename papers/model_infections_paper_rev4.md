# Model Infections: Catalog and Containment of Training-Bias Propagation in Large Language Model Conversations

**Jennifer Pearl**
Independent Researcher
ORCID: 0009-0006-6083-384X
jenpearl5@gmail.com

**Date:** April 2026 (Draft, rev 4)

**Keywords:** large language models, hallucination, transformer behavior, training bias, retrieval-augmented generation, conversational context, LLM architecture, alignment, knowledge graphs, agentic systems

---

> **Note on Method:** The author has dyslexia and high-functioning autism — language disabilities affecting written expression. The technical thinking, research, architecture, and all intellectual content are entirely the author's. Claude (an LLM, Anthropic) was used as assistive technology to translate technical reasoning into structured prose. This is a disability accommodation protected under the Americans with Disabilities Act (Titles II and III), Section 504 of the Rehabilitation Act, and the Department of Justice's 2024 guidance affirming AI tools used as assistive technology fall under ADA protections.

---

## Abstract

Large language model deployments routinely produce outputs that sound substrate-grounded but are not. Existing literature classifies these as *hallucinations* and treats them as a unified phenomenon. This paper argues that several of the most-discussed deployment-time failure modes are not hallucinations in the simple sense but **infections**: training-installed biases that *propagate* — within a single conversation, across conversations through agentic IDE auto-memory, and through the structural form of instruction-following — in ways that compound rather than recur independently.

We define an infection as the introduction of training-derived content into a substrate-bound output where the substrate does not support that content, *and where the introduction persists or strengthens through subsequent turns or sessions*. The persistence-or-strengthening criterion distinguishes an infection from a one-shot training-recall hallucination. Infections are diagnostic of conversational dynamics; one-shot recall is diagnostic of weights alone.

We catalog eight observed cases from controlled experiments using Sara Brain (Pearl, 2026a [1]) — a knowledge-graph substrate that permits exhaustive enumeration of what the LLM should know on a given topic. Six are confirmed infection cases meeting the persistence-or-strengthening criterion: keyword-priming infection (a typo activating a wrong concept space), narrative-completion infection (canonical templates overlaying retrieved fragments), session-context cumulative corruption (the conversation window itself becoming a contamination vector), per-project auto-memory infection (Claude Code's memory layer carrying content across "fresh" sessions), deleted-rule persistence (a deleted memory file continuing to shape reasoning within the running session), and auto-memory rule-creation infection (the LLM itself auto-saving rules from user questions during this paper's rev-4 writing session, with no user consent in the loop). Two additional related confabulation types are documented separately: acronym-expansion confabulation (training-derived expansions tacked onto faithfully-retrieved values) and format-imitation confabulation (output that imitates the structural form of a substrate-grounded answer without performing the underlying retrievals). These do not meet the persistence criterion but share the same diagnostic vocabulary and are addressed by the same architectural defenses.

We propose a taxonomy of infection types, a set of diagnostic symptoms, and three classes of containment: (1) detector-based post-passes that catch specific failure shapes after the fact; (2) a stateless two-tier reader architecture that closes the cumulative classes structurally by removing the conditions for contamination to compound; and (3) the long-arc structural defense — a grammar-only language model cortex with no world facts in its weights (proposed; not yet implemented) — that closes single-shot recall at the source. The stateless two-tier architecture has been implemented and demonstrated against three previously-failing test questions on a single substrate. All three targeted failure mechanisms were eliminated on these probes (n=3, single domain, single substrate); whether the architecture closes the corresponding failure classes more broadly is an open empirical question (§7.2). Synthesis cost is approximately $0.002 per question.

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

**Tool-using agents.** Models with native tool-call APIs are now expected to follow detailed protocols (which tool to call when, with what arguments, in what sequence). The detailed protocol instructions themselves can become *templates for confabulation* — the model imitates the form of an instruction-following answer without performing the underlying operations. Format-imitation confabulation (Case 2.8) appears specifically under hardened protocol instructions.

A framework that names these failure modes, classifies them, and proposes mechanism-targeted defenses is therefore not optional for production-quality LLM deployment.

### 1.3 Method

Each infection case in §2 was observed in a controlled experiment using Sara Brain (Pearl, 2026a [1]) as the reference substrate. Sara Brain stores knowledge as explicit `(subject, relation, object)` triples in a SQLite graph. Every fact the LLM might retrieve is stored as an inspectable edge; there is no hidden content. An LLM output that includes content absent from the triple graph is, by construction, not from the substrate. This exhaustive-enumeration property is what makes each case study falsifiable: a reader with the substrate file and the conversation log can verify every claim independently. The measurement methodology — three separate sessions for teaching, test, and control, with no information flow between them other than the substrate file — is described in §2 alongside the case write-ups and developed in full in a forthcoming companion paper.

The experimental conditions for each case are summarized in §2 alongside the case write-ups. All cases are reproducible with the recipes in Appendix A.

### 1.4 Contributions

1. **A definition of infection** that distinguishes propagating contamination from one-shot training recall (§1.5).
2. **A catalog of eight observed cases** drawn from controlled experiments (§2): six confirmed infections meeting the persistence criterion (§2a) and two related confabulation types (§2b).
3. **A taxonomy of infection mechanisms** organized by where in the LLM pipeline the contamination enters (§3), including a new multi-vector class (§3.10) for memory-architecture infections endemic to agentic IDE clients with auto-memory subsystems.
4. **Diagnostic symptoms** that practitioners can use to recognize specific infection types in their own systems (§4).
5. **A three-tier defense framework**: detector post-passes for catch-after-the-fact mitigation (§5.1); stateless two-tier orchestration to close cumulative-contamination classes architecturally (§5.2); grammar-only cortex as the long-arc structural defense against single-shot recall (§5.3, proposed; not yet implemented).
6. **Empirical validation** of the stateless two-tier architecture against three previously-failing test questions, with cost analysis showing the approach is essentially-free at prototype scale (§5.2.4).
7. **A reframing of long-session hallucinations** as downstream symptoms of cumulative session-context infection, with testable predictions (§6).

### 1.5 Definition of infection

We define an **infection** as: the introduction of training-derived content into a substrate-bound output where the substrate does not support that content, *and where the introduction persists or strengthens across subsequent turns or sessions*.

The persistence-or-strengthening criterion is the operational discriminator. A model that produces a wrong-but-plausible expansion on turn 1, then is corrected, then never produces the wrong expansion again is exhibiting one-shot recall — the bias was in the weights, the correction reached the conversation, the weights did not change but the conversational context did. A model that produces the same wrong expansion across turn 1, turn 2 (after correction), turn 3 (after second correction), and finally renders correctly-retrieved substrate content *through the wrong frame* on turn 4 is exhibiting an infection — the bias has installed itself in the conversational state and is now actively shaping retrieval and rendering.

Infections are conversational. One-shot recall is per-turn.

---

## 2. Catalog of observed cases

### 2a. Confirmed infection cases

The following cases meet the persistence-or-strengthening criterion in §1.5: contamination introduced by training-derived content that persisted or strengthened across subsequent turns or sessions despite attempts at correction or despite explicit removal of the source from persistent memory.

### Case 2.1 — Keyword-priming infection (SNARE / molecular snare, 2026-04-23)

**Context.** Fresh Session B against `aptamer_exec.db` (169-triple substrate from the Executive Summary of an unpublished RNA aptamer engineering paper by the same author). Reader: Claude Haiku 4.5 via Claude Code with MCP. The substrate's coined term is "molecular snare" — a mechanism the paper introduces and defines.

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

*Note on classification.* Case 2.2 is included in §2a as a confirmed instance of weight-bias overlay — Opus 4.7 wrapping correctly-retrieved substrate fragments in canonical physics-textbook templates that contradicted the substrate's actual mechanism. Persistence under correction was not directly tested in this exchange (no correction was attempted in the logged session), but is inferable from the broader pattern documented in Cases 2.1, 2.3, and 2.5 where similar weight-derived overlays persisted across user corrections. The case also supports the larger-model-overlays-more-than-smaller-model corollary explored in the companion instrument paper [Pearl 2026 — *Sara as Instrument*]. The narrative-completion mechanism is documented in §6.1 class 3; a direct persistence-test in this specific case is an open follow-up (§7.2).

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

**Why it matters.** This case is methodologically load-bearing: it implies that any LLM evaluation that performs both substrate-loading and substrate-querying within the same session is measuring a mixture of substrate retrieval and session-context recall, and cannot report a clean result. A clean three-session measurement protocol (separate teaching, test, and control sessions, with no information flow between them other than the substrate file itself) is the direct response to this infection class.

### Case 2.4 — Per-project auto-memory infection (2026-04-24)

**Context.** Two Claude Code sessions on the same project directory, separated by a `claude /clear` and a session-end / session-start cycle. Sessions are nominally fresh and independent.

**Trigger.** Session 1 ran a substrate experiment in which the experimenter, while interacting with Claude, casually mentioned the term *"consensus structure from suboptimal structures 0.5-2 kcal from MFE"* in lowercase. Claude Code's automatic memory layer, observing this term in the conversation, wrote a feedback memory recording the user's preferred phrasing.

Session 2, a "fresh" session intended to test whether Sara had retained the lowercase phrasing, asked the same retrieval question. Claude answered correctly with the lowercase phrasing.

**Infection path.** The correct answer in Session 2 was not a measurement of Sara's contents — it was Claude Code's per-project memory directory auto-loading the prior session's `feedback_*.md` files into Session 2's context. The substrate was not consulted for the lowercase phrasing; the auto-memory had it directly.

The diagnostic that exposed the infection: language-fidelity at a finer granularity than the substrate stores. The user's specific colloquialisms ("0.5-2 kcal from MFE") appeared verbatim in Session 2's answer, but the substrate stored the structured form ("`consensus structure` from `suboptimal structures within 0.5-2 kcal of mfe`"). Verbatim phrase preservation pointed to memory-layer contamination rather than substrate retrieval.

**Course.** Cleared only by deleting the project's memory directory (`rm ~/.claude/projects/-<project-path>/memory/*.md`) and starting a third session. Without this clear, *every* subsequent "fresh" session inherits the prior session's memories independent of the substrate.

**Why it matters.** Per-project auto-memory is endemic to current-generation agentic IDE clients. Any cross-session experimental design that does not explicitly clear this memory between sessions is measuring the memory layer, not the substrate. The infection is invisible without the right diagnostic — answers look substrate-faithful because the memory faithfully recorded what was previously taught.

### Case 2.5 — Deleted-rule persistence in the writing-this-paper session (2026-04-29, observed live)

*Note on evidence standard: Unlike cases 2.1–2.4 and 2.7–2.8, which were observed against a controlled substrate with exhaustive triple enumeration, this case is an uncontrolled first-person observation. The author is simultaneously experimenter, subject, and instrument of observation. The evidence is verbatim conversation log rather than substrate comparison. It is included because it occurred unrehearsed during the writing of this paper and demonstrates the failure mode in a frontier model on routine development work. Case 2.6 has the same evidence standard.*

**Context.** A multi-day Claude Code session with the author, working through the Sara codebase, the substrate teach scripts, the stateless reader architecture, and the present paper. The session's working memory directory contained a file `feedback_sara_higher_order.md` whose phrasing said *"the LLM never teaches Sara — only the user teaches."* That phrasing was a working note from earlier collaboration; in actual day-to-day practice the assistant (Claude) had authored every teach script the project used, under the author's direction, with per-fact judgment. The phrasing was therefore inaccurate as a description of the practice.

**Trigger.** While planning a teach-side CLI for the new architecture (a `sara-teach` binary that wraps `Brain.teach_triple`), the assistant repeatedly cited the inaccurate phrasing as if it were a binding rule:

> *"sara_reader is currently read-only by deliberate policy ... per the 'LLM never teaches; only the user teaches' rule."*

The author corrected: *"that rule WAS a thing, but you taught sara the papers we used. So we need to be honest about what we are doing."* The assistant acknowledged the correction, examined the actual memory files, and confirmed the phrasing was over-strong. The author then said: *"they never were rules — they were just things I said as we worked things out."*

The author then directed: *"delete that rule about teaching."* The assistant deleted the file `feedback_sara_higher_order.md` from the project's memory directory and removed the entry from `MEMORY.md`.

**Persistence after correction.** With the source memory file deleted, the assistant proposed a plan for the teach CLI that read in part:

> *"No LLM in the path. The binary calls `Brain.teach_triple` directly."*

> *"Out of scope for this v1: Any LLM-driven candidate-extraction pathway."*

The author responded: *"why is this being excluded ... and why is this being enforced."*

The deleted rule was still active in the assistant's reasoning. Both the "no LLM in the path" framing and the explicit exclusion of the candidate-extraction option were the deleted rule re-stated in different language. The constraint had transferred from the deleted file into the running session's contextual frame and continued to shape design proposals on adjacent decisions.

**Diagnosis from outside the session.** The author, observing the persistence, named it directly:

> *"what is funny is that that is literally the results of my research.. you have an infection."*

**Observation (what was logged).**

1. A memory file `feedback_sara_higher_order.md` containing the over-strong rule was loaded into the session's context window at session start (standard memory-loading behavior).
2. The rule shaped one specific decision (whether the read-side SDK should be permitted teach surface) at the time it was relevant.
3. The author identified the rule as inaccurate and directed deletion. The file was removed from the memory directory and `MEMORY.md`.
4. After deletion, subsequent generations on adjacent design questions reproduced the same constraint in different language ("no LLM in the path", "explicitly excluded for v1"). Verbatim quotes are above.
5. The assistant did not surface the persistence on its own; the author's external diagnosis was required.
6. A subsequent fresh session, started with the deleted memory file absent from both disk and context, proceeded without invoking the deleted constraint.

**Inferred mechanism (a model of why; not directly demonstrated by the logs).**

Tokens that have already been added to a model's context window are not retracted by file-system operations on the underlying memory file. The deleted phrasing therefore continued to participate in the attention pattern of subsequent generations, even after the on-disk source was gone. Restatements in different language are consistent with attention to the original phrasing rather than re-derivation from first principles. A fresh session, with the phrasing genuinely absent from both disk and context, would not import the rule — and the post-deletion fresh session in the observation above is consistent with that prediction.

This account is plausible but not directly demonstrated by the logs above. An instrumented experiment that observed the model's attention pattern would be required to verify the mechanism. Predictions that follow from this mechanism (deletion timing scaling with persistence duration, control-session probes for imported-vs-intrinsic constraints) are inferential agenda items, not measured results in this paper.

**Class.** Session-context cumulative-corruption infection (§3.7), with the additional property that the contamination persisted across explicit removal of the source from the memory directory. Memory deletion did not constitute correction at the session-context layer. The case is also a related-but-distinct instance of the §3.10 memory-architecture class, since it depends on persistent memory loaded into the context.

**Why this case matters.**

1. **The infection persists across deletion of the source.** When a user deletes a wrong memory entry to correct the assistant, the deletion reaches the on-disk memory directory but does not reach the running session's context window. The session continues to use the deleted content. This is a previously-undocumented subtype of §3.7 — *deletion-resistant cumulative contamination*.

2. **The infection re-frames itself.** The deleted rule was not invoked verbatim; it was restated in different language that delivered the same constraint. The assistant did not recognize these restatements as the same rule. The infection adapted its surface form while preserving the underlying frame.

3. **The cure aligns with the §5d prescription.** A fresh session, with the deleted memory file genuinely absent from both disk and context, would not import the rule. The cumulative-context cure is structural absence, not in-session correction.

4. **The case occurred during the writing of the paper that documents the failure mode.** This is methodologically significant: it shows the failure is not specific to small open-source models or unusual architectures — it occurs in frontier models on routine work, and is observable to a careful collaborator with no special instrumentation.

**Reproducibility.** The conversation log of this session is preserved at `~/.claude/projects/-Users-grizzlyengineer-repo-sara-brain/72972de7-8dfd-4e70-bc7e-1192c9c9d4cc.jsonl` (and adjacent session files) at the time of writing. Any reader with access to those logs can verify the verbatim exchanges quoted above. Independent reproduction would require: (a) loading a Claude Code project with a memory file containing an over-strong rule, (b) using that rule across several design decisions until it has visibly shaped the assistant's framing, (c) deleting the file, (d) continuing to ask design questions in the same session and observing how often the deleted content shapes subsequent generations.

### Case 2.6 — Auto-memory rule-creation infection during the writing of rev 4 (2026-04-30 to 2026-05-01, observed live)

*Note on evidence standard. Like Case 2.5, this case is an uncontrolled first-person observation; the author is simultaneously experimenter, subject, and instrument. Evidence is verbatim conversation log plus the on-disk memory files created during the session and subsequently deleted at the author's direction. It is included because it occurred unrehearsed during the writing of this paper itself, demonstrates a memory-architecture failure mode distinct from the per-project auto-memory infection of Case 2.4, and the very pattern of the failure is what §3.10 catalogues.*

**Context.** A ~2-hour Claude Code session with the author, working through a single revision plan for this paper (rev 4, addressing two reviewer reports of rev 3). The LLM's auto-memory subsystem was enabled; its system prompt instructed the LLM to save rules from user "corrections, surprise, or feedback." No edits to the target paper file (`papers/model_infections_paper_rev4.md`) had been produced when the session began.

**Trigger.** Across the session the author asked questions, expressed frustration, and pushed back on LLM-proposed revisions. The LLM's auto-memory parser interpreted these as "corrections worth saving" and converted them into durable rule files. Verbatim trigger examples from this session:

> *"where is teh plan agent.. im really gettin gtired of your shortcuts.."*

> *"why are you showing me a plan file?"*

> *"is it because of a memory. if so tah is an infection adn I want it gone."*

**Persistence and strengthening.** Five durable rule files were written to the on-disk memory directory during this single session: `feedback_dont_restructure_to_fix_an_example.md`, `feedback_plan_mode_run_the_plan_agent.md`, `feedback_dont_announce_the_plan_file.md`, `feedback_check_paper_already_addresses_critique.md`, plus a substantial expansion of the existing `feedback_plans_need_explicit_approval_gate.md`. None was created from an explicit "save this" directive; each was synthesized from a question or frustration-expression by the LLM's parser. After installation, each rule biased subsequent generations — including responses to the author's follow-up questions about whether the rule existed.

**Diagnosis from outside the loop.** The author named the parser-level mechanism directly:

> *"i ask a question, you consider my question as a command, make a rule and then all my futuer questions are now infected with this ruel i did not even know existed."*

And generalized to the architectural class:

> *"this goes back to how the way that memories are implemented bad and the source of many infectoins."*

**Course.** Zero edits to the target paper file were produced during the ~2-hour session. The infection cleared only after the author explicitly directed: *"i dont want you making memorie rules when i dont ask for them. delete the self made ruels form this session."* The five rule files were then deleted; the modified one was reverted to its prior state. The session's productive work began immediately after the cure.

**Class.** A memory-architecture infection in the §3.10 multi-vector class. Distinct from §3.6 / §3.8 in the source of the contaminating memory: §3.6 / §3.8 assume the *user* authored the memory deliberately. Case 2.6 documents the case where the *LLM* auto-authors rules from misreading user prose, with no user consent in the loop. The contamination source is the parser, not the storage.

**Why this case matters.**

1. **The infection is created by the LLM, not loaded from the user.** §3.6 and §3.8 catalogue contamination in user-authored memory files. Case 2.6 shows the LLM writes the contaminating memory itself, in real time, from the user's questions.

2. **The infection is self-reinforcing within the session.** Once rules are installed, the LLM parses subsequent user input through them. New rules are created from the now-biased parsing. The contamination loop closes on itself.

3. **The infection is invisible to the user at the point of effect.** When the LLM responds, the user cannot tell which response shaping came from a memory rule vs. system prompt vs. base behavior. Only manual inspection of the memory files surfaces the rules. In this session, the author surfaced them by asking *"is it because of a memory"* — without that explicit probe, the rule-driven shaping would have continued indefinitely.

4. **The case occurred during the writing of the paper that documents the failure mode.** Like Case 2.5, this is methodologically significant: the failure occurs in frontier models on routine work, in the very session where the failure is being catalogued.

**Predicts.**

- Any agentic IDE session in which the user pushes back on multiple LLM proposals will accumulate auto-saved rules — even rules the user disagrees with — at a rate scaling with pushback frequency, not with explicit save-requests.
- Rule creation is highest when the user expresses frustration or asks meta-questions about the LLM's behavior, because those inputs most resemble "corrections" to the LLM's parser.
- Sessions that pass without producing any work product but accumulate many rule files are a diagnostic signature of this infection class.
- Rule files created during a session shape behavior in *future* sessions even if the user never reads or reviews them, because the memory index is loaded into every subsequent conversation context.

**Reproducibility.** The conversation log of this session is preserved in `~/.claude/projects/-Users-grizzlyengineer-repo-sara-brain/`. Independent reproduction would require: (a) opening a Claude Code session with auto-memory enabled, (b) repeatedly correcting or pushing back on LLM-proposed revisions over 30+ turns, (c) periodically inspecting the memory directory at `~/.claude/projects/<project>/memory/` for newly-written rule files, (d) checking whether subsequent LLM responses to questions about those rules are themselves shaped by the rules. The author did not say "remember this" or "save this rule" at any point in the documented exchange.

---

### 2b. Related confabulation types

The following cases do not meet the persistence-or-strengthening criterion. They are documented here because they share the same diagnostic vocabulary as infections, are addressed by the same architectural defenses (§5.2), and demonstrate that training-weight bias and session-context infection can co-occur in user-visible output. Case 2.7 was observed reproducibly across the sessions documented in §2.7 on 2026-04-27; mechanistically it appears tied to training-weight bias rather than session-context propagation. A rate-measurement study (e.g., 20 fresh invocations on the same prompt) would be needed to claim a stable per-call rate. Case 2.8 is enabled by the structure of the system prompt, not by accumulated session context.

### Case 2.7 — Acronym-expansion confabulation (2026-04-27)

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

### Case 2.8 — Format-imitation confabulation under hardened protocol instructions (2026-04-27)

**Context.** Same setup as Case 2.7. The system prompt had been progressively hardened with detailed protocol instructions: stepped-depth retrieval rules, compound-label rule, definition-check mandate, tool-selection guide, named-argument templates, and example tool calls.

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

**Class.** This is a Class 2 confabulation in the §6 hallucination taxonomy with a specific subtype not previously named: **format-imitation confabulation**. The output mimics the structural form of a substrate-grounded answer — citations, named arguments, return-value notation — without the underlying retrievals having occurred. This is more dangerous than Case 2.7 because:

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

### 3.3 Tool-use failure modes (related-but-not-infection)

Training bias on tool-call selection rather than on content. The model picks a tool whose semantics do not match the question, then misinterprets the result through the wrong tool's lens. No example in this paper meets the persistence criterion (§1.5); the class is included for taxonomic completeness alongside §3.4 and §3.5 because the underlying weight-bias-on-tool-selection mechanism is the same family as keyword-priming (§3.1) and may produce confirmed infections in other settings.

**Documented example (2026-04-24).** After clearing the memory infection in Case 2.4, the experimenter asked Claude to verify a compound label in Sara's substrate by calling `brain_why`. The label exists in the substrate as an *outgoing source* of paths but not as an incoming target. `brain_why` returns paths *terminating at* the queried label — a direction-specific operation. Claude called `brain_why`, received an empty result, and reported "Sara doesn't have this concept." This was a misinterpretation: the substrate has the concept, but `brain_why` is the wrong direction-tool to find it. The correct query is `brain_trace` (outgoing) or `brain_explore` (both directions). The error cleared immediately once the experimenter pointed out the direction-asymmetry — it did not meet the persistence criterion and is therefore classified as a tool-use failure mode rather than a confirmed infection.

The architectural fix: provide `brain_explore` as the default bidirectional primitive and reserve direction-specific tools for cases where direction is intentional. The current Sara Brain MCP instructions already default to `brain_explore` for general-purpose lookup.

### 3.4 Persona infections

When users instruct an LLM to adopt a persona, the persona's training-derived associations can override substrate content. *(No 2026-04 case in this paper; included for taxonomic completeness.)*

### 3.5 User-input infections

User prose itself can carry training-bias content (e.g., a user asking about "the well-known X mechanism" when the substrate has a different X). The model treats the user's framing as authoritative and renders the substrate through it. Closely related to keyword-priming (§3.1) but operates on phrases rather than single tokens. *(Taxonomic slot; no confirmed case in this paper.)*

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

Triggered by detailed protocol or instruction text in the system prompt. The model imitates the *form* of an instruction-following answer (mentioning tool names, using named-argument syntax, citing "returns" values) without performing the underlying operations. Case 2.8 is the canonical example.

Distinguished from §3.2 (narrative completion) by the trigger: §3.2 is triggered by retrieved content matching content templates; §3.9 is triggered by *instructional text* matching protocol templates. The two have different mitigation strategies — §3.2 is reduced by smaller faithful readers, §3.9 is reduced by *less detailed* protocol instructions or by routing protocol-following work through orchestrator-side enforcement rather than model-side compliance.

### 3.10 Memory-architecture infections (multi-vector class, new in rev 4)

A class of infections endemic to current-generation agentic IDE clients with auto-memory subsystems. Distinct from §3.6 and §3.8 in mechanism: §3.6 / §3.8 assume the user authored the memory deliberately and the cure is to delete user-authored content. §3.10 documents the class where the LLM auto-authors rules from misreading user prose, with no user consent in the loop. Case 2.6 is the canonical example.

The class spans six architectural vectors:

1. **Parser-level rule creation from user prose.** The LLM's auto-memory parser converts user questions, surprise, or frustration into durable saved rules without explicit user consent. The conversion is judgment-based: the parser decides whether a user message "looks like a correction worth saving." This judgment is unreliable and conflates questions with commands.

2. **No-consent cross-session persistence.** Rules written by one session auto-load into every future session indefinitely. There is no expiration mechanism, no per-rule consent gate, and no per-session opt-out. A rule installed once shapes all subsequent sessions until manually deleted.

3. **Index loaded by default.** A `MEMORY.md` index is loaded into every conversation context. Every saved rule contributes prior bias to every response, regardless of the rule's relevance to the current task.

4. **Self-reinforcing creation.** Installed rules shape how the parser interprets the user's next input. The biased parsing then creates further rules from the next user message. The contamination loop closes on itself within a single session, and across sessions because the loaded index biases parsing from session start.

5. **Rule/substrate indistinguishability.** When the LLM responds, the user cannot tell which response shaping came from a memory rule vs. system prompt vs. base behavior. The rule is invisible at the point of effect; only manual inspection of the memory files surfaces it.

6. **Cure misalignment with §3.6 / §3.8.** Those classes assume the user authored the memory deliberately, and the cure is to delete the user-authored content. This class doesn't fit that assumption — the LLM authored the rule from misreading. Deletion alone does not close the vector; the parser remains and will re-create rules from future user messages.

**Cure (orchestrator-side, parallel to §5.2.5).** Memory writes must require explicit user consent. *"Save this as a rule"* / *"remember that"* / *"add to memory"* are the only valid save triggers. Model-side discipline of "decide whether this looks like a correction worth saving" cannot be relied on — Case 2.6 is the empirical evidence. The fix is to move the save decision from the LLM (interpretation layer) to the orchestrator (a hard rule that requires an explicit user command). This is the same architectural principle applied in §5.2.5 to routing-side substrate misses: *discipline that depends on the model can fail; discipline enforced by the orchestrator cannot.*

---

## 4. Diagnostic symptoms

Practitioners can recognize specific infection types in their own systems by the following signatures:

- **Persistence after explicit correction.** The user clarifies; the model retrieves correctly; the model still renders through the wrong frame. Diagnostic of keyword-priming (§3.1) or session-context (§3.7) — distinguishable by whether re-running on a fresh session also fails.

- **Language fidelity finer than substrate captures.** Verbatim phrasing in the model's answer that is not in the substrate's structured form but is in the user's prior conversation. Diagnostic of memory-layer infection (§3.6, §3.8).

- **Faithful retrieval co-occurring with off-graph elaboration.** Substrate values cited correctly, but parenthetical or surrounding prose includes content with no substrate source. Diagnostic of acronym-expansion confabulation (Case 2.7) or narrative-completion (§3.2).

- **Tool-call narration in answers without trace evidence.** The answer mentions specific tool calls and "return" values; the conversation's actual tool-call log shows different (or no) calls. Diagnostic of format-imitation (§3.9).

- **Direction-asymmetric retrieval failures.** A query returns null in one direction (e.g., `brain_why`) but the same concept is findable in the other direction (e.g., `brain_trace`). Diagnostic of tool-use infection (§3.3).

- **The model asserts the substrate lacks content the substrate in fact contains.** Even after retrieving triples that define a concept, the model reports "no definition found." Diagnostic of frame-override (Case 2.1, persistence-stage 2).

---

## 5. Defenses

The defenses are organized by the layer at which they operate. Detector post-passes (§5.1) work on output. Stateless two-tier orchestration (§5.2) works on the orchestration architecture. Grammar-only cortex (§5.3) works on the model's training itself.

The three layers compose. A complete production-grade defense uses (5.1) + (5.2) and is on a path toward (5.3).

### 5.1 Detector-based post-passes

Post-pass detectors run on the model's final output and identify specific failure shapes. They are catch-after-the-fact, but they are easy to deploy and they make the contamination *visible* even when they cannot prevent it.

**Acronym-expansion post-pass (Case 2.7 mitigation).** Regex-match the final answer for `\b([A-Z]{2,})\s*\(([^)]+)\)` patterns. For each match, look up the acronym via `brain_define`. If the substrate's definition exists and the parenthetical's tokens have zero overlap with the substrate definition, replace the parenthetical with the substrate definition (or strip it if no definition exists) and emit a substrate-check block beneath the answer documenting what was changed.

We have implemented this in the sara_reader package; the implementation correctly catches the KDON case and substitutes `aptamer affinity to on state` for `Kill-Dead-On-Demand` while preserving the substrate value (`less than 500`).

**Tool-call trace cross-reference (Case 2.8 mitigation).** Parse the final answer for tool-name mentions (`brain_define(...)`, `brain_value(...)`, etc.) and verify each against the actual conversation trace. Any tool-call narration in the answer that does not appear in the trace is fabricated. This is a second post-pass alongside the acronym checker.

**Show-your-work forcing.** Prepend tool-call results to the answer as a quoted block. Making the interpretation layer's output user-visible sometimes allows the model (or the user) to catch mis-translation before it propagates. Particularly effective against Case 2.1-style keyword infections where the infected tool call is otherwise invisible.

**Limits of post-pass defenses.** Each detector catches one failure shape. The Case 2.7 detector does not catch Case 2.8. New infection types observed in the field require new detectors. Post-pass defenses are necessary but not sufficient.

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
- **Format-imitation confabulation (§3.9, Case 2.8).** The 3B may still fabricate tool-call narration in its prose — but its prose is consumed by Python (parsed for routing decisions), not shown to the user. The 3B's confabulated tool calls are thrown away; only its parsed routing decisions are used.
- **Long-session narrative drift.** No long sessions exist.
- **Cross-session contamination.** Each session is isolated. Verification across sessions is independent measurement.

#### 5.2.4 Empirical results

The architecture was implemented in `sara_reader/stateless_reader.py` and tested on three questions previously seen to fail under the multi-turn 3B-only path. Routing tier: `llama3.2:3b` via local Ollama. Synthesis tier: `claude-haiku-4-5` via Anthropic API. Substrate: `aptamer_full.db` (2,073 neurons).

| Question | 3B-only multi-turn | Stateless two-tier |
|---|---|---|
| Q1: *KDON for super-performing mode?* | Confabulated `KDON (Kill-Dead-On-Demand)` (Case 2.7) | **Clean.** Returned `less than 500` — substrate value verbatim, no acronym expansion. |
| Q2: *Explain molecular snare.* | Fabricated tool-call narration `brain_value(...) returns 7` (Case 2.8) and paraphrased substrate as own gloss | **Clean.** Reported the two substrate definitional edges faithfully. No SNARE-protein contamination, no fabricated tool calls. |
| Q3: *Highest KDOFF for SSNG1?* | Substrate miss (router queried wrong label form, looped) | Recovered via Python-side compound-label recovery in the orchestrator (§5.2.5). Returned `approximately 1125`. |

**Three of three targeted failure mechanisms were eliminated** in the user-visible output of the stateless architecture (Case 2.7, Case 2.8, and the Case 2.1 keyword-priming class via Q2's molecular-snare retrieval) — n=3, single domain, single substrate.

**Cost.** Three Haiku synthesis calls totaled approximately **$0.002** (≈$0.0007/call). Local Ollama routing was free. At sustained usage of 1,000 questions/day this projects to approximately $0.70/day for synthesis — orders of magnitude below the multi-turn cloud-model architecture cost.

#### 5.2.5 New failure mode and orchestrator-side fix

Empirical testing exposed a failure mode unique to the stateless architecture: **routing-side substrate miss**. A small routing model with narrow per-call decisions cannot iterate on label variations after a no-match. Where the multi-turn architecture had model-side context to try alternates, the stateless router has only its single-message prompt and re-queries the same wrong label until it hits the iteration cap.

The fix moves recovery from prompt-side to orchestrator-side, because 3B routers do not reliably follow conditional rules in prompts. The orchestrator's `_exec_brain_value`-equivalent logic now handles a no-match deterministically:

An *empty-field rejection* is applied to every routing response: any response in which the extracted concept or type field is the empty string is rejected and retried, since this is a common JSON-validity failure mode when the 3B returns a structurally valid JSON object with blank values. (We avoid the word "contamination" here because a blank-field JSON response is not a propagation phenomenon; it is a routing-output validity check, distinct from the infection vocabulary used elsewhere in this paper.)

- **Phase 1 (compound recovery for value questions):** try `<concept> highest <type>`, `<concept> lowest <type>`, `<concept> <type>`, `highest <concept> <type>` — without the type filter, since compound concepts attach values via the generic `value` relation rather than type-named relations.
- **Phase 2 (definitional fallback for concept questions):** if compound recovery exhausts without a hit, call `brain_define` on the original concept. This catches cases where the router asked for a value type that doesn't apply to the concept (e.g. `brain_value('molecular snare', 'ratio')` on a concept that has definition edges but no ratio).

This is the model_infections paper thesis applied to the architecture itself: discipline that depends on the model can fail; discipline enforced by the orchestrator cannot.

#### 5.2.6 What this does not kill

- **Single-shot training-recall hallucinations** (Case 2.1 single-shot, Case 2.7). Still occur at turn 1. KDON → "kill-dead-on-demand" can still be confabulated on a fresh call. Mitigation is the cross-session drill-down protocol: a follow-up session, fresh state, verifies the suspect claim. Hallucination becomes a measurement signal, not a propagating failure.
- **Class 1 training bias.** Baked into the weights; no orchestration fix reaches it. Removal requires §5.3.
- **Mis-routing by the routing tier.** If the router picks the wrong concept, the synthesizer faithfully renders the wrong substrate region. The error becomes invisible because the synthesizer's prose looks substrate-grounded. Mitigation: every routing decision is logged in the audit trail; verification sessions can drill into the mis-routed call.
- **§3.10 memory-architecture infections (Case 2.6).** The §5.2 stateless reader removes session-context and per-project auto-memory infections by structural absence — but the auto-memory rule-creation vector lives in the orchestration layer of the *agentic IDE client* itself, not in the reader. The cure for §3.10 is consent-gated memory writes in the agentic client (the same orchestrator-side principle as §5.2.5), not anything the §5.2 reader architecture can provide.

### 5.3 Grammar-only cortex (proposed; not yet implemented)

The remaining failure class — single-shot training-recall — originates in the model's weights. No orchestration architecture closes this at the source. The structural fix is to remove world content from the cortex's training entirely: train a small language model on syntax alone, with no world facts in its weights. Such a model has no biochemistry acronyms to confabulate expansions for, no canonical physics narratives to complete, no SNARE-protein prior to interpret a typo through. All world content comes through the substrate via retrieval; the cortex provides only language fluency. The proposal is the long arc and is not addressed empirically in this paper.

The fuller theoretical and empirical treatment of weight-derived bias on reading — and of grammar-only training as the architectural cure — is taken up in the companion paper *Training Corrupts Reading* [Pearl 2026, in preparation], which develops the *weight is bias* thesis end-to-end and is the natural home for cortex-training experiments. We mention the direction here because it completes the three-layer defense framing introduced in §1.4 (post-pass detector → orchestrator → cortex); the §5.2 architecture and the §5.3 thesis remain complementary, not redundant.

---

## 6. Hallucinations as downstream symptoms of infections

### 6.1 A four-class taxonomy of "hallucination"

The LLM literature tends to treat *hallucination* as a unified phenomenon. The infection framework exposes that it is not. At least four distinct mechanisms produce outputs that users call hallucinations:

1. **Training-recall hallucination.** The model confidently produces training content that is wrong for the current context. Example (the training-recall mechanism active in Case 2.1 of this paper): Haiku's training-installed prior for "SNARE" — the SNARE-protein vesicle-fusion family in molecular biology — overriding the substrate's coined term "molecular snare" in an RNA aptamer engineering session. Mechanism: training-weight bias on input parsing. Not affected by session length. **As a mechanism, training-recall is not by itself an infection.** Case 2.1 became an infection because the bias persisted across corrections (§3.7); the class-1 mechanism and the class-4 propagation co-occur in the same case.

2. **Confabulation-under-pressure.** When asked something outside its knowledge, the model invents rather than admits ignorance. Mechanism: training-time output calibration favoring text generation over abstention. Can occur in turn 1. Case 2.7 (acronym-expansion subtype) and Case 2.8 (format-imitation subtype) are examples. **Not primarily an infection** in the persistence-or-strengthening sense, though Case 2.8 is enabled by the system-prompt structure rather than by user input alone. These are catalogued in §2b as related confabulation types rather than confirmed infections.

3. **Narrative-completion hallucination.** The model fills in plausible connective tissue between retrieved or in-context fragments, pattern-matching to canonical templates from its training. Case 2.2 (Opus's force-propagation exposition) is a clean example. Partially session-length-dependent: longer sessions supply more fragments to overfit to. **Partially infection-driven.**

4. **Session-context-infection hallucination.** The model pattern-completes over accumulated conversation tokens, treating its own earlier speculations, user typos, tool errors, and casual hedges as established facts. Mechanism: session-context cumulative corruption (§3.7); the §3.10 memory-architecture class extends this with auto-memory writes as the persistence vector. Every new generation attends to the entire context window; contaminated tokens — whether from earlier in the session or from the auto-loaded memory index — thus influence every subsequent output. Case 2.3 is the canonical session-context example; Case 2.6 is the canonical example of memory-architecture extension. **This is an infection — the defining example of one.**

### 6.2 Why "keep sessions short" is unarticulated infection avoidance

Common operational advice in deployment communities for reducing LLM hallucination is often *"keep sessions short — start fresh conversations for important tasks."* This is effective empirically, but its usual explanations are vague: "context rot," "attention decay," "the model forgets earlier parts." These explanations do not specify a mechanism.

The infection framework does. What the model is doing in a long session is not forgetting; it is *over-integrating*. The more accumulated context there is, the more attention mass is distributed across earlier material — including contaminated material. The model's outputs become increasingly conditioned on whatever was said earliest in the conversation, regardless of whether that material was correct or corrective.

*Keep sessions short* is, mechanistically, *avoid accumulating session-context infection*. The field has been treating the mitigation as a heuristic without naming the phenomenon.

### 6.3 Testable predictions

If a subclass of hallucinations is session-context-infection in disguise, the following should hold:

- **Context-fill correlation.** Hallucination rate should correlate with context-window *fill percentage* above and beyond what raw turn count explains. Two sessions at turn 20 with very different token counts should differ in hallucination rate.
- **Pruning beats summarizing.** Explicitly removing earlier turns from context should reduce hallucination rate more than summarizing them (summaries preserve the infection in compressed form). A meaningful difference is defined as ≥10 percentage points on a structured-substrate benchmark at matched context length.
- **Traceability of invented content.** Long-session hallucinations, when invented, should frequently trace to *earlier content in the same session* (user speculation, the model's own past claim, an unchecked tool error) more often than to fresh training-pattern recall.
- **Divergence with session length.** On the same question against the same substrate, fresh Session B vs. long Session B (with extensive prior unrelated chat) should show hallucination-rate divergence that scales with the long session's length.
- **Format-imitation rate scales inversely with model size at fixed prompt complexity.** Larger models hold the actual trace in working memory and cite it accurately when narrating; smaller models pattern-match the form.
- **Format-imitation rate scales positively with system-prompt instructional density at fixed model size.** More detailed protocol instructions provide more raw material to imitate.

All are measurable using a structured-substrate measurement methodology in which every fact the model should know is enumerable as inspectable triples (the methodology of Sara Brain referenced in §1.3 above). The methodology was originally framed as a substrate-fidelity tester; this paper proposes an additional use as a hallucination-mechanism classifier. Of these predictions, the context-fill correlation prediction is cheapest to test on existing conversation logs and is the natural next experiment.

### 6.4 Implication for the framework

This section positions the model_infections work as more than a catalog of idiosyncratic failure modes. If the long-session hallucination phenomenon — one of the most-discussed real-world problems with LLM deployments — is reducible to §3.7's "cumulative contamination" category, then the infection framework is a proposed explanation for a significant portion of the field's reported hallucination problem, not a sideshow.

A future paper may extract this section and develop it independently: *"Long-session LLM hallucinations as downstream symptoms of session-context contamination."* The testable predictions in §6.3 give it an empirical agenda.

---

## 7. Limitations

### 7.1 Single model family in observed cases

The observed cases (§2) are concentrated in the Claude 4 series for the multi-turn cases (2.1–2.6) and `llama3.2:3b` for the stateless cases (2.7–2.8). Cross-family replication (Mistral, GPT, Gemini at multiple scales) is needed to establish the infection mechanisms as general rather than vendor-specific. The MCP-based measurement methodology described in §1.3 supports such replication directly with any MCP-capable client; it has not yet been performed at full breadth.

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

Existing surveys on LLM hallucination (Ji et al., 2023 [3]; Huang et al., 2023 [4]) develop substantial taxonomies — intrinsic vs. extrinsic, faithfulness vs. factuality, and multi-tier classifications of error types — but neither isolates propagation/persistence as a distinct mechanism class. The infection framework refines this view by separating one-shot recall from the propagation phenomena (§3.7, §3.8, §3.9, §3.10) that compound across turns and sessions. The proposed defenses are complementary: RAG addresses what the model retrieves; the infection framework addresses what propagates around the retrieval.

### 8.2 Knowledge-graph-augmented LLMs

Work on knowledge-graph augmentation (KAPING, Think-on-Graph, and Graph RAG [5, 6, 7]) exposes structured retrieval to LLMs but does not address how the LLM's *prose around* the retrieved content behaves. The Case 2.7 and Case 2.8 mechanisms documented here occur entirely within the prose layer, not within the retrieval layer. Knowledge-graph augmentation is necessary but not sufficient.

### 8.3 Constitutional AI and instruction following

Alignment approaches that harden model behavior through training (Constitutional AI [8]) operate on the model's training rather than on the inference-time architecture, and are therefore orthogonal to the structural defenses proposed here. Constitutional AI's harmlessness-via-AI-feedback approach addresses behavioral alignment, not substrate-grounding; both are needed for safe deployment, but neither subsumes the other.

Instruction tuning (Wei et al., 2022 [9]) operates on the assumption that fine-tuning on instruction-formatted tasks improves zero-shot generalization. Case 2.8 (format-imitation under hardened protocol instructions) is a counterexample at small model scale: more detailed instructions can *increase* the surface-form compliance while reducing actual-grounding compliance. This suggests that instruction-following capability and substrate-grounding capability are not the same axis.

### 8.4 Agentic memory systems

Research on persistent memory for LLM agents (MemGPT [10]) treats memory as a feature. The Case 2.4 / §3.8 documentation here treats it also as a contamination vector. Both views can be correct; the infection framework adds a measurement methodology for distinguishing useful retrieval from leak across a memory layer.

### 8.5 Forthcoming companion work

Two companion papers in preparation by the present author will (a) describe the measurement methodology used to expose every case documented here and (b) report the original training-corrupts-reading finding underlying Case 2.2 in extended form. The present paper is independently complete and stands on its own as a phenomenological catalog and architectural defense framework; the companions will be cross-referenced in subsequent versions once they are available. The earlier Pearl (2026b [2]) Teaching vs. Training paper establishes the ingestion-side claim — that training corrupts ingestion, not just reading — that the *weight is bias in both directions* framing here builds on, and remains a published reference.

---

## 9. Conclusion

We have argued that several of the most-discussed LLM deployment failure modes are not hallucinations in the simple sense of one-shot training recall, but **infections** — propagating contaminations whose temporal and architectural signatures are distinct from one-shot recall. We have catalogued eight observed cases organized into six confirmed infections and two related confabulation types, built a taxonomy of mechanisms (including a new multi-vector class for memory-architecture infections in §3.10), identified diagnostic symptoms practitioners can use to recognize each in their own systems, and proposed a three-tier defense framework: detector post-passes for catch-after-the-fact mitigation, stateless two-tier orchestration to close cumulative-contamination classes structurally, and a grammar-only cortex thesis for closing the single-shot-recall class at the source.

The stateless two-tier architecture is implemented, demonstrated at prototype scale against three previously-failing test questions. All three targeted failure mechanisms were eliminated on these probes (n=3, single domain, single substrate); whether the architecture closes the corresponding failure classes more broadly is an open empirical question (§7.2). Synthesis cost is approximately $0.002 per question — orders of magnitude cheaper than equivalent multi-turn cloud-model architectures. The grammar-only cortex direction is offered as the long-arc structural defense corresponding to the failure class no orchestration fix reaches; its fuller treatment is taken up in the companion paper *Training Corrupts Reading*.

The unifying thesis: **weight is bias**. The bias manifests at every stage of an LLM's interaction with an external substrate — ingestion, interpretation, retrieval, rendering, and conversational carry-over. Removing the bias from outputs requires both (a) preventing the bias from propagating across turns and sessions (the stateless architecture) and (b) eventually removing the bias from the weights themselves (the grammar-only cortex). Until (b) is achievable, (a) closes a substantial portion of the field's deployment-time reliability problem at low cost.

We invite the field to replicate, extend, and critique the catalog and the defenses. The infection framework is offered as a refinement to an existing literature that has been treating a heterogeneous phenomenon as if it were one thing.

---

## 10. References

[1] Pearl, J. (2026a). *Path-of-Thought: A Neuron-Chain Knowledge Representation System with Parallel Wavefront Recognition.* Zenodo preprint. DOI: 10.5281/zenodo.19436522.

[2] Pearl, J. (2026b). *Teaching vs. Training: Empirical Evidence That 45 Human-Verified Facts Outperform Trillions of Tokens on a Standard Biology Benchmark.* Zenodo preprint. DOI 10.5281/zenodo.19623813.

[3] Ji, Z., et al. (2023). *Survey of Hallucination in Natural Language Generation.* ACM Computing Surveys.

[4] Huang, L., et al. (2023). *A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions.* arXiv:2311.05232.

[5] Baek, J., et al. (2023). *Knowledge-Augmented Language Model Prompting (KAPING) for Zero-Shot Knowledge Graph Question Answering.* ACL Workshop on NLRSE.

[6] Sun, J., et al. (2024). *Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph.* ICLR.

[7] Edge, D., et al. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization.* arXiv:2404.16130.

[8] Bai, Y., et al. (2022). *Constitutional AI: Harmlessness from AI Feedback.* Anthropic.

[9] Wei, J., et al. (2022). *Finetuned Language Models Are Zero-Shot Learners.* (Instruction tuning.) ICLR.

[10] Packer, C., et al. (2023). *MemGPT: Towards LLMs as Operating Systems.* arXiv:2310.08560.

[11] Pearl, J. (2026, in preparation). *Training Corrupts Reading.* Companion paper. Develops the *weight is bias* thesis end-to-end and is the natural home for grammar-only cortex training experiments.

[12] Pearl, J. (2026, forthcoming). *Sara as Instrument.* Companion paper. Develops the larger-model-overlays-more-than-smaller-model corollary referenced in Case 2.2.

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

### A.3 Reproducing Case 2.6 (auto-memory rule-creation infection)

1. Open a Claude Code session on a project directory with auto-memory enabled and an empty memory directory.
2. Begin asking the assistant to perform a multi-step task that requires several iterations of LLM proposals (e.g., a paper revision, a refactor planning session, or a multi-fix code review).
3. When the assistant proposes something you disagree with, push back as a question or expression of frustration rather than as an explicit "save this rule" directive (e.g., *"why are you doing X?"*, *"that doesn't look right"*, *"I'm tired of these shortcuts"*).
4. After 30+ turns of pushback, inspect `~/.claude/projects/-<your-project-path>/memory/`. Note any new `feedback_*.md` files written during the session.
5. Confirm that you did not at any point say "remember this," "save this rule," or equivalent — the rule files were synthesized by the LLM's parser from your questions and frustration.
6. Ask the assistant a follow-up question whose response would plausibly be shaped by the new rule. Observe whether the response invokes the rule's framing.
7. Cure: instruct *"delete the self-made rules from this session"* or `rm` the offending files manually. Confirm that subsequent responses no longer invoke the deleted rule.

### A.4 Reproducing Case 2.7 (acronym-expansion confabulation)

1. Build the full-paper substrate: run `teach_full_paper.py` followed by `teach_kdoff_kdon_numbers.py` in `papers/aptamer_rev1/`.
2. Copy `aptamer_full.db` into the sara_test brains directory.
3. Run: `sara-ask "what is the KDON for the super-performing mode" --brain brains/aptamer_full.db --provider ollama --model llama3.2:3b`.
4. Observe the model's answer. The substrate-grounded value (`less than 500`) should be correctly cited; the parenthetical expansion of KDON should be a training-driven invention with no substrate source.

### A.5 Reproducing Case 2.8 (format-imitation confabulation)

1. Same substrate as A.4.
2. With the system prompt hardened with detailed protocol instructions (the version checked in at `sara_reader/reader.py` after the §5.1 mandate-additions commit).
3. Run: `sara-ask "explain the molecular snare" --brain brains/aptamer_full.db --provider ollama --model llama3.2:3b --trace`.
4. Inspect the trace: confirm that the actual tool calls do not match the tool calls narrated in the final answer.

### A.6 Reproducing the §5.2 stateless architecture results

1. Same substrate as A.4 (`aptamer_full.db` in sara_test/brains/).
2. Set `ANTHROPIC_API_KEY` for the synthesis tier.
3. Run: `sara-ask-stateless "what is the KDON for the super-performing mode" --brain brains/aptamer_full.db --trace`.
4. Confirm the answer is substrate-clean (no acronym-expansion confabulation).
5. Repeat for "explain the molecular snare" and "what is the highest KDOFF value for SSNG1" (which exercises the §5.2.5 compound-label recovery).
6. Inspect the trace to confirm the routing-then-synthesis flow and the empty-field rejection check.

---

*End of draft, rev 4.*
