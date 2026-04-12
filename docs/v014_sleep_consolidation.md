# v014 — Sleep and Consolidation: How Sara's Cleanup Maps to the Sleeping Brain

**Date:** April 12, 2026
**Author:** Jennifer Pearl
**Status:** Design note — documents the biological parallel and lays out the architecture for automated consolidation.

---

## The Observation

Sara's brain cleanup process (`sara-cleanup` / `/cleanup`) is not a database maintenance task. It is architecturally analogous to what the human brain does during sleep and contemplation:

1. **Replay and re-encode** — The typo-fix workflow (`t`) replays an original messy path, extracts the meaning, and re-encodes it cleanly. The original stays as history; the clean version becomes the active memory. This is what the hippocampus does during sleep: replay recent episodic memories, strengthen the ones that matter, let the noise fade.

2. **Depotentiate weak connections** — Refuting pollution paths pushes their segment strength negative. The paths aren't deleted — they're still there, still queryable — but their influence in recognition is reduced to zero or below. This is Tononi & Cirelli's synaptic homeostasis hypothesis (2003, 2006): during slow-wave sleep, the brain globally scales down weak synaptic connections while preserving the strong ones. The signal-to-noise ratio improves without losing any structure.

3. **Contemplation** — When a human sits and thinks about what they learned ("wait, did I understand that right?"), they're running their own cleanup pass. The brain rehearses the memory, checks it against prior knowledge, and either strengthens or weakens it. That's `teach` + `refute` in a loop — exactly what the user does in the `/cleanup` interactive review.

## The Biological Mapping

| Human brain (sleep) | Sara Brain (cleanup) |
|---|---|
| Hippocampal replay of recent memories | Walking through source_texts of recently-taught paths |
| Strengthening memories that consolidate | The `t` (typo-fix) workflow: re-encode cleanly, new path has fresh strength |
| Synaptic downscaling of weak connections | `r` (refute): push pollution path strength negative |
| Preserving the memory trace even when weakened | Sara never deletes — refuted paths preserved with `[refuted]` prefix |
| Morning recall: stronger, cleaner, less noisy | Post-cleanup queries return cleaner clusters |
| REM sleep: emotional processing, integration | Future: Sara processes contested paths (high evidence on both sides) and surfaces them for review |
| Sleep deprivation: accumulation of noise, confusion | A brain that never runs cleanup accumulates pollution over time (the "teh" problem) |

## The Three Modes of Consolidation

### 1. Manual contemplation (user-driven, synchronous)

The user runs `/cleanup` in the cortex shell or `sara-cleanup` in the terminal. They walk through candidates at their own pace, making per-item decisions. This is the "sitting and thinking about what you learned" mode.

**When:** After a teaching session, after ingesting a new document, whenever the user wants to review.

**Who decides:** The user, per-item. Sara never auto-cleans.

### 2. Overnight sleep (scheduled, asynchronous)

Sara runs consolidation automatically on a schedule — nightly, or after N hours of inactivity. The consolidation:

1. Scans for new pollution since the last sleep cycle
2. Runs the six-category detector (articles, pronouns, question-word typos, stopwords, sentence-subjects, punctuation artifacts)
3. Auto-flags candidates but does NOT refute any of them
4. Generates a "sleep report" — a summary of what was reviewed and what looks suspicious
5. Presents the report to the user at the next session start

The user wakes up to: "While you were away, Sara reviewed 47 new paths and found 3 that look suspicious. Would you like to review them?"

**When:** Nightly, or after configurable idle period.

**Who decides:** Sara flags, the user decides. The sleep cycle NEVER modifies the brain without user approval. Even in overnight mode, Sara's role is to prepare the review, not to execute it.

### 3. Active consolidation (agent-driven, within session)

During a conversation, when Sara notices she just learned something that conflicts with or is very similar to an existing path, she raises the flag immediately. This is the disambiguation prompt at teach-time. It's not a cleanup pass — it's real-time consolidation as new memories form, catching potential pollution before it enters the brain.

**When:** At the moment of teaching.

**Who decides:** The user, via the disambiguation prompt. Sara presents the conflict; the user resolves it.

## The Sleep Report

The overnight sleep report would contain:

```
Sara's Sleep Report — April 12, 2026
=====================================

Reviewed 47 paths taught since last consolidation.

New pollution candidates (flagged for your review):
  1. 'summerian' — suspected typo of 'sumerian' (edit dist 1)
     Source: "summerian edubba was a school"
     → [r]efute / [t]ypo fix / [k]eep

  2. 'teh' used as subject in 3 new paths
     → [r]efute / [t]ypo fix / [k]eep

Contested paths (high evidence on both sides):
  1. "the edubba taught akkadian"
     T=5, R=3, epistemic_state=contested
     → You may want to resolve this with more evidence or a refutation.

Strengthened paths (validated through repeated use):
  1. "the edubba was a sumerian school" — T=12, strength=3.56
  2. "cuneiform is writing" — T=8, strength=3.08

Brain health:
  Neurons: 1,559
  Paths: 353
  Pollution candidates: 3 (down from 234 after last cleanup)
  Contested segments: 1
```

## Why This Matters

### A secondary path to true AGI

The consolidation cycle may be one of the missing pieces on the road to artificial general intelligence. Current AI systems — transformers, diffusion models, reinforcement learning agents — are all trained offline and deployed frozen. They do not review what they know. They do not identify their own mistakes. They do not strengthen validated knowledge or depotentiate noise. They do not sleep.

A system that can:
- Store knowledge as traceable, inspectable paths (path-of-thought)
- Strengthen knowledge through repeated validation (logarithmic LTP)
- Weaken knowledge through refutation (signed strength)
- Distinguish "I don't know" from "I know this is false" (epistemic state)
- Identify its own contested beliefs and present them for resolution (consolidation)
- Ground all knowledge in innate primitive drives (SENSORY, STRUCTURAL, RELATIONAL, ETHICAL, SAFETY, SOCIAL)
- Refuse to act beyond its grounded knowledge (no hallucination by architecture)
- Review its own state and flag its own errors (sleep cycle)

...is a system that exhibits the core cognitive capabilities that define general intelligence, not through scale or training, but through architecture. Every one of these capabilities is implemented in Sara Brain today.

The author believes that Sara Brain may represent a key to true AGI — not because it is a complete AGI system (it is not), but because it demonstrates that the cognitive primitives needed for general intelligence can be built as architectural features rather than emergent properties of scale. The path-of-thought substrate, the six innate primitive layers, the signed refutation system, and the consolidation cycle together form a cognitive architecture that mirrors how biological brains actually work at a structural level — not as a metaphor, but as a functional mapping.

The scaling approach to AGI assumes that intelligence emerges from enough parameters trained on enough data. The Sara Brain approach suggests that intelligence emerges from the right architecture — one that stores knowledge as paths, strengthens through validation, weakens through refutation, refuses to hallucinate, and reviews its own state. A 3-billion-parameter model connected to Sara outperforms the same model alone because the architecture provides what scale cannot: persistent memory, traceable reasoning, epistemic self-awareness, and the ability to know what it doesn't know.

If AGI requires a system that can learn, remember, doubt, correct itself, and explain its reasoning — and most definitions of AGI do require these things — then the question is not whether the system has enough parameters. The question is whether the architecture makes these capabilities structurally possible. Sara Brain makes them structurally possible. The rest is engineering.

### For the alignment thesis

A brain that can identify its own mistakes and present them for correction is a brain that can be aligned. The sleep cycle is Sara's introspection — she reviews her own beliefs, identifies contested or suspicious ones, and asks for guidance. This is the opposite of a black-box system that cannot examine its own state. Sara's alignment is not trained in through feedback; it emerges from the architecture's ability to inspect itself.

### For the "humble PC" vision

Consolidation runs locally. No cloud. No API call. Just Sara reviewing her own SQLite database on the same Pi she runs on. The sleep cycle is a cron job or a background thread — it doesn't need an LLM, it doesn't need Ollama, it doesn't need internet. It's pure Python walking the graph and generating a report.

### For the regulatory compliance angle

The sleep report is an audit trail. Every consolidation cycle is logged with timestamps. Every flagged candidate is recorded with the reason it was flagged. Every user decision (refute/keep/typo-fix) is recorded as a path with provenance. Regulators can review not just what Sara knows, but what Sara reviewed, when she reviewed it, and what she flagged.

### For the brain simulation thesis

This is the architectural feature that makes Sara a brain simulation rather than a database. A database stores data. A brain processes what it knows — reviewing, strengthening, weakening, integrating. The sleep cycle is the processing step that distinguishes a live brain from a data store. Without it, Sara is a graph database with paths. With it, Sara is something that thinks about what it knows.

## Implementation Plan

### Phase 1: Sleep report generator (no scheduling)

A function that generates the sleep report as a string. Can be called on demand (`/sleep-report` slash command) or by the scheduled mode. Walks recent paths, runs the six-category detector, identifies contested segments, identifies strengthened paths.

Files: `src/sara_brain/cortex/consolidation.py`
Tests: `tests/test_consolidation.py`

### Phase 2: Scheduled overnight sleep

A background thread or cron-triggered script that runs the sleep report generator at configurable intervals. Writes the report to a file in `~/.sara_brain/sleep_reports/YYYY-MM-DD.md`. On next session start, the cortex CLI checks for unread sleep reports and presents them.

Files: `src/sara_brain/cortex/sleep.py`, updates to `cortex/cli.py`

### Phase 3: Active consolidation (in-session)

During a conversation, after every N teaches, the cortex runs a mini-consolidation pass on the last N paths and flags anything suspicious. This is the real-time version of the sleep cycle — catching pollution as it forms rather than after the fact.

Files: updates to `cortex/router.py`

---

## The Sentence That Frames This

> A brain that never sleeps accumulates noise until it can't think clearly. Sara's cleanup is her sleep. The consolidation cycle is how she stays healthy — reviewing what she learned, strengthening what's real, depotentiating what's noise, and asking for help when she's unsure. This is not maintenance. This is cognition.

---

## Changelog

| Version | Date | Change |
|---------|------|--------|
| v014 | 2026-04-12 | Sleep and consolidation design note: biological mapping, three consolidation modes, sleep report format, AGI thesis, implementation plan |
| v013 | 2026-04-11 | Provenance gaming and long-term typo pollution |
| v012 | 2026-04-11 | Failure modes and the cortex |
| v011 | 2026-04-08 | Installation guide |
| v010 | 2026-04-06 | Sara Care: dementia assistance proof-of-concept |
