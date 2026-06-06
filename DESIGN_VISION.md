# Sara Brain — Design Vision

**Author:** Jennifer Pearl
**Date:** 2026-06-03
**Status:** Active — this is the target for all development sessions

---

## What Sara Brain Is

Sara Brain is a **persistent, teachable, inspectable knowledge system** that runs on minimal hardware with no internet dependency. Anyone can teach it. Anyone can ask it. Every answer is traceable to the facts it was taught. It never invents knowledge it wasn't given.

Sara is not a general-purpose reasoning engine or code generator. Sara is a **knowledge assistant** — a chatbot with persistent memory, retrieval, and teaching. You teach it information, it tells you that information back when you ask. It doesn't reason about novel problems or write code — it remembers what it was taught and retrieves it faithfully.

## The Core Interaction

```
TEACH:  "Sara, the Carbon Helix is a sentient combat ship."
        → stored permanently, traceable, correctable

ASK:    "What is the Carbon Helix?"
        → wavefront finds the path → renders the answer
        → "The Carbon Helix is a sentient combat ship."

REFUTE: "Sara, the Carbon Helix is NOT a space station."
        → counter-path recorded, original claim marked false
```

Teach. Ask. Correct. That's it.

## Architecture

```
Teaching:
  Document or voice → [Extractor 115M] → triples → Sara Brain (SQLite)
  Human correction  → teach_triple / refute → immediate update

Retrieval:
  Question → Wavefront Propagation → Rendered Facts → [Synthesizer] → Answer

The wavefront IS the reasoning. It is not a tool to be selected.
It runs first, always, automatically.
```

Two from-scratch models (115M each, zero domain knowledge):
- **Extractor:** reads text, produces structured triples
- **Synthesizer:** reads wavefront output, produces prose answers

Both trained on synthetic nonsense. Neither knows anything about any domain. All domain knowledge lives in Sara's substrate.

## Design Principles

1. **The wavefront is the brain.** It is not one retrieval option among many. It runs first, always.

2. **Knowledge belongs in the substrate, not in weights.** Models provide language competence. Sara provides knowledge. Neither does the other's job.

3. **Teach at runtime, not at training time.** Adding a fact: microseconds. Retraining a model: impossible/unnecessary.

4. **Every answer is traceable.** Query any fact back to its source: who taught it, when, from what document.

5. **Correct, don't retrain.** Wrong fact? Refute it. Missing fact? Teach it. No model surgery needed.

6. **Runs on hardware people already own.** Raspberry Pi. Arduino Uno Q. Old laptop. Phone. $60, not $60 million.

7. **No internet required.** The system works offline. Knowledge doesn't leave the device unless you copy it.

8. **Smaller models are more faithful.** The industry scales for knowledge storage. We scale for language competence only. 115M is the right size for structural tasks.

## Use Cases (Same Architecture, Different Substrate)

### Household
- "Going to the park, back at 5" → roommate asks "where's Alex?" → "At the park, back at 5"
- "Did we feed the cat?" → "Yes, fed at 3pm by Morgan"
- Shared family memory on a Pi on the shelf

### Medical / Diagnostic Lab (Remote)
- Expert arrives with protocols and reference ranges
- Feeds documentation to Sara, teaches/corrects until satisfied
- Leaves — local staff query test results against the substrate
- Every interpretation traces to the expert's teaching
- Next expert can update guidelines by teaching new facts

### Construction Site
- Specifications, load ratings, permit locations, safety protocols
- Foreman teaches site-specific facts
- Workers query: "What's the max load on beam 4?" → traceable answer

### Education
- Teacher teaches curriculum facts
- Students ask questions
- Answers trace to specific taught content — not hallucinated
- Different grade levels = different substrates

### Cultural Preservation
- Elders teach traditional knowledge, oral history, practices
- Community accesses it across generations
- Knowledge persists in a .db file — no server, no subscription

### Engineering / Maintenance
- Feed it a manual → extractor learns the system
- Technicians ask diagnostic questions
- "What causes fault code E47?" → answers from the manual, traceable

## What Must Work for This Vision

### Teaching Pipeline
1. Feed a document → extractor produces triples → Sara learns
2. Human reviews, corrects, teaches what the extractor missed
3. Result: high-quality substrate with full provenance

### Retrieval Pipeline
1. Ask a question → wavefront propagates from question concepts
2. Converged neurons' facts are rendered as readable sentences
3. Synthesizer (or small LLM) produces prose answer
4. If insufficient data: "I don't have enough information. Teach me."

### Correction Pipeline
1. Wrong answer → human says "No, that's wrong because X"
2. System refutes the wrong fact (counter-path)
3. System learns the correct fact (new path)
4. Immediate — no retraining, no waiting

### Hardware Target
- Minimum: Raspberry Pi 4 (4GB) or Arduino Uno Q ($59)
- Inference: CPU only, < 2 seconds per question
- Storage: SQLite .db file, megabytes to low gigabytes
- Training the models: RTX 3070 (one-time, 2 hours)
- Running the models: any CPU

## Current State (2026-06-03)

### Working
- Extractor (115M): reads documents, produces structured triples
- Wavefront propagation + renderer: finds facts, produces readable output
- Sara-demo CLI: teach, ask, compare
- Full pipeline (sara-pipeline): document → teach → ask
- Benchmark: Sara (0 params) beats 1B model on biology
- Novel concept retrieval: answers about content not in any training data

### Needs Improvement
- Extractor misses key definitions (captures relations but not "X is_a Y")
- Synthesizer produces fragments (vocab limitation, needs retraining)
- No voice interface yet
- No temporal facts ("back at 5") — only static relationships
- No confidence indicator ("I'm not sure about this")

### Not Yet Built
- Voice input/output (teach by speaking, ask by speaking)
- Temporal/event facts with timestamps
- Multi-user teaching with attribution
- Mobile deployment (Android/iOS app)
- Streaming teach from live conversation

## How to Work on This

Every development session should:
1. Read this document to understand the target
2. Pick one thing from "Needs Improvement" or "Not Yet Built"
3. Build it, test it, commit it
4. Never demote the wavefront. Never bypass it. It IS the brain.

## The Thesis

The AI industry spends billions compressing knowledge into opaque weight matrices. Sara Brain demonstrates an alternative: knowledge in an inspectable, correctable, persistent substrate; minimal models for language competence only; reasoning through deterministic graph traversal.

A $60 board with Sara Brain, taught for 30 minutes by a domain expert, produces correct, traceable, correctable answers that a billion-dollar model cannot match on novel content.

Knowledge belongs in the hippocampus, not compressed into the cortex.

---

*— Jennifer Pearl, 2026*

## Next Engineering Priority: Multi-Pass Teaching

The extractor bottleneck: one pass tries to extract everything at once.
Better approach — multiple focused passes over the same document:

- **Pass 1 (Definitions):** What is X? X is_a Y. Focus only on definitions.
- **Pass 2 (Relationships):** What does X do? X produces/requires/contains Y.
- **Pass 3 (Bridges):** What connects X to Z? Conceptual bridges for wavefront.

Each pass uses the same extractor model but with different prompt/framing,
or a dedicated lightweight model per pass type. The result is a richer
substrate with better coverage than a single pass.

This mirrors how humans read: first pass for "what is this about?",
second pass for "how does it work?", third pass for "how does it connect
to what I already know?"
