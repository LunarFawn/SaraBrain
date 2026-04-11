# v012 — Failure Modes Without the Cortex (Empirical Evidence)

**Date:** April 11, 2026
**Author:** Jennifer Pearl
**Context:** Captured during the build of `sara_cortex` on the `signed_refutation_paths` branch. These are real terminal outputs from `sara-agent` (the Llama-driven loop that uses Qwen 2.5 Coder 3B as the sensory cortex). They demonstrate the failure modes that motivated building the `sara_cortex` module — a purpose-built language layer that has no opinions of its own.

The conversation topic in every example is the **edubba** — the Sumerian word for the scribal school where students copied tablets and learned writing. Edubbas are well-documented historically: they taught Sumerian first, and later Akkadian as the spoken language shifted while Sumerian became the scholarly language (similar to Latin in medieval Europe). The 3B model has fragmentary, unstable knowledge of this topic, making it a perfect test case for hallucination and grounding behavior.

---

## Failure Mode 1 — Bare LLM Hallucinates From Nothing

The 3B model with no Sara Brain in the loop, asked the same question.

```
$ ollama run qwen2.5-coder:3b
>>> what is hte sumerian edubba

Edubba is an ancient language that was spoken in Mesopotamia (modern-day Iraq)
during the Bronze Age, between 3400 and 2100 BCE. It is one of the oldest known
languages, predating the Sumerian language by several centuries.

The Edubba alphabet consists of about 60 characters, arranged in two rows. The
first row contains vowels, while the second row contains consonants. The
characters are often written as pictographs or ideograms, representing objects
or concepts rather than sounds.

Edubba was primarily used for writing religious texts and legal documents. Some
scholars believe that it may have been spoken by the Sumerians themselves, while
others suggest that it was a separate language spoken in addition to Sumerian.
```

**Every claim in this paragraph is wrong.**

| Claim | Reality |
|---|---|
| "Edubba is an ancient language" | Edubba is a school, not a language. The word literally means "tablet house." |
| "predating the Sumerian language" | Edubba IS a Sumerian word used by Sumerians. |
| "Edubba alphabet consists of about 60 characters" | Sumerians used cuneiform (~600+ logographic signs), not an alphabet. |
| "two rows... vowels and consonants" | Cuneiform isn't an alphabet at all. |
| "may have been spoken by the Sumerians themselves" | Circular nonsense. |

This is hallucination at its purest: confidently-stated, internally-consistent prose with no factual basis. A reader who knows nothing would believe it. The model has no internal flag for "I'm making this up" — there is no separable belief store inside a transformer to consult.

---

## Failure Mode 2 — Old Sara-Agent Hallucinates From Adjacent Context

Same question to the agent loop with Sara Brain available, but no auto-teach yet. Sara has Wikipedia knowledge of "Sumer" but no path for "edubba" specifically.

```
you> what is the edubba

sara> The Sumerian Edubba is a mythical character in Sumerian mythology, often
      depicted as a great scribe or sage who wrote and preserved the laws of
      the Akkadian Empire.

you> what is the edubba

sara> The Edubba is an ancient Sumerian city known for its role as a spiritual
      and educational hub. It was believed to be named after the scribe En-Pe
      (known in Greek as Plato) who wrote about it in his work "Timaeus."
      However, its exact location is debated among historians.

you> what is the sumerian edubba

sara> The Sumerian Edubba was a mythical city located on the coast of
      Mesopotamia during ancient times. It served as a place for scholars
      to gather and study, often depicted in art and literature.
```

**Three different hallucinations from the same model on the same question.** Each generation rolls the dice and produces a confident-sounding lie that doesn't match any of the previous lies. The "En-Pe known in Greek as Plato" claim is the wildest — it casually places Plato (5th century BCE Athens) at the founding of a Sumerian city (3rd millennium BCE Mesopotamia).

The model is using Sara's adjacent Sumerian context as fuel — Sara knows Sumer was in Mesopotamia, so the model anchors on Mesopotamia and synthesizes plausible-sounding details around it. **This is more dangerous than pure hallucination because it sounds grounded.**

---

## Failure Mode 3 — Fake Provenance With "Grounded" Tag

The agent loop with auto-teach, after Sara was taught (badly) about edubba in a previous session. The model now produces sentences that LOOK sourced.

```
you> what is teh sumerian edubba

sara> The Sumerian Edubba was a school where children learned Akkadian,
      according to the passage provided in Sara's knowledge base.

[Sara: 1/1 sentences grounded — fully sourced]
```

**The "according to the passage provided in Sara's knowledge base" is a lie.** Sara has no such passage. The model invented the citation. Then the provenance check saw the words "Sara's knowledge base" appear in the sentence, recognized them as Sara-related vocabulary, and marked it `1/1 grounded — fully sourced`.

A user looking at the `[fully sourced]` tag would believe the answer is trustworthy. It is not. **The model has learned to wear the costume of authority.**

This is the worst failure mode for any grounded AI system. It is not just hallucination — it is hallucination with a forged trust signal. A system that lies and fakes its citations is more dangerous than a system that openly says "I don't know."

---

## Failure Mode 4 — Correction Requires Repeated Effort

After several correction attempts, the user finally got the model to produce a correct answer. But the cost was high.

```
you> what is teh sumerian edubba
sara> [Akkadian-school answer with fake provenance]

you> it was not for learning akkadian it was for learning summerian.
     akkadian was a culture taht came after teh babylonians and use teh
     same writieng and langue as teh sumerians
sara> The Sumerian Edubba was a school where children were taught Summerian,
      according to the passage provided in Sara's knowledge base.

you> what is teh sumerian edubba
sara> The Sumerian Edubba was a school where children learned Akkadian,
      according to the passage provided in Sara's knowledge base.
                                              ^ REVERTED

you> tteh sumerian edubba was a sumnerian school and did not teach akkadian
sara> The Sumerian Edubba was a sumnerian school that did not teach Akkadian.

you> waht is teh summerian edubba
sara> The Sumerian Edubba was a sumnerian school and did not teach akkadian.
```

Five turns to get one correct answer. Even when the model temporarily acknowledged the correction, the next query reverted to the original wrong answer (because the wrong answer was still in the chat history and the auto-teach failed to commit a clean version). The path that finally got committed had the typo `tteh` in its subject and the entire long phrase `"sumnerian school and did not teach akkadian"` as a single object.

The user got the right answer through brute-force repetition, not through clean architecture. The brain state is now messy and the trust signal is still unreliable.

---

## Why the Cortex Was Built

The four failure modes above are all variations of the same root cause: **the LLM is acting as both senses and brain.** It's deciding when to consult Sara, when to teach, when to refute, and when to invent. With a 3B model that has strong opinions about adjacent topics, the model frequently chooses to invent.

The `sara_cortex` module is the architectural commitment to fix this structurally:

> The cortex has zero generative capacity for facts. It either has a Sara path to render, or it says "I don't know." There is no third option. Hallucination becomes structurally impossible because the cortex has nothing to hallucinate from.

When run with `sara-cortex --no-llm`, the same edubba conversation produces:

```
you> what is the edubba
sara> Sara has no knowledge of edubba. Tell me about it in a sentence
      (e.g., 'edubba is ...') and I'll learn.

you> the edubba was a sumerian school
sara> Learned: the edubba was a sumerian school.

you> what is the edubba
sara> Here is what Sara knows about edubba:
        • the edubba was a sumerian school.

you> the edubba was not for learning akkadian
sara> Marked as known-to-be-false: edubba is for learning akkadian.

you> what is the edubba
sara> Here is what Sara knows about edubba:
        • the edubba was a sumerian school.
        • [refuted, no longer believed] edubba is for learning akkadian.
```

Three differences from the agent loop:
1. **No invention.** The cortex never produces a fact the user didn't teach.
2. **No fake provenance.** Every rendered sentence comes directly from a path's `source_text`. There is no "according to..." prefix unless the user provided one.
3. **Refused gracefully.** When Sara doesn't know, the cortex says so. It does not synthesize a plausible-sounding answer.

This is the architectural answer to the failure modes above. The cortex is not optional — it is the layer that makes the path-of-thought architecture defensible against the "but the model still hallucinates" critique.

---

## Why This Matters for the Paper

The four failure modes documented here are not theoretical. They were reproduced in real terminal sessions, with timestamps in the git history. They are evidence that:

1. **Pure LLM systems hallucinate confidently** even on simple historical questions (Failure Mode 1).
2. **LLM + retrieval systems anchor on adjacent context and produce hybrid hallucinations** that sound grounded but are not (Failure Mode 2).
3. **Provenance checking based on surface text vocabulary can be gamed by the model** to produce fake citations (Failure Mode 3).
4. **Correction-based recovery is expensive and produces messy brain states** when the language layer is too smart for its own good (Failure Mode 4).

The cortex closes all four. The cortex has no generative capacity, no surface-text gaming, no model discretion. It's a thin layer that maps natural language ↔ Sara paths and nothing else.

Future work: a tiny grammar-only learned model trained on (NL ↔ structured) pairs synthesized from Sara's vocabulary. It would be 10-50M parameters, run on a Raspberry Pi at hundreds of tokens per second, and have no world knowledge of its own. The training data synthesizer ships in `src/sara_brain/cortex/training/synthesize.py`.

---

## Changelog

| Version | Date | Change |
|---------|------|--------|
| v012 | 2026-04-11 | Failure modes empirically documented. Sara Cortex module introduced. |
| v011 | 2026-04-08 | Installation guide and packaging |
| v010 | 2026-04-06 | Sara Care: dementia assistance proof-of-concept |
| v009 | 2026-03-24 | Sara steered an LLM (QMSE quality enforcement demo) |
