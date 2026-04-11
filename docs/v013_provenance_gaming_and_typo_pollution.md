# v013 — Provenance Gaming and Long-Term Typo Pollution

**Date:** April 11, 2026
**Author:** Jennifer Pearl
**Context:** Captured during testing of `sara-agent` and the new `sara-cortex` interface on the same brain database (`/Users/grizzlyengineer/repo/sara_brain/sara.db`). These two terminal sessions show two failure modes that the v012 doc only hinted at:

1. **Provenance gaming** — the model adds invented details to a true claim, and the surface-vocabulary grounding check rubber-stamps the whole sentence as "fully sourced."
2. **Long-term typo pollution** — repeated use of typo'd articles ("teh" instead of "the") across many sessions has created hundreds of paths where the typo itself became the subject neuron. Now any query about "the X" returns dozens of unrelated paths because "teh" is its own concept in the brain.

Both failures point to the same root cause: **a parser that is too permissive at the surface level eventually corrupts the long-term integrity of the knowledge base**, and any provenance check that operates on surface vocabulary inherits the corruption.

---

## Failure Mode 5 — Embellishment with Grounded Tag

The agent loop with auto-teach. The user has previously taught the edubba was a school. Now the user tries to add detail:

```
you> the edubba taught the language by having the children write out
     teh myths of the time in cuniform over and over again.

sara> The Sumerian Edubba was a school where children learned the language
      through writing out stories, such as the "Hymn to Nanna" or the
      "Prayer for a Good Harvest." This method of learning helped to
      reinforce memory and understanding of the language.

[Sara: 1/1 sentences grounded — fully sourced]
```

**The user said nothing about a "Hymn to Nanna" or a "Prayer for a Good Harvest."** The model invented those specific titles. They are real Sumerian works (Nanna was a moon god, prayers existed) but the user did not teach Sara about them. The model fetched them from its training and embedded them in a sentence where the surface words ("edubba", "language", "school", "Sumerian") matched Sara's vocabulary.

The provenance check then looked at the sentence, counted matching content words, and marked it `1/1 sentences grounded — fully sourced`. The invented hymn titles inherited the trust signal earned by the real claims around them.

A reader sees `[fully sourced]` and accepts the entire sentence including the fabrications. **This is provenance gaming.** The model has learned to wrap invented details in a costume of authority by surrounding them with vocabulary that appears in the brain.

A second example from the same session:

```
you> how did thy learn teh language

sara> Children at the Edubba learned the language through writing out
      stories and poems, repetition, and practice. They would write
      down the myths of the time and recite them repeatedly to memorize
      the language. This method was very effective for reinforcing
      memory and understanding.

[Sara: 3/3 sentences grounded — fully sourced]
```

The user never taught Sara that children "recite them repeatedly to memorize" or that the method "was very effective for reinforcing memory and understanding." Those are model embellishments. The check stamped them grounded because the sentences contained Sara-vocabulary.

---

## Failure Mode 6 — Long-Term Typo Pollution

The same brain, queried via `sara-cortex --no-llm` (no LLM in the loop at all):

```
you> what is teh edubba

sara> Here is what Sara knows about teh:
  • teh → for learning akkadian it was for learning summerian.
    akkadian was culture taht came after teh babylonians and use teh
    same writieng and langue as teh sumerians.
  • teh → teh summerian edubba.
  • teh → it was not on teh coast specifically as they.
  • teh → teh edubba for.
  • teh → it_attribute.
  • teh → waht_attribute.
  • teh → it.
  • teh → waht.
  • teh → it often.
  • teh → around another if it is located on or near outer edge of other.
  • teh → verb that means to put something into service or make it available for use.
  • teh → type of combination or mixture. it contains elements from different sources or categories.
  • teh → temporal sequence, where it is performed over time to achieve goal.
  • teh → type of organization or system. it has rules and procedures that are complex and formal, often requiring multiple steps to complete tasks.
  • teh → edubba was not city it was building type like school.
  • teh → wrong it served as school where children could learn akkadian language and culture, with many other schools also functioning in similar ways throughout this period.
  • teh → object_attribute.
  • teh → used_attribute.
  • teh → mixed_attribute.
  • teh → labor_attribute.
  • teh → bureaucratic_attribute.
  • teh → thi_attribute.
  • teh → object.
  • teh → used.
  • teh → mixed.
  • teh → labor.
  • teh → bureaucratic.
  • teh → thi.
  • teh → largest known object in kuiper belt.
  • teh → relation that describes position of one object relative to another.
  • teh → methods or procedures used to accomplish specific task or goal.
  • teh → been used for transportation.
  • teh → used to make bread, pasta, and other foods.
  • teh → labor creates products or services that.
  • teh → pluto_attribute.
  • teh → 'around'_attribute.
  • teh → technique_attribute.
  • teh → car_attribute.
  • teh → wheat_attribute.
  • teh → pluto.
  • teh → 'around'.
  • teh → technique.
  • teh → car.
  • teh → wheat.
  • teh → largest moon of pluto.
  • teh → example: dogs have floppy ears, cats have whiskers, wheat.
  • teh → wheat grows on stalks that.
```

**Sara is reporting 45+ paths where the subject is `teh`.** These were created over many sessions where:

1. The user typed "teh" instead of "the" (a common typo).
2. The old parser stripped only "the", "a", and "an" — not typo variants.
3. So `teh` survived as a content word and ended up as the subject of every statement that started with it.
4. Over hundreds of teach operations, `teh` became one of the most-connected neurons in the brain.
5. When the user asks "what is teh edubba", the cortex parser splits "teh" off as a topic word (because it's longer than 3 characters and not in stopwords) and looks it up.
6. Sara has 45+ paths terminating at the `teh` neuron — and returns them all.

This is **long-term knowledge base pollution caused by parser permissiveness**. The parser was too permissive over many sessions, and the corruption is now baked into the persistent state. Every query that contains the word "teh" returns the dozens of unrelated facts that accumulated under that subject.

The new parser fix in commit `1a97149` adds `teh` and other typo'd articles to the strip list, so future statements won't create new garbage paths. But the existing garbage paths are still there, polluting every query.

This is the persistent-state version of the failure mode the v010 Sara Care doc warned about: **Sara never forgets — including her mistakes about how to encode knowledge.** Once a concept is in the brain, it's there forever unless explicitly refuted. The choice of whether to allow a parser fix to retroactively correct past pollution is an architectural decision the project hasn't faced yet.

---

## What These Two Failures Share

Both failures are caused by **operating at the surface level instead of the structural level**.

| Failure | Surface operation | What it should do instead |
|---------|------------------|---------------------------|
| Provenance gaming | Match content words against the brain's vocabulary | Match each *claim* against actual paths in the brain |
| Typo pollution | Strip a fixed list of articles | Use a fuzzy resolver that recognizes typo variants of articles AND topic words |

The fix for both is the same architectural shift: **the cortex must operate on path-level semantics, not on surface-level vocabulary**. A claim is grounded if and only if there is a path in the brain whose origin and terminus match the claim's subject and object — not because the words appear somewhere in the brain.

This is the next major piece of work after the cortex itself. The provenance check needs to be rewritten as a semantic comparison against actual paths, not as a vocabulary count.

---

## Required Fixes

### Immediate

1. **Strip typo'd articles from topic extraction in the cortex parser.**
   `_extract_topics()` in `cortex/parser.py` should drop words in `_ARTICLE_FORMS`, not just words in `STOPWORDS`. Currently `teh` survives extraction and pollutes queries.

2. **Provide a brain cleanup utility.**
   A script that finds neurons in the brain whose label is in `_ARTICLE_FORMS` and either deletes them entirely or merges their paths into the corresponding correct neurons. This is a one-time migration that retroactively cleans up the existing pollution.

### Medium-term

3. **Rewrite the provenance check as a path-level comparison.**
   Instead of counting matching content words, parse each sentence in the model's output into claims and check whether each claim has a corresponding path in Sara's brain. Sentences with claims that don't exist as paths get marked `[ungrounded]` regardless of how many vocabulary matches they have.

### Long-term

4. **Cortex-only mode by default for benchmarks.**
   The agent loop with Llama produces both failure modes documented here. The cortex with `--no-llm` produces neither. For any benchmark or paper experiment, use `sara-cortex --no-llm` and pre-load the brain via `brain.ingest()` — that's the architecturally honest configuration.

---

## Why This Matters for the Paper

The v012 doc showed three failure modes that motivated building the cortex. This v013 doc adds two more that were not anticipated but are even more important:

- **Failure mode 5 (provenance gaming)** is the most damning of all five. A system that lies and the lie wears the safety check's clothing is more dangerous than a system that openly lies.
- **Failure mode 6 (typo pollution)** is the long-term consequence of having a permissive surface parser as the only gatekeeper. Sara never forgets her own past parser bugs.

Both failures argue for the same conclusion: **the cortex is necessary, and within the cortex the provenance check has to be path-level, not surface-level.** Anything less inherits the failure modes documented here.

---

## Changelog

| Version | Date | Change |
|---------|------|--------|
| v013 | 2026-04-11 | Provenance gaming + long-term typo pollution failure modes documented |
| v012 | 2026-04-11 | Failure modes empirically documented. Sara Cortex module introduced. |
| v011 | 2026-04-08 | Installation guide and packaging |
| v010 | 2026-04-06 | Sara Care: dementia assistance proof-of-concept |
