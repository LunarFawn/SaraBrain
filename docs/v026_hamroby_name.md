# v026 — The name: HamRobyLLM

**Date:** 2026-05-03
**Replaces working name:** HamlinLLM (used in v025)

A grammar transformer plus small task heads. 125 M parameters. Runs on
a 3070. Knows nothing about the world.

This document is about the name and the architectural commitment the
name encodes. The technical handoff lives in
[v025_hamlinllm_status.md](v025_hamlinllm_status.md).

---

## The name

**HamRoby** is a play on two things at once.

### Hamlin Robinson

The Seattle school for dyslexic learners. Dyslexia is, at its core,
trouble with the *symbolic form* of written language — meaning is
intact, but the surface code resists. Hamlin (the model) is the
inverse case: it is competent at the form and refuses to fake the
meaning. It will not invent a fact to fill a sentence. It only
renders what its substrate says. A small engine that handles the
symbolic form well and is honest about not knowing the rest.

### Hammurabi

Hammurabi's code (~1750 BCE) is the earliest substantial body of
written law we have recovered. The point of writing law down on a
stele was that the rules became *inspectable*. A magistrate could
not invent law in their head; they had to point at the stone. The
verdict was accountable to a record anyone literate could read.

HamRoby uses the same separation. The grammar model is the engine —
it knows the *rules* of how English is structured, nothing else.
Facts live externally, in `brain.db`, the substrate. Anyone can
read the substrate. No claim about the world is made unless an
edge in the substrate supports it.

---

## Why the link matters

Most LLMs work the other way around. They memorize the world inside
their weights, ship the weights through a small number of central
pipelines, and update what is "true" by retraining. This makes a
small number of operators the de facto magistrates of meaning. The
world the model describes is the world their training run produced.

HamRoby refuses that arrangement. The model engine is shared —
everyone runs the same grammar weights. But the **substrate is
owned**. Each user keeps their own `brain.db`. Their facts. Their
refutations. Their definitions. There is no central authority that
can push a patch tomorrow morning and change what your HamRoby
believes about the world.

In Hammurabi's time, writing law down was already a kind of
democratization — it lifted law out of any single ruler's head and
fixed it to a public surface. The HamRoby twist goes further: not
one stele in the public square, but **a stele in every household**.
A distributed Hammurabi. The architecture itself returns the power
to set the record to the person who keeps it.

---

## What this looks like in practice

- No API key, no external service, no telemetry.
- The grammar model never carries world knowledge; you cannot
  extract facts from its weights because they were never put there.
- Every assertion the model surfaces traces back to an edge in
  *your* substrate.
- If you disagree with a fact, `/refute` it. If you want to add
  one, `/teach` it. Your record is yours to amend.
- The model is small enough and local enough that, although the
  weights are called `HamRobyLLM`, the *deployment* is yours.

---

## On the name itself

Phonetically: **Ham**lin **Rob**inson collapses into "HamRoby",
which sounds like Hammurabi without being it. Both layers are
audible. The first is who built the architecture's ethos (a school
that respects the gap between form and meaning). The second is
what the architecture commits to (rules externalized to a record,
accountable to anyone who can read).

Short form: **HamRoby**.
