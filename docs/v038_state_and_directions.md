# v038 — state of the project after v037, and where to push next

**Date:** 2026-05-03
**Branch:** `feature/grammar-cortex`
**Companions:** [v036_v3_findings.md](v036_v3_findings.md),
[v037_layered_synth_architecture.md](v037_layered_synth_architecture.md)

This doc records where we are after v037 ships — *honestly*, not
to flatter the result — and the three directions the project can
push from here.

## What we have

A working layered synthesis stack:

- **L1 grammar** (UD-trained, 76 structural tokens, no English words)
- **L2-en function-word overlay** (175 tokens including `a / an / the / is / has`)
- **HamRoby-Sum-Core** (slot composition, zero real-language exposure
  in weights, observable verb-agnosticism via `<unk>` predicates)
- **HamRoby-Sum-EN** (English verb overlay on top of Core, 12-verb
  pool, learns which English verbs slot where)
- **Substrate (L3)** (per-user `brain.db`, the world content)

End-to-end reproducible from open data + a GPU. No internet
training, no API, no third party in the loop. Cross-brain proven:
EN trained on synthetic substrates emits clean English prose for
real-substrate concepts it has never seen.

## What this proves — narrowly but rigorously

For one specific LLM capability — **rendering structured facts as
prose** — we have demonstrated working alternatives to five
defaults of the current paradigm:

1. "Knowledge must be in weights" → false. Knowledge can live in
   queryable storage with structural slot-emission.
2. "Hallucination is unavoidable" → false. v037 cannot hallucinate
   facts. There is no token in its vocabulary that decodes to a
   real-world claim.
3. "Bigger is the answer" → false at this task. 125M + the right
   architecture beats frontier scale on substrate-stitching.
4. "Opacity is the price of capability" → false. Every layer of
   v037 is individually inspectable. Core's verb-agnosticism is
   directly observable.
5. "Centralized infrastructure is necessary" → false. End-to-end
   reproducible from open data on a 3070.

These defaults shape the LLM discourse. v037 has empirical pushback
on each, for one task class.

## What this does NOT prove

We have not refuted the internet-trained-LLM paradigm in general.
v037 covers the **narrowest, easiest** thing LLMs do. It does not
attempt:

- Open-domain reasoning (no substrate to retrieve from)
- Code generation
- Translation
- Long-context summarization
- Conversation across topics
- Creative composition
- Following novel instructions

For all of these, internet-scale training is what gives capability.
v037 has no answer for them. **Generalizing v037's success to "all
of LLM capability could be substrate-bound" is a multi-year research
program that hasn't started.**

The honest framing: v037 is a *principled alternative for one
task*, not a refutation of the paradigm. Internet-trained LLMs
solve a real problem v037 doesn't attempt — letting people get
useful answers without first building a substrate. That free lunch
is real, even if the cost (hallucination, opacity, centralization)
is also real.

## The architectural surprise worth recording

Core's predicate slots came out as `<unk>` — not as the nonsense
verbs we trained on. Mechanism: nonsense relation names aren't in
`vocab_synth`, so they UNK'd at encode time during training. Core
learned that the predicate position emits `<unk>` — i.e. has no
preference for what fills it.

This is **stronger** than the design intended. Two implications:

1. **EN doesn't fight Core.** Phase EN trains verb embeddings into
   rows that have no prior signal. No competition between layers.
2. **Per-language overlays are truly orthogonal.** A future Spanish
   overlay starts from a verb-agnostic Core; English priors don't
   bias the Spanish training. Same for any other language or
   domain-specific overlay.

Verb-agnosticism is observable: sample Core, watch it emit `<unk>`
in every predicate position. It's not a claim, it's a property.

## Where we push next

Three honest directions, in increasing scope:

### 1. Finish synthesis

- **v037.1** — extend `_RELATIONS_POOL` to cover all the verbs real
  brains use (mine from existing brains; ~30 min mechanical work).
  Retrain EN. Should close the predicate-coverage gap and make EN
  output match v032 templates' verb fidelity.
- **Chat REPL `--use-hamrobysum`** — wire EN into `chat.py` with
  template fallback for uncovered cases. Ship as a real
  user-visible feature.
- **Document the slot-expansion + article post-processor** — apply
  the v032 article heuristic to slot-expanded prose so `is a
  organism` becomes `is an organism`.

Closes the "is the synthesis layer shippable" loop. Days of work,
not weeks.

### 2. Scale the principle to a second capability

Pick another LLM behavior and ask: *can it be substrate-bound +
slot-emitted the same way?* Candidates:

- **Substrate question-answering** — given a question, retrieve
  relevant substrate edges (already exists), compose the answer
  via slot mechanism (HamRoby-Sum). Closer to a real "ask Sara
  a question" loop.
- **Multi-hop reasoning** — chain two substrate retrievals through
  the model: "Why does X happen?" → fetch X's mechanism →
  fetch the mechanism's reason → compose. Requires a planner of
  some kind on top of the substrate.
- **Substrate-grounded summarization** — take a many-edge cluster,
  compose a paragraph. Already partially what HamRoby-Sum does for
  large clusters; could be made first-class.

Each is its own architectural project. Doing one is weeks; doing
all of LLM capability this way is the multi-year program. The
question is which capability moves the project's pitch the most:
"v037 + substrate question-answering" is a real product;
"v037 + multi-hop" is a research demonstration.

### 3. Publish what we have

The instrument paper, the training-corrupts-reading paper, and now
v037 form a coherent argument: knowledge-in-weights is structurally
wrong, and we have working alternatives for at least one capability
class. A v037 paper would:

- Document the layered Core + EN architecture
- Show the cross-brain genericness result with side-by-side
  comparisons against frontier-LLM output on the same prompts
- Argue from concrete artifact, not from theory: "here is a 125M
  model, trained on UD treebanks plus pronounceable nonsense, that
  cannot hallucinate facts about your data"
- Position this as the principled alternative, distinct from
  retrieval-augmented generation (which still has an
  internet-trained generator in the loop)

This is the "make the world know it exists" path. Without it,
v037 sits in a private repo and doesn't shift the broader
discourse. With it, the architectural alternative becomes
defensible as a research direction other people can build on.

## The user's stance

> "We have the first bit that shows we need to push for more."

Recorded as the working position post-v037: **confirmation that the
alternative paradigm is constructible for at least one capability,
and intent to push the principle further** — pragmatically (finish
synthesis), architecturally (scale to a second capability), and
publicly (publish the demonstration).

Direction to be picked next; this doc records the choice point, not
the choice.
