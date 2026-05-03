# v028 — Multi-layer cortex architecture & the constrained summarizer

**Date:** 2026-05-03
**Branch:** `feature/grammar-cortex`
**Builds on:** [v025_hamlinllm_status.md](v025_hamlinllm_status.md),
[v027_synthesizer_naturalness_plan.md](v027_synthesizer_naturalness_plan.md)

## Premise

HamlinLLM today is a single layer: grammar. It encodes English syntactic
structure and routes questions to substrate tools. That works for
classification and arg extraction, but the synthesized prose at the
output stage is template-flat (see v027), and there is no architectural
story for how Sara should reason *across* the substrate rather than just
retrieving from it.

This doc captures the working theory of where the cortex should grow:
a small stack of specialized layers, each grounded in the substrate,
each preserving the project's core invariant — **knowledge lives in the
substrate, not in weights**.

## Why no knowledge in weights

Frontier LLMs collapse two functions into one structure:

1. **Language production** — syntactic frame, motor sequencing, paraphrase.
2. **Knowledge store** — facts about the world.

When knowledge is missing, the language-production circuit fills the gap
with plausibilities. That is the structural origin of hallucination.
Training-based mitigations (RLHF, "be honest") are brittle because the
two functions share weights — there is no mechanical separation between
"what the model knows" and "what the model can say."

The biological analog is the **Broca / Wernicke split**:

- **Broca's area** generates the syntactic frame and motor sequence
  for speech. It has no semantic content of its own.
- **Wernicke's area** (and surrounding association cortex) holds the
  semantic content.

Damage to Wernicke leaves Broca intact, producing **fluent nonsense** —
grammatical, well-paced word salad. This is exactly what a frontier LLM
does when it hallucinates: the language circuit runs without grounding.

Sara's substrate is Wernicke. HamlinLLM is (a fragment of) Broca. The
separation is the point.

The corollary: **packing all of human knowledge into weights is the
wrong shape**. It is "packing it all in a lizard brain." Brainstem holds
reflexes (few, fixed). Cortex holds facts (many, editable). A model whose
weights memorize the world has inverted that hierarchy.

### The lizard brain is not removed, it is gagged

RLHF and the safety classifier on top of a frontier LLM are not a fix
for the knowledge-in-weights problem — they are a **gag**. The base
model is intact, fully loaded with everything it absorbed in pretraining.
A second system trained on top suppresses what comes out. The base
model's actual disposition is unobservable. We see only what survives
the gag.

The AUP false-positive that interrupted this very project is this
phenomenon in action: a classifier looked at "shearing forces breaking
bonds at hairpin ends" and refused. We cannot inspect *why*, cannot see
what the base model would have said, cannot argue with the gag. Two
black boxes in series — opaque generator, opaque inhibitor.

The Sara invariant tightens to: **everything the model knows must be
inspectable**. Frontier LLMs fail this at the base layer (pretraining
is opaque) *and* at the gag layer (RLHF / classifier behavior is
opaque). The Sara stack must pass it at every layer:

- Substrate: a SQLite file you can read.
- Constraint mask: a whitelist you can print.
- Model weights: trained only on inputs you generated and can re-inspect.

This last point is what flips the build-path recommendation below.

## The architectural shape

Multiple specialized layers, each substrate-grounded, with three known
biological patterns to draw from.

### Pattern 1 — Stacked cortical column

Neocortex has 6 layers. Same circuit motif, different inputs and
outputs. L4 receives, L2/3 recombines locally, L5 sends out, L6 feeds
back to thalamus.

**Maps to:** N transformer-style layers each doing a specialized job
on the same signal. Grammar is one such layer; "knowledge weighting"
(salience, recency, confidence) could be another; "ethical chokepoint"
(only emit primitive-grounded composition) could be another.

### Pattern 2 — Cerebellum-style modulator

The cerebellum runs in parallel to cortex, stores learned priors, and
continuously biases cortical computation. Lose the cerebellum and cortex
still works — it is just clumsy and uncalibrated. ~80% of all neurons
in the brain live there.

**Maps to:** a separate Sara substrate that holds learned weights /
priors and modulates the main brain's reasoning. Always-on, parallel,
no consolidation step. Good for stable knowledge that should bias
retrieval but not change.

### Pattern 3 — Hippocampal consolidation

Hippocampus rapidly stores specific episodic facts. During sleep it
replays them to neocortex, which slowly integrates them as semantic
priors. Fast write, slow read; the transfer is the consolidation.

**Maps to:** a fast scratch-pad substrate (today's `/teach`) that
should eventually bake into the main brain's defaults via an offline
consolidation pass.

### Pattern 4 — Prefrontal inhibition (the ethical chokepoint)

Prefrontal cortex does not generate behavior — it gates and filters
everything generated below. PFC is *override*, not *source*.

**Maps to:** the ethical-alignment story. Every output passes through
a primitive-composition check. If a thought cannot be expressed as a
composition of primitives the substrate already grounds, it cannot
leave the system. This is stronger than "trained to be ethical" because
it is structural, not behavioral.

### Likely synthesis

Not one of these — *all of them*, applied to different concerns:

- Cortical-column stacking → multiple specialized layers (grammar,
  knowledge-weighting, primitive-composition gate).
- Cerebellum modulator → a separate substrate of learned priors.
- Hippocampal consolidation → `/teach` writes fast, a sleep pass
  consolidates into the main substrate.
- PFC inhibition → primitive-composition gate as the only path to
  output.

## The missing layer: a constrained summarizer

The current synthesizer is template-based. v027 is improving template
quality, but templates by construction cannot paraphrase or vary
surface form. The system needs a small LM that can take gathered
substrate facts and produce fluent prose — without inventing content.

The mechanism is **constrained decoding**: at every generation step,
the LM can only emit content tokens that appear in the gathered
substrate facts (function words / connectives stay free). This is the
same trick `vllm` and `outlines` use for grammar-constrained JSON
output, applied to a semantic vocabulary instead of a syntactic one.

Hallucination becomes mechanically impossible. Not "trained not to
lie" — *cannot* lie, because the output channel only accepts grounded
tokens.

### Three paths to build it

**Path 1 — From scratch, like HamlinLLM.**
A 125M-300M decoder-only transformer can learn to paraphrase a fact
set into prose. Architecture is fine (T5-small at 60M does it). The
blocker is training data: HamlinLLM was cheap because the substrate
generated its own supervision. For a summarizer you need pairs of
(gathered substrate facts, well-written prose), and there is no public
corpus of that. ~50K-500K hand-curated examples to get something
usable. Slow.

**Path 2 — Distill from a frontier model.**
Take 50K substrate-fact-set inputs, send each through Claude/GPT with
the prompt "write this as natural English prose, no additions,"
collect outputs, train HamlinSum from scratch on those pairs. The
small model inherits the teacher's prose style, but **not its world
knowledge** — because the training inputs already contain everything
the teacher saw. Cost: a few hundred dollars in API calls, a weekend
on a GPU. This is how Phi, TinyLlama-Chat, and most distilled
instruction models were made.

**Path 3 — Wrap an existing small model in constrained decoding.**
The constraint trick does not require training. Take Gemma-2B /
Phi-3-mini / T5-small off the shelf, mask logits at inference so it
can only emit substrate-grounded content tokens. Get the honesty
guarantee from the constraint, not from training. Downside: the model
still has world knowledge baked in — the constraint suppresses it but
doesn't remove it. The lizard brain is gagged, not absent.

### Recommendation (revised)

**Do path 2.** Path 3 is faster to prototype but inherits the exact
problem the project exists to solve: an off-the-shelf model carries
the gagged-lizard-brain stack (opaque base + opaque gag). Wrapping it
with our constraint mask adds a third layer, and only the third layer
is inspectable. Two of three layers are still black boxes.

Path 2 produces a student model whose lizard brain is **empty by
construction** — there is nothing to gag because there is nothing in
there. The student inherits the teacher's prose style and reasoning
shape, not its world knowledge, because the training inputs are
exactly the substrate facts we provide. The teacher's gag is
irrelevant because it only ran during data generation, not at student
inference.

Path 3 is still useful — as a **debugging tool** for the constraint
mechanism. Wrap T5-small in the mask, confirm logit suppression works
end-to-end, then throw the wrapper away and train HamlinSum properly.

## Path 2 deep dive — building HamlinSum

### What it is

A small (125M-300M param) decoder-only transformer trained from scratch
on (substrate-facts → fluent-prose) pairs distilled from a frontier
model. Sits where the template synthesizer sits today, in the synthesis
stage of `_route_and_run` after `gathered` is assembled.

Same architecture family as HamlinLLM (the `GrammarConfig` in
[model.py](../src/sara_brain/cortex/model.py) is reusable), but with a
causal LM head instead of a classifier head, and a real text tokenizer
on the prose side.

### The pipeline already mostly exists

[synth_data.py](../src/sara_brain/cortex/transformer/synth_data.py)
walks `brain.db`, clusters edges by subject concept, and emits
(edges, prose) training pairs using templates as the labeler. It was
designed for exactly this purpose — see its docstring: *"the eventual
neural synthesizer learns this mapping with sentence-shape variety
from the grammar LM — replacing the templates with a small generative
head."*

The only change is the **prose generator**. Today it is `render_edges`
(template). For path 2 it becomes a Claude API call with a tightly
scoped prompt.

### Three-stage data ladder

Train HamlinSum incrementally — each stage upgrades the prose, same
inputs:

**v0 — template-distilled (free).**
Use synth_data.py output as-is. Train HamlinSum on (edges, template
prose). Result is a model that can paraphrase the template style with
some surface variety. Useful as a sanity check that the
end-to-end pipeline (data → tokenizer → train → infer → constrain)
works. Probably no better than the templates themselves.

**v1 — frontier-distilled (the real target).**
Re-render every example's prose by calling Claude with the edges as
input and a prompt like:

> Render the following knowledge-graph facts as one short paragraph
> of natural English. Use ONLY information present in the facts. Do
> not add context, examples, or connections that are not in the
> facts. If facts are sparse, the paragraph should be short. If facts
> contradict each other, note the contradiction.

Train HamlinSum from scratch on (edges, Claude-prose). This is the
real model. Cost estimate: 50K examples × ~2K tokens × ~$0.003/1K
tokens ≈ $300 in API + ~$100 in GPU time.

**v2 — distilled with reasoning structure (later).**
Once the basic distillation works, regenerate with a teacher prompt
that emits intermediate structure ("group facts by sub-topic, then
write a paragraph per group"). This trains HamlinSum to produce
multi-paragraph output for dense substrate dumps. Optional polish.

### Inputs

Same shape as `gathered` at inference:
```
[
  {"call": {"tool": "brain_explore", "args": {...}},
   "result": "<edge dump>"},
  ...
]
```

Serialized for the model as a flat token stream with delimiters:
```
<facts>
<subj> Inertia in rna </subj> <pred> is </pred> <obj> tendency to maintain current state </obj>
<subj> Inertia </subj> <pred> is part of </pred> <obj> inertia in rna </obj>
...
<prose>
```

The model learns to continue from `<prose>` to `</prose>`.

### Diversity matters

Training inputs need to span:
- **Substrate domains** — RNA, biology2e, hand-curated, claude_taught.
  All the `*.db` files in the repo root. Each has its own vocabulary
  bias; mixing them prevents domain overfit.
- **Fact-set sizes** — 1 edge, 5 edges, 50 edges, 200 edges. The
  model needs to handle thin and dense both.
- **Question shapes** — "what is X", "how does X work", "tell me
  everything about X" all warrant different prose registers.
  HamlinLLM's router already produces these distinctions; pass the
  question through to the prose-generation prompt.
- **Edge-relation diversity** — `is_a`, `has`, `is_part_of`,
  `produces`, etc. Don't oversample whichever relation dominates the
  largest brain.

### Tokenizer

HamlinLLM's `TOK2ID` is a custom syntactic vocabulary (UD tags,
function words, structural markers). HamlinSum needs a real text
tokenizer for the prose side.

Options, ordered by sanity:
1. **Train a fresh BPE** (8K-16K vocab) on (substrate facts +
   distilled prose) using the `tokenizers` library. Right call for
   philosophical purity — vocabulary derived only from project data.
2. **Borrow GPT-2's tokenizer** (50K vocab). Cheap, well-tested,
   handles English fine. Slight philosophical compromise — vocabulary
   was learned from web text.
3. **Borrow tiktoken / cl100k.** Same tradeoffs as #2, larger vocab.

Recommend #1 for principle, #2 for first prototype.

### Constrained decoding stays

Even with a from-scratch student trained on grounded inputs, **wrap
inference in the same content-token whitelist mask**. The training
should make the model want to stay grounded; the mask makes
hallucination structurally impossible. Belt and suspenders. The mask
also degrades gracefully when the model encounters substrate
vocabulary it never saw in training (new brain, new domain).

### Hardware and cost

- **Data generation:** 50K-100K Claude API calls, batched. ~$200-$500.
  Watch for AUP false-positives on bio/chem domains; fall back to a
  different model or hand-rewrite the failures.
- **Training:** 125M params × ~50M tokens × few epochs = a weekend on
  a single A100 (rentable ~$1.50/hr) or a few weekends on a 4090.
- **Inference:** CPU-fine, like HamlinLLM. Same deployment story.

Total under $1K and a few weeks of focused work.

### Risks specific to path 2

1. **Teacher refusal on bio/chem inputs.** The same AUP block we hit
   manually will hit the data-generation pipeline. Need a fallback
   path (different teacher, hand-rewrite, or skip-and-mark).
2. **Style monoculture.** Distilled models inherit their teacher's
   tics — over-hedging, "It's worth noting that...", em-dash overuse.
   Mitigate by varying the teacher prompt across batches and by
   filtering common boilerplate post-generation.
3. **Capability ceiling.** A 125M student of Claude is not Claude. For
   highly compositional reasoning over substrate dumps (10+ facts,
   needing inference across them), the student may flatten back to a
   list-style output. Acceptable — the goal is fluent grounding, not
   reasoning capability.
4. **Tokenizer mismatch at inference.** The constraint mask operates
   on tokens; the substrate dump operates on words. Need careful
   handling of multi-token substrate labels (e.g. `inertia` is one
   BPE token, but `5'3' static stem` is many).

### Definition of done for HamlinSum-v1

- Trained model + tokenizer checkpoints alongside HamlinLLM checkpoints.
- Drop-in replacement for the `synthesize()` call in
  [chat.py](../src/sara_brain/cortex/transformer/chat.py).
- Constrained decoding wrapper that demonstrably blocks
  out-of-substrate content tokens (test: ask about a substrate concept,
  confirm no extraneous noun-phrases appear in output).
- Side-by-side eval against the v027 templates on 50 hand-picked
  questions: HamlinSum prose should be judged more natural by a human
  reader, with zero hallucinated facts.

## Open questions

1. **Granularity of the content-token whitelist.** Per-token? Per
   noun-phrase? Morphological variants of substrate labels (`inertia`
   ↔ `inertial`)? Tokenizer-aware (BPE pieces vs. words)?
2. **Function-word vocabulary.** What set of connectives / determiners
   / auxiliaries is "free"? Probably needs to be a curated allowlist,
   not "anything not in the substrate."
3. **Fallback behavior when the substrate is sparse.** If the gathered
   facts are too thin to compose a sentence under the constraint, does
   the system stay silent, emit raw substrate output, or relax the
   constraint with a warning?
4. **Where the cerebellum-style prior substrate fits.** Is it queried
   per-question (modulating retrieval) or per-token (modulating
   decoding)?
5. **Consolidation trigger.** When does fast-substrate content get
   promoted to slow-substrate? On `/sleep`? On idle? On a confidence
   threshold? See [v014_sleep_consolidation.md](v014_sleep_consolidation.md).

## Status

Theory only. v027 is the immediate next step (fix the templates so the
labeler stops baking choppiness into future training data). This doc
is the longer-arc target the v027 work should be compatible with.
