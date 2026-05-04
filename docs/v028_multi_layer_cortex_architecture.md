# v028 — Multi-layer cortex architecture & the constrained summarizer

**Date:** 2026-05-03
**Branch:** `feature/grammar-cortex`
**Builds on:** [v025_hamlinllm_status.md](v025_hamlinllm_status.md),
[v026_hamroby_name.md](v026_hamroby_name.md),
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

## Sub-architecture of the language side: L1 / L2 / L3

The Broca/Wernicke split says language production is separate from
knowledge. It does not say what is *inside* the language production
side. Watching what kids actually lose if they miss the critical
period for language exposure suggests language production is itself
layered:

- **L1 — universal grammatical capacity.** The deep structural
  competence that makes any human language learnable. Innate, baked
  in by genetics + early development. Use-it-or-lose-it: kids who do
  not get sufficient language input during the critical window
  (Genie, late-acquired sign language in deaf children) never fully
  recover full grammatical fluency in any language. The scaffold is
  not built later.

- **L2 — language-specific overlay.** Surface-form competence in a
  particular spoken language: function-word vocabulary
  (`a / an / the / of / is`), morphological inflection, idiomatic
  ordering. Per language. Acquired on top of L1 during normal
  childhood exposure. Replaceable — bilinguals run multiple L2s on
  the same L1.

- **L3 — content / substrate.** The world model. What you know about
  things. Already separate from both L1 and L2 in the brain (Wernicke
  + association cortex). Already separate in the project (`brain.db`).

The Broca/Wernicke split argued for separating language production
from knowledge. The L1/L2/L3 view argues for separating *within*
language production: the structural capacity from the surface-form
overlay from the content. Each layer has a different lifecycle, a
different training source, and (critically) different
**inspectability** properties.

### Mapping into the project

Today's grammar transformer (HamRobyLLM) is an early prototype of L1:
76 structural tokens (UPOS + UD deps + slots), trained on 6 English
UD treebanks, no actual words. Its vocabulary *cannot* express
English-specific function words, which is why the template-rendered
prose feels stilted and why v027's article heuristic was even
considered.

The architectural fix is to grow the cortex into the layered shape
the brain uses:

| Layer | Vocab | Trained on | Per | Frozen at runtime |
|---|---|---|---|---|
| L1 | UPOS + UD deps + slots (no words) | UD treebanks across many languages | All users | Yes (shipped once) |
| L2 | UPOS for content + literal function words for one language | UD treebanks for that language with content words abstracted | Language | Yes (per language) |
| L3 (substrate) | Content labels | User's own teaching | User | No (always-mutable) |
| Synthesizer head (HamRobySum / A1) | L2's output token space | (edges, prose) pairs | Stitcher | Yes (per stitcher) |

The synthesizer head sits on top of the frozen L1+L2 stack and
stitches substrate content into L2's grammatical frames. The
constrained-decoding mask described below operates at the head's
output layer, ensuring content tokens are substrate-grounded.

### Why this matters: multi-language as a first-class property

Universal Dependencies was designed for cross-lingual portability.
~100+ languages have UD treebanks today, all annotated with the same
universal POS tags and the same dependency labels. The L1 layer can
be trained once on a cross-lingual mix and used by every L2.

Practical consequence: **anyone can train an L2 for their own
language** and run their own substrate through it. Spanish L2 plus a
substrate of Spanish-language facts → Spanish HamRoby. No retraining
of L1. No per-user model surgery. Just the small adapter.

This is the same principle as substrate-per-user, applied one layer
up. The architecture is symmetric: shared scaffolding (L1), pluggable
overlay (L2), owned content (L3). The HamRoby naming carries the same
commitment — universal grammar engine, per-instance everything else.

### Honest caveats

- **Word order varies (SVO / SOV / VSO).** UD encodes structure via
  the dependency tree, not the linear order, but L1 trained on a
  cross-lingual mix may benefit from a language tag as a conditioning
  input rather than learning all orders implicitly.
- **Morphology varies wildly.** UD has a `FEATS` field for
  morphological annotation but it is coarse; agglutinative languages
  (Turkish, Finnish) need extra L2 work for inflection.
- **Cross-lingual L1 is a weaker per-language signal** than a
  monolingual model would be. Trade-off accepted in exchange for
  universality.
- **UD coverage is uneven.** Major languages have rich treebanks;
  smaller ones have small or none. An L2 trained on a 5k-token
  treebank will be weaker than one trained on millions.

None of these are blockers. They are knowable trade-offs.

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
collect outputs, train HamRobySum from scratch on those pairs. The
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
end-to-end, then throw the wrapper away and train HamRobySum properly.

### How L1+L2 changes the HamRobySum job

The L1/L2/L3 view above splits "the language model" into three
concerns. HamRobySum, as originally framed, was a single 125M-300M
student doing **all three** at once: function-word grammar,
content-stitching, and (implicitly) cross-lingual portability via
whatever language the teacher happened to use.

With L1+L2 in place, HamRobySum's job shrinks substantially:

- L1 contributes the structural priors. Frozen encoder.
- L2 contributes function-word grammar (`a` vs `an`, prep choice,
  agreement, conjunction patterns). Frozen overlay.
- HamRobySum (the synthesizer head) is left with: content-ordering,
  edge-clustering decisions, lexicalization of substrate labels into
  L2's token space, and pronoun/connective stitching.

This is a smaller learning problem. Two practical consequences:

1. **Path 2 still applies, but the data target changes.** The
   teacher prompt should ask for prose that respects the substrate
   *and* uses idiomatic L2 grammar. The student no longer has to
   absorb function-word grammar from prose — it inherits that from
   L2. The training signal can focus on stitching quality.

2. **Path 1 (template-distilled) becomes more viable.** With L2
   handling function-word grammar at runtime, the v0 template-
   distilled student is no longer "no better than templates" —
   templates produce structural skeletons, L2 fills in articles
   and agreement, and the student learns to vary surface form
   within those frames. Path 1 may be enough for English-only
   prototype work.

The recommendation reorders: **build L1+L2 first (open data, no
API cost, multi-language by construction), then evaluate whether
the resulting v0 stitcher is good enough before committing to the
Path 2 distillation budget.**

## Path 2 deep dive — building HamRobySum

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

Train HamRobySum incrementally — each stage upgrades the prose, same
inputs:

**v0 — template-distilled (free).**
Use synth_data.py output as-is. Train HamRobySum on (edges, template
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

Train HamRobySum from scratch on (edges, Claude-prose). This is the
real model. Cost estimate: 50K examples × ~2K tokens × ~$0.003/1K
tokens ≈ $300 in API + ~$100 in GPU time.

**v2 — distilled with reasoning structure (later).**
Once the basic distillation works, regenerate with a teacher prompt
that emits intermediate structure ("group facts by sub-topic, then
write a paragraph per group"). This trains HamRobySum to produce
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
function words, structural markers). HamRobySum needs a real text
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

### Definition of done for HamRobySum-v1

- Trained model + tokenizer checkpoints alongside HamlinLLM checkpoints.
- Drop-in replacement for the `synthesize()` call in
  [chat.py](../src/sara_brain/cortex/transformer/chat.py).
- Constrained decoding wrapper that demonstrably blocks
  out-of-substrate content tokens (test: ask about a substrate concept,
  confirm no extraneous noun-phrases appear in output).
- Side-by-side eval against the v027 templates on 50 hand-picked
  questions: HamRobySum prose should be judged more natural by a human
  reader, with zero hallucinated facts.

## L2 implementation plan (the next concrete step)

Build the L2 layer for English first. It is the smallest meaningful
slice of the L1/L2/L3 architecture, validates the layered design end
to end, and removes the v027 article-heuristic problem permanently.
Spanish (or any second L2) becomes a copy-and-retrain of the same
recipe.

### Vocabulary changes

Today's `vocab.py` defines a 76-token structural vocabulary —
`UPOS_*`, `DEP_*`, slot markers, BOS/EOS/PAD/UNK. No actual words.

L2 needs an **augmented vocabulary**: structural tokens stay (so L1
checkpoints remain compatible) plus a curated set of English
function-word literals.

Function-word allowlist for L2-en (closed-class, finite):

- **Determiners**: `a`, `an`, `the`, `this`, `that`, `these`,
  `those`, `some`, `any`, `every`, `each`, `no`
- **Prepositions**: `of`, `in`, `on`, `at`, `by`, `for`, `with`,
  `to`, `from`, `as`, `into`, `over`, `under`, `between`, `through`,
  `against`, `about`, `before`, `after`
- **Conjunctions**: `and`, `or`, `but`, `nor`, `so`, `yet`, `because`,
  `although`, `if`, `when`, `while`, `since`, `unless`, `until`
- **Auxiliaries**: `is`, `are`, `was`, `were`, `be`, `been`, `being`,
  `has`, `have`, `had`, `do`, `does`, `did`, `can`, `could`, `will`,
  `would`, `shall`, `should`, `may`, `might`, `must`
- **Pronouns**: `it`, `they`, `them`, `their`, `its`, `which`,
  `that`, `who`, `whose`, `whom`
- **Negation / particles**: `not`, `n't`, `also`, `only`, `just`,
  `then`, `now`, `here`, `there`

Estimate ~150 function-word tokens. Total L2-en vocab ~225 (76
structural + 150 function-word + buffer for added markers).

Files to add / change:
- `vocab.py` — keep as is (L1 vocab, frozen surface).
- `vocab_en.py` (new) — re-exports L1 tokens + adds the function-word
  allowlist with stable IDs.
- A small `vocab_for_l2(lang: str)` factory so adding `vocab_es.py`
  later is a one-file change.

### UD ingestion changes

`ud.py` today produces sequences of structural tokens. For L2 it
needs to produce **mixed sequences**: function words kept literal
when they appear in the treebank, content words abstracted to UPOS.

A token in the UD treebank becomes:
- Its literal lowercased form, **if** the form is in the L2-en
  function-word allowlist
- Its UPOS tag otherwise

This preserves L1 compatibility (the structural skeleton is
identical) while teaching L2 where function words slot in.

Files to add / change:
- `ud.py` — add a `lexicalize_function_words: bool = False` flag and
  a `function_word_set: set[str] | None = None` parameter. Default
  off (existing L1 training path unchanged); on for L2 training.
- Add `ud_l2.py` (new) or just a `prepare_l2_corpus()` entry point
  that runs the existing UD ingestion with the L2 flag set.

### L2 training

Architecture: small adapter on top of frozen L1. Concretely, the
simplest first cut is a **vocabulary-projection adapter** — keep the
L1 transformer frozen, add a new embedding for the function-word
tokens, retrain only the new embedding rows + the LM head.

If that under-performs, escalate to:
- LoRA adapters on a subset of L1's attention layers (still cheap)
- Full fine-tune of L1 on the lexicalized corpus (loses some
  cross-lingual benefit; treat as fallback)

Files to add:
- `train_l2.py` — mirrors `train_router.py`'s structure. Loads frozen
  L1 checkpoint, attaches L2 adapter, trains on the lexicalized UD
  corpus, evaluates dev perplexity on a held-out portion.

Training budget estimate: ~150 new embedding rows + LM head fine-tune
on the same 6 English UD treebanks (~1.3M tokens). Probably 5-15
minutes on a 3070, similar order to the router head. Adapter
checkpoint: `l2_en.pt`, ~1-5 MB depending on what we choose to
fine-tune.

### Touch points for the existing synthesizer pipeline

Once L2-en is trained, the v027 article heuristic comes out and the
synthesizer head loads `(L1, L2-en)` instead of just L1. The
synthesizer head doesn't exist yet (path 2 / HamRobySum is next), so
this integration step is deferred — but the labeler in `synth_data.py`
should start emitting prose that uses L2-en's vocabulary so that when
HamRobySum is trained, the training labels are already L2-grammatical.

In other words: when L2-en is shipped, **update the templates in
`synthesizer.py` to use the L2-en function-word allowlist**. This is
a labeling change, not a runtime change. Templates become slightly
richer; they pick from L2-en's allowed function words rather than
hard-coding strings. Wave 2 of v027 (the article heuristic) is
replaced by `(L2-en allowlist) × (small templating logic)` instead.

### Definition of done for L2-en

- `vocab_en.py` exists, exports IDs for ~150 function-word tokens
  plus the 76 L1 structural tokens.
- `ud.py` ingests UD treebanks with `lexicalize_function_words=True`,
  producing mixed sequences.
- `train_l2.py` trains an adapter to dev perplexity at least as good
  as L1's structural perplexity (no degradation; ideally lower since
  function words are predictable).
- `l2_en.pt` checkpoint shipped alongside L1.
- A small CLI demo: load `(L1, L2-en)`, give it a structural prompt
  like `DET NOUN VERB DET NOUN`, sample, confirm it produces fluent
  English skeletons (`The cat saw the dog`) rather than pure
  structural tokens.

### Order of operations

1. Save this plan (this commit).
2. `vocab_en.py` — define the function-word allowlist with stable
   IDs. Standalone change, easy to review.
3. Extend `ud.py` to support function-word lexicalization. Run it
   once, manually inspect a sample of mixed sequences, sanity-check
   that the abstraction looks right.
4. `train_l2.py` — train the adapter. Compare dev ppl against L1.
5. CLI demo to confirm fluent-English skeleton sampling.
6. Update v027 templates to use L2-en allowlist (deferring or
   removing the Wave 2 article heuristic).
7. Then: pick up HamRobySum / path 2 with L2-en already in place.

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

- v027 Wave 1 shipped (commit `9b267bf`): decomposition filter,
  sentence combining, labeler/inference noise filters synced. Remaining
  v027 item (Wave 2 article heuristic) is **superseded by L2** — the
  function-word allowlist replaces it permanently.
- L1/L2/L3 architecture documented above.
- **L2-en operational end-to-end:**
  - `vocab_en.py` (commit `6d19e97`, plan v029) — 175-token vocab
    (76 L1 + 99 EN function words), L1 IDs preserved.
  - `ud.py` lexicalization (commit `6bc080c`, plan v030) — opt-in
    flag preserves byte-for-byte L1 ingestion.
  - `train_l2.py` (commit `52d176b`, plan v031) — 134K-param
    adapter on frozen L1; trained checkpoint
    `l2_en_003000.pt` reaches dev_ppl=4.127 from pre-train 38.462
    in ~5.5 min on a 3070.
  - `inference_l2.py` (commit `ad1abf8`) — sample + score wrapper.
    L2 produces recognisable English sentence skeletons with
    function words in plausible structural positions
    (`det the`, `case from / to`, `cop are`, `mark to / that`).
- HamRobySum (path 2) deferred until the synthesizer-pipeline
  integration. The layered architecture lets the synthesizer head be
  a smaller learning problem and gives it function-word grammar for
  free.

**Immediate next step:** synthesizer-pipeline integration. Two
sub-steps:

1. Update `synthesizer.py`'s template renderer to draw function words
   from L2-en (replaces v027 Wave 2 article heuristic) — this is the
   labeler change. Templates become richer; the labeler emits
   L2-grammatical prose.
2. HamRobySum / path 2 — train the actual synthesizer head on top of
   `(L1, L2-en)`. Now the head doesn't have to learn function-word
   grammar from prose; L2-en already owns it.
