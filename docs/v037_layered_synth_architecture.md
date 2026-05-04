# v037 — Layered HamRobySum: structural core + per-language verb overlay

**Date:** 2026-05-03
**Branch:** `feature/grammar-cortex`
**Builds on:** [v035_generic_slot_hamrobysum.md](v035_generic_slot_hamrobysum.md),
[v036_synthetic_corpus_training.md](v036_synthetic_corpus_training.md)
**Aspires to:** an AI that thinks for itself but **honestly** — not
what frontier models are. Knowledge structurally separated from
language production; honesty enforced by architecture, not by
behavioral training.

## The framing

Frontier LLMs collapse two functions into one set of weights:

1. Language production (syntax, sequencing, paraphrase)
2. World knowledge (facts about the world)

When the model is asked something it doesn't know, the language
circuit fills the gap with plausibilities pattern-matched from
training. That's structural hallucination, not a behavior bug. RLHF
and safety classifiers paper over it with a second model on top —
"the lizard brain is gagged, not removed." Two black boxes in series
(opaque generator + opaque inhibitor). No way to inspect what the
base model would have said.

HamRoby's whole point is to refuse that arrangement:

- **Substrate (L3)** holds the world. SQLite file you can read.
  Per-user, fully editable.
- **Grammar L1 + L2** (already shipped) hold structural and
  function-word competence. UD-trained, no real-world content in
  weights, ~270 tokens total.
- **Synthesizer (this doc)** stitches them together. The v037
  architecture splits the synthesizer into two layers so the
  separation extends all the way down.

## v3 told us the principle

v3 (commit `c015247`-ish) trained on synthetic nonsense substrates
with real English relation names (`is_a`, `produces`, `requires`...)
and proved the slot architecture works on never-seen brains:

> `Multicellular organism produces cell division, is a organism, is a individual, and is a sea urchin.`

But v3 conflated two layers. It learned **slot composition** AND
**which real English verbs go where** in one training pass. That
worked for the 12 verbs in `_RELATIONS_POOL` but failed on
real-brain predicates outside that set (`forms`, `role_in`,
`applies_to`, `has_formed`, etc. all UNK'd).

v037's insight: split the two concerns into two checkpoints.

## The architecture

| Layer | Trained on | What it knows | Per-user / per-language? |
|---|---|---|---|
| **HamRoby-Sum-Core** | nonsense concepts + **nonsense relations** | pure slot composition: which slot goes where, sentence frames, function words | universal (ships once) |
| **HamRoby-Sum-EN** | nonsense concepts + **real English relations** | which English verbs slot where | per language (Spanish gets its own) |
| **Substrate (L3)** | user's own teaching | actual world content | per user |

Each layer is the same `GrammarModel` architecture; what differs is
the training corpus and which checkpoint a third party loads.

### What this enables

- **Anyone can ship a Core checkpoint as universal infrastructure.**
  Trained once on nonsense-only synthetic substrates. No real English
  appears in its training data. By construction it cannot have
  memorized any fact, English idiom, brand name, or anything else
  about the real world.
- **Anyone can build a per-language overlay** by:
  1. Translating the `_REAL_RELATIONS` pool to their language
  2. Generating synthetic substrates with that pool (still nonsense
     content, real-language verbs)
  3. Training an overlay adapter resuming from Core
  4. Shipping their `hamroby_sum_<lang>.pt`
- **No layer requires retraining the layer below it.** Adding
  Spanish is a Spanish-only training run; Core stays untouched.
  Adding new English domain verbs (medical, legal) is a re-run of
  EN with an extended pool, Core stays untouched.

### What this forecloses

- **Memorization-based hallucination at the synth layer**: Core has
  zero exposure to real content; it can't pattern-match because
  there's no pattern to match.
- **Safety-classifier-on-top architecture**: there's nothing for a
  classifier to suppress — the model only emits substrate slots
  plus closed-class English glue. If a sentence comes out the model
  it traces to a substrate edge.
- **Vendor lock-in at any layer**: Core trains from open data
  (`generate_synthetic_substrate.py`), no API. EN overlay trains
  from same. Substrate is yours. End to end reproducible by anyone
  with a GPU.

## Implementation

Three changes to the existing v036 pipeline.

### 1. `generate_synthetic_substrate.py` — `--nonsense-relations` flag

Today: `_RELATIONS_POOL` is hardcoded to 12 real English verbs.

Change: add a `--nonsense-relations` flag (and `nonsense_relations`
function arg). When set, the generator produces relation names from
the same pronounceable-nonsense alphabet as the concepts. So instead
of `(zilkrap, produces, bortle)` you get `(zilkrap, vlinkop, bortle)`
— pure nonsense triples.

Existing v036 behavior (real English verbs) stays the default for
backward compat.

### 2. `scripts/build_layered_corpus.sh` — new script

Generates two cumulative training corpora:

- **`/tmp/synth_pairs_core.jsonl`** — fully synthetic (nonsense
  concepts + nonsense relations). Used to train Core.
- **`/tmp/synth_pairs_en.jsonl`** — synthetic concepts + REAL
  relations. Used to train EN, resuming from Core.

Each corpus uses the same size mixing as v036
(small/medium/large) for variety.

### 3. Two-phase training (matches v036's chain pattern)

```
# Phase Core: cold-start from L2-en, train on nonsense-everything
PAIRS=/tmp/synth_pairs_core.jsonl \
CKPT_NAME=hamroby_sum_core \
STEPS=2500 \
SESSION=sara-synth-core \
./scripts/train_hamrobysum.sh

# Phase EN: resume from Core, train on real-English-verbs
PAIRS=/tmp/synth_pairs_en.jsonl \
CKPT_NAME=hamroby_sum_en \
STEPS=2500 \
SESSION=sara-synth-en \
RESUME_FROM=src/sara_brain/cortex/checkpoints/hamroby_sum_core_002500.pt \
./scripts/train_hamrobysum.sh
```

No code changes to `train_synth.py` or `train_hamrobysum.sh` — v036
already supports `--resume-from` and the `RESUME_FROM` env var.

## Files

**Modified:**
- [papers/instrument_validation/generate_synthetic_substrate.py](../papers/instrument_validation/generate_synthetic_substrate.py)
  — add `--nonsense-relations` flag and `nonsense_relations` arg.

**New:**
- [scripts/build_layered_corpus.sh](../scripts/build_layered_corpus.sh)
  — generates the two cumulative corpora.

**Reused unchanged:**
- `vocab_synth.py` — the substrate verb list is already there;
  Core training just won't use most of those positions, EN training
  fills them in.
- `train_hamrobysum.sh` — `RESUME_FROM` already supported (v036).
- `train_synth.py` — `--resume-from` already supported (v036).
- `inference_synth.py` — slot expansion already works with any
  generic-flavored synth ckpt.

## Order of operations

1. Save plan + script + generator change → commit (this slice).
2. **You run** `./scripts/build_layered_corpus.sh` (CPU).
3. **You run Core training** in tmux (~10 min on 3070).
4. **You run EN training** resuming from Core (~10 min on 3070).
5. When done, ping me; I run inference comparison:
   - Core alone (won't emit real English — confirms zero memorization)
   - EN on demo brain (compare to v3, expecting better predicate coverage)
   - EN on `brain.db.drifted_s1` (cross-brain — should match or exceed v3)

## Verification

### 1. Core does NOT emit real English

After Core training, run inference on the demo brain. Output should
be GIBBERISH (nonsense verbs, since that's all Core saw). This is the
proof Core has zero real-content memorization. If Core somehow emits
real English, something leaked.

### 2. EN does emit real English (and more verbs than v3)

After EN training, run inference on the demo brain. Output should
include real verbs from the extended `_REAL_RELATIONS` pool, including
the ones v3 missed (`forms`, `role_in`, etc.).

### 3. EN on never-seen brain produces clean prose

Same `brain.db.drifted_s1` cross-brain test as v3. Should produce
prose at least as clean — and ideally cleaner because EN has more
verbs to choose from.

### 4. Core ships standalone (the architectural milestone)

`hamroby_sum_core_*.pt` is by construction free of real-world
content in weights. It's the universal infrastructure layer. v037 is
done when this checkpoint exists, loads cleanly, and produces
gibberish-but-grammatical output (proving the structural-composition
half of the architecture works without any real-language exposure).

## Future overlays (out of scope for v037 itself)

Anyone wanting to add a language overlay follows the same recipe:

```
# Spanish — example
1. Add _RELATIONS_POOL_ES = ["es_un", "produce", "requiere", ...] to the generator
2. Build a Spanish synth corpus: nonsense concepts + Spanish verbs
3. Resume from hamroby_sum_core_*.pt, train on the Spanish corpus
4. Save as hamroby_sum_es_*.pt
5. (also need vocab_es.py with Spanish function words — see v028)
```

The architecture supports this as a clean copy-and-modify of v037's
EN path. If you generated v037's EN with `nonsense_relations=True`
already in the generator, every L2 follows the exact same recipe
with only the relations pool changed.

This is what "anyone can overlay" looks like in practice. The
project ships Core; the world ships overlays. The substrate is
yours.

## Out of scope

- Actual non-English overlays. v037 ships Core + EN; everything else
  is an exercise for downstream users following the recipe above.
- Domain-specific overlays (medical-EN, legal-EN). Same recipe; just
  a different `_RELATIONS_POOL`.
- Chat REPL `--use-hamrobysum` integration. Same gating: needs to
  beat v032 templates on real prose. EN should clear that bar with
  the extended verb coverage.
- Inspectability tooling for the substrate (already a separate
  thread; v037 just preserves the property).
