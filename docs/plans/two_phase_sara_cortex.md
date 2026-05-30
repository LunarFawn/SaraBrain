# Plan: Two-Phase From-Scratch Sara Cortex

**Date:** 2026-05-28
**Goal:** A pure from-scratch model that reads Sara Brain wavefront
output. No borrowed weights. No inherited knowledge.

---

## Why Single-Phase Failed

Training a model to do MCQ substrate reasoning from scratch with 10k
examples failed (26.5% — random) because the model had to learn TWO
things simultaneously:

1. How to read substrate format (what tokens mean, what structure means)
2. How to reason about it (which facts answer which questions)

10k labeled examples isn't enough for both. TinyLlama succeeded (50%)
because it already knew #1 (English comprehension) and only had to
learn #2.

## The Fix: Two Phases (Same Pattern as HamRobyLLM)

The grammar backbone worked because it was trained in phases:
- Phase 1: Learn UD tag sequences (unsupervised, next-token prediction)
- Phase 2: Add task heads on top (router, synthesizer)

We do the same for substrate format:

### Phase 1 — Substrate Language Model (unsupervised)

**Task:** Predict the next token in substrate text.
**Data:** Unlimited — generate wavefront outputs from synthetic brains.
No labels needed. Just raw substrate text.
**What it learns:** The structure of wavefront output — what follows
what, how labels relate to strengths, how intersections are formatted,
what neuron names look like.

```
Input:  "wavefront from 2 seed(s) ['zelpak', 'moridu']: 3 intersection"
Target: "(s), 8 neuron(s) reached.\n\nintersections"
```

**Config:**
- Model: ~30M params (d_model=512, 6 layers, 8 heads)
- Vocab: 4096 tokens (built from substrate text)
- Sequence length: 512
- Training data: 100k+ substrate text chunks (generated in minutes)
- Training time: ~1 hour on 3070
- Objective: next-token prediction (causal LM)

### Phase 2 — Substrate Reasoning Head (supervised)

**Task:** Given substrate + question, pick the right answer.
**Data:** The 10k synthetic MCQ examples we already have.
**What it learns:** Which substrate facts are relevant to which
questions. The reasoning skill.

**Method:** Freeze most of the Phase 1 backbone. Train a classification
head + maybe unfreeze the last 2 layers. Same pattern as the router
head on the grammar backbone, but the right task this time.

**Config:**
- Base: Phase 1 checkpoint (frozen or partially frozen)
- Head: Linear(d_model, 4) for MCQ
- Training data: 10k synthetic examples
- Training time: ~15-20 min on 3070

## Data Generation

### Phase 1 data (substrate language)

Generate raw wavefront outputs — no questions, no answers needed.
Just run wavefronts on random synthetic brains and save the text.

```bash
python scripts/generate_substrate_lm_data.py \
    --num-substrates 5000 \
    --queries-per-substrate 20 \
    --out training_data/substrate_lm_100k.txt
```

Each substrate: create a random brain, run 20 random wavefront queries,
save the output text. 5000 × 20 = 100k text chunks. Generation time:
~30 min (no GPU needed).

### Phase 2 data (reasoning)

Already have it: `training_data/sara_cortex_synthetic_10k.jsonl`

## Timeline

| Step | Time | GPU? |
|------|------|------|
| Generate Phase 1 data (100k chunks) | ~30 min | No |
| Train Phase 1 (substrate LM) | ~1 hour | Yes |
| Train Phase 2 (reasoning head) | ~20 min | Yes |
| Test on held-out synthetic | 2 min | Yes |
| Test on real bio brain | 5 min | Yes |
| **Total** | **~2 hours** | |

## Expected Outcome

Phase 1 gives the model substrate "fluency" — it understands the
format the way the grammar backbone understands UD tags. Phase 2
teaches it to reason. With proper language understanding as a
foundation, the 10k reasoning examples should be enough.

Target: **>40% held-out accuracy** (matching or beating TinyLlama's
50% would prove a pure model can match a borrowed one).

## What Makes This Pure

- No pretrained weights from anyone else's model
- No English training data (only substrate format)
- No real-world knowledge in any training phase
- Every weight learned from Sara's own output format
- Vocabulary built from substrate text, not English
- The model literally cannot know anything except how Sara talks

## Files to Create

1. `scripts/generate_substrate_lm_data.py` — Phase 1 data generator
2. `scripts/train_substrate_lm.py` — Phase 1 training (next-token)
3. Update `scripts/train_sara_cortex_scratch.py` — Phase 2 (load
   Phase 1 checkpoint, add head, train reasoning)
