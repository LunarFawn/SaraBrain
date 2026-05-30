# Sara-Cortex Experiments: Findings Summary

**Date:** 2026-05-30
**Author:** Jennifer Pearl

---

## The Thesis

A model trained only on language competence + Sara Brain as its
knowledge substrate = equivalent or better performance than a model
with knowledge baked into weights. No datacenter. No $100M training
run. Knowledge belongs in the hippocampus (Sara), not compressed
into the cortex (LLM weights).

## Key Demonstrations

### 1. Novel Concept Retrieval (Aptamer Substrate)

Tested on paper-coined terms that cannot exist in any training data:

**"What is the molecular snare?"**
- 1B alone: "a 3D ring structure of flavonoid moieties" (confabulation)
- 1B + Sara: "a mechanism involved in detecting and binding" (substrate-grounded)
- 3B + Sara: "a mechanism or hypothesis involving detecting and binding
  a target molecule, related to RNA aptamers" (correct)

**"What is marker theory?"**
- 1B alone: "proposes that spread of ideas is influenced by markers" (invented)
- 1B + Sara: "a concept introduced in her study using crowdsourcing" (correct)

The models have ZERO training on these concepts. All correct answers
come from Sara's substrate.

### 2. Wavefront Propagation Works on Large Brains

The substrate-aware seed extraction fix (implemented 2026-05-26)
resolved the BFS explosion that was stalling wavefront on the 132k
neuron bio brain. Wavefront now completes in 1.5-6 seconds on large
brains.

### 3. The Wavefront Renderer

Raw wavefront output (convergence maps, strength scores) is not
LLM-readable. The breakthrough: pull `source_text` from paths of
converged neurons. This gives the LLM actual readable sentences:

```
Wavefront from seeds ['molecular snare']: 4 neurons reached.
Facts (9):
  - the paper describes molecular snare mechanism
  - molecular snare mechanism involves molecular snare
  - molecular snare mechanics hypothesis emphasizes tension
  - molecular snare mechanics hypothesis emphasizes axial forces
  - molecular snare mechanics hypothesis explains structural changes
```

The wavefront does the thinking. The renderer makes it readable.
The LLM renders it as prose.

### 4. From-Scratch Model Experiments

| Experiment | Result | Learning |
|-----------|--------|----------|
| TinyLlama 1.1B fine-tune (MCQ) | 50% held-out (vs 25% random) | Substrate reasoning IS learnable |
| Phase 1 substrate LM | Perplexity 1.2 | From-scratch model learns substrate format perfectly |
| Phase 2 MCQ head | 27% (random) | MCQ is wrong task — wavefront already does matching |
| Text-gen cortex | Loss 0.01, generates concept labels | Model learns to produce substrate-grounded text |
| From-scratch extractive QA | Overfits, can't generalize | Needs copy mechanism for rare tokens |
| **Sara-Cortex-Copy (pointer network)** | **95% exact match on held-out** | **THE RESULT** |

### 4a. The Breakthrough: Sara-Cortex-Copy

**5.6M parameters. Trained in 6 minutes. 95% exact match on held-out data.**

Architecture: encoder-decoder with copy mechanism (pointer network).
- Encoder (4 layers, bidirectional): reads rendered wavefront facts + question
- Decoder (2 layers, causal): generates answer by COPYING tokens from input
- Copy gate: learns when to copy from input vs generate from vocab
- Base vocab: 81 tokens (relations + punctuation only)
- All concept labels are copied from the input, never generated

What it does:
1. Wavefront renderer produces readable facts (source_text from paths)
2. The answer IS one of those facts
3. The model identifies which fact answers the question
4. It copies that fact token-by-token to the output

Result on 20 held-out examples (different random seed, never seen):
- 19/20 exact match (95%)
- 1/20 wrong fact selected (still substrate-grounded, just wrong triple)
- Subject correct on 20/20 (100%)

Why this works when everything else failed:
- MCQ classification failed because it's a matching task (wavefront does that)
- Standard text generation failed because rare tokens get <unk>'d
- The copy mechanism solves both: it's extractive (find the right fact)
  AND handles rare tokens (points to them in the input)

**The model has zero domain knowledge. All correct answers come from Sara.**

### 5. Architecture Insight

The correct architecture has three layers:

```
Question → Wavefront propagation (Sara's native reasoning)
    → Convergence map (which neurons are relevant)
    → Renderer (pull source_text from paths of converged neurons)
    → Readable facts (actual sentences from the substrate)
    → Cortex LLM (renders facts as prose answer)
```

The wavefront IS the thought. The cortex just reads it.

### 6. MCQ Benchmarks

The wavefront-direct scorer (no LLM, just label matching) is the
correct approach for MCQ. The April 80% result used hand-curated
bridge facts. Auto-extracted brains score ~30-37% because the
extractor doesn't produce bridge facts.

### 7. What "Thought" Is

A thought is a wavefront converging on paths through recorded
knowledge. You start with observations (seeds). You propagate
through what you know (the graph). Where paths converge — that's
the conclusion. Every step is visible, traceable, explainable.

Transformers do the same computation internally (attention heads
propagating activation through weight matrices) but you can't see
it. Sara makes thought inspectable.

## The Economic Argument

| | Sara Brain | Training Weights |
|---|---|---|
| Teach one fact | microseconds, $0 | retrain, $10k-$100M |
| Correct a fact | one refutation, immediate | impossible without retraining |
| Add a domain | minutes of teaching | weeks of GPU, dataset curation |
| Inspect knowledge | SQL query | mechanistic interpretability (research-grade) |
| Biological analog | hippocampal memory formation | nothing — no biological system works this way |

## What's Needed for the Paper

1. ✅ Novel concept demo (aptamer substrate, 1B/3B models)
2. ✅ A/B comparison (with Sara vs without Sara)
3. ✅ Wavefront renderer producing readable facts
4. ⬜ Clean benchmark numbers (wavefront scorer on curated brain)
5. ⬜ Custom from-scratch cortex model (the funding ask)

## What's Needed for Funding

A working custom model (even at 40% accuracy) that:
- Has zero domain knowledge in its weights
- Reads Sara's wavefront output
- Produces correct domain answers
- Demonstrates: all knowledge comes from the substrate

This proves the concept is buildable and worth investing in.
The model doesn't need to be perfect — it needs to show the signal.

## Files

| Path | What |
|------|------|
| `src/sara_brain/core/wavefront_renderer.py` | Renders wavefront → readable facts |
| `src/sara_reader/stateless_reader.py` | Substrate-aware seed extraction |
| `scripts/generate_synthetic_finetune.py` | Synthetic MCQ training data |
| `scripts/train_substrate_lm.py` | Phase 1: substrate format LM |
| `scripts/train_cortex_textgen.py` | Text generation from substrate |
| `scripts/finetune_sara_cortex.py` | TinyLlama LoRA fine-tune |
| `training_data/sara_cortex_synthetic_10k.jsonl` | 10k MCQ examples |
| `training_data/substrate_lm_100k.txt` | 100k substrate text chunks |
| `models/sara-cortex-lm-v1/` | Phase 1 LM checkpoint |
| `models/sara-cortex-1b-v2/` | TinyLlama fine-tune (50% held-out) |
| `docs/architecture_sara_as_weights.md` | Architecture direction |
| `docs/plans/hamroby_wavefront_first_rewrite.md` | Wavefront-first plan |
| `docs/plans/two_phase_sara_cortex.md` | Two-phase training plan |
