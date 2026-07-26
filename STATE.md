# Sara Brain — Project State

**Last updated:** 2026-07-25
**Branch:** feature/sara-cortex-development

---

## Goal

Build a multi-billion parameter LLM that uses Sara Brain as its weights.
The model knows only grammar and logic — all factual knowledge comes from
the Sara Brain substrate (SQLite graph of neuron-segment chains). No
pre-trained factual weights. Knowledge is inspectable, correctable, and
teachable at runtime.

## Architecture

```
Teaching:   Text → [115M Extractor] → triples → Sara Brain (SQLite)
Retrieval:  Question → Wavefront Propagation → Rendered Facts → [Cortex LLM] → Answer
```

The cortex (currently 115M, target multi-B) receives ONLY wavefront output.
It never sees original source text. It should understand grammar and logical
structure enough to select/synthesize answers from structured facts.

## What Works

### Core Sara Brain (src/sara_brain/)
- Path-of-thought recognition via parallel wavefront propagation
- Bidirectional propagation, true backwave, echo modes
- IS-A inheritance (separate from propagation)
- Hub discrimination in scoring
- Negation-aware question handling
- 290+ tests passing

### C++ Engine (src/sara_brain/core/engine.cpp)
- 35-40x faster than Python BFS (2.3s vs 83s per question on 952k edges)
- Compiled as sara_engine.so, loaded via ctypes
- Supports forward, bidirectional, and backwave propagation modes
- Parity-tested against Python implementation

### 115M Extractor (scripts/train_sara_extractor_scratch.py)
- Copy-pointer encoder-decoder (768d, 8 enc layers, 6 dec layers)
- Cannot hallucinate — only copies tokens from input
- Best model: models/sara-extractor-v2-clean/best.pt
- Trained on extractor_v2_500k.jsonl (synthetic nonsense substrates)

### 115M Synthesizer (scripts/train_sara_extractor_scratch.py, same arch)
- Reads wavefront-rendered facts + question → produces answer
- Best model: models/sara-synthesizer-115m/best.pt
- Trained on synthesizer_500k_v3.jsonl (jibberish concepts, grammar-only)

### Jibberish Cipher (tools/translate_mmlu_to_jibberish.py)
- 7098 noun→nonsense mappings (data/biology_short_cipher_nouns.json)
- All 47 biology chapters translated (data/biology_short_jibberish/)
- MMLU questions translated (data/mmlu_biology_short_jibberish.json)
- Purpose: proves wavefront works because no model can use memorized weights

### Wavefront Scorer (src/sara_brain/core/wavefront_scorer.py)
- Pure graph-intersection scoring for MCQ (no LLM needed)
- score_choices() + pick_choice() — honest, no answer leakage
- This is the ground-truth measurement of Sara's knowledge quality

### Full Pipeline (scripts/sara_pipeline.py)
- End-to-end: document → extractor → teach Sara → wavefront → synthesizer → answer
- Clean architecture, no knowledge leakage

## What's Broken / Needs Work

### Synthesizer Training Data Quality
The MCQ training data (complex_jibberish_mcq_500k.jsonl) trains the model
to output flourishing English prose like "Based on the substrate, X relation
Y. Therefore, the correct choice is A." A 115M model should output minimal
tokens — just the answer or a single supporting fact + letter. Needs retraining
with tighter output format.

### Hub Penalty Accuracy Issue
The hub discrimination penalty (weight / (connectivity + 1)) actively hurts
pure wavefront accuracy (drops from 28% to 24%). TODOs exist in the code.
The penalty may need to be removed or replaced with a smarter approach.

### Scale
Current honest benchmark scores:
- Pure wavefront (no LLM): ~24-28% on full 310Q MMLU biology
- Sara + 3B Ollama model: ~52% (lower than 3B alone at 63%)
- The wavefront provides too much noise for the 3B model currently

The signal-to-noise ratio in the full 952k-edge brain is the bottleneck.
The extractor puts too many low-quality triples into the graph.

## Key Databases

| Database | Neurons | Segments | Description |
|----------|---------|----------|-------------|
| data/biology_full_v2_clean.db | 102,319 | 952,883 | Full 47-chapter biology (English, v2-clean extractor) |
| data/jibberish_biology_v2_stable.db | ~80k | ~700k | Jibberish biology (31/47 chapters, lemma-consistent) |
| data/jibberish_biology_full.db | ~100k | ~950k | Full jibberish biology (all 47 chapters) |

## Model Checkpoint Map

| Model | Path | Size | Purpose | Status |
|-------|------|------|---------|--------|
| Extractor v2-clean | models/sara-extractor-v2-clean/best.pt | 462MB | Text → triples | ✓ Production |
| Extractor 115m-v2 | models/sara-extractor-115m-v2/best.pt | 462MB | Previous version | Archive |
| Synthesizer 115m | models/sara-synthesizer-115m/best.pt | 461MB | Facts → prose | ✓ Production |
| Synthesizer jibberish | models/sara-synthesizer-115m-jibberish/best.pt | 464MB | Jibberish-trained | Testing |
| Synthesizer jibberish-qa | models/sara-synthesizer-115m-jibberish-qa/best.pt | 465MB | MCQ-focused | Testing |

## Running Benchmarks

```bash
# Pure wavefront (honest, no LLM, no cheating)
.venv/bin/python benchmarks/run_mmlu_biology.py --db data/biology_full_v2_clean.db --model wavefront_only

# Sara + external 3B model (Ollama)
.venv/bin/python benchmarks/run_mmlu_biology.py --db data/biology_full_v2_clean.db --model llama3.2:3b

# Jibberish version (proves no weight cheating)
.venv/bin/python benchmarks/run_mmlu_biology.py --db data/jibberish_biology_full.db --model wavefront_only --json data/mmlu_biology_short_jibberish.json

# Baseline (no Sara, just LLM)
.venv/bin/python benchmarks/run_mmlu_biology.py --baseline --model llama3.2:3b
```

## Next Steps (toward multi-B cortex)

1. **Fix signal-to-noise**: Improve extractor quality so the graph contains
   fewer garbage triples and more precise factual chains.
2. **Retrain synthesizer**: Minimal output format (fact + letter, no prose).
3. **Scale cortex**: Once the 115M proves the pipeline works honestly on
   jibberish, train a larger (1B→3B→multi-B) cortex that ONLY knows grammar
   and logic, with all knowledge from Sara's wavefront output.
4. **Inference target**: Raspberry Pi / phone deployment. The cortex weights
   are small (grammar only), Sara Brain is SQLite (runs anywhere).

## Repo Structure (key paths)

```
src/sara_brain/core/         — Brain, Recognizer, FastRecognizer, wavefront_scorer
src/sara_brain/core/engine.cpp — C++ propagation engine
scripts/sara_pipeline.py     — Full end-to-end pipeline
scripts/train_sara_extractor_scratch.py — Training script for both models
benchmarks/run_mmlu_biology.py — Benchmark runner (honest, no answer leakage)
benchmarks/run_mmlu_pipeline_115m.py — 115M-only benchmark (no external LLM)
data/                        — Databases, cipher files, benchmark logs
models/                      — Checkpoints (gitignored, ~92GB total)
tools/                       — Cipher translation utilities
```
