# Sara Brain — Kiro Onboarding Document
**Last updated: 2026-08-17**

## What Is This Project

Sara Brain is a path-of-thought cognitive architecture that separates knowledge from reasoning. Knowledge lives in SQLite (0 trainable parameters). A language model (the "cortex") reads the knowledge and reasons over it. The model contributes grammar and logic ONLY — zero domain knowledge.

**Key result: 93% precision at 84% coverage on MMLU Biology using Qwen 2.5 7B as cortex + 532 hand-curated facts in a JSON file.**

## The Architecture

```
User question → Confidence check → Fact retrieval → Cortex reasoning → Answer (or "I don't know")
```

1. **Sara Brain** (SQLite): stores facts as triples (subject, relation, object) with source text
2. **Confidence check**: keyword matching determines if Sara has relevant knowledge
3. **Cortex** (Qwen 2.5 7B Instruct via Ollama): reads facts, picks answer
4. **Abstention**: if confidence low OR cortex uncertain → says "I don't know"

## Current Best Results

| Configuration | Coverage | Precision |
|---------------|----------|-----------|
| Qwen 7B + Sara (English, single ask) | 84% | 93% |
| Qwen 7B + Sara (Jibberish cipher) | 76% | 92% |
| Llama 3B + Sara (ask-twice) | 40% | 90% |
| Qwen 7B alone (no Sara, jibberish) | - | 18% (random) |

## Key Proofs

1. **Jibberish proof**: All biology nouns ciphered to nonsense. Qwen alone = 18% (random). Qwen + Sara's ciphered facts = 92%. Knowledge comes ONLY from substrate.
2. **100% precision**: With ask-twice consistency on llama3b, 12/12 correct at low coverage.
3. **Scaling curve**: 194→532 facts, precision holds at 88-93% as coverage grows from 12% to 84%.
4. **Domain agnostic**: Same code works for medical facts (demonstrated with 25 medical facts).

## Repository Structure

```
scripts/
  sara_ask.py              ← USER INTERFACE (teach/ask/cite)
  sara_pipeline.py         ← End-to-end pipeline
  train_sara_extractor_scratch.py  ← 115M model training
  generate_english_extractor_data.py
  generate_gold_extraction_data.py
  pretrain_mlm_wiki.py     ← Custom cortex pretraining
  lora_finetune_qwen.py    ← LoRA fine-tuning (WIP, OOM on 3070)

data/
  biology_hand_curated.db  ← PRODUCTION brain (532 facts, 90% precision)
  medical_demo.db          ← Medical domain demo
  biology_8b_full.db       ← Full 8B extraction (35k triples)
  jibberish_*.db           ← Cipher proof databases

training_data/
  sara_hand_curated_facts.jsonl  ← The 532 gold facts (IMPORTANT)
  extractor_kiro_gold.jsonl      ← Extractor training data
  lora_substrate_obedience.jsonl ← LoRA training data for Qwen

papers/
  sara_precision_paper_2026.md   ← Main paper draft
  compute_proposal.md            ← A100 compute request
  publication_funding_strategy_20260815.md

models/ (gitignored, local only)
  sara-extractor-v4-kiro/   ← Best extractor (multi-triple)
  sara-cortex-pretrained/   ← Wiki-pretrained 100M (36% on MMLU)
  sara-cortex-curriculum/   ← Curriculum-trained logic model
```

## How To Run

```bash
# Setup
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# Ensure Ollama is running with Qwen
ollama pull qwen2.5:7b-instruct-q4_K_M

# Ask Sara a question (uses biology_hand_curated.db by default)
.venv/bin/python scripts/sara_ask.py "what does the electron transport chain produce"

# Teach Sara a new fact
.venv/bin/python scripts/sara_ask.py --teach "Aspirin inhibits platelet aggregation" --brain data/medical_demo.db

# Run benchmark
# (see the inline scripts in git history for full benchmark code)
```

## What Was Being Worked On (WIP)

### LoRA Fine-tuning Qwen 7B
**Status**: Script written (`scripts/lora_finetune_qwen.py`), training data ready (`lora_substrate_obedience.jsonl`), but OOM crashes on RTX 3070 (8GB VRAM). Needs RTX 3080 (10GB) or reduce `max_len` to 256 and LoRA `r` to 8.

**Goal**: Make Qwen even MORE substrate-obedient. Train on jibberish facts so it ALWAYS reads from Sara, never from training weights.

**Training data**: 498 examples (199 English, 199 jibberish, 100 abstention).

### Custom 100M Cortex
**Status**: Wiki-pretrained (101M words, 200k steps), gets 36% on MMLU. Can't bridge paraphrase gap without more pretraining. Needs A100 compute (proposal written).

### Scaling Coverage
**Status**: 532 facts = 40% coverage at 90% precision on full 310q. Each ~50 facts adds ~10% coverage. Could push to 60%+ with more facts.

## Key Technical Decisions

1. **Qwen 2.5 7B is the cortex** — vastly outperforms llama3.2:3b (93% vs 76% precision)
2. **Hand-curated facts beat automated extraction** — 532 curated facts at 90% precision vs 35k extracted at 51%
3. **Ask-twice consistency** — eliminates wrong answers but reduces coverage (use with llama, unnecessary with qwen)
4. **Smart confidence check** — requires 2+ question words to match facts (not just choice words)
5. **Sara's `_build_chain` stores triples inverted** — property→relation→concept (object→subject). Forward semantic edges were added to fix this.
6. **`part_of` decomposition creates hub noise** — function words filtered in `_link_sub_concepts`

## The Thesis (What We're Proving)

1. Knowledge and reasoning can be completely separated
2. A system that knows what it doesn't know is more trustworthy than one that guesses
3. Every error is deterministically diagnosable and fixable
4. Current LLMs cannot handle genuinely novel information (jibberish proof)
5. The architecture scales linearly (more facts = more coverage, precision stable)

## Contact

Jennifer Pearl — github.com/LunarFawn/SaraBrain — Path of Thought Research
