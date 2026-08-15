# Proposal: Compute Resources for Sara Brain Custom Cortex Training

## Principal Investigator
Jennifer Pearl — Path of Thought Research

## Summary

We request GPU compute time to pretrain a custom 100M–500M parameter reasoning model that serves as the "cortex" for Sara Brain, a path-of-thought cognitive architecture. The system has demonstrated **89% precision** on MMLU High School Biology when an abstention mechanism is employed, using only 103 facts stored in a non-neural SQLite knowledge graph. We seek to replace the current borrowed 3B commercial model with a fully independent, phone-deployable custom model.

## Background

Sara Brain is a 0-parameter knowledge store that separates domain knowledge from the reasoning model entirely. Knowledge lives as directed neuron-segment chains in SQLite — inspectable, correctable, and teachable at runtime. A separate "cortex" model reads Sara's output and performs language reasoning.

### Proven Results

| Configuration | Precision | Coverage | Notes |
|--------------|-----------|----------|-------|
| Sara + 3B cortex (with abstention) | **89%** | 28% | Only answers when confident |
| Sara + 3B cortex (threshold 0.30) | **100%** | 5% | Perfect when highly confident |
| Sara + 3B cortex (no abstention) | 54% | 100% | +18% over 3B alone |
| 3B model alone (no Sara) | 33% | 100% | Training weights only |
| Random chance | 25% | 100% | Baseline |

### Preliminary Custom Cortex Results

| Model | Pretraining | MMLU Score | Notes |
|-------|-------------|------------|-------|
| 5M (no pretrain) | None | 27% | Cannot learn reasoning from scratch |
| 100M (wiki-pretrained) | 101M words, 11.6h | **36%** | Outperforms 3B alone (33%) |
| Target | 1B+ words, multi-day | >50% | Close gap to 3B cortex |

The 100M wiki-pretrained model already outperforms the 3B commercial model when both operate without Sara (36% vs 33%). With Sara's facts, the custom model is expected to approach the 89% precision the borrowed 3B achieves.

## What We Need

### Compute Requirements

- **GPU:** 1-4x A100 (80GB) or equivalent
- **Duration:** 5-7 days continuous training
- **Task:** Masked Language Model pretraining on 1-10B words of English text

### Training Plan

**Phase 1: Language Pretraining (5-7 days)**
- Model: 200M–500M parameters (768-1024d, 12-16 layers, 12-16 heads)
- Data: English Wikipedia + BookCorpus + OpenWebText (~5B words)
- Objective: Masked Language Modeling (BERT-style)
- Target: 60%+ MLM accuracy (vs 52% achieved with 101M words on RTX 3070)

**Phase 2: Reasoning Fine-tuning (1 day)**
- Data: 50k+ MCQ examples (curriculum: simple logic → complex reasoning)
- Objective: 4-way classification from structured facts + question
- Target: >50% on MMLU biology with Sara's facts

**Phase 3: Deployment Validation**
- Test on full 310-question MMLU benchmark
- Verify abstention mechanism maintains 89%+ precision
- Benchmark inference speed for phone/Raspberry Pi deployment

## Why This Matters

1. **Separation of knowledge from intelligence:** Sara Brain proves that domain knowledge does not need to be embedded in model weights. All knowledge lives in an inspectable, correctable graph. The model contributes ONLY grammar and logical reasoning.

2. **Trustworthiness through abstention:** Unlike standard LLMs that hallucinate when uncertain, Sara + cortex achieves 89% precision by honestly admitting ignorance on topics not in its knowledge base.

3. **Deployability:** A 200-500M parameter cortex runs on mobile devices and edge hardware. Combined with Sara's SQLite brain, the entire system is self-contained with no cloud dependency.

4. **Independence from commercial APIs:** The custom cortex eliminates reliance on OpenAI, Anthropic, or Meta models for inference, while maintaining the demonstrated 89% precision capability.

5. **Scientific contribution:** This work demonstrates that a 0-parameter knowledge graph with well-curated facts outperforms models with billions of parameters, validating the path-of-thought architecture as an alternative to scaling model weights.

## Existing Infrastructure

- Sara Brain codebase: production-ready (290+ tests passing)
- Training pipeline: proven (115M extractor, curriculum learning, MLM pretraining)
- Benchmark framework: automated MMLU testing with abstention metrics
- C++ wavefront engine: 35-40x speedup for graph propagation
- RTX 3070 (local): sufficient for fine-tuning, insufficient for large pretraining

## Publication Plan

Results will be published as an extension to the existing Sara Brain paper (DOI: 10.5281/zenodo.19436522) demonstrating:
- 89% precision with knowledge-reasoning separation
- Custom cortex achieving parity with commercial 3B model
- Full system deployable on consumer hardware

## Contact

Jennifer Pearl  
GitHub: github.com/LunarFawn/SaraBrain  
Substack: Path of Thought
