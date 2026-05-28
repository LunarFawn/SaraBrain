# Plan: Sara-Native Cortex Model (1B Fine-Tune)

**Goal:** Fine-tune llama3.2:1b to reason over Sara Brain wavefront output.
The model learns to READ substrates, not to KNOW facts.

**Hardware:** RTX 3070 (8GB VRAM) — sufficient for 1B LoRA.

---

## Phase 1: Training Data Generation (~90 min)

Generate (substrate, question, answer) examples from the bio brain.

```bash
# Full MMLU biology set (310 questions)
python scripts/generate_finetune_data.py \
    --brain /home/grizzlyengineer/repo/debug_sara/sara_bio.db \
    --questions benchmarks/ch10_test_questions.json \
    --out training_data/sara_cortex_bio_ch10.jsonl

# TODO: need to create a full 310-question JSON from the MMLU set
# in the same format as ch10_test_questions.json
```

**Target:** 300+ examples minimum. Can augment with:
- Rephrase questions (same substrate, different wording)
- Wrong-answer examples (teach the model what "not in substrate" looks like)
- Multi-brain examples (different substrates, same questions)

## Phase 2: Data Formatting for LoRA

Convert to chat-completion format (what fine-tuning frameworks expect):

```json
{
  "messages": [
    {"role": "system", "content": "<SYSTEM_INSTRUCTION>"},
    {"role": "user", "content": "SUBSTRATE:\n<wavefront output>\n\nQUESTION:\n<question with choices>"},
    {"role": "assistant", "content": "C"}
  ]
}
```

Short answers (just the letter) for MCQ. For open-ended questions,
the answer would be a substrate-grounded paragraph.

## Phase 3: LoRA Fine-Tune (~1-2 hours on 3070)

**Framework:** unsloth (fastest for consumer GPUs) or huggingface PEFT

```bash
pip install unsloth peft trl bitsandbytes
```

**Config:**
- Base model: `unsloth/Llama-3.2-1B-Instruct` (4-bit quantized for training)
- LoRA rank: 16-32 (small model, don't need high rank)
- Learning rate: 2e-4
- Epochs: 3-5 (small dataset, watch for overfit)
- Max sequence length: 2048 (substrate + question fits)
- Batch size: 2-4 (8GB VRAM constraint)
- Gradient accumulation: 4 (effective batch 8-16)

**What the model learns:**
- Parse structured wavefront output (intersections, convergence maps)
- Identify which substrate facts are relevant to the question
- Select the answer supported by substrate evidence
- Say "insufficient substrate" when the data isn't there

## Phase 4: Export to Ollama

```bash
# Export to GGUF for Ollama serving
python -c "from unsloth import FastLanguageModel; ..."  # export script
ollama create sara-cortex-1b -f Modelfile
```

Then use it as the synthesis model:
```bash
sara-ask-stateless "What is mitosis?" \
    --brain sara_bio.db \
    --synthesis-model sara-cortex-1b
```

## Phase 5: Benchmark

Compare on MMLU Biology (310 questions):

| Config | Expected |
|--------|----------|
| llama3.2:1b alone (no Sara) | ~40-45% |
| llama3.2:3b alone (no Sara) | ~58% |
| llama3.2:3b + Sara (April baseline) | ~80% |
| **sara-cortex-1b + Sara** | **target: ≥80%** |

If 1B + Sara matches or beats 3B + Sara, we've proven:
the model doesn't need knowledge in its weights — Sara IS the weights.

## Phase 6: Open-Ended Answers (after MCQ works)

Expand training data to include open-ended questions where the
answer is a substrate-grounded paragraph, not just a letter.
This teaches the model to RENDER substrate content as prose.

Training examples:
```json
{
  "messages": [
    {"role": "system", "content": "<SYSTEM_INSTRUCTION>"},
    {"role": "user", "content": "SUBSTRATE:\n<wavefront>\n\nQUESTION:\nExplain DNA replication."},
    {"role": "assistant", "content": "Based on the substrate: DNA replication involves [facts from substrate]..."}
  ]
}
```

---

## Dependencies to Install

```bash
pip install unsloth "peft>=0.7" "trl>=0.7" bitsandbytes datasets
# OR if unsloth doesn't support 3070:
pip install peft trl bitsandbytes datasets accelerate
```

## Risks

- 310 examples may be too few → augment with rephrased questions
- Substrate noise may confuse the model → the noise IS the training signal
- 8GB VRAM tight for training → 4-bit quantization + gradient checkpointing
- Model may memorize answers instead of learning substrate reasoning →
  validate on held-out questions from a DIFFERENT Sara brain

## Success Criteria

1. sara-cortex-1b + Sara bio brain ≥ 75% on MMLU Biology
2. sara-cortex-1b + Sara aptamer brain answers aptamer questions correctly
   (proves it learned substrate reasoning, not biology facts)
3. sara-cortex-1b WITHOUT Sara ≤ 45% (proves knowledge comes from Sara)
