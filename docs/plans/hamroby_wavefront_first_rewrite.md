# Plan: HamRobyLLM Wavefront-First Rewrite

**Date:** 2026-05-27
**Problem:** The previous LLM session built HamRobyLLM with a router
head that picks tools (`brain_explore`, `brain_define`, `brain_value`,
`brain_did_you_mean`). This bypasses wavefront propagation — the
brain's defining mechanism. The router turns Sara into a database to
be queried, not a brain that thinks.

**Fix:** Remove the router. The wavefront runs FIRST and ALWAYS. 
HamRobyLLM's job is to READ the wavefront output, not to decide
what to query.

---

## The Wrong Architecture (what was built)

```
Question → spaCy parse → HamRobyLLM router head → picks a tool
    → calls brain_explore / brain_define / brain_value
    → gets narrow result back
    → template synthesizer renders it

Problem: the wavefront never runs. The brain never "thinks."
Sara is reduced to a lookup table.
```

## The Right Architecture (what to build)

```
Question → wavefront propagation (AUTOMATIC, always)
    → full convergence neighborhood (the "noise IS the data")
    → HamRobyLLM substrate-reasoning head reads the neighborhood
    → produces answer grounded in the substrate

No routing. No tool selection. The brain thinks first.
The model reads what the brain thought.
```

## What Changes

### Remove
- `router_head.py` — the 4-way tool classifier (wrong task)
- `router_data.py` — tool-call training data generation
- `train_router.py` — router head training script
- The concept of "the LLM picks which Sara tool to call"

### Keep
- `model.py` — the 125M grammar backbone (GrammarModel)
- `train.py` — grammar LM training on UD treebanks
- `vocab.py` — UD tag vocabulary
- `synthesizer.py` — template renderer (useful for output)
- The spaCy parse pipeline (for seed extraction from questions)

### Add
- `substrate_head.py` — new head on top of the grammar backbone
  that reads tokenized wavefront output and produces an answer
- `train_substrate.py` — trains the substrate head on synthetic
  nonsense data (same data we generated for the TinyLlama fine-tune)
- Integration in `stateless_reader.py`: wavefront runs first,
  output goes to HamRobyLLM substrate head, answer comes out

## The Substrate-Reasoning Head

**Input:** tokenized wavefront output (intersection labels, strengths,
convergence map) + the question

**Output:** answer (MCQ letter for now, free-text later)

**Architecture:**
```python
class SubstrateHead(nn.Module):
    """Reads wavefront output through the frozen grammar backbone."""
    
    def __init__(self, grammar_model, n_choices=4):
        self.encoder = grammar_model  # frozen 125M backbone
        self.projection = nn.Linear(d_model, n_choices)
    
    def forward(self, substrate_tokens, question_tokens):
        # Encode the combined input through the grammar backbone
        combined = concat(substrate_tokens, question_tokens)
        hidden = self.encoder.encode(combined)  # frozen, no grad
        # Pool and classify
        pooled = hidden.mean(dim=1)
        logits = self.projection(pooled)
        return logits
```

The grammar backbone provides structural understanding of the
token sequence. The substrate head learns what convergence patterns
mean — which intersection labels relate to which question concepts.

## Training Data

Same synthetic nonsense data we already generated:
- `training_data/sara_cortex_synthetic_10k.jsonl` (10,000 examples)
- Pure nonsense-word substrates, no real knowledge
- The head learns substrate reasoning, not domain facts

**Tokenization change:** instead of using the chat template (designed
for LLMs), we tokenize the wavefront output into UD-style tags that
the grammar backbone understands. This means a custom tokenizer that
maps substrate structure (labels, strengths, relations) into the
vocabulary the backbone was trained on.

OR: retrain the backbone on a mixed corpus (UD grammar + substrate
format) so it learns both English structure and substrate structure.
This is ~50 min additional training.

## Implementation Steps

1. **Verify tonight's TinyLlama result** — confirms the task is
   learnable (substrate → answer)

2. **Design substrate tokenization** — how to represent wavefront
   output in a format the grammar backbone can process. Options:
   - Raw text tokens (treat substrate output as English text)
   - Structured tokens (special tokens for strength, intersection, etc.)
   - Hybrid (label text as English, structure markers as special tokens)

3. **Retrain grammar backbone on mixed corpus** — add substrate-format
   text to the UD training data so the backbone learns both

4. **Train substrate head** — freeze backbone, train the head on
   10k synthetic examples (~10-20 min on 3070, head is tiny)

5. **Integrate into stateless_reader.py** — replace the router path
   with: wavefront → tokenize → substrate head → answer

6. **Benchmark** — test on MMLU biology with the bio brain

## Timeline

- Step 1: tonight (TinyLlama run finishes ~1 AM)
- Steps 2-4: one session (~2-3 hours total)
- Step 5-6: one session (~1-2 hours)

Total: the from-scratch Sara-native cortex in 2 sessions after tonight.

## The Principle

**Wavefront IS the brain.** It is not one tool among many. It is not
something the LLM "decides" to use. It runs first, always, automatically.
The LLM's only job is to read what the wavefront returned and render
it as a human-readable answer. The brain thinks. The cortex reads.
