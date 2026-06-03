# Sara Extractor v2 — Reproduction Guide

**Model:** 115M parameter encoder-decoder with copy mechanism
**Result:** Extracts definitions first ("carbon helix is_a sentient combat ship")
**Training time:** 2 hours on RTX 3070
**Training data:** 500k synthetic examples (definitions-first)

---

## How to Reproduce

### 1. Generate training data

```bash
python scripts/generate_extractor_v2_data.py \
    --num-examples 500000 \
    --out training_data/extractor_v2_500k.jsonl \
    --seed 314159
```

Key properties of the data:
- 58% of all triples are `is_a` definitions
- Every example starts with a definition triple
- Multi-word compound concepts (2-3 words)
- Complex grammar: nested clauses, passive voice, relative clauses
- Structured output format: `t_start subject t_rel relation t_obj object t_end`
- 500k examples, generated in ~43 seconds

### 2. Train the model

```bash
python scripts/train_sara_extractor_scratch.py \
    --data training_data/extractor_v2_500k.jsonl \
    --out models/sara-extractor-115m-v2 \
    --steps 100000 \
    --batch-size 4 \
    --max-enc 300 \
    --max-dec 150
```

Note: The script uses SaraExtractor with d_model=768, enc_layers=8,
dec_layers=6, n_heads=12. These are set via the class override in the
training launch script (see /tmp/train_ext_v3.py pattern or modify
the script directly).

### 3. Test

```python
from train_sara_extractor_scratch import SaraExtractor, build_vocab, encode_with_oov
import torch

tok2id = build_vocab()
model = SaraExtractor(len(tok2id) + 300, d_model=768, enc_layers=8,
                      dec_layers=6, n_heads=12, max_enc=300, max_dec=150)
ckpt = torch.load('models/sara-extractor-115m-v2/best.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()

# Test sentence
enc_ids, oov, oov_map = encode_with_oov(
    "The Carbon Helix is an accidentally sentient combat ship.", tok2id, 300)
enc_t = torch.tensor([enc_ids]).unsqueeze(0) if enc_ids else torch.tensor([enc_ids])
# ... generate and parse output
```

Expected output: `carbon helix | is_a | accidentally sentient combat ship`

## Model Architecture

```
SaraExtractor (115M params):
  Encoder:
    - Embedding: vocab_size × 768
    - Positional: 300 × 768
    - 8 TransformerEncoderLayers (d=768, heads=12, ff=3072, norm_first=True)
    - LayerNorm
  Decoder (with copy mechanism):
    - Embedding: vocab_size × 768
    - Positional: 150 × 768
    - 6 TransformerDecoderLayers (d=768, heads=12, ff=3072, norm_first=True)
    - LayerNorm
    - Generate head: Linear(768, vocab_size)
    - Copy gate: Linear(768, 1) → sigmoid
    - Copy attention: Linear(768, 768)
  Base vocab: 151 tokens (relations + English function words)
  Extended vocab: base + 300 OOV slots (for content words copied from input)
```

## What Made This Work (Lessons)

1. **Definitions first (58% is_a)** — previous version had ~10% is_a.
   The model now learns "extract what X IS before what X does."

2. **Structured delimiters (t_start/t_rel/t_obj/t_end)** — earlier
   attempt used `<triple>` which the tokenizer split into 3 tokens.
   Underscore-joined tokens stay as single tokens.

3. **Copy mechanism** — concept labels are copied from input, not
   generated from vocabulary. Can handle any domain without retraining.

4. **Synthetic training only** — no real-world knowledge in the
   training data. The model learns the SKILL of extraction, not
   any domain facts.

5. **Complex grammar in training data** — nested clauses, passive
   voice, multi-verb sentences. Without these, the model fails on
   real text that uses complex structures.

## Files

- `scripts/generate_extractor_v2_data.py` — training data generator
- `scripts/train_sara_extractor_scratch.py` — model + training loop
- `training_data/extractor_v2_500k.jsonl` — the training data
- `models/sara-extractor-115m-v2/best.pt` — trained checkpoint
