# Sara Extractor v2-Clean — Reproduction Guide

**Model:** 115M parameter encoder-decoder with copy mechanism
**Result:** Extracts definitions first, perfectly handles both long paragraphs and isolated short sentences, and ignores English "noise".
**Training time:** ~1.3 hours on RTX 3070
**Training data:** 500k synthetic examples (mix of paragraphs, sentences, and noise)

---

## How to Reproduce

### 1. Generate training data

```bash
python scripts/generate_extractor_v2_data.py \
    --num-examples 500000 \
    --out training_data/extractor_v2_noise_500k.jsonl
```

Key properties of the data:
- **Sequence Length Robustness:** 60% of examples are full paragraphs (3-6 sentences), 30% are isolated short sentences (1-2 sentences), and 10% are purely "noise" (e.g., "The result was 123.") to explicitly teach the model when NOT to extract triples.
- **Simple Declaratives:** Contains simple "X involves Y" templates in addition to complex grammar, ensuring the model doesn't fail on basic facts.
- **Structured output format:** `t_start subject t_rel relation t_obj object t_end`
- 500k examples, generated in ~46 seconds.

*Note: The generated JSONL is ~237MB. For repository storage, it is compressed via zip into `training_data/extractor_v2_noise_500k.zip` (~50MB).*

### 2. Train the model

```bash
python scripts/train_sara_extractor_scratch.py \
    --data training_data/extractor_v2_noise_500k.jsonl \
    --out models/sara-extractor-v2-clean \
    --steps 50000 \
    --batch-size 8 \
    --d-model 768 \
    --enc-layers 8 \
    --dec-layers 6 \
    --n-heads 12
```

**CRITICAL RULE:** The training script no longer has default architecture arguments. You **must** explicitly pass the `--d-model`, `--enc-layers`, `--dec-layers`, and `--n-heads` arguments to prevent accidentally training a tiny debug model. The 115M configuration above is the production standard.

### 3. Test

```python
import torch
from scripts.train_sara_extractor_scratch import SaraExtractor, build_vocab, encode_with_oov

tok2id = build_vocab()
ext_vocab = len(tok2id) + 300

model = SaraExtractor(
    ext_vocab, 
    d_model=768, 
    enc_layers=8, 
    dec_layers=6,
    n_heads=12, 
    max_enc=400, 
    max_dec=100
).to('cuda')

ckpt = torch.load('models/sara-extractor-v2-clean/best.pt', map_location='cuda', weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()

# Test sentence
enc_ids, oov, oov_map = encode_with_oov("Meiosis involves prophase.", tok2id, 400)
enc_t = torch.tensor([enc_ids]).to('cuda')
pm = torch.zeros(1, len(enc_ids), dtype=torch.bool).to('cuda')

with torch.no_grad():
    out_ids = model.generate(enc_t, pm, max_len=100)[0].tolist()

# ... map out_ids back to tokens using id2tok and oov_map ...
```

Expected output: `t_start meiosis t_rel involves t_obj prophase t_end`

### 4. Running the Model in Production (Ingestion)

To use the 115M model to ingest documents into a brain:

```bash
python src/sara_reader/cli_teach_book.py \
    --brain /tmp/my_book.db \
    --extractor sara \
    path/to/my/book.txt
```

*Note: The `cli_teach_book.py` pipeline currently pre-parses documents into isolated grammar clauses before feeding them to the extractor. While the v2-clean model handles isolated sentences well, extremely fragmented clauses can still degrade output quality compared to feeding full paragraphs.*

## Model Architecture

```
SaraExtractor (115.5M params):
  Encoder:
    - Embedding: vocab_size × 768
    - Positional: 400 × 768
    - 8 TransformerEncoderLayers (d=768, heads=12, ff=3072, norm_first=True)
    - LayerNorm
  Decoder (with copy mechanism):
    - Embedding: vocab_size × 768
    - Positional: 100 × 768
    - 6 TransformerDecoderLayers (d=768, heads=12, ff=3072, norm_first=True)
    - LayerNorm
    - Generate head: Linear(768, vocab_size)
    - Copy gate: Linear(768, 1) → sigmoid
    - Copy attention: Linear(768, 768)
  Base vocab: 185 tokens (relations + English function words)
  Extended vocab: base + 300 OOV slots (for content words copied from input)
```

## What Made This Work (Lessons)

1. **Varying Sequence Length** — The model's attention mechanism breaks if it only sees long paragraphs during training but is fed single clauses in production. Mixing paragraphs, sentences, and noise-only sequences ensures robust inference at any length.
2. **Explicit Architecture Arguments** — Defaulting to tiny "debug" sizes in training scripts leads to hours of wasted time if a session restarts blindly. Forcing explicit parameter counts guarantees the production 115M model is always used.
3. **Copy mechanism** — Concept labels are copied from input, not generated from vocabulary. The model cannot hallucinate facts, only pointing indices.
4. **Structured delimiters (t_start/t_rel/t_obj/t_end)** — Using underscore-joined structural tokens prevents the tokenizer from splitting delimiters.
