# Training Data

Large training data files are split into 50MB parts for GitHub storage.

## Reassemble

```bash
# Extractor v2 (500k examples, definitions-first)
cat training_data/extractor_v2_500k_part_* > training_data/extractor_v2_500k.jsonl
```

## Regenerate from scratch

If you prefer to regenerate rather than reassemble:

```bash
python scripts/generate_extractor_v2_data.py \
    --num-examples 500000 \
    --out training_data/extractor_v2_500k.jsonl \
    --seed 314159
```

This is deterministic — same seed produces identical data.
