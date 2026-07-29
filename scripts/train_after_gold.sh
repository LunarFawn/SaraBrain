#!/usr/bin/env bash
# Wait for gold extraction to finish, then build mixed data and train
set -e
cd /home/grizzlyengineer/repo/SaraBrain

echo "Waiting for gold extraction to finish..."
while ps -p 1439 > /dev/null 2>&1; do
    sleep 30
done
echo "Gold extraction done!"

GOLD=training_data/extractor_gold_multi.jsonl
echo "Gold examples: $(wc -l < $GOLD)"

# Build mixed training dataset
echo "Building mixed training data..."
.venv/bin/python -c "
import json, random

rng = random.Random(2026)

# Load gold multi-triple data (highest quality — repeat 50x)
gold = []
with open('training_data/extractor_gold_multi.jsonl') as f:
    for line in f:
        gold.append(json.loads(line))
print(f'Gold: {len(gold)} examples (will repeat 50x = {len(gold)*50})')

# Load existing jibberish data (generalization)
jib = []
with open('training_data/extractor_english_v2_500k.jsonl') as f:
    for line in f:
        ex = json.loads(line)
        # Only keep jibberish examples (have nonsense words)
        if any(c in ex['paragraph'][:20] for c in ['uu','zz','ww','xx']) or \
           any(len(w) > 8 and not any(v in w for v in 'aeiou') for w in ex['paragraph'].split()[:3]):
            jib.append(ex)
        if len(jib) >= 100000:
            break
print(f'Jibberish: {len(jib)} examples')

# Build final dataset: gold x50 + jibberish
examples = []
for _ in range(50):
    examples.extend(gold)
examples.extend(jib[:100000])

rng.shuffle(examples)
print(f'Total: {len(examples)} examples')

with open('training_data/extractor_gold_mixed_500k.jsonl', 'w') as f:
    for ex in examples:
        f.write(json.dumps(ex) + '\n')
print('Written to training_data/extractor_gold_mixed_500k.jsonl')
"

echo "Starting training..."
.venv/bin/python scripts/train_sara_extractor_scratch.py \
    --data training_data/extractor_gold_mixed_500k.jsonl \
    --out models/sara-extractor-gold-v3 \
    --steps 50000 \
    --batch-size 8 \
    --d-model 768 \
    --enc-layers 8 \
    --dec-layers 6 \
    --n-heads 12 \
    --checkpoint-every 5000 \
    2>&1 | tee training_data/extractor_gold_v3.log

echo "Training complete!"
