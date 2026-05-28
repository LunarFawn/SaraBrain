#!/bin/bash
# Sara-Cortex-1B v2 training run.
# Run this tonight after data generation finishes.
#
# Expected: ~2-3 hours on RTX 3070
# Result: models/sara-cortex-1b-v2/
#
# Usage: bash scripts/train_tonight.sh

set -e
cd /home/grizzlyengineer/repo/SaraBrain

echo "=== Sara-Cortex-1B v2 Training ==="
echo "Start: $(date)"
echo ""

# Verify data exists
DATA="training_data/sara_cortex_synthetic_10k.jsonl"
if [ ! -f "$DATA" ]; then
    echo "ERROR: $DATA not found. Data generation may still be running."
    echo "Check: tail -3 training_data/gen_10k.log"
    exit 1
fi
LINES=$(wc -l < "$DATA")
echo "Training data: $DATA ($LINES examples)"

# Clean previous run
rm -rf models/sara-cortex-1b-v2

# Train: 10k examples, 5 epochs, larger effective batch
# More data + fewer epochs = better generalization (less overfit)
.venv/bin/python scripts/finetune_sara_cortex.py \
    --data "$DATA" \
    --out models/sara-cortex-1b-v2 \
    --epochs 5

echo ""
echo "=== Training complete. Testing... ==="
echo ""

# Quick accuracy test on held-out data
.venv/bin/python -c "
import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from collections import Counter

tokenizer = AutoTokenizer.from_pretrained('models/sara-cortex-1b-v2')
base_model = AutoModelForCausalLM.from_pretrained(
    'TinyLlama/TinyLlama-1.1B-Chat-v1.0', torch_dtype=torch.float16, device_map='auto')
model = PeftModel.from_pretrained(base_model, 'models/sara-cortex-1b-v2')
model.eval()

correct = 0
total = 100
with open('training_data/sara_cortex_synthetic_400.jsonl') as f:
    for i, line in enumerate(f):
        if i >= total: break
        ex = json.loads(line)
        messages = [
            {'role': 'system', 'content': ex['system']},
            {'role': 'user', 'content': f\"SUBSTRATE:\n{ex['substrate']}\n\nQUESTION:\n{ex['question']}\"},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=3, do_sample=False)
        answer = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        if answer.startswith(ex['answer']): correct += 1

print(f'Held-out accuracy: {correct}/{total} = {correct/total*100:.0f}%')
print(f'(Random baseline: 25%)')
print(f'(v1 result: 42%)')
"

echo ""
echo "=== Done: $(date) ==="
echo "Model saved to: models/sara-cortex-1b-v2/"
echo ""
echo "Next: test on real biology brain:"
echo "  .venv/bin/sara-ask-stateless 'What is DNA replication?' \\"
echo "      --brain /home/grizzlyengineer/repo/debug_sara/sara_bio.db \\"
echo "      --synthesis-model models/sara-cortex-1b-v2"
