import json
import re
from sara_brain.cortex.transformer.vocab_en import TOK2ID_EN as L2_TOK2ID

with open('data/biology_mcq_50k.jsonl', 'r') as f:
    ex = json.loads(f.readline())

text = f"{ex['system']}\n\n{ex['prompt']}"
tokens = re.findall(r"[a-z_]+|[0-9]+\.[0-9]+|[0-9]+|[^\s]", text.lower())

print("Tokens:")
print(tokens)
