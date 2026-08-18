#!/usr/bin/env python3
"""LoRA fine-tune Qwen 2.5 7B to be substrate-obedient for Sara Brain.

Trains the model to:
1. ALWAYS derive answers from provided facts (never training weights)
2. Say 'E' when facts don't support an answer
3. Handle jibberish/cipher text by reading logical structure

Usage:
    python scripts/lora_finetune_qwen.py
"""
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset, DataLoader

# Config
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "models/sara-cortex-qwen-lora"
DATA_PATH = "training_data/lora_substrate_obedience.jsonl"
EPOCHS = 3
BATCH_SIZE = 1  # Small batch for 8GB GPU
GRAD_ACCUM = 8  # Effective batch = 8
LR = 2e-4
MAX_LEN = 512

print("Loading training data...")
examples = []
with open(DATA_PATH) as f:
    for line in f:
        examples.append(json.loads(line))
print(f"Training examples: {len(examples)}")

print(f"Loading {MODEL_NAME} in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Dataset
class SubstrateDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        # Format as chat
        text = f"<|im_start|>user\n{ex['prompt']}<|im_end|>\n<|im_start|>assistant\n{ex['response']}<|im_end|>"
        encoded = self.tokenizer(text, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt")
        input_ids = encoded["input_ids"].squeeze()
        # Only compute loss on the response part
        labels = input_ids.clone()
        # Find where assistant response starts
        response_text = f"<|im_start|>assistant\n{ex['response']}"
        response_tokens = self.tokenizer(response_text, add_special_tokens=False)["input_ids"]
        # Mask everything before the response
        response_start = len(input_ids) - len(response_tokens) - 1
        labels[:max(0, response_start)] = -100
        return {"input_ids": input_ids, "labels": labels, "attention_mask": encoded["attention_mask"].squeeze()}

dataset = SubstrateDataset(examples, tokenizer, MAX_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Training
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
model.train()

print(f"\nTraining for {EPOCHS} epochs ({len(dataloader)} steps/epoch)...")
import time
t0 = time.time()

for epoch in range(EPOCHS):
    total_loss = 0
    optimizer.zero_grad()
    for step, batch in enumerate(dataloader):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss / GRAD_ACCUM
        loss.backward()
        total_loss += loss.item() * GRAD_ACCUM
        
        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        if (step + 1) % 50 == 0:
            avg_loss = total_loss / (step + 1)
            print(f"  Epoch {epoch+1} step {step+1}/{len(dataloader)} loss={avg_loss:.4f}")
    
    avg_loss = total_loss / len(dataloader)
    print(f"  Epoch {epoch+1} complete. Avg loss: {avg_loss:.4f} [{time.time()-t0:.0f}s]")

# Save
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nSaved LoRA adapter to {OUTPUT_DIR}")
print(f"Total time: {(time.time()-t0)/60:.1f} minutes")
