"""Fine-tune Llama-3.2-1B-Instruct as a Sara-native cortex model using LoRA.

Trains the model to read Sara Brain wavefront output and select
substrate-grounded answers. Training data is synthetic nonsense-word
substrates — no real knowledge in the weights.

Prerequisites:
    pip install torch transformers peft trl datasets accelerate bitsandbytes

Usage:
    python scripts/finetune_sara_cortex.py \
        --data training_data/sara_cortex_synthetic_400.jsonl \
        --out models/sara-cortex-1b \
        --epochs 5

After training, export to GGUF for Ollama:
    python scripts/finetune_sara_cortex.py --export models/sara-cortex-1b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_dataset_from_jsonl(path: str) -> list[dict]:
    """Load JSONL training data into chat-format messages."""
    examples = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            messages = [
                {"role": "system", "content": ex["system"]},
                {
                    "role": "user",
                    "content": (
                        f"SUBSTRATE:\n{ex['substrate']}\n\n"
                        f"QUESTION:\n{ex['question']}"
                    ),
                },
                {"role": "assistant", "content": ex["answer"]},
            ]
            examples.append({"messages": messages})
    return examples


def train(args):
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer
    from datasets import Dataset

    model_name = args.model

    print(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 1.1B model fits in fp16 on 8GB GPU (~2.2GB model + training overhead)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Enable gradient computation on the base model
    model.enable_input_require_grads()

    # LoRA config
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    print(f"Loading training data: {args.data}")
    raw = load_dataset_from_jsonl(args.data)
    print(f"  {len(raw)} examples loaded")

    # Format into chat template
    def format_example(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = Dataset.from_list(raw).map(format_example)

    # Training args — tuned for RTX 3070 (8GB)
    output_dir = args.out
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        optim="adamw_torch",
        seed=42,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=2048,
        packing=True,
    )

    print(f"Starting training: {args.epochs} epochs, output={output_dir}")
    print(f"  Effective batch size: {1 * 8} (batch=1, grad_accum=8)")
    trainer.train()

    # Save
    print(f"Saving LoRA adapter to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete.")


def export(args):
    """Merge LoRA adapter and export to GGUF for Ollama."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    adapter_path = args.export

    # Read the adapter config to find the base model
    import json
    config_path = Path(adapter_path) / "adapter_config.json"
    with open(config_path) as f:
        adapter_config = json.load(f)
    model_name = adapter_config.get("base_model_name_or_path", args.model)

    print(f"Loading base model + LoRA adapter from {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cpu",
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()

    # Save merged model
    merged_path = Path(adapter_path) / "merged"
    merged_path.mkdir(exist_ok=True)
    print(f"Saving merged model to {merged_path}")
    model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)

    print(f"\nTo convert to GGUF and load in Ollama:")
    print(f"  1. pip install llama-cpp-python")
    print(f"  2. python -m llama_cpp.convert {merged_path} --outfile sara-cortex-1b.gguf --outtype q4_k_m")
    print(f"  3. Create a Modelfile:")
    print(f'     echo \'FROM ./sara-cortex-1b.gguf\nPARAMETER temperature 0.1\nPARAMETER num_ctx 2048\' > Modelfile')
    print(f"  4. ollama create sara-cortex-1b -f Modelfile")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", help="Training data .jsonl path")
    ap.add_argument("--out", default="models/sara-cortex-1b",
                    help="Output directory (default: models/sara-cortex-1b)")
    ap.add_argument("--epochs", type=int, default=5, help="Training epochs (default: 5)")
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    help="Base model (default: TinyLlama-1.1B-Chat. "
                         "Use meta-llama/Llama-3.2-1B-Instruct if you have HF token)")
    ap.add_argument("--export", help="Merge LoRA + export (pass adapter dir)")
    args = ap.parse_args()

    if args.export:
        export(args)
    elif args.data:
        train(args)
    else:
        ap.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
