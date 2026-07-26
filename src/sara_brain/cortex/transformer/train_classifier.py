import json
import torch
import torch.nn as nn
import argparse
import re
from pathlib import Path
from tqdm import tqdm
import os
import torch.optim as optim

from sara_brain.cortex.transformer.model import GrammarConfig, GrammarModel
from sara_brain.cortex.transformer.vocab_en import TOK2ID_EN as L2_TOK2ID

class L3WithHead(nn.Module):
    def __init__(self, bb, d_model):
        super().__init__()
        self.backbone = bb
        self.cls_head = nn.Linear(d_model, 4)
        
    def forward(self, input_ids):
        x = self.backbone.tok_embed(input_ids)
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        if T <= self.backbone.pos_embed.weight.shape[0]:
            x = x + self.backbone.pos_embed(pos)
        x = self.backbone.drop(x)
        attn_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        for block in self.backbone.blocks:
            x = block(x, attn_mask=attn_mask)
        x = self.backbone.ln_f(x)
        
        # Use the representation of the last actual token (<eos>)
        # by finding the sequence length (where padding is 0)
        lengths = (input_ids != 0).sum(dim=1)
        B_idx = torch.arange(B, device=x.device)
        x_last = x[B_idx, lengths - 1, :]
        
        return self.cls_head(x_last)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load L2 backbone
    print(f"Loading backbone from {args.l2_ckpt}")
    ck = torch.load(args.l2_ckpt, map_location="cpu", weights_only=False)
    cfg = GrammarConfig(**ck["config"])
    bb = GrammarModel(cfg)
    bb.load_state_dict(ck["state_dict"], strict=False)
    
    # Load data
    print(f"Loading data from {args.data}")
    examples = []
    with open(args.data, 'r') as f:
        for line in f:
            if not line.strip(): continue
            examples.append(json.loads(line))
            
    print(f"Loaded {len(examples)} examples.")
    
    # Resize embeddings for substrate vocabulary
    print("Building vocabulary from dataset...")
    substrate_vocab = {}
    for ex in examples:
        text = f"{ex['system']}\n\n{ex['prompt']}"
        tokens = re.findall(r"[a-z_]+|[0-9]+\.[0-9]+|[0-9]+|[^\s]", text.lower())
        for tok in tokens:
            if tok not in L2_TOK2ID and tok not in substrate_vocab:
                substrate_vocab[tok] = len(substrate_vocab)
                
    new_vocab_size = max(L2_TOK2ID.values()) + 1 + len(substrate_vocab)
    
    old_embed = bb.tok_embed.weight.data
    bb.tok_embed = nn.Embedding(new_vocab_size, cfg.d_model)
    bb.tok_embed.weight.data[:old_embed.size(0)] = old_embed
    bb.cfg.vocab_size = new_vocab_size
    
    # Freeze backbone
    for param in bb.parameters():
        param.requires_grad = False
        
    # Unfreeze top layers and new embeddings
    bb.tok_embed.weight.requires_grad = True
    for block in bb.blocks[-args.unfreeze_top_n:]:
        for param in block.parameters():
            param.requires_grad = True
    bb.ln_f.weight.requires_grad = True
    bb.ln_f.bias.requires_grad = True
    
    model = L3WithHead(bb, cfg.d_model).to(device)
    

    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    
    batch_size = args.batch
    model.train()
    
    out_dir = Path("src/sara_brain/cortex/checkpoints")
    out_dir.mkdir(exist_ok=True, parents=True)
    
    print("Training...")
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Shift substrate IDs so they don't collide with L2 vocab
    base_id = max(L2_TOK2ID.values()) + 1
    
    for step in range(args.steps):
        batch_ex = [examples[i % len(examples)] for i in range(step * batch_size, (step + 1) * batch_size)]
        
        input_ids = []
        labels = []
        
        max_len = 0
        for ex in batch_ex:
            text = f"{ex['system']}\n\n{ex['prompt']}"
            tokens = re.findall(r"[a-z_]+|[0-9]+\.[0-9]+|[0-9]+|[^\s]", text.lower())[:cfg.max_seq - 2]
            ids = [1]
            for tok in tokens:
                if tok in substrate_vocab:
                    ids.append(base_id + substrate_vocab[tok])
                elif tok in L2_TOK2ID:
                    ids.append(L2_TOK2ID[tok])
                else:
                    ids.append(4)
            ids.append(2)
            input_ids.append(ids)
            max_len = max(max_len, len(ids))
            labels.append(ex['answer'])
            
        # Pad
        padded = []
        for ids in input_ids:
            padded.append(ids + [0] * (max_len - len(ids)))
            
        x = torch.tensor(padded, dtype=torch.long, device=device)
        y = torch.tensor(labels, dtype=torch.long, device=device)
        
        logits = model(x)
        loss = loss_fn(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += len(y)
        
        if (step + 1) % 50 == 0:
            print(f"Step {step+1}/{args.steps} - Loss: {running_loss/50:.4f} - Acc: {correct/total:.4f}")
            running_loss = 0.0
            correct = 0
            total = 0
            
    # Save model
    out_path = out_dir / "hamroby_cls_biology_50k.pt"
    torch.save({
        "step": args.steps,
        "config": bb.cfg.__dict__,
        "model_state_dict": model.state_dict(),
        "substrate_vocab": {k: base_id + v for k, v in substrate_vocab.items()}
    }, out_path)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--l2-ckpt", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--unfreeze-top-n", type=int, default=8)
    args = p.parse_args()
    train(args)
