"""Pretrain 100M model on Wikitext-103 (101M words) for language understanding."""
import torch, torch.nn as nn, re, random, time, math, json
from pathlib import Path
from datasets import load_dataset

# Load data
print("Loading Wikitext-103...")
ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
texts = [ex['text'] for ex in ds if len(ex['text'].strip()) > 50]
print(f"Usable lines: {len(texts):,}")

# Build vocab from first 1M words
rng = random.Random(2026)
word_counts = {}
for line in texts[:100000]:
    for w in re.findall(r'[a-zA-Z]+', line.lower()):
        word_counts[w] = word_counts.get(w, 0) + 1
sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
vocab = {'<pad>': 0, '<unk>': 1, '<mask>': 2}
for word, _ in sorted_words[:16000]:
    vocab[word] = len(vocab)
for ch in '.,?!;:-()\'\"': vocab[ch] = len(vocab)
print(f"Vocab: {len(vocab)} tokens")

def tokenize(text, max_len=128):
    words = re.findall(r'[a-zA-Z]+|[.,?!;:()\-\'"]', text.lower())
    return [vocab.get(w, vocab['<unk>']) for w in words[:max_len]]

# Model: 100M params
class MLMModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_heads=12, n_layers=8, max_seq=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq, d_model)
        self.drop = nn.Dropout(0.1)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.mlm_head = nn.Linear(d_model, vocab_size)
    def forward(self, input_ids, pad_mask=None):
        B, T = input_ids.shape
        x = self.embed(input_ids) + self.pos(torch.arange(T, device=input_ids.device))
        x = self.drop(x)
        if pad_mask is not None: x = self.encoder(x, src_key_padding_mask=pad_mask)
        else: x = self.encoder(x)
        x = self.ln(x)
        return self.mlm_head(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLMModel(len(vocab), d_model=768, n_heads=12, n_layers=8).to(device)
params = sum(p.numel() for p in model.parameters())
print(f"Model: {params/1e6:.0f}M params on {device}")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss(ignore_index=-100)
MASK_ID = vocab['<mask>']

model.train()
t0 = time.time()
Path('models/sara-cortex-pretrained').mkdir(exist_ok=True)

for step in range(1, 200001):
    batch_lines = rng.sample(texts, 32)
    batch_ids = [tokenize(line, 128) for line in batch_lines]
    max_l = max(len(x) for x in batch_ids)
    if max_l == 0: continue
    padded = [x + [0]*(max_l-len(x)) for x in batch_ids]
    masks = [[False]*len(x) + [True]*(max_l-len(x)) for x in batch_ids]
    
    input_t = torch.tensor(padded, dtype=torch.long, device=device)
    pad_mask_t = torch.tensor(masks, dtype=torch.bool, device=device)
    
    labels = torch.full_like(input_t, -100)
    mask_pos = (torch.rand_like(input_t.float()) < 0.15) & ~pad_mask_t & (input_t != 0)
    labels[mask_pos] = input_t[mask_pos]
    input_t[mask_pos] = MASK_ID
    
    logits = model(input_t, pad_mask_t)
    loss = criterion(logits.view(-1, len(vocab)), labels.view(-1))
    
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if step % 20000 == 0:
        model.eval()
        with torch.no_grad():
            acc = (logits.argmax(-1) == labels)[labels != -100].float().mean().item()
        elapsed = time.time() - t0
        remaining = (200000 - step) / step * elapsed
        print(f"  step={step} loss={loss.item():.3f} acc={acc*100:.0f}% [{elapsed/3600:.1f}h, ~{remaining/3600:.1f}h left]")
        torch.save({'model': model.state_dict(), 'vocab': vocab, 'step': step},
                   'models/sara-cortex-pretrained/wiki_mlm.pt')
        model.train()

print(f"\nDone in {(time.time()-t0)/3600:.1f}h")
torch.save({'model': model.state_dict(), 'vocab': vocab, 'step': 200000},
           'models/sara-cortex-pretrained/wiki_mlm_final.pt')
