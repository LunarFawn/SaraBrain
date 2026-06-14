import os
import sys
import torch
from pathlib import Path

# Add scripts to sys.path to find train_sara_extractor_scratch
sys.path.insert(0, os.path.abspath("scripts"))
from train_sara_extractor_scratch import SaraExtractor, build_vocab, encode_with_oov
from sara_brain.cortex.transformer.v2.normalize import normalize_label

def load_sara_model(ckpt_path, device):
    tok2id = build_vocab()
    ext_vocab = len(tok2id) + 300
    
    # Use exact 347M architecture from the training run
    model = SaraExtractor(
        ext_vocab, 
        d_model=1024, 
        enc_layers=14, 
        dec_layers=10,
        n_heads=16, 
        max_enc=400, 
        max_dec=100
    ).to(device)
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
        
    model.eval()
    return model, tok2id

def sara_extract(model, tok2id, clause, device):
    enc_ids, oov, oov_map = encode_with_oov(clause, tok2id, 400)
    enc_t = torch.tensor([enc_ids], dtype=torch.long, device=device)
    pm = torch.zeros(1, len(enc_ids), dtype=torch.bool, device=device)
    with torch.no_grad():
        out_ids = model.generate(enc_t, pm, max_len=100)[0].tolist()
    
    id2tok = {v: k for k, v in tok2id.items()}
    for t, idx in oov_map.items():
        id2tok[idx] = t
    
    gen_tokens = []
    for i in out_ids:
        if i not in (0, 2):
            gen_tokens.append(id2tok.get(i, f"?({i})"))
    gen = " ".join(gen_tokens)
    print(f"  Raw output: {gen}")

    triples = []
    for part in gen.split("t_end"):
        if "t_start" in part and "t_rel" in part and "t_obj" in part:
            try:
                after = part.split("t_start")[1]
                subj = after.split("t_rel")[0].strip()
                rel = after.split("t_rel")[1].split("t_obj")[0].strip()
                obj = after.split("t_obj")[1].strip()
                subj = normalize_label(subj)
                obj = normalize_label(obj)
                if subj and rel and obj and len(subj) > 1 and len(obj) > 1 and subj != obj:
                    triples.append((subj, rel, obj))
            except (IndexError, ValueError):
                pass
    return triples

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    new_model_path = "models/sara-extractor-340m-v2/best.pt"
    
    model, tok2id = load_sara_model(new_model_path, device)
    
    test_sentences = [
        # OOD (too short)
        "Meiosis involves prophase.",
        # In distribution (looks like training data)
        "Meiosis is a biological process. The result was 123. Meiosis involves prophase. This is a common word.",
        "carbon helix is a sentient combat ship. 5 and 6 are common words. carbon helix includes guns as a component. generally we see this.",
    ]
    
    for text in test_sentences:
        print(f"\nText: {text}")
        triples = sara_extract(model, tok2id, text, device)
        if not triples:
            print("  (No triples extracted)")
        for s, r, o in triples:
            print(f"  [{s}] --({r})--> [{o}]")
