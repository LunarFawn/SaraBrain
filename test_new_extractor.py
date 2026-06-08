
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
    # Match architecture from cli_teach_book.py or train script
    # cli_teach_book uses d_model=768, enc_layers=8, dec_layers=6, n_heads=12
    # train_sara_extractor_scratch uses d_model=256, enc_layers=4, dec_layers=3, n_heads=8
    # The log says: Params: 6,887,398 (6.9M)
    # Let's check which one matches 6.9M. 
    # d_model=256, L=4/3, H=8:
    # 256*256*12 (layers) ~ 0.8M. Plus embeddings. 
    # 768*768*14 ~ 8M.
    # Actually, let's try to detect from checkpoint or just try the train script defaults first.
    
    try:
        model = SaraExtractor(ext_vocab, d_model=256, enc_layers=4, dec_layers=3,
                              n_heads=8, max_enc=400, max_dec=100).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
    except RuntimeError:
        # Try the other common config
        model = SaraExtractor(ext_vocab, d_model=768, enc_layers=8, dec_layers=6,
                              n_heads=12, max_enc=300, max_dec=150).to(device)
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
    gen = " ".join(id2tok.get(i, "?") for i in out_ids if i not in (0, 2))

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
    new_model_path = "models/sara-extractor-v2-clean/best.pt"
    
    print(f"Loading new model from {new_model_path}...")
    model, tok2id = load_sara_model(new_model_path, device)
    
    test_sentences = [
        "Meiosis involves prophase.",
        "The result was 123.",
        "It is able to obtain 5 along with others.",
        "DNA and RNA share base pairing.",
        "The carbon helix is a sentient combat ship."
    ]
    
    for text in test_sentences:
        print(f"\nText: {text}")
        triples = sara_extract(model, tok2id, text, device)
        if not triples:
            print("  (No triples extracted)")
        for s, r, o in triples:
            print(f"  [{s}] --({r})--> [{o}]")
