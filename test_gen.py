import sys, os, re
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import torch
from train_sara_extractor_scratch import SaraExtractor, encode_with_oov, build_vocab
tok2id = build_vocab()
max_enc = 512; max_dec = 128
ext_vocab = len(tok2id) + 512
model = SaraExtractor(ext_vocab, d_model=768, enc_layers=8, dec_layers=6, n_heads=12, max_enc=max_enc, max_dec=max_dec).to('cuda')
ckpt = torch.load("models/sara-synthesizer-115m-jibberish/best.pt", map_location='cuda')
model.load_state_dict(ckpt.get("model", ckpt.get("state_dict", {})))
model.eval()
input_text = "SUBSTRATE:\n  - mazo resvaclid involves levtoanij\n\nQUESTION:\nWhat does mazo resvaclid involve?\nA. levtoanij\nB. foo\nC. bar\nD. baz"
enc_ids, oov, oov_map = encode_with_oov(input_text, tok2id, max_enc)
enc_t = torch.tensor([enc_ids], dtype=torch.long, device='cuda')
pm = torch.zeros(1, len(enc_ids), dtype=torch.bool, device='cuda')
out_ids = model.generate(enc_t, pm, max_len=50)[0].tolist()
id2tok = dict(tok2id)
for t, idx in oov_map.items(): id2tok[idx] = t
response = " ".join(id2tok.get(i, "") for i in out_ids if i not in (0, 1, 2)).strip()
print("RAW GENERATION:", repr(response))
response = response.strip().upper()
print("UPPERCASE:", repr(response))
match = re.search(r"\b([A-D])\b", response)
print("MATCH:", match.group(1) if match else "NONE")
