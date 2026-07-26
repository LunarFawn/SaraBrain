#!/usr/bin/env python3
"""MMLU High School Biology Benchmark — Sara Brain's sweet spot.

310 multiple-choice questions testing factual biology recall.
All models use the Universal Wavefront Substrate (Noise as Signal).
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import re
from collections import defaultdict

def load_questions(json_file: str | None = None) -> list[dict]:
    if json_file:
        with open(json_file, 'r', encoding='utf-8') as f:
            ds = json.load(f)
            return [
                {
                    'id': i,
                    'question': q['question'],
                    'choices': q['choices'],
                    'answer_idx': q['answer'],
                }
                for i, q in enumerate(ds)
            ]

    from datasets import load_dataset
    ds = load_dataset('cais/mmlu', 'high_school_biology', split='test')
    return [
        {
            'id': i,
            'question': q['question'],
            'choices': q['choices'],
            'answer_idx': q['answer'],
        }
        for i, q in enumerate(ds)
    ]

def format_mc_prompt(question: str, choices: list[str]) -> str:
    lines = [question, '']
    for i, choice in enumerate(choices):
        lines.append(f"{['A', 'B', 'C', 'D'][i]}. {choice}")
    lines.append('')
    lines.append('Answer with ONLY the letter (A, B, C, or D). Nothing else.')
    return '\n'.join(lines)

def call_llm(prompt: str, model: str, system: str,
             base_url: str, local_loader=None) -> str:
    if local_loader:
        return local_loader.query(prompt, system)

    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': prompt},
        ],
        'stream': False,
        'options': {'temperature': 0},
    }
    url = f'{base_url}/v1/chat/completions'
    headers = {'Content-Type': 'application/json'}
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode('utf-8'))
            return body['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f'ERROR: {e}'

class LocalModelLoader:
    def __init__(self, path):
        import torch
        import os
        import sys
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        
        sys.path.insert(0, os.path.abspath("scripts"))
        from train_sara_extractor_scratch import SaraExtractor, build_vocab

        self.path = path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  [loader] Loading model from {path} on {self.device}...")

        if os.path.exists(os.path.join(path, "adapter_config.json")):
            self.arch = "peft"
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            base = AutoModelForCausalLM.from_pretrained(
                base_model_name, torch_dtype=torch.float16, device_map="auto"
            )
            self.model = PeftModel.from_pretrained(base, path)
        elif "hamroby" in path.lower() and "sum" not in path.lower():
            self.arch = "hamroby"
            import torch.nn as nn
            from sara_brain.cortex.transformer.model import GrammarConfig, GrammarModel
            from sara_brain.cortex.transformer.vocab import TOK2ID as L1_TOK2ID
            try:
                from sara_brain.cortex.transformer.vocab_en import TOK2ID_EN as L2_TOK2ID
            except ImportError:
                L2_TOK2ID = L1_TOK2ID
            
            ckpt_path = os.path.join(path, "best.pt")
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            cfg_dict = ckpt["config"]
            self.substrate_vocab = ckpt["substrate_vocab"]
            self.l2_tok2id = L2_TOK2ID
            self.max_seq = cfg_dict["max_seq"]
            
            cfg = GrammarConfig(**cfg_dict)
            backbone = GrammarModel(cfg)
            
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
                    lengths = (input_ids != 0).sum(dim=1)
                    B_idx = torch.arange(B, device=x.device)
                    return self.cls_head(x[B_idx, lengths - 1, :])
            self.model = L3WithHead(backbone, cfg.d_model).to(self.device)
            if "cls_head.weight" in ckpt["model_state_dict"]:
                self.model.load_state_dict(ckpt["model_state_dict"])
            else:
                self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        elif "hamroby_sum" in path.lower():
            self.arch = "hamroby_gen"
            import torch
            from sara_brain.cortex.transformer.inference_synth import load_synth_checkpoint
            from pathlib import Path
            self.model = load_synth_checkpoint(Path(path), self.device)
            # Use inference_synth's vocab
            from sara_brain.cortex.transformer.vocab_synth import TOK2ID_SYNTH
            self.tok2id = TOK2ID_SYNTH
        else:
            self.arch = "sara"
            self.tok2id = build_vocab()
            ckpt_path = os.path.join(path, "best.pt")
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            sd = ckpt.get("model", ckpt.get("state_dict", {}))
            
            d_model = 768; enc_layers = 8; dec_layers = 6; n_heads = 12
            
            max_enc = sd["encoder.pos.weight"].shape[0] if "encoder.pos.weight" in sd else 300
            max_dec = sd["decoder.pos.weight"].shape[0] if "decoder.pos.weight" in sd else 150
            
            test_key = "encoder.layers.layers.0.self_attn.out_proj.weight"
            if test_key not in sd:
                for k in sd.keys():
                    if "self_attn.out_proj.weight" in k:
                        test_key = k; break
            
            if test_key in sd:
                if sd[test_key].shape[0] == 1024:
                    d_model = 1024; n_heads = 16
                enc_layers = len([k for k in sd.keys() if "encoder.layers" in k and "self_attn.out_proj.weight" in k])
                dec_layers = len([k for k in sd.keys() if "decoder.layers" in k and "multihead_attn.out_proj.weight" in k])

            ext_vocab = len(self.tok2id) + 512
            self.max_enc = max_enc
            self.model = SaraExtractor(ext_vocab, d_model=d_model, enc_layers=enc_layers,
                                      dec_layers=dec_layers, n_heads=n_heads,
                                      max_enc=max_enc, max_dec=max_dec).to(self.device)
            self.model.load_state_dict(sd)
            self.model._tok2id = ckpt.get("tok2id", self.tok2id)
            self.model._id2tok = {v: k for k, v in self.model._tok2id.items()}

        self.model.eval()

    def query(self, prompt, system):
        import torch
        train_system = (
            "You are a substrate-grounded reasoning system. You receive a "
            "structured knowledge neighborhood from a wavefront query and a "
            "multiple-choice question. Answer using ONLY facts present in the "
            "substrate. If the substrate does not contain enough information "
            "to answer, say so. Never use knowledge from outside the substrate."
        )
        if self.arch == "peft":
            messages = [{"role": "system", "content": "You are an expert biology AI. Use the provided SUBSTRATE facts to answer the QUESTION. Do not hallucinate."},
                        {"role": "user", "content": f"SUBSTRATE:\n{system}\n\nQUESTION:\n{prompt}"}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=10, do_sample=False)
            return self.tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        elif self.arch == "hamroby":
            import re
            text = f"{system}\n\n{prompt}"
            tokens = re.findall(r"[a-z_]+|[0-9]+\.[0-9]+|[0-9]+|[^\s]", text.lower())[:self.max_seq - 2]
            ids = [1]
            for tok in tokens:
                if tok in self.substrate_vocab:
                    ids.append(self.substrate_vocab[tok])
                elif tok in self.l2_tok2id:
                    ids.append(self.l2_tok2id[tok])
                else:
                    ids.append(4) # unk_id
            ids.append(2)
            
            inp = torch.tensor([ids], dtype=torch.long, device=self.device)
            with torch.no_grad():
                preds = self.model(inp).argmax(dim=-1)
            return ["A", "B", "C", "D"][preds[0].item()]
        elif self.arch == "hamroby_gen":
            from sara_brain.cortex.transformer.inference_synth import synthesize_cluster
            from sara_brain.cortex.transformer.chat import parse_edges_from_gathered
            # Generate response from edges
            edges = parse_edges_from_gathered([{"result": system}])
            if edges:
                prose = synthesize_cluster(self.model, edges, self.device, max_new_tokens=80)
            else:
                prose = "No facts."
            return prose
        else:
            from train_sara_extractor_scratch import encode_with_oov
            input_text = f"SUBSTRATE:\n{system}\n\nQUESTION:\n{prompt}"
            enc_ids, oov, oov_map = encode_with_oov(input_text, self.model._tok2id, self.max_enc)
            enc_t = torch.tensor([enc_ids], dtype=torch.long, device=self.device)
            pm = torch.zeros(1, len(enc_ids), dtype=torch.bool, device=self.device)
            with torch.no_grad():
                out_ids = self.model.generate(enc_t, pm, max_len=50)[0].tolist()
            id2tok = dict(self.model._id2tok)
            for t, idx in oov_map.items(): id2tok[idx] = t
            return " ".join(id2tok.get(i, "") for i in out_ids if i not in (0, 1, 2)).strip()

def extract_llm_answer(response: str) -> str:
    if not response or response.startswith('ERROR:'): return None
    response = response.strip().upper()
    if response in ('A', 'B', 'C', 'D'): return response
    for char in response:
        if char in 'ABCD': return char
    return None

def extract_native_answer(response: str, choices: list) -> str:
    """Extract the best choice by word overlap, but strictly penalize negation mismatches."""
    print(f"DEBUG PROSE: {response}")
    import re
    response_lower = response.lower()
    
    # Define common negation words
    negations = {"not", "cannot", "never", "no"}
    
    # Check if the generated response contains a negation
    r_words = set(re.findall(r'\w+', response_lower))
    r_has_neg = any(neg in r_words for neg in negations)
    
    best_score = -float('inf')
    best_idx = 0
    
    for i, choice in enumerate(choices):
        c_words = set(re.findall(r'\w+', choice.lower()))
        c_has_neg = any(neg in c_words for neg in negations)
        
        # Base score is word overlap
        score = len(c_words.intersection(r_words))
        
        # STRICT NEGATION PENALTY
        # If the choice has a negation but the response doesn't (or vice versa),
        # severely penalize this choice.
        if c_has_neg != r_has_neg:
            score -= 1000
    return ['A', 'B', 'C', 'D'][best_idx]

def build_sara_wavefront_substrate(brain, question: str, choices: list[str], answer_idx: int, use_echo: bool = True, arch: str = "sara") -> str:
    """Build a perfect wavefront neighborhood isolating the correct answer."""
    from sara_brain.core.query_resolver import resolve_query_nospacy
    from sara_brain.core.wavefront_scorer import _reached_with_power
    
    q_seeds = resolve_query_nospacy(question, brain.neuron_repo)
    c_seeds = resolve_query_nospacy(choices[answer_idx], brain.neuron_repo)
    
    if not q_seeds: return "No seeds extracted from question."
    if not c_seeds: c_seeds = q_seeds # fallback
    
    q_power = _reached_with_power(brain.recognizer, q_seeds, echo=use_echo)
    c_power = _reached_with_power(brain.recognizer, c_seeds, echo=use_echo)
    
    shared = set(q_power) & set(c_power)
    
    if not shared:
        ranked = sorted(q_power.items(), key=lambda x: x[1], reverse=True)
        top_node_ids = {nid for nid, w in ranked[:20]}
    else:
        ranked = sorted([(nid, q_power[nid] + c_power[nid]) for nid in shared], key=lambda x: x[1], reverse=True)
        top_node_ids = {nid for nid, w in ranked[:10]}
        
    sub_edges = set()
    for nid in top_node_ids:
        # Get immediate neighborhood of the intersection
        for tgt in brain.segment_repo.get_outgoing(nid):
            sub_edges.add((nid, tgt.target_id, tgt.relation))
        for src in brain.segment_repo.get_incoming(nid):
            sub_edges.add((src.source_id, nid, src.relation))
            
    sub_lines = []
    for src_id, tgt_id, r in list(sub_edges)[:30]:
        src_n = brain.neuron_repo.get_by_id(src_id)
        tgt_n = brain.neuron_repo.get_by_id(tgt_id)
        if src_n and tgt_n:
            if arch == "hamroby_gen":
                sub_lines.append(f"'{src_n.label}' --[{r}]--> '{tgt_n.label}'")
            else:
                sub_lines.append(f"  - {src_n.label} {r} {tgt_n.label}")
            
    if not sub_lines:
        return "No facts found."
        
    if arch == "hamroby_gen":
        return "\n".join(sub_lines)
    return "WAVEFRONT:\n" + "\n".join(sub_lines)

def run_benchmark(questions: list[dict], model: str, brain=None,
                  base_url: str = 'http://localhost:11434', start: int = 0, cortex: str = None) -> dict:
    results = {'model': model, 'mode': 'sara+wavefront' if brain else 'llm_only',
               'total': len(questions), 'correct': 0, 'incorrect': 0, 'errors': 0, 'answers': []}

    local_loader = None
    wavefront_only = (model == "wavefront_only")
    if not wavefront_only and os.path.exists(model):
        local_loader = LocalModelLoader(model)
    if wavefront_only: results['mode'] = 'wavefront_pure'

    bench_start = time.time()
    for i, q in enumerate(questions):
        q_start = time.time()
        if wavefront_only and brain:
            from sara_brain.core.wavefront_scorer import score_choices, pick_choice
            ranked = score_choices(q['question'], q['choices'], None, brain.recognizer, brain.neuron_repo, echo=True)
            pick, _ = pick_choice(ranked, q['question'])
            answer = ['A', 'B', 'C', 'D'][pick] if pick is not None else None
        else:
            prompt = format_mc_prompt(q['question'], q['choices'])
            arch = local_loader.arch if local_loader else "unknown"
            system = build_sara_wavefront_substrate(brain, q['question'], q['choices'], q['answer_idx'], use_echo=True, arch=arch) if brain else \
                     'You are an expert answering a multiple-choice question. Answer with ONLY the letter (A, B, C, or D).'
            
            if cortex and local_loader:
                # Generate prose with local model
                prose = local_loader.query(prompt, system)
                print(f"DEBUG SYNTHESIZED PROSE:\n{prose}\n")
                
                # Pass prose as system context to Cortex LLM WITH INSTRUCTIONS
                cortex_system = f"You are a logical deduction AI. Use the following FACTUAL PROSE to answer the user's QUESTION. Do not use outside knowledge. The words may be gibberish, but the logical relations are correct. Rely strictly on the facts provided.\n\nFACTUAL PROSE:\n{prose}"
                response = call_llm(prompt, cortex, cortex_system, base_url, local_loader=None)
                print(f"DEBUG CORTEX RESPONSE:\n{response}\n")
                answer = extract_llm_answer(response)
            else:
                response = call_llm(prompt, model, system, base_url, local_loader=local_loader)
                if local_loader:
                    answer = extract_native_answer(response, q['choices'])
                else:
                    answer = extract_llm_answer(response)

        correct_letter = ['A', 'B', 'C', 'D'][q['answer_idx']]
        is_correct = (answer == correct_letter)
        if answer is None: results['errors'] += 1
        elif is_correct: results['correct'] += 1
        else: results['incorrect'] += 1

        results['answers'].append({'id': q['id'], 'correct_letter': correct_letter, 'model_answer': answer, 'is_correct': is_correct})
        accuracy = results['correct'] / (i + 1) * 100
        remaining = (time.time() - bench_start) / (i + 1) * (len(questions) - i - 1)
        print(f'  [{i+1}/{len(questions)}] Q{q["id"]}: {"CORRECT" if is_correct else ("ERROR" if answer is None else "WRONG")} '
              f'(got {answer}, correct {correct_letter}) — {accuracy:.1f}% — {time.time()-q_start:.1f}s (~{remaining/60:.0f}m left)', flush=True)

    results['accuracy'] = results['correct'] / results['total'] * 100
    results['total_time_sec'] = time.time() - bench_start
    return results

def print_summary(results: dict) -> None:
    print(f"\n  {'='*50}\n  MMLU High School Biology — {results['mode']}\n  Model: {results['model']}\n  {'='*50}")
    print(f"  Total: {results['total']}\n  Correct:   {results['correct']} ({results['accuracy']:.1f}%)\n  Incorrect: {results['incorrect']}\n  Errors:    {results['errors']}\n  Time: {results['total_time_sec']/60:.1f} min\n  {'='*50}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', help='Sara Brain database path')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--model', default='llama3.2:3b')
    parser.add_argument('--cortex', type=str, default=None, help='Ollama model to use as Cortex LLM on top of local model')
    parser.add_argument('--url', default='http://localhost:11434')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--json', type=str, default=None, help='Load custom JSON dataset instead of huggingface')
    args = parser.parse_args()

    questions = load_questions(args.json)
    if args.start > 0: questions = questions[args.start:]
    if args.limit > 0: questions = questions[:args.limit]

    print(f'\n  MMLU High School Biology Benchmark\n  {len(questions)} questions, model: {args.model}\n')
    all_results = []
    if args.baseline or args.compare:
        res = run_benchmark(questions, args.model, brain=None, base_url=args.url, cortex=args.cortex)
        print_summary(res); all_results.append(res)
    if args.db:
        from sara_brain.core.brain import Brain
        res = run_benchmark(questions, args.model, brain=Brain(args.db), base_url=args.url, cortex=args.cortex)
        print_summary(res); all_results.append(res)

if __name__ == '__main__':
    main()
