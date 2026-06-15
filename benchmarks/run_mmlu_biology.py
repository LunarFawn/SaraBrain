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

def load_questions() -> list[dict]:
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

            ext_vocab = len(self.tok2id) + 300
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
            messages = [{"role": "system", "content": train_system},
                        {"role": "user", "content": f"SUBSTRATE:\n{system}\n\nQUESTION:\n{prompt}"}]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=10, do_sample=False)
            return self.tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        else:
            from train_sara_extractor_scratch import encode_with_oov
            input_text = f"SUBSTRATE:\n{system}\n\nQUESTION:\n{prompt}"
            enc_ids, oov, oov_map = encode_with_oov(input_text, self.model._tok2id, self.max_enc)
            enc_t = torch.tensor([enc_ids], dtype=torch.long, device=self.device)
            pm = torch.zeros(1, len(enc_ids), dtype=torch.bool, device=self.device)
            with torch.no_grad():
                out_ids = self.model.generate(enc_t, pm, max_len=10)[0].tolist()
            id2tok = dict(self.model._id2tok)
            for t, idx in oov_map.items(): id2tok[idx] = t
            return "".join(id2tok.get(i, "") for i in out_ids if i not in (0, 1, 2)).strip()

def extract_answer(response: str) -> str | None:
    if not response or response.startswith('ERROR:'): return None
    response = response.strip().upper()
    if response in ('A', 'B', 'C', 'D'): return response
    for char in response:
        if char in 'ABCD': return char
    return None

def build_sara_wavefront_substrate(brain, question: str, use_echo: bool = True) -> str:
    """Build a structured wavefront neighborhood (noisy signal) for the cortex."""
    import re
    from sara_brain.cortex.cleanup import STOPWORD_SUBJECTS
    from sara_brain.core.wavefront_scorer import _reached_with_power
    from sara_brain.core.query_resolver import ResolvedSeed

    words = re.findall(r"[a-z][a-z']+", question.lower())
    seeds = [w for w in words if len(w) > 3 and w not in STOPWORD_SUBJECTS]
    if not seeds: return "No seeds extracted from question."

    q_seeds = [ResolvedSeed(label=s, power=1, is_compound=False) for s in seeds]
    power_map = _reached_with_power(brain.recognizer, q_seeds, echo=use_echo)

    resolved = []
    for nid, weight in power_map.items():
        n = brain.neuron_repo.get_by_id(nid)
        if n: resolved.append((n.label, weight))
    resolved.sort(key=lambda x: -x[1])

    lines = [f"Wavefront from {len(seeds)} seed(s) {seeds}: {len(power_map)} neuron(s) activated.",
             "", "Reached (full convergence map, top 30 of {len(resolved)}):"]
    for label, w in resolved[:30]:
        lines.append(f"  - '{label}' (strength={w:.2f})")
    return "\n".join(lines)

def run_benchmark(questions: list[dict], model: str, brain=None,
                  base_url: str = 'http://localhost:11434', start: int = 0) -> dict:
    results = {'model': model, 'mode': 'sara+wavefront' if brain else 'llm_only',
               'total': len(questions), 'correct': 0, 'incorrect': 0, 'errors': 0, 'answers': []}

    local_loader = None
    wavefront_only = (model == "wavefront_only")
    if not wavefront_only and os.path.isdir(model):
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
            system = build_sara_wavefront_substrate(brain, q['question'], use_echo=True) if brain else \
                     'You are an expert answering a multiple-choice question. Answer with ONLY the letter (A, B, C, or D).'
            response = call_llm(prompt, model, system, base_url, local_loader=local_loader)
            answer = extract_answer(response)

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
    parser.add_argument('--url', default='http://localhost:11434')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--start', type=int, default=0)
    args = parser.parse_args()

    questions = load_questions()
    if args.start > 0: questions = questions[args.start:]
    if args.limit > 0: questions = questions[:args.limit]

    print(f'\n  MMLU High School Biology Benchmark\n  {len(questions)} questions, model: {args.model}\n')
    all_results = []
    if args.baseline or args.compare:
        res = run_benchmark(questions, args.model, brain=None, base_url=args.url)
        print_summary(res); all_results.append(res)
    if args.db:
        from sara_brain.core.brain import Brain
        res = run_benchmark(questions, args.model, brain=Brain(args.db), base_url=args.url)
        print_summary(res); all_results.append(res)

if __name__ == '__main__':
    main()
