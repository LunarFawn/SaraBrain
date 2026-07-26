#!/usr/bin/env python3
"""Generate training data for substrate-aware readers using Jibberish cipher."""

import os
import sys
import json
import time
from pathlib import Path

# Add project src to path
sys.path.insert(0, os.path.abspath("src"))
from datasets import load_dataset

sys.path.insert(0, os.path.abspath("scripts"))
from consistent_cipher import ConsistentCipher

def generate_jibberish_noise_data(db_path, cipher_path, output_path, limit=0):
    print(f"Loading cipher from {cipher_path}...")
    cipher = ConsistentCipher()
    cipher.load_cipher(cipher_path)

    print(f"Loading brain from {db_path}...")
    sys.path.insert(0, os.path.abspath("benchmarks"))
    from run_mmlu_biology import build_sara_wavefront_substrate

    # Manually activate C++ engine in Brain before loading
    from sara_brain.core.brain import Brain
    from sara_brain.core.fast_recognizer import FastRecognizer
    brain = Brain(db_path)
    if not isinstance(brain.recognizer, FastRecognizer):
        print("Activating C++ FastRecognizer...")
        brain.recognizer = FastRecognizer(brain.neuron_repo, brain.segment_repo)

    print("Loading MMLU Biology dataset...")
    ds = load_dataset('cais/mmlu', 'high_school_biology', split='test')
    
    examples = []
    t0 = time.time()
    
    count = len(ds) if limit == 0 else limit
    print(f"Generating jibberish noise features for {count} questions...")

    for i in range(count):
        q = ds[i]
        question_text = q['question']
        choices = q['choices']
        labels = ["A", "B", "C", "D"]
        
        # Translate to jibberish
        jibberish_q = cipher.translate_text(question_text)
        jibberish_choices = [cipher.translate_text(c) for c in choices]
        
        full_q = jibberish_q + "\n" + "\n".join([f"{labels[j]}. {jibberish_choices[j]}" for j in range(4)])
        correct_answer = labels[q['answer']]
        
        # Get the "Noise" (Echo wavefront substrate) using the jibberish question
        substrate = build_sara_wavefront_substrate(brain, jibberish_q, use_echo=True)
        
        examples.append({
            "instruction": "Answer the multiple-choice question based on the provided wavefront substrate.",
            "substrate": substrate,
            "question": full_q,
            "answer": correct_answer,
            "original_question": question_text
        })
        
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{count}] done... ({elapsed:.1f}s)")

    print(f"Saving {len(examples)} examples to {output_path}...")
    with open(output_path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
            
    print(f"Done in {time.time()-t0:.1f}s.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/jibberish_biology_full.db")
    ap.add_argument("--cipher", default="data/biology_cipher.json")
    ap.add_argument("--out", default="training_data/jibberish_noise_310.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    
    os.makedirs("training_data", exist_ok=True)
    generate_jibberish_noise_data(args.db, args.cipher, args.out, args.limit)
