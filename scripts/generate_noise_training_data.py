#!/usr/bin/env python3
"""Generate training data for substrate-aware readers.

This script runs wavefronts for MMLU questions and saves the 
raw activation map (the noise) as a training feature.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add project src to path
sys.path.insert(0, os.path.abspath("src"))
from sara_brain.core.brain import Brain
from datasets import load_dataset

def generate_noise_data(db_path, output_path, limit=0):
    print(f"Loading brain from {db_path}...")
    # Add benchmarks to path to find build_sara_wavefront_substrate
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
    print(f"Generating noise features for {count} questions...")

    for i in range(count):
        q = ds[i]
        question_text = q['question']
        choices = q['choices']
        labels = ["A", "B", "C", "D"]
        full_q = question_text + "\n" + "\n".join([f"{labels[j]}. {choices[j]}" for j in range(4)])
        correct_answer = labels[q['answer']]
        
        # Get the "Noise" (Echo wavefront substrate)
        # Using depth 3 echo for maximum signal density
        substrate = build_sara_wavefront_substrate(brain, question_text, use_echo=True)
        
        examples.append({
            "instruction": "Answer the multiple-choice question based on the provided wavefront substrate.",
            "substrate": substrate,
            "question": full_q,
            "answer": correct_answer
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
    ap.add_argument("--db", default="data/biology_full_v2_clean.db")
    ap.add_argument("--out", default="training_data/biology_noise_310.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    
    os.makedirs("training_data", exist_ok=True)
    generate_noise_data(args.db, args.out, args.limit)
