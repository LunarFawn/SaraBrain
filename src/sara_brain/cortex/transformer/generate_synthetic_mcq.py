import json
import random
from pathlib import Path
import argparse
from tqdm import tqdm
from sara_brain.core.brain import Brain
from benchmarks.run_mmlu_biology import build_sara_wavefront_substrate

def generate(brain_path: str, out_path: str, num_questions: int = 50000):
    brain = Brain(brain_path)
    all_neurons = [n for n in brain.neuron_repo.list_all() if n.label]
    
    # Pre-collect all relations and targets for fast sampling
    all_relations = list(set([seg.relation for seg in brain.segment_repo.list_all()]))
    
    pairs = []
    
    print("Generating synthetic multiple-choice questions...")
    for _ in tqdm(range(num_questions)):
        # 1. Pick a random source node that has edges
        src = random.choice(all_neurons)
        edges = brain.segment_repo.get_outgoing(src.id)
        if not edges:
            continue
            
        # 2. Pick a true fact
        true_edge = random.choice(edges)
        true_tgt = brain.neuron_repo.get_by_id(true_edge.target_id)
        if not true_tgt:
            continue
            
        # Clean labels
        src_label = src.label.replace('_attribute', '')
        tgt_label = true_tgt.label.replace('_attribute', '')
        rel = true_edge.relation.replace('_', ' ')
        
        # 3. Create question
        q_type = random.choice(["relation", "true_false", "about"])
        
        if q_type == "relation":
            question = f"What is the relationship between '{src_label}' and '{tgt_label}'?"
            correct = f"'{src_label}' {rel} '{tgt_label}'"
            
            # Distractors
            distractors = []
            for _ in range(3):
                bad_rel = random.choice(all_relations).replace('_', ' ')
                if bad_rel == rel: bad_rel = "does not " + rel
                distractors.append(f"'{src_label}' {bad_rel} '{tgt_label}'")
                
        elif q_type == "true_false":
            question = f"Which of the following is TRUE?"
            correct = f"'{src_label}' {rel} '{tgt_label}'"
            
            # Distractors
            distractors = []
            # Negation trap
            distractors.append(f"'{src_label}' does not {rel} '{tgt_label}'")
            # Random other targets
            for _ in range(2):
                rand_tgt = random.choice(all_neurons).label.replace('_attribute', '')
                distractors.append(f"'{src_label}' {rel} '{rand_tgt}'")
                
        else:
            question = f"Regarding '{src_label}', which statement is correct?"
            correct = f"It {rel} '{tgt_label}'"
            
            distractors = []
            distractors.append(f"It does not {rel} '{tgt_label}'")
            for _ in range(2):
                rand_rel = random.choice(all_relations).replace('_', ' ')
                distractors.append(f"It {rand_rel} '{tgt_label}'")
        
        choices = [correct] + distractors
        random.shuffle(choices)
        correct_idx = choices.index(correct)
        
        # Get substrate
        # Using the same wavefront logic as benchmark
        wavefront = build_sara_wavefront_substrate(brain, src_label, use_echo=True, arch="hamroby")
        
        prompt = f"{question}\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}"
        
        pairs.append({
            "system": wavefront,
            "prompt": prompt,
            "answer": correct_idx
        })
        
    print(f"Generated {len(pairs)} questions.")
    with open(out_path, 'w') as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--brain", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--n", type=int, default=50000)
    args = p.parse_args()
    generate(args.brain, args.out, args.n)
