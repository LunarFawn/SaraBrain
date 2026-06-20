#!/usr/bin/env python3
"""Investigate the impact of Hub Dampening and Logarithmic Scoring.

Runs the MMLU Biology benchmark using the Pure Wavefront scorer (C++ engine)
across different permutations of the dampening logic to see what actually
drives accuracy.

Permutations tested:
1. Hub Penalty: ON vs OFF (does penalizing highly-connected nodes help?)
2. Log Scoring: ON vs OFF (does math.log1p dampening during score summation help?)
3. Traversal: Echo vs Basic Bidirectional
"""

import argparse
import time
from collections import defaultdict
from datasets import load_dataset
from sara_brain.core.brain import Brain
from sara_brain.core.fast_recognizer import FastRecognizer
from sara_brain.core.wavefront_scorer import score_choices, pick_choice
import sara_brain.core.wavefront_scorer as wfs

def load_questions(limit=0):
    ds = load_dataset('cais/mmlu', 'high_school_biology', split='test')
    qs = [{'id': i, 'question': q['question'], 'choices': q['choices'], 'answer_idx': q['answer']}
          for i, q in enumerate(ds)]
    if limit > 0: qs = qs[:limit]
    return qs

# We will monkeypatch _reached_with_power to toggle Hub Penalty
original_reached_with_power = wfs._reached_with_power

def patched_reached_with_power(recognizer, seeds, echo=False, use_hub_penalty=True):
    if echo:
        from sara_brain.core.short_term import ShortTerm
        st = ShortTerm(event_id=f"score-{time.time()}", event_type="score")
        seed_labels = [s.label for s in seeds]
        recognizer.propagate_echo(seed_labels, st, max_rounds=2)
        
        power = defaultdict(float)
        for nid, weight in st.convergence_map.items():
            if use_hub_penalty:
                out_count = len(recognizer.segment_repo.get_outgoing(nid))
                in_count = len(recognizer.segment_repo.get_incoming(nid))
                connectivity = out_count + in_count
                h_weight = 1.0 / (connectivity + 1)
                power[nid] = weight * h_weight
            else:
                power[nid] = weight
        return dict(power)
    else:
        power = defaultdict(float)
        import ctypes
        from sara_brain.core.fast_recognizer import ResultNode
        max_res = 50000
        result_buffer = (ResultNode * max_res)()
        for seed in seeds:
            n = recognizer.neuron_repo.resolve(seed.label, exact_only=True)
            if n is None: continue
            count = recognizer._lib.engine_propagate(
                recognizer._engine, n.id, recognizer.max_depth, ctypes.c_float(0.5),
                True, result_buffer, max_res
            )
            targets = [result_buffer[i].id for i in range(count) if result_buffer[i].id != n.id]
            nodes_to_power = [n.id] + targets
            total_witnesses = len(nodes_to_power)
            base_power_per_witness = seed.power / total_witnesses
            
            for nid in nodes_to_power:
                if use_hub_penalty:
                    connectivity = CONNECTIVITY_CACHE.get(nid, 0)
                    weight = 1.0 / (connectivity + 1)
                    power[nid] += base_power_per_witness * weight
                else:
                    power[nid] += base_power_per_witness
        return dict(power)

def run_experiment(name, questions, brain, echo, use_hub_penalty, use_log_scoring):
    # Apply monkeypatch
    wfs._reached_with_power = lambda rec, s, echo=False: patched_reached_with_power(rec, s, echo, use_hub_penalty)
    
    correct = 0
    errors = 0
    t0 = time.time()
    
    for q in questions:
        try:
            print(f"DEBUG: Starting Q {q['id']}", flush=True)
            ranked = score_choices(q['question'], q['choices'], None, brain.recognizer, brain.neuron_repo, 
                                   echo=echo, dampened=use_log_scoring)
            print(f"DEBUG: Finished score_choices Q {q['id']}", flush=True)
            pick, _ = pick_choice(ranked, q['question'])
            if pick == q['answer_idx']:
                correct += 1
            elif pick is None:
                errors += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error: {e}")
            errors += 1
            break
            
    elapsed = time.time() - t0
    acc = correct / len(questions) * 100
    
    print(f"| {name:<20} | {str(echo):<5} | {str(use_hub_penalty):<11} | {str(use_log_scoring):<11} | {acc:>6.1f}% | {errors:>6} | {elapsed:>5.1f}s |")
    
    # Restore original
    wfs._reached_with_power = original_reached_with_power
    return acc

CONNECTIVITY_CACHE = {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/biology_full_v2_clean.db")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    print(f"\nLoading brain {args.db} ...")
    brain = Brain(args.db)
    
    print("Precomputing connectivity cache...", flush=True)
    rows = brain.segment_repo.conn.execute("SELECT source_id, COUNT(*) FROM segments GROUP BY source_id").fetchall()
    for row in rows: CONNECTIVITY_CACHE[row[0]] = CONNECTIVITY_CACHE.get(row[0], 0) + row[1]
    rows = brain.segment_repo.conn.execute("SELECT target_id, COUNT(*) FROM segments GROUP BY target_id").fetchall()
    for row in rows: CONNECTIVITY_CACHE[row[0]] = CONNECTIVITY_CACHE.get(row[0], 0) + row[1]

    brain.recognizer = FastRecognizer(brain.neuron_repo, brain.segment_repo)
    qs = load_questions(args.limit)
    print(f"Testing on {len(qs)} questions with C++ FastRecognizer.\n")
    
    print(f"| {'Experiment':<20} | {'Echo':<5} | {'Hub Penalty':<11} | {'Log Scoring':<11} | {'Acc':>7} | {'Errors':>6} | {'Time':>6} |")
    print(f"|{'-'*22}|{'-'*7}|{'-'*13}|{'-'*13}|{'-'*8}|{'-'*8}|{'-'*8}|")
    
    experiments = [
        ("Forward Base",       False, True,  False),
        ("Forward + Log",      False, True,  True),
        ("Forward No Hub",     False, False, False),
        ("Forward NoHub+Log",  False, False, True),
    ]
    
    for name, e, h, l in experiments:
        run_experiment(name, qs, brain, echo=e, use_hub_penalty=h, use_log_scoring=l)
        
    print()

if __name__ == "__main__":
    main()
