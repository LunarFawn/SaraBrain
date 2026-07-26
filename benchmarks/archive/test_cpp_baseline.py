import sys
import os
import time
import ctypes
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from sara_brain.core.brain import Brain
from sara_brain.core.query_resolver import resolve_query_nospacy
from benchmarks.investigate_dampening import pick_choice, load_questions
from collections import defaultdict

def engine_score(question, choices, recognizer, neuron_repo, bidirectional=False):
    q_seeds = resolve_query_nospacy(question, neuron_repo)
    power = defaultdict(float)
    
    import ctypes
    from sara_brain.core.fast_recognizer import ResultNode
    max_res = 100000
    result_buffer = (ResultNode * max_res)()
    
    for seed in q_seeds:
        n = neuron_repo.resolve(seed.label, exact_only=True)
        if not n: continue
        
        count = recognizer._lib.engine_propagate(
            recognizer._engine, n.id, recognizer.max_depth, ctypes.c_float(0.5),
            bidirectional, result_buffer, max_res
        )
        
        for i in range(count):
            nid = int(result_buffer[i].id)
            weight = result_buffer[i].weight
            power[nid] += weight
            
    # Score choices
    ranked = []
    for idx, text in enumerate(choices):
        c_seeds = resolve_query_nospacy(text, neuron_repo)
        c_score = 0.0
        for s in c_seeds:
            n = neuron_repo.resolve(s.label, exact_only=True)
            if n and n.id in power:
                c_score += power[n.id]
        ranked.append((c_score, idx, text))
    ranked.sort(reverse=True, key=lambda x: x[0])
    return ranked

def main():
    brain = Brain("data/biology_full_v2_clean.db")
    from sara_brain.core.fast_recognizer import FastRecognizer
    brain.recognizer = FastRecognizer(brain.neuron_repo, brain.segment_repo)
    questions = load_questions(5)
    
    # 1. C++ Forward Only (bidirectional=False)
    correct = 0
    for q in questions:
        ranked = engine_score(q['question'], q['choices'], brain.recognizer, brain.neuron_repo, bidirectional=False)
        pick = ranked[0][1] if ranked else -1
        if pick == q['answer_idx']: correct += 1
    print(f"C++ Forward Only: {correct}/5")
    
    # 2. C++ Simultaneous Flood (bidirectional=True)
    correct = 0
    for q in questions:
        ranked = engine_score(q['question'], q['choices'], brain.recognizer, brain.neuron_repo, bidirectional=True)
        pick = ranked[0][1] if ranked else -1
        if pick == q['answer_idx']: correct += 1
    print(f"C++ Simultaneous Flood: {correct}/5")

if __name__ == "__main__":
    main()
