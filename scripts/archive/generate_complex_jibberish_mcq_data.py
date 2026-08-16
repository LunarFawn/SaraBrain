#!/usr/bin/env python3
"""Generate 500k complex MCQ jibberish wavefronts from the DB in-memory."""
import sqlite3
import json
import random
import time
from collections import defaultdict
import os

def generate():
    print("Loading DB into memory...")
    conn = sqlite3.connect("data/jibberish_biology_full.db")
    c = conn.cursor()
    c.execute("SELECT id, label FROM neurons")
    nodes = {r[0]: r[1] for r in c.fetchall()}
    c.execute("SELECT source_id, target_id, relation FROM segments")
    edges = c.fetchall()
    
    adj = defaultdict(list)
    for s, t, r in edges:
        adj[s].append((t, r))
    
    valid_edges = [(s,t,r) for s,t,r in edges if s in nodes and t in nodes]
    node_ids = list(nodes.keys())
    
    print(f"Loaded {len(nodes)} nodes, {len(edges)} edges.")
    
    examples = []
    t0 = time.time()
    
    QUESTION_FORMS = [
        "What does {s} {r}?",
        "What {r} {o}?",
        "What is {s} related to?",
        "Which of the following does {s} {r}?",
        "What is associated with {s}?",
    ]
    
    print("Generating 500k examples...")
    for i in range(500000):
        # Pick random target fact
        s_id, t_id, rel = random.choice(valid_edges)
        target_s = nodes[s_id]
        target_o = nodes[t_id]
        
        # Build fast BFS substrate (depth 2)
        sub_edges = set()
        queue = [s_id]
        for depth in range(2):
            next_q = []
            for n in queue:
                for nxt, r in adj[n]:
                    if (n, nxt, r) not in sub_edges:
                        sub_edges.add((n, nxt, r))
                        next_q.append(nxt)
                        if len(sub_edges) > 50: break
                if len(sub_edges) > 50: break
            queue = next_q
            if len(sub_edges) > 50: break
            
        # Guarantee target edge is in substrate
        sub_edges.add((s_id, t_id, rel))
        
        # Render substrate
        sub_lines = []
        for src, tgt, r in list(sub_edges)[:50]: # cap at 50 to match real wavefront limits
            if src in nodes and tgt in nodes:
                sub_lines.append(f"  - {nodes[src]} {r} {nodes[tgt]}")
        
        random.shuffle(sub_lines)
        substrate = "WAVEFRONT:\n" + "\n".join(sub_lines)
        
        # Build question
        q_text = random.choice(QUESTION_FORMS).format(s=target_s, r=rel, o=target_o)
        
        # Build choices
        n_choices = 4
        choices = [target_o]
        while len(choices) < n_choices:
            distractor = nodes[random.choice(node_ids)]
            if distractor not in choices:
                choices.append(distractor)
        
        random.shuffle(choices)
        correct_idx = choices.index(target_o)
        labels = ["A", "B", "C", "D"]
        correct_letter = labels[correct_idx]
        
        full_q = q_text + "\n" + "\n".join([f"{labels[j]}. {choices[j]}" for j in range(4)])
        
        # Build PROSE answer to force grammar and OOV copying
        prose_answer = f"Based on the substrate, {target_s} {rel} {target_o}. Therefore, the correct choice is {correct_letter}."
        
        examples.append({
            "substrate": substrate,
            "question": full_q,
            "answer": prose_answer
        })
        
        if (i+1) % 100000 == 0:
            print(f"  {i+1}/500000 ({(time.time()-t0):.1f}s)")
            
    out_path = "training_data/complex_jibberish_mcq_500k.jsonl"
    print(f"Saving to {out_path}...")
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
            
    print("Done!")

if __name__ == "__main__":
    os.makedirs("training_data", exist_ok=True)
    generate()
