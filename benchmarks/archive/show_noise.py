import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from sara_brain.core.brain import Brain
from benchmarks.investigate_dampening import load_questions
from sara_brain.core.query_resolver import resolve_query_nospacy
from sara_brain.core.wavefront_scorer import _reached_with_power

def print_activations(question, brain, echo_mode):
    print(f"\n=============================================")
    print(f"Mode: {'True Backwave' if echo_mode else 'Old Flood (Simultaneous Bidirectional)'}")
    print(f"Question: {question}")
    print(f"=============================================")
    
    q_seeds = resolve_query_nospacy(question, brain.neuron_repo)
    power_dict = _reached_with_power(brain.recognizer, q_seeds, echo=echo_mode)
    
    # Sort by weight
    sorted_acts = sorted(power_dict.items(), key=lambda x: x[1], reverse=True)[:20]
    for nid, weight in sorted_acts:
        n = brain.neuron_repo.get_by_id(nid)
        print(f"  {n.label.ljust(30)} (Power: {weight:.4f})")

def main():
    brain = Brain("data/biology_full_v2_clean.db")
    questions = load_questions(2)
    q = questions[1]['question'] # Use the second question to get a fresh look
    
    print_activations(q, brain, echo_mode=False)
    print_activations(q, brain, echo_mode=True)

if __name__ == "__main__":
    main()
