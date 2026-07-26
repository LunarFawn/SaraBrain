import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from sara_brain.core.brain import Brain
from sara_brain.core.wavefront_scorer import score_choices
from benchmarks.investigate_dampening import pick_choice, load_questions

def main():
    brain = Brain("data/biology_full_v2_clean.db")
    questions = load_questions(5)
    print(f"Testing 5 questions. Comparing Old Flooding vs True Backwave (Python Best-Weight)\n")
    
    # 1. Old Single-Round Flood (Already ran, scored 0/5)
    # correct = 0
    # t0 = time.time()
    # for q in questions:
    #     print(f"Processing Old Flood Q {q['id']}...", flush=True)
    #     ranked = score_choices(q['question'], q['choices'], None, brain.recognizer, brain.neuron_repo, echo=False, dampened=False)
    #     pick, _ = pick_choice(ranked, q['question'])
    #     if pick == q['answer_idx']: correct += 1
    # t1 = time.time()
    # print(f"Old Flood: {correct}/5 ({(correct/5)*100:.1f}%) in {t1-t0:.1f}s\n")
    
    # 2. True Backwave
    correct = 0
    t0 = time.time()
    for q in questions:
        print(f"Processing True Backwave Q {q['id']}...", flush=True)
        ranked = score_choices(q['question'], q['choices'], None, brain.recognizer, brain.neuron_repo, echo=True, dampened=False)
        pick, _ = pick_choice(ranked, q['question'])
        if pick == q['answer_idx']: correct += 1
    t1 = time.time()
    print(f"True Backwave: {correct}/5 ({(correct/5)*100:.1f}%) in {t1-t0:.1f}s\n")

if __name__ == "__main__":
    main()
