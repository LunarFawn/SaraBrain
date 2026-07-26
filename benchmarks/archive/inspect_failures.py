import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from sara_brain.core.brain import Brain
from benchmarks.investigate_dampening import load_questions
from benchmarks.run_mmlu_biology import build_sara_wavefront_substrate, format_mc_prompt, call_llm

def inspect_question(brain, q_dict, base_url="http://localhost:11434"):
    print(f"\n{'='*60}")
    print(f"QUESTION ID: {q_dict['id']}")
    print(f"QUESTION: {q_dict['question']}")
    for i, c in enumerate(q_dict['choices']):
        print(f"  {['A','B','C','D'][i]}. {c}")
    print(f"CORRECT ANSWER: {['A','B','C','D'][q_dict['answer_idx']]}")
    print(f"{'-'*60}")
    
    print("Generating Substrate via True Backwave...")
    system = build_sara_wavefront_substrate(brain, q_dict['question'], use_echo=True)
    prompt = format_mc_prompt(q_dict['question'], q_dict['choices'])
    
    print(f"SUBSTRATE SENT TO LLM:\n{system}")
    print(f"{'-'*60}")
    
    print("Asking Llama 3.2 3B...")
    response = call_llm(prompt, "llama3.2:3b", system, base_url)
    print(f"LLM RESPONSE: {response}")
    print(f"{'='*60}\n")

def main():
    brain = Brain("data/biology_full_v2_clean.db")
    questions = load_questions(5)
    
    # The user asked about "2, 4, and 5", which correspond to indices 1, 3, and 4
    for idx in [1, 3, 4]:
        inspect_question(brain, questions[idx])

if __name__ == "__main__":
    main()
