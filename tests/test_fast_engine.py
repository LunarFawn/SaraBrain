import time
import sys
import os
from sara_brain.core.brain import Brain
from sara_brain.core.short_term import ShortTerm
from sara_brain.core.recognizer import Recognizer
from sara_brain.core.fast_recognizer import FastRecognizer

def test_speed():
    db_path = "data/biology_full_v2_clean.db"
    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found.")
        return

    brain = Brain(db_path)
    
    seeds = ["mitosis", "cell cycle", "eukaryotic", "chromosome"]
    
    print(f"\n--- Testing Propagation Speed (Brain: {db_path}) ---")
    
    # Python Recognizer
    py_rec = Recognizer(brain.neuron_repo, brain.segment_repo, max_depth=3)
    st_py = ShortTerm(event_id="test-py", event_type="test")
    
    t0 = time.time()
    py_rec.propagate_into(seeds, st_py)
    py_time = time.time() - t0
    print(f"Python (Depth 3): {py_time:.4f}s")
    
    # Fast Recognizer (using the one already in brain if possible)
    if isinstance(brain.recognizer, FastRecognizer):
        fast_rec = brain.recognizer
    else:
        fast_rec = FastRecognizer(brain.neuron_repo, brain.segment_repo)
    
    # Ensure depth is same
    fast_rec.max_depth = 3
    st_fast = ShortTerm(event_id="test-fast", event_type="test")
    
    t0 = time.time()
    fast_rec.propagate_into(seeds, st_fast)
    fast_time = time.time() - t0
    print(f"C++ (Depth 3):    {fast_time:.4f}s")
    
    speedup = py_time / fast_time if fast_time > 0 else 0
    print(f"Speedup:          {speedup:.1f}x")
    
    # Validate results
    py_conv = st_py.convergence_map
    fast_conv = st_fast.convergence_map
    print(f"\nPython reached {len(py_conv)} neurons.")
    print(f"C++ reached    {len(fast_conv)} neurons.")
    
    common = set(py_conv.keys()) & set(fast_conv.keys())
    print(f"Intersection:  {len(common)} neurons.")

    print(f"\n--- Testing Echo Mode Speed ---")
    st_py_echo = ShortTerm(event_id="test-py-echo", event_type="test")
    t0 = time.time()
    py_rec.propagate_echo(seeds, st_py_echo)
    py_echo_time = time.time() - t0
    print(f"Python (Echo): {py_echo_time:.4f}s")

    st_fast_echo = ShortTerm(event_id="test-fast-echo", event_type="test")
    t0 = time.time()
    fast_rec.propagate_echo(seeds, st_fast_echo)
    fast_echo_time = time.time() - t0
    print(f"C++ (Echo):    {fast_echo_time:.4f}s")
    
    speedup_echo = py_echo_time / fast_echo_time if fast_echo_time > 0 else 0
    print(f"Speedup:       {speedup_echo:.1f}x")

if __name__ == "__main__":
    test_speed()
