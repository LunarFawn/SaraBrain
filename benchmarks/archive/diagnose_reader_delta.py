#!/usr/bin/env python3
"""Diagnose weight deltas between Python and C++ BFS propagation.

Runs a single MMLU question and compares the per-seed propagation results
at the individual neuron level to show exactly where and why weights differ.
"""

from __future__ import annotations

import re
import sys
import time
from collections import defaultdict


def main():
    db_path = sys.argv[1] if len(sys.argv) > 1 else "data/biology_full_v2_clean.db"
    q_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    # Load one question
    from datasets import load_dataset
    ds = load_dataset('cais/mmlu', 'high_school_biology', split='test')
    q = ds[q_idx]
    print(f"\n  Question {q_idx}: {q['question'][:100]}...")
    print(f"  Choices: {q['choices']}")
    print(f"  Correct: {chr(65 + q['answer'])}\n")

    # Load brain
    from sara_brain.core.brain import Brain
    from sara_brain.core.recognizer import Recognizer
    from sara_brain.core.fast_recognizer import FastRecognizer
    from sara_brain.core.short_term import ShortTerm
    from sara_brain.cortex.cleanup import STOPWORD_SUBJECTS

    brain = Brain(db_path)
    stats = brain.stats()
    print(f"  Brain: {stats['neurons']:,} neurons, {stats['segments']:,} segments\n")

    py_rec = Recognizer(brain.neuron_repo, brain.segment_repo, max_depth=3, min_strength=0.5)
    cpp_rec = FastRecognizer(brain.neuron_repo, brain.segment_repo, max_depth=3, min_strength=0.5)

    # Extract concepts (same as benchmark)
    stopwords = set(STOPWORD_SUBJECTS) | {
        "following", "most", "many", "some", "each", "every", "both",
        "which", "what", "when", "where", "would", "could", "should",
        "about", "above", "below", "these", "those", "between",
        "question", "answer", "choice", "correct", "example",
    }
    words = re.findall(r"[a-z][a-z'-]+", q['question'].lower())
    concepts = [w for w in words if len(w) >= 4 and w not in stopwords]
    print(f"  Extracted concepts: {concepts}\n")

    # Resolve to neurons
    seeds = []
    for label in concepts:
        n = brain.neuron_repo.resolve(label.strip().lower(), exact_only=True)
        if n is not None:
            seeds.append(n)
            print(f"    Resolved: '{label}' → neuron {n.id} ('{n.label}')")
        else:
            print(f"    NOT found: '{label}'")

    if not seeds:
        print("  No seeds resolved. Exiting.")
        return

    print(f"\n  {len(seeds)} seeds resolved. Comparing per-seed propagation...\n")
    print(f"  {'='*80}")

    # -----------------------------------------------------------------------
    # Compare per-seed propagation
    # -----------------------------------------------------------------------
    import ctypes
    from sara_brain.core.fast_recognizer import ResultNode

    total_py_only = 0
    total_cpp_only = 0
    total_weight_diffs = 0
    total_shared = 0
    all_weight_diffs = []

    for seed in seeds:
        print(f"\n  Seed: '{seed.label}' (id={seed.id})")
        print(f"  {'-'*60}")

        # Python BFS
        t0 = time.time()
        py_reached = py_rec._propagate(seed, min_strength=0.5)
        py_time = time.time() - t0

        # Build Python weight map: {target_id: best_avg_weight}
        py_weights = {}
        for target_id, path_lists in py_reached.items():
            if target_id == seed.id:
                continue
            best_weight = max(py_rec._path_weight(p) for p in path_lists)
            py_weights[target_id] = best_weight

        # C++ BFS
        max_res = 50000
        result_buffer = (ResultNode * max_res)()
        t0 = time.time()
        count = cpp_rec._lib.engine_propagate(
            cpp_rec._engine, seed.id, 3, ctypes.c_float(0.5),
            False, result_buffer, max_res
        )
        cpp_time = time.time() - t0

        cpp_weights = {}
        for i in range(count):
            res = result_buffer[i]
            if res.id == seed.id:
                continue
            cpp_weights[res.id] = res.weight

        # Compare
        py_keys = set(py_weights.keys())
        cpp_keys = set(cpp_weights.keys())
        shared = py_keys & cpp_keys
        only_py = py_keys - cpp_keys
        only_cpp = cpp_keys - py_keys

        total_py_only += len(only_py)
        total_cpp_only += len(only_cpp)
        total_shared += len(shared)

        print(f"    Python reached: {len(py_keys):,} nodes in {py_time:.3f}s")
        print(f"    C++    reached: {len(cpp_keys):,} nodes in {cpp_time:.3f}s")
        print(f"    Shared:         {len(shared):,}")
        print(f"    Only in Python: {len(only_py)}")
        print(f"    Only in C++:    {len(only_cpp)}")

        # Show nodes only in one side
        if only_py:
            print(f"\n    Nodes ONLY in Python:")
            for nid in sorted(only_py)[:10]:
                n = brain.neuron_repo.get_by_id(nid)
                label = n.label if n else "???"
                # Show the path Python took
                paths = py_reached.get(nid, [])
                path_strs = []
                for p in paths[:2]:
                    path_strs.append(" → ".join(nn.label for nn in p))
                print(f"      id={nid} '{label}' weight={py_weights[nid]:.4f} "
                      f"path: {path_strs[0] if path_strs else '?'}")
            if len(only_py) > 10:
                print(f"      ... and {len(only_py) - 10} more")

        if only_cpp:
            print(f"\n    Nodes ONLY in C++:")
            for nid in sorted(only_cpp)[:10]:
                n = brain.neuron_repo.get_by_id(nid)
                label = n.label if n else "???"
                print(f"      id={nid} '{label}' weight={cpp_weights[nid]:.4f}")
            if len(only_cpp) > 10:
                print(f"      ... and {len(only_cpp) - 10} more")

        # Weight differences on shared nodes
        weight_diffs = []
        for nid in shared:
            py_w = py_weights[nid]
            cpp_w = cpp_weights[nid]
            diff = abs(py_w - cpp_w)
            if diff > 0.0001:
                n = brain.neuron_repo.get_by_id(nid)
                label = n.label if n else "???"
                weight_diffs.append((nid, label, py_w, cpp_w, diff))
                all_weight_diffs.append(diff)

        total_weight_diffs += len(weight_diffs)

        if weight_diffs:
            weight_diffs.sort(key=lambda x: -x[4])
            print(f"\n    Weight differences ({len(weight_diffs)} nodes):")
            print(f"    {'ID':>8} {'Label':<30} {'Python':>10} {'C++':>10} {'Delta':>10}")
            print(f"    {'-'*8} {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
            for nid, label, py_w, cpp_w, diff in weight_diffs[:15]:
                # Also show what paths Python took to this node
                paths = py_reached.get(nid, [])
                path_len = len(paths[0]) - 1 if paths else "?"
                print(f"    {nid:>8} {label:<30} {py_w:>10.4f} {cpp_w:>10.4f} {diff:>10.4f}  (depth {path_len})")
            if len(weight_diffs) > 15:
                print(f"    ... and {len(weight_diffs) - 15} more")
        else:
            print(f"\n    No weight differences on shared nodes.")

        # Show depth distribution for this seed (Python only since it has paths)
        depth_counts = defaultdict(int)
        for target_id, path_lists in py_reached.items():
            for p in path_lists:
                depth = len(p) - 1
                depth_counts[depth] += 1
        print(f"\n    Python depth distribution: {dict(sorted(depth_counts.items()))}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n  {'='*80}")
    print(f"  SUMMARY ACROSS ALL SEEDS")
    print(f"  {'='*80}")
    print(f"  Total shared nodes:        {total_shared:,}")
    print(f"  Total only in Python:      {total_py_only}")
    print(f"  Total only in C++:         {total_cpp_only}")
    print(f"  Total weight differences:  {total_weight_diffs}")
    if all_weight_diffs:
        print(f"  Max weight delta:          {max(all_weight_diffs):.6f}")
        print(f"  Mean weight delta:         {sum(all_weight_diffs)/len(all_weight_diffs):.6f}")
        print(f"  Min weight delta:          {min(all_weight_diffs):.6f}")

    print(f"\n  ALGORITHMIC DIFFERENCE EXPLANATION:")
    print(f"  Python BFS: strict visited set — first path wins, never re-explores.")
    print(f"  C++ BFS:    best-weight tracking — re-explores if a higher-weight path")
    print(f"              is found to the same node, and continues BFS from there.")
    print(f"  {'='*80}\n")


if __name__ == "__main__":
    main()
