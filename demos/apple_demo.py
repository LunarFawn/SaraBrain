#!/usr/bin/env python3
"""Apple vs Circle demo — walkthrough of Sara Brain's path-of-thought model."""

from sara_brain.core.brain import Brain
from sara_brain.visualization.text_tree import render_paths_from, render_graph_dot


def main():
    print("=" * 60)
    print("  Sara Brain — Path of Thought Demo")
    print("=" * 60)

    brain = Brain(":memory:")

    # --- Teaching Phase ---
    print("\n--- Teaching Phase ---\n")

    teachings = [
        "apples are red",
        "apples are round",
        "apples are sweet",
        "circles are round",
        "bananas are yellow",
        "bananas are sweet",
        "lemons are yellow",
        "lemons are sour",
    ]

    for statement in teachings:
        result = brain.teach(statement)
        print(f'  teach "{statement}"')
        print(f"    → {result.path_label} ({result.neurons_created} new neurons, {result.segments_created} new segments)")

    # --- Stats ---
    print("\n--- Brain Stats ---\n")
    stats = brain.stats()
    print(f"  Neurons:  {stats['neurons']}")
    print(f"  Segments: {stats['segments']}")
    print(f"  Paths:    {stats['paths']}")

    # --- Recognition Phase ---
    print("\n--- Recognition: 'red, round' ---\n")
    results = brain.recognize("red, round")
    for i, r in enumerate(results, 1):
        print(f"  #{i} {r.neuron.label} ({r.confidence} converging paths)")
        for trace in r.converging_paths:
            print(f"      {trace}")

    print("\n--- Recognition: 'yellow, sweet' ---\n")
    results = brain.recognize("yellow, sweet")
    for i, r in enumerate(results, 1):
        print(f"  #{i} {r.neuron.label} ({r.confidence} converging paths)")
        for trace in r.converging_paths:
            print(f"      {trace}")

    print("\n--- Recognition: 'yellow, sour' ---\n")
    results = brain.recognize("yellow, sour")
    for i, r in enumerate(results, 1):
        print(f"  #{i} {r.neuron.label} ({r.confidence} converging paths)")
        for trace in r.converging_paths:
            print(f"      {trace}")

    # --- Why ---
    print('\n--- Why "apple"? ---\n')
    traces = brain.why("apple")
    for i, trace in enumerate(traces, 1):
        src = f' (from: "{trace.source_text}")' if trace.source_text else ""
        print(f"  {i}. {trace}{src}")

    # --- Trace ---
    print('\n--- Trace from "red" ---\n')
    traces = brain.trace("red")
    for trace in traces:
        print(f"  {trace}")

    # --- Path Tree ---
    print('\n--- Path Tree from "round" ---\n')
    print(render_paths_from(brain, "round"))

    # --- Similarity ---
    print("\n--- Similarity Analysis ---\n")
    links = brain.analyze_similarity()
    for link in links:
        print(f"  {link.neuron_a_label} ↔ {link.neuron_b_label} (shared: {link.shared_paths}, overlap: {link.overlap_ratio:.1%})")

    print("\n--- Similar to 'red' ---\n")
    similar = brain.get_similar("red")
    for s in similar:
        print(f"  {s.neuron_b_label} (shared: {s.shared_paths}, overlap: {s.overlap_ratio:.1%})")

    # --- Graphviz export ---
    print("\n--- Graphviz DOT (first 10 lines) ---\n")
    dot = render_graph_dot(brain)
    for line in dot.split("\n")[:10]:
        print(f"  {line}")
    print("  ...")

    brain.close()
    print("\n" + "=" * 60)
    print("  Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
