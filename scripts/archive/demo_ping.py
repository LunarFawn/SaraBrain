import os
from sara_brain.core.brain import Brain

def demo_ping():
    db_path = "ping_demo.db"
    if os.path.exists(db_path): os.remove(db_path)
    
    brain = Brain(db_path)
    
    print("Teaching Sara some facts...")
    brain.teach_triple("apple", "is", "round")
    brain.teach_triple("apple", "is", "red")
    brain.teach_triple("orange", "is", "round")
    brain.teach_triple("orange", "is", "orange_color")
    brain.teach_triple("ball", "is", "round")
    brain.teach_triple("ball", "is", "toy")
    
    print("\nRunning 'brain_ping' on ['apple', 'orange', 'ball']...")
    # Simulate what brain_ping does in MCP
    with brain.short_term("demo_ping") as st:
        brain.propagate_echo(["apple", "orange", "ball"], st, max_rounds=2)
        intersections = st.intersections(min_sources=2)
        
        if not intersections:
            print("No intersections found.")
        else:
            print("\nFound convergence points:")
            for nid, weight, count in intersections:
                neuron = brain.neuron_repo.get_by_id(nid)
                if neuron:
                    print(f"  - {neuron.label} ({count} connections, score: {weight:.2f})")

    brain.close()
    if os.path.exists(db_path): os.remove(db_path)

if __name__ == "__main__":
    demo_ping()
