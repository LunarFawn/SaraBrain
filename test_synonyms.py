
from sara_brain.core.brain import Brain
from sara_brain.bootstrap import ensure_dictionary
import os

DB_PATH = "/tmp/test_synonym.db"
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

brain = Brain(DB_PATH)
print("Bootstrapping dictionary...")
ensure_dictionary(brain, limit=5000) # Load first 5k entries for speed in test

# Teach a fact
print("Teaching fact: 'directional selection leads to extreme phenotype'")
brain.teach_triple("directional selection", "leads_to", "extreme phenotype")

# Query with a synonym
query = "tallest"

print(f"Recognizing: {query}")
# 'tallest' --synonym_of--> 'extreme' should allow finding 'extreme phenotype'
# or at least intersecting with it.
# Brain.recognize(string_labels, min_strength)
results = brain.recognize(query)
if not results:
    print(f"No results found for '{query}'")
else:
    for res in results:
        print(f"Recognized: {res.neuron.label} (confidence: {res.confidence:.2f})")
        for p in res.converging_paths:
            labels = [n.label for n in p.neurons]
            print(f"  Path: {' -> '.join(labels)}")

brain.close()
