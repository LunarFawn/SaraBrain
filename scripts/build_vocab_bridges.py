#!/usr/bin/env python3
"""Build synonym and antonym links in the brain substrate.

Uses Moby Thesaurus for synonyms and a curated list for common antonyms.
Creates 'synonym_of' and 'antonym_of' segments.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
from sara_brain.core.brain import Brain
from sara_brain.models.neuron import NeuronType

from sara_brain.models.segment import Segment

# Common antonyms to seed the 'antonym_of' understanding
COMMON_ANTONYMS = [
    ("hot", "cold"), ("fast", "slow"), ("big", "small"), ("large", "small"),
    ("increase", "decrease"), ("higher", "lower"), ("more", "less"),
    ("start", "stop"), ("begin", "end"), ("first", "last"),
    ("always", "never"), ("sometimes", "rarely"),
    ("positive", "negative"), ("active", "passive"),
    ("parent", "daughter"), ("male", "female"),
    ("presence", "absence"), ("absent", "present"),
    ("inhibit", "activate"), ("block", "allow"),
    ("mitosis", "meiosis"), # Biological contrasts
    ("haploid", "diploid"), ("dominant", "recessive"),
]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--moby", type=Path, help="Path to moby_thesaurus.txt")
    ap.add_argument("--limit", type=int, default=5000, help="Limit Moby entries")
    args = ap.parse_args()

    brain = Brain(str(args.db))
    print(f"Opening {args.db} for vocabulary bridging...")
    
    # Load spaCy for morphology
    import spacy
    nlp = spacy.load("en_core_web_sm")

    # 1. Add curated antonyms
    print("Adding common antonyms...")
    ant_count = 0
    for a, b in COMMON_ANTONYMS:
        n1, _ = brain.neuron_repo.get_or_create(a, NeuronType.CONCEPT)
        n2, _ = brain.neuron_repo.get_or_create(b, NeuronType.CONCEPT)
        # Bidirectional antonym edges
        brain.segment_repo.get_or_create(n1.id, n2.id, "antonym_of")
        brain.segment_repo.get_or_create(n2.id, n1.id, "antonym_of")
        ant_count += 2
    print(f"  Added {ant_count} antonym edges.")

    # 2. Add Morphology bridges (e.g. mitotic -> mitosis)
    print("Adding Morphology bridges...")
    morph_count = 0
    all_neurons = brain.neuron_repo.list_all()
    for n in all_neurons:
        if n.neuron_type != NeuronType.CONCEPT:
            continue
        label = n.label
        doc = nlp(label)
        if len(doc) != 1:
            continue
        token = doc[0]
        lemma = token.lemma_.lower().strip()
        if lemma != label:
            # Found a variant! mitotic -> mitosis
            lemma_n, _ = brain.neuron_repo.get_or_create(lemma, NeuronType.CONCEPT)
            # Bidirectional alias edges
            brain.segment_repo.get_or_create(n.id, lemma_n.id, "alias_of")
            brain.segment_repo.get_or_create(lemma_n.id, n.id, "alias_of")
            morph_count += 2
    print(f"  Added {morph_count} morphology edges.")

    # 3. Add Moby synonyms
    if args.moby and args.moby.exists():
        print(f"Adding synonyms from {args.moby}...")
        syn_count = 0
        entries = 0
        with open(args.moby) as f:
            for line in f:
                if entries >= args.limit:
                    break
                parts = [w.strip().lower() for w in line.strip().split(",")]
                if len(parts) < 2:
                    continue
                
                root = parts[0]
                # Only add if the root exists in our brain (to keep it focused)
                root_n = brain.neuron_repo.get_by_label(root)
                if not root_n:
                    continue
                
                synonyms = parts[1:10] # limit to top 10
                for syn in synonyms:
                    if not syn or syn == root:
                        continue
                    syn_n, _ = brain.neuron_repo.get_or_create(syn, NeuronType.CONCEPT)
                    # Bidirectional synonym edges
                    brain.segment_repo.get_or_create(root_n.id, syn_n.id, "synonym_of")
                    brain.segment_repo.get_or_create(syn_n.id, root_n.id, "synonym_of")
                    syn_count += 2
                
                entries += 1
                if entries % 500 == 0:
                    brain.conn.commit()
                    print(f"  Processed {entries} Moby entries, {syn_count} edges...")
        print(f"  Added {syn_count} synonym edges.")
    
    brain.conn.commit()
    print("Done.")
    brain.close()

if __name__ == "__main__":
    main()
