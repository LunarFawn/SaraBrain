#!/usr/bin/env python3
"""Build the dictionary region from Moby Thesaurus II.

Loads ~30K synonym groups into a dictionary region within a brain DB.
Each root word becomes a CONCEPT neuron. Synonyms connect via
'synonym_of' segments. No LLM needed — pure structured data.

This is foundational vocabulary infrastructure. Built ONCE, used by
every domain layer. The echo propagation bridges "faster" → "rapidly"
through synonym edges automatically.

Usage:
    python benchmarks/build_dictionary.py --db layer_vocab.db
    python benchmarks/build_dictionary.py --db bio_full.db --region dictionary
"""

from __future__ import annotations

import argparse
import os
import time


THESAURUS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data", "moby_thesaurus.txt"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True,
                        help="Brain database to add dictionary to")
    parser.add_argument("--region", default="",
                        help="Region prefix (empty = default tables)")
    parser.add_argument("--max-synonyms", type=int, default=15,
                        help="Max synonyms per root word (Moby can have 100+)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to N entries (0 = all)")
    args = parser.parse_args()

    from sara_brain.core.brain import Brain
    from sara_brain.models.neuron import NeuronType
    from sara_brain.storage.neuron_repo import NeuronRepo
    from sara_brain.storage.segment_repo import SegmentRepo

    brain = Brain(args.db)

    # Use region tables if specified
    if args.region:
        brain.db.create_region(args.region, "Synonym dictionary from Moby Thesaurus II")
        neuron_repo = NeuronRepo(brain.conn, prefix=args.region)
        segment_repo = SegmentRepo(brain.conn, prefix=args.region)
    else:
        neuron_repo = brain.neuron_repo
        segment_repo = brain.segment_repo

    if not os.path.exists(THESAURUS_PATH):
        print(f"  ERROR: {THESAURUS_PATH} not found")
        print(f"  Run: curl -sL https://raw.githubusercontent.com/words/moby/master/words.txt "
              f"-o {THESAURUS_PATH}")
        return

    print(f"\n  Building dictionary from Moby Thesaurus II")
    print(f"  DB: {args.db}")
    if args.region:
        print(f"  Region: {args.region}")
    print()

    start = time.time()
    total_neurons = 0
    total_segments = 0
    entries = 0

    with open(THESAURUS_PATH) as f:
        for line_num, line in enumerate(f):
            if args.limit and entries >= args.limit:
                break

            parts = [w.strip().lower() for w in line.strip().split(",")]
            if len(parts) < 2:
                continue

            root = parts[0]
            synonyms = parts[1:args.max_synonyms + 1]

            # Create root neuron
            root_n, created = neuron_repo.get_or_create(
                root, NeuronType.CONCEPT
            )
            if created:
                total_neurons += 1

            # Create synonym neurons and link
            for syn in synonyms:
                if not syn or syn == root:
                    continue
                syn_n, created = neuron_repo.get_or_create(
                    syn, NeuronType.CONCEPT
                )
                if created:
                    total_neurons += 1

                # Bidirectional synonym edges
                _, created = segment_repo.get_or_create(
                    root_n.id, syn_n.id, "synonym_of"
                )
                if created:
                    total_segments += 1

                _, created = segment_repo.get_or_create(
                    syn_n.id, root_n.id, "synonym_of"
                )
                if created:
                    total_segments += 1

            entries += 1

            if entries % 5000 == 0:
                brain.conn.commit()
                elapsed = time.time() - start
                rate = entries / elapsed
                remaining = (30259 - entries) / rate if rate > 0 else 0
                print(f"  [{entries}/30259] {total_neurons} neurons, "
                      f"{total_segments} segments "
                      f"({rate:.0f}/sec, ~{remaining:.0f}s left)",
                      flush=True)

    brain.conn.commit()
    elapsed = time.time() - start

    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Entries processed: {entries}")
    print(f"  Neurons created: {total_neurons}")
    print(f"  Synonym edges: {total_segments}")
    print(f"  Avg synonyms per word: {total_segments / entries / 2:.1f}")
    brain.close()


if __name__ == "__main__":
    main()
