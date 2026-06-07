#!/usr/bin/env python3
"""Teach fact files into a brain.db using the 115M Hamroby Neural Extractor.
"""
from __future__ import annotations
import argparse
import time
import os
from pathlib import Path
import spacy
from sara_brain.core.brain import Brain
from sara_brain.cortex.transformer.hamroby_extractor_v1.inference import extract_triples
from sara_brain.cortex.parser import EnhancedParser

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--sources", nargs="+", required=True, type=Path)
    args = ap.parse_args()

    if args.db.exists():
        print(f"Adding to existing {args.db}")
    brain = Brain(str(args.db))
    
    # Load spaCy for clause splitting
    nlp = spacy.load("en_core_web_sm")

    for source in args.sources:
        print(f"Neural Teaching from {source}...")
        lines = [l.strip() for l in source.read_text().splitlines() if l.strip() and not l.startswith("#")]
        
        print(f"{len(lines)} lines found.")

        total_triples = 0
        t0 = time.time()
        for i, line in enumerate(lines, 1):
            # Split line into sentences just in case
            for sentence in nlp(line).sents:
                # Split into clauses using the EnhancedParser's logic
                clauses = EnhancedParser._split_compound(sentence.text)
                for clause in clauses:
                    if not clause.strip():
                        continue
                    
                    # Truncate clause to avoid IndexError in transformer (max_seq=128)
                    # We'll be conservative and use 100 words.
                    words = clause.split()
                    if len(words) > 100:
                        clause = " ".join(words[:100])

                    # Use the 115M Neural Head
                    try:
                        triples = extract_triples(clause, nlp)
                        for tri in triples:
                            # Teach as a triple to avoid secondary parsing
                            brain.teach_triple(
                                tri.subject, tri.relation, tri.object,
                                source_text=tri.source_clause
                            )
                            total_triples += 1
                    except Exception as e:
                        print(f"  [error] line {i} clause {clause[:30]}...: {e}")
            
            if i % 100 == 0 or i == len(lines):
                print(f"  [{i}/{len(lines)}] triples so far: {total_triples}")
                
        elapsed = time.time() - t0
        print(f"  elapsed: {elapsed:.1f}s, triples: {total_triples}")

    print()
    print(f"Final Neural Stats for {args.db}:")
    print(f"neurons:  {brain.conn.execute('SELECT COUNT(*) FROM neurons').fetchone()[0]}")
    print(f"paths:    {brain.conn.execute('SELECT COUNT(*) FROM paths').fetchone()[0]}")
    print(f"segments: {brain.conn.execute('SELECT COUNT(*) FROM segments').fetchone()[0]}")

if __name__ == "__main__":
    main()
