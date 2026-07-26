import argparse
import sqlite3
import spacy
from pathlib import Path

from sara_brain.core.brain import Brain
from sara_brain.models.neuron import NeuronType



def extract_triplets(doc):
    triplets = []
    for token in doc:
        if token.pos_ == "VERB" or token.pos_ == "AUX":
            subject = None
            obj = None
            
            # Find subject
            for child in token.lefts:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subject = child
                    break
                    
            # Find object
            for child in token.rights:
                if child.dep_ in ("dobj", "attr", "acomp"):
                    obj = child
                    break
                elif child.dep_ == "prep":
                    for grandchild in child.rights:
                        if grandchild.dep_ == "pobj":
                            obj = grandchild
                            break
                    if obj:
                        break
                        
            if subject and obj:
                # Clean up noun chunks by extracting just the core phrase without deep subtrees
                subj_text = " ".join([t.text for t in subject.subtree if t.pos_ not in ("PUNCT", "SYM")]).strip()
                obj_text = " ".join([t.text for t in obj.subtree if t.pos_ not in ("PUNCT", "SYM")]).strip()
                
                # Truncate overly long subjects/objects (protects against bad dependency parses)
                if len(subj_text.split()) > 15 or len(obj_text.split()) > 15:
                    continue
                    
                verb_lemma = token.lemma_.lower()
                
                # Use open vocabulary verbs! Filter out meaningless structural verbs.
                if verb_lemma not in ("be", "do", "have", "seem"):
                    triplets.append((subj_text.lower(), verb_lemma, obj_text.lower()))
                    
    return triplets

def main():
    parser = argparse.ArgumentParser(description="Extract semantic graph from text using hybrid spaCy approach.")
    parser.add_argument("--text", type=str, required=True, help="Input text file")
    parser.add_argument("--db", type=str, required=True, help="Output brain.db path")
    args = parser.parse_args()

    print("Loading spaCy model...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
        return

    print(f"Reading from {args.text}...")
    text_path = Path(args.text)
    texts_to_process = []
    if text_path.is_file():
        texts_to_process.append(text_path)
    elif text_path.is_dir():
        texts_to_process.extend(text_path.glob("*.txt"))
        
    print("Parsing documents (this runs purely on CPU)...")
    triplets = []
    for p in texts_to_process:
        with open(p, "r", encoding="utf-8") as f:
            text = f.read()
            # If text is still too long, split by chunks (1M char limit). 
            # A simple chunking by 900k chars is safe enough for textbooks.
            chunk_size = 900000
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i+chunk_size]
                doc = nlp(chunk)
                triplets.extend(extract_triplets(doc))

    print(f"Found {len(triplets)} semantic triplets. Ingesting to {args.db}...")
    
    # Initialize Brain Database
    brain = Brain(args.db)

    neurons_created = 0
    segments_created = 0

    for subj, rel, obj in triplets:
        n_subj, c_subj = brain.neuron_repo.get_or_create(subj, NeuronType.CONCEPT)
        n_obj, c_obj = brain.neuron_repo.get_or_create(obj, NeuronType.CONCEPT)
        
        if c_subj: neurons_created += 1
        if c_obj: neurons_created += 1
        
        seg, c_seg = brain.segment_repo.get_or_create(n_subj.id, n_obj.id, rel)
        if c_seg:
            segments_created += 1
        else:
            brain.segment_repo.strengthen(seg)
            
    brain.conn.commit()
    print(f"Ingestion complete!")
    print(f"  Neurons created: {neurons_created}")
    print(f"  Segments created: {segments_created}")

if __name__ == "__main__":
    main()
