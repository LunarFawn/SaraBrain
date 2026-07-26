import argparse
import json
import os
import sys
from pathlib import Path
import sqlite3
import urllib.request

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from sara_brain.core.brain import Brain
from sara_brain.models.neuron import NeuronType
from sara_brain.nlp.reader import DocumentReader

def query_llm_for_triplets(text_chunk: str, base_url: str = "http://localhost:11434") -> list[dict]:
    system_prompt = """You are an expert biological data extractor.
Read the following text and extract all factual statements as Subject-Relation-Object triplets.
Rules:
1. Subject and Object MUST be simple noun phrases (entities).
2. Relation MUST be a single open-vocabulary verb (e.g., encodes, produces, regulates).
3. Output ONLY valid JSON in this exact format:
[
  {"subject": "entity1", "relation": "verb", "object": "entity2"}
]
Do not include any markdown formatting, explanations, or text outside the JSON array."""

    payload = {
        "model": "llama3.2:3b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text_chunk}
        ],
        "stream": False,
        "options": {"temperature": 0.0}
    }
    
    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            content = result["message"]["content"].strip()
            
            # Clean up potential markdown formatting
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            try:
                triplets = json.loads(content)
                if isinstance(triplets, list):
                    valid_triplets = []
                    for t in triplets:
                        if isinstance(t, dict) and 'subject' in t and 'relation' in t and 'object' in t:
                            # Clean up and normalize
                            s = t['subject'].strip().lower()
                            r = t['relation'].strip().lower()
                            o = t['object'].strip().lower()
                            if s and r and o:
                                valid_triplets.append({"subject": s, "relation": r, "object": o})
                    return valid_triplets
            except json.JSONDecodeError:
                print(f"Failed to parse JSON: {content[:100]}...")
                pass
    except Exception as e:
        print(f"LLM API error: {e}")
        
    return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="Directory containing txt files")
    parser.add_argument("--db", required=True, help="Database output path")
    args = parser.parse_args()

    in_dir = Path(args.text)
    if not in_dir.exists():
        print(f"Directory {in_dir} not found")
        return

    # Initialize Brain
    if os.path.exists(args.db):
        os.remove(args.db)
    brain = Brain(args.db)

    total_triplets = 0
    files = list(in_dir.glob("*.txt"))
    
    for i, txt_file in enumerate(files):
        print(f"Processing {txt_file.name} ({i+1}/{len(files)})...")
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()
            
        chunks = DocumentReader._chunk_text(text, max_chars=1500)
        
        for j, chunk in enumerate(chunks):
            print(f"  Chunk {j+1}/{len(chunks)}...")
            triplets = query_llm_for_triplets(chunk)
            
            for t in triplets:
                # Insert directly into the database as a fact
                n_subj, c_subj = brain.neuron_repo.get_or_create(t['subject'], NeuronType.CONCEPT)
                n_obj, c_obj = brain.neuron_repo.get_or_create(t['object'], NeuronType.CONCEPT)
                seg, c_seg = brain.segment_repo.get_or_create(n_subj.id, n_obj.id, t['relation'])
                if not c_seg:
                    brain.segment_repo.strengthen(seg)
                total_triplets += 1
                
    brain.conn.commit()
    print(f"Extraction complete! Found {total_triplets} triplets.")
    
if __name__ == "__main__":
    main()
