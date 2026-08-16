#!/usr/bin/env python3
"""Consistent Cipher for Jibberishing Text.

Takes a source text and replaces every unique word with a consistent 
pronounceable nonsense word. Preserves the structure and grammar while 
stripping real-world semantic meaning.
"""

import os
import sys
import json
import random
import re
from pathlib import Path

import spacy
from pathlib import Path

# Load spaCy for POS tagging
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # Fallback to download if missing
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

CONSONANTS = "bcdfghjklmnpqrstvwxyz"
VOWELS = "aeiou"

def _syllable(rng):
    return rng.choice(CONSONANTS) + rng.choice(VOWELS) + rng.choice(CONSONANTS + "")

def generate_nonsense(rng, length=None):
    if length is None:
        length = rng.randint(2, 4)
    return "".join(_syllable(rng) for _ in range(length))

class ConsistentCipher:
    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.mapping = {}
        self.used_nonsense = set()
        # POS tags to jibberish
        self.target_tags = {"NOUN", "PROPN", "ADJ", "PRON"}

    def translate_word(self, token):
        # Keep whitespace and punctuation as is
        if not token.text.isalpha():
            return token.text
        
        # Use lemma for the mapping key to ensure consistency (e.g., cell/cells)
        lemma = token.lemma_.lower()
        
        # Decide if we should jibberish based on POS tag
        should_jibberish = token.pos_ in self.target_tags or lemma in self.mapping
        
        if not should_jibberish:
            return token.text_with_ws
        
        if lemma not in self.mapping:
            while True:
                nonsense = generate_nonsense(self.rng)
                if nonsense not in self.used_nonsense:
                    self.used_nonsense.add(nonsense)
                    self.mapping[lemma] = nonsense
                    break
        
        # Get the nonsense root
        root = self.mapping[lemma]
        
        # Attempt to preserve pluralization in a simple way
        # If the original word ends in 's' and the lemma doesn't, append 's'
        result = root
        if token.text.lower().endswith('s') and not lemma.endswith('s'):
            result += 's'
            
        # Match capitalization of the original word
        if token.text.isupper():
            result = result.upper()
        elif token.text.istitle():
            result = result.capitalize()
            
        return result + token.whitespace_

    def translate_text(self, text):
        doc = nlp(text)
        translated = [self.translate_word(token) for token in doc]
        return "".join(translated)

    def save_cipher(self, path):
        with open(path, 'w') as f:
            json.dump(self.mapping, f, indent=2)

    def load_cipher(self, path):
        with open(path, 'r') as f:
            self.mapping = json.load(f)
            self.used_nonsense = set(self.mapping.values())

def jibberish_biology(facts_dir, output_dir, cipher_path):
    cipher = ConsistentCipher()
    input_files = sorted(Path(facts_dir).glob("ch*_facts.txt"))
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Jibberishing {len(input_files)} chapters...")
    for f in input_files:
        print(f"  Processing {f.name}...")
        text = f.read_text()
        translated = cipher.translate_text(text)
        
        out_f = Path(output_dir) / f.name
        out_f.write_text(translated)
        
    print(f"Saving cipher to {cipher_path}...")
    cipher.save_cipher(cipher_path)
    print("Done.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="benchmarks/biology2e_facts")
    ap.add_argument("--output", default="data/biology_jibberish")
    ap.add_argument("--cipher", default="data/biology_cipher.json")
    args = ap.parse_args()
    
    jibberish_biology(args.input, args.output, args.cipher)
