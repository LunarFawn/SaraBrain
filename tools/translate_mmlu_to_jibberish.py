import json
import re
from pathlib import Path
from datasets import load_dataset

def encrypt_text(text, cipher_dict):
    # We need to replace words but keep case and punctuation
    def replace_word(match):
        word = match.group(0)
        lower_word = word.lower()
        if lower_word in cipher_dict:
            short_word = cipher_dict[lower_word]
        elif lower_word.endswith('s') and lower_word[:-1] in cipher_dict:
            short_word = cipher_dict[lower_word[:-1]] + 's'
        elif lower_word.endswith('es') and lower_word[:-2] in cipher_dict:
            short_word = cipher_dict[lower_word[:-2]] + 'es'
        else:
            return word
            
        if word.isupper():
            return short_word.upper()
        elif word.istitle():
            return short_word.capitalize()
        else:
            return short_word

    # Use regex to find word tokens
    return re.sub(r'[a-zA-Z]+', replace_word, text)

def main():
    cipher_path = Path("data/biology_short_cipher_nouns.json")
    if not cipher_path.exists():
        print(f"Cipher file not found at {cipher_path}")
        return
        
    with open(cipher_path, "r") as f:
        cipher_dict = json.load(f)
        
    print("Loading MMLU High School Biology test set...")
    ds = load_dataset('cais/mmlu', 'high_school_biology', split='test')
    
    encrypted_ds = []
    
    print("Translating questions and choices into Jibberish...")
    for idx, item in enumerate(ds):
        q = item['question']
        choices = item['choices']
        answer = item['answer']
        
        enc_q = encrypt_text(q, cipher_dict)
        enc_choices = [encrypt_text(c, cipher_dict) for c in choices]
        
        encrypted_ds.append({
            'question': enc_q,
            'choices': enc_choices,
            'answer': answer
        })
        
        if idx < 3:
            print(f"\n--- Example {idx} ---")
            print(f"Original Q: {q}")
            print(f"Encrypted Q: {enc_q}")
            
    out_path = Path("data/mmlu_biology_short_jibberish.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(encrypted_ds, f, indent=2)
        
    print(f"\nTranslated {len(encrypted_ds)} questions. Saved to {out_path}")

if __name__ == "__main__":
    main()
