import json
import random
from pathlib import Path

def generate_short_word(existing):
    consonants = list("bdfgklmnprstvz")
    vowels = list("aeiou")
    while True:
        r = random.random()
        # Generate CVC, CVCV, or CVCVC
        if r < 0.33:
            word = random.choice(consonants) + random.choice(vowels) + random.choice(consonants)
        elif r < 0.66:
            word = random.choice(consonants) + random.choice(vowels) + random.choice(consonants) + random.choice(vowels)
        else:
            word = random.choice(consonants) + random.choice(vowels) + random.choice(consonants) + random.choice(vowels) + random.choice(consonants)
        
        if word not in existing:
            return word

def main():
    cipher_path = Path("data/biology_cipher.json")
    if not cipher_path.exists():
        print("Original cipher not found.")
        return
        
    with open(cipher_path, "r") as f:
        old_cipher = json.load(f)
        
    new_cipher = {}
    used_words = set()
    
    for eng_word in old_cipher.keys():
        short_word = generate_short_word(used_words)
        used_words.add(short_word)
        new_cipher[eng_word] = short_word
        
    out_path = Path("data/biology_short_cipher.json")
    with open(out_path, "w") as f:
        json.dump(new_cipher, f, indent=2)
        
    print(f"Generated {len(new_cipher)} short cipher words.")

if __name__ == "__main__":
    main()
