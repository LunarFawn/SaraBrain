import json
import re
from pathlib import Path

def encrypt_text(text, cipher_dict):
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
    return re.sub(r'[a-zA-Z]+', replace_word, text)

def main():
    with open("data/biology_short_cipher.json", "r") as f:
        cipher_dict = json.load(f)
        
    in_dir = Path("data/biology_english")
    out_dir = Path("data/biology_short_jibberish")
    out_dir.mkdir(exist_ok=True)
    
    for txt_file in in_dir.glob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()
            
        encrypted = encrypt_text(text, cipher_dict)
        
        out_path = out_dir / txt_file.name
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(encrypted)
            
    print(f"Encrypted {len(list(in_dir.glob('*.txt')))} files into {out_dir}")

if __name__ == "__main__":
    main()
