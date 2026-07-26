import json
import re
from pathlib import Path

def decrypt_text(text, cipher_dict):
    # Invert the cipher (jibberish -> english)
    decipher = {v: k for k, v in cipher_dict.items()}
    
    # We need to replace words but keep case and punctuation
    def replace_word(match):
        word = match.group(0)
        lower_word = word.lower()
        if lower_word in decipher:
            english_word = decipher[lower_word]
        elif lower_word.endswith('s') and lower_word[:-1] in decipher:
            english_word = decipher[lower_word[:-1]] + 's'
        elif lower_word.endswith('es') and lower_word[:-2] in decipher:
            english_word = decipher[lower_word[:-2]] + 'es'
        else:
            return word
            
        # Restore case
        if word.isupper():
            return english_word.upper()
        elif word.istitle():
            return english_word.capitalize()
        else:
            return english_word

    # Use regex to find word tokens
    return re.sub(r'[a-zA-Z]+', replace_word, text)

def main():
    cipher_path = Path("data/biology_cipher.json")
    if not cipher_path.exists():
        print(f"Cipher file not found at {cipher_path}")
        return
        
    with open(cipher_path, "r") as f:
        cipher_dict = json.load(f)
        
    in_dir = Path("data/biology_jibberish")
    out_dir = Path("data/biology_english")
    out_dir.mkdir(exist_ok=True)
    
    for txt_file in in_dir.glob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()
            
        decrypted = decrypt_text(text, cipher_dict)
        
        out_path = out_dir / txt_file.name
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(decrypted)
            
    print(f"Decrypted {len(list(in_dir.glob('*.txt')))} files into {out_dir}")

if __name__ == "__main__":
    main()
