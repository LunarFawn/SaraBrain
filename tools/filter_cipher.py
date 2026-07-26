import json
import spacy

def main():
    nlp = spacy.load("en_core_web_sm")
    with open("data/biology_short_cipher.json") as f:
        cipher = json.load(f)
        
    filtered = {}
    for word, code in cipher.items():
        doc = nlp(word)
        if len(doc) == 1:
            token = doc[0]
            if token.pos_ in ("NOUN", "PROPN") and not token.is_stop:
                filtered[word] = code
        else:
            # If it's a multi-word term, it's likely a specific biological concept
            filtered[word] = code
            
    with open("data/biology_short_cipher_nouns.json", "w") as f:
        json.dump(filtered, f, indent=2)
        
    print(f"Reduced cipher from {len(cipher)} to {len(filtered)} words")

if __name__ == "__main__":
    main()
