import json
import re
from pathlib import Path
import sys
sys.path.insert(0, 'src')
from sara_brain.core.brain import Brain

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
    with open("data/biology_short_cipher_nouns.json", "r") as f:
        cipher_dict = json.load(f)
        
    src_db = Brain("data/biology_llm_english.db")
    dst_path = "data/biology_llm_short_perfect.db"
    
    if Path(dst_path).exists():
        Path(dst_path).unlink()
        
    dst_db = Brain(dst_path)
    
    cursor = src_db.segment_repo.conn.cursor()
    cursor.execute('''
        SELECT n1.label, n2.label, s.relation 
        FROM segments s
        JOIN neurons n1 ON s.source_id = n1.id
        JOIN neurons n2 ON s.target_id = n2.id
    ''')
    
    rows = cursor.fetchall()
    print(f"Encrypting {len(rows)} triplets...")
    
    for src_label, tgt_label, relation in rows:
        enc_src = encrypt_text(src_label, cipher_dict)
        enc_tgt = encrypt_text(tgt_label, cipher_dict)
        enc_rel = encrypt_text(relation, cipher_dict)
        
        from sara_brain.models.neuron import NeuronType
        n_src, _ = dst_db.neuron_repo.get_or_create(enc_src, NeuronType.CONCEPT)
        n_tgt, _ = dst_db.neuron_repo.get_or_create(enc_tgt, NeuronType.CONCEPT)
        dst_db.segment_repo.get_or_create(n_src.id, n_tgt.id, enc_rel)
        
    dst_db.segment_repo.conn.commit()
    print(f"Successfully encrypted database to {dst_path}")

if __name__ == '__main__':
    main()
