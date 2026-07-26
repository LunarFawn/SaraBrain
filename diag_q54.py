import sys
import os
from pathlib import Path
import torch

# Add benchmarks and scripts to path
sys.path.insert(0, os.path.abspath("benchmarks"))
sys.path.insert(0, os.path.abspath("scripts"))

from run_mmlu_biology import build_sara_wavefront_substrate, LocalModelLoader
from sara_brain.core.brain import Brain

def diagnostic():
    db_path = "data/biology_full_v2_clean.db"
    synth_path = "models/sara-synthesizer-115m"
    question = "During which phase of the cell cycle does the quantity of DNA in a eukaryotic cell typically double?"
    
    brain = Brain(db_path)
    synth_loader = LocalModelLoader(synth_path)
    
    print("\n--- WAVEFRONT SUBSTRATE (RAW) ---")
    substrate_raw = build_sara_wavefront_substrate(brain, question, use_prose=False)
    print(substrate_raw)
    
    print("\n--- NEURAL SYNTHESIZED PROSE ---")
    substrate_prose = build_sara_wavefront_substrate(brain, question, use_prose=True, synth_loader=synth_loader)
    print(substrate_prose)

if __name__ == "__main__":
    diagnostic()
