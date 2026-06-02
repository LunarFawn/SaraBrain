"""sara-pipeline — Full end-to-end: document → teach → ask → answer.

No external LLM needed. Uses from-scratch models:
  - Extractor (115M): reads document, produces triples
  - Wavefront: selects relevant facts for the question
  - Synthesizer (115M): renders selected facts as prose

Usage:
    # Teach from a document then ask questions
    sara-pipeline teach document.txt --brain my.db
    sara-pipeline ask "What is X?" --brain my.db

    # One-shot: teach and immediately ask
    sara-pipeline run document.txt "What is X?" --brain my.db
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from sara_brain.core.brain import Brain
from sara_brain.core.wavefront_renderer import render_wavefront_facts
from sara_reader.stateless_reader import _extract_seed_concepts, _filter_seeds_by_substrate
from train_sara_extractor_scratch import SaraExtractor, build_vocab, encode_with_oov, tokenize


class SaraPipeline:
    """Full pipeline: document → extractor → Sara → wavefront → synthesizer → answer."""

    def __init__(self, brain_path: str,
                 extractor_path: str = "models/sara-extractor-115m-v2/best.pt",
                 synthesizer_path: str = "models/sara-synthesizer-115m/best.pt",
                 device: str = "cpu"):
        self.device = torch.device(device)
        self.tok2id = build_vocab()
        self.ext_vocab = len(self.tok2id) + 300
        self.id2tok = {v: k for k, v in self.tok2id.items()}

        # Load brain
        self.brain_path = brain_path
        Path(brain_path).parent.mkdir(parents=True, exist_ok=True)
        self.brain = Brain(brain_path)

        # Load extractor
        if Path(extractor_path).exists():
            self.extractor = self._load_model(extractor_path, max_enc=300, max_dec=150)
            print(f"  Extractor: loaded ({extractor_path})")
        else:
            self.extractor = None
            print(f"  Extractor: not found ({extractor_path})")

        # Load synthesizer
        if Path(synthesizer_path).exists():
            self.synthesizer = self._load_model(synthesizer_path, max_enc=200, max_dec=80)
            print(f"  Synthesizer: loaded ({synthesizer_path})")
        else:
            self.synthesizer = None
            print(f"  Synthesizer: not found ({synthesizer_path})")

    def _load_model(self, path, max_enc, max_dec):
        model = SaraExtractor(self.ext_vocab, d_model=768, enc_layers=8, dec_layers=6,
                              n_heads=12, max_enc=max_enc, max_dec=max_dec).to(self.device)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model

    def _run_model(self, model, input_text, max_enc, max_len):
        enc_ids, oov, oov_map = encode_with_oov(input_text, self.tok2id, max_enc)
        enc_t = torch.tensor([enc_ids], dtype=torch.long, device=self.device)
        pm = torch.zeros(1, len(enc_ids), dtype=torch.bool, device=self.device)
        with torch.no_grad():
            out_ids = model.generate(enc_t, pm, max_len=max_len)[0].tolist()
        id2tok = dict(self.id2tok)
        for t, idx in oov_map.items():
            id2tok[idx] = t
        return " ".join(id2tok.get(i, "?") for i in out_ids if i not in (0, 2))

    def teach_document(self, text: str) -> list[tuple[str, str, str]]:
        """Extract triples from text and teach them to Sara."""
        if not self.extractor:
            print("ERROR: No extractor model loaded.")
            return []

        # Split into sentences
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        all_triples = []

        for sent in sentences:
            output = self._run_model(self.extractor, sent, max_enc=300, max_len=100)

            # Parse structured triples
            parts = output.split("t_end")
            for part in parts:
                if "t_start" in part and "t_rel" in part and "t_obj" in part:
                    try:
                        after = part.split("t_start")[1]
                        subj = after.split("t_rel")[0].strip()
                        rel = after.split("t_rel")[1].split("t_obj")[0].strip()
                        obj = after.split("t_obj")[1].strip()
                        if subj and rel and obj and len(subj) > 1 and len(obj) > 1:
                            # Clean: remove self-references
                            if subj != obj:
                                all_triples.append((subj, rel, obj))
                    except (IndexError, ValueError):
                        pass

        # Teach to Sara
        taught = 0
        for s, r, o in all_triples:
            self.brain.teach_triple(s, r, o, source_text=f"{s} {r} {o}")
            taught += 1

        return all_triples

    def ask(self, question: str) -> str:
        """Ask a question — wavefront retrieves, synthesizer renders."""
        # Get seeds from question
        candidates = _extract_seed_concepts(question)
        seeds = _filter_seeds_by_substrate(self.brain, candidates)
        if not seeds:
            seeds = candidates[:3]
        if not seeds:
            return "I don't have enough information to answer that."

        # Run wavefront
        facts = render_wavefront_facts(self.brain, seeds, depth=2, max_facts=10)
        fact_lines = [l.strip("- ").strip() for l in facts.split("\n") if l.strip().startswith("- ")]

        if not fact_lines:
            return "I don't have enough information to answer that."

        # If synthesizer available, render as prose
        if self.synthesizer:
            # Take top 3 facts, synthesize each
            answers = []
            for fact_text in fact_lines[:3]:
                # Convert to structured format for synthesizer
                structured = f"t_start {fact_text} t_end"
                input_text = f"{structured} {question}"
                prose = self._run_model(self.synthesizer, input_text, max_enc=200, max_len=40)
                if prose and "?" not in prose[:5]:
                    answers.append(prose.rstrip(". ") + ".")
            if answers:
                return " ".join(answers)

        # Fallback: return raw facts
        return "Based on Sara Brain: " + "; ".join(fact_lines[:5]) + "."

    def close(self):
        self.brain.close()


def cmd_teach(args):
    pipeline = SaraPipeline(args.brain, device=args.device)
    text = Path(args.document).read_text()
    print(f"\nTeaching from: {args.document} ({len(text)} chars)")
    t0 = time.time()
    triples = pipeline.teach_document(text)
    dt = time.time() - t0
    print(f"\nExtracted and taught {len(triples)} triples in {dt:.1f}s")
    print(f"Brain: {pipeline.brain.stats()}")
    for s, r, o in triples[:10]:
        print(f"  {s} | {r} | {o}")
    if len(triples) > 10:
        print(f"  ... and {len(triples)-10} more")
    pipeline.close()


def cmd_ask(args):
    pipeline = SaraPipeline(args.brain, device=args.device)
    answer = pipeline.ask(args.question)
    print(f"\nQ: {args.question}")
    print(f"A: {answer}")
    pipeline.close()


def cmd_run(args):
    pipeline = SaraPipeline(args.brain, device=args.device)
    text = Path(args.document).read_text()
    print(f"Teaching from: {args.document}")
    triples = pipeline.teach_document(text)
    print(f"Taught {len(triples)} triples.\n")
    answer = pipeline.ask(args.question)
    print(f"Q: {args.question}")
    print(f"A: {answer}")
    pipeline.close()


def main():
    ap = argparse.ArgumentParser(prog="sara-pipeline",
                                 description="Full Sara Brain pipeline: teach + ask")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("teach", help="Teach Sara from a document")
    p.add_argument("document", help="Path to text file")
    p.add_argument("--brain", default="pipeline.db")

    p = sub.add_parser("ask", help="Ask Sara a question")
    p.add_argument("question")
    p.add_argument("--brain", default="pipeline.db")

    p = sub.add_parser("run", help="Teach then ask in one shot")
    p.add_argument("document")
    p.add_argument("question")
    p.add_argument("--brain", default="pipeline.db")

    args = ap.parse_args()
    if args.cmd == "teach":
        cmd_teach(args)
    elif args.cmd == "ask":
        cmd_ask(args)
    elif args.cmd == "run":
        cmd_run(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
