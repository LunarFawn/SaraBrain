"""End-to-end neural router: question (English) -> tool call JSON.

Combines:
  spaCy parser            (English -> UD-style tag stream)
  Grammar LM encoder      (frozen, learned during cortex training)
  RouterHead classifier   (tiny, trained on substrate-derived examples)
  Rule-based arg extractor (substrate-aware concept/type/label/term)

Single-shot: emits one tool call per question. Multi-step composition,
if needed later, wraps this in an outer loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import spacy
import torch

from .model import GrammarConfig, GrammarModel
from .router_args import ArgResolution, SubstrateIndex, extract_args
from .router_data import ID2TOOL, N_TOOLS, map_dep
from .router_head import RouterHead
from .vocab import BOS_ID, EOS_ID, PAD_ID, TOK2ID, UNK_ID


@dataclass
class RouterDecision:
    tool: str
    args: dict
    classifier_confidence: float    # softmax prob of the chosen class
    extractor_confidence: float     # rule-based confidence
    rationale: str
    raw_logits: list[float]


class CortexRouter:
    def __init__(
        self,
        grammar_ckpt: Path,
        head_ckpt: Path,
        substrate_db: Path | None = None,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        spacy_model: str = "en_core_web_sm",
    ):
        self.device = torch.device(device)

        gck = torch.load(grammar_ckpt, map_location=self.device, weights_only=False)
        cfg = GrammarConfig(**gck["config"])
        encoder = GrammarModel(cfg).to(self.device)
        encoder.load_state_dict(gck["state_dict"])
        encoder.eval()
        self.encoder = encoder
        self.max_seq = cfg.max_seq

        hck = torch.load(head_ckpt, map_location=self.device, weights_only=False)
        head = RouterHead(encoder, N_TOOLS, freeze_encoder=True).to(self.device)
        head.classifier.load_state_dict(hck["head_state_dict"])
        head.eval()
        self.head = head
        self.head_max_seq = hck.get("max_seq", self.max_seq)

        self.nlp = spacy.load(spacy_model)
        self.substrate = SubstrateIndex(substrate_db) if substrate_db else None

    @torch.no_grad()
    def route(self, question: str) -> RouterDecision:
        doc = self.nlp(question)
        tags: list[str] = []
        for t in doc:
            tags.append(map_dep(t.dep_))
            tags.append(t.pos_)
        ids = self._encode_tags(tags)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        logits, _ = self.head(x)
        probs = torch.softmax(logits[0].float(), dim=-1)
        cls = int(probs.argmax().item())
        tool = ID2TOOL[cls]
        cls_conf = float(probs[cls].item())

        extracted: ArgResolution = extract_args(question, tool, self.nlp, self.substrate)

        return RouterDecision(
            tool=tool,
            args=extracted.args,
            classifier_confidence=cls_conf,
            extractor_confidence=extracted.confidence,
            rationale=extracted.rationale,
            raw_logits=logits[0].tolist(),
        )

    def _encode_tags(self, tags: list[str]) -> list[int]:
        ids = [BOS_ID] + [TOK2ID.get(t, UNK_ID) for t in tags] + [EOS_ID]
        if len(ids) > self.head_max_seq:
            ids = ids[:self.head_max_seq]
        return ids + [PAD_ID] * (self.head_max_seq - len(ids))


__all__ = ["CortexRouter", "RouterDecision"]
