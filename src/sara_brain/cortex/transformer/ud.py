"""Universal Dependencies (English EWT) ingestion for the Grammar Cortex.

Downloads the CoNLL-U treebank on first use and parses sentences into
delexicalized (UPOS, DEPREL) token streams. Word forms are discarded —
the cortex must learn from structure alone, per v024.
"""
from __future__ import annotations

import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

UD_EWT_BASE = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master"
UD_EWT_FILES = {
    "train": "en_ewt-ud-train.conllu",
    "dev":   "en_ewt-ud-dev.conllu",
    "test":  "en_ewt-ud-test.conllu",
}

DEFAULT_CACHE = Path("data/ud/en_ewt")


@dataclass
class UDToken:
    upos: str
    dep: str        # base relation, language-specific subtypes stripped
    head: int       # 1-indexed head id, 0 = root
    is_q_marker: bool   # WH-pronoun / question-word heuristic flag
    is_neg: bool        # negation particle heuristic flag


@dataclass
class UDSentence:
    tokens: list[UDToken]


def ensure_split(split: str = "train", cache_dir: Path = DEFAULT_CACHE) -> Path:
    if split not in UD_EWT_FILES:
        raise ValueError(f"unknown split: {split}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = UD_EWT_FILES[split]
    out = cache_dir / fname
    if out.exists() and out.stat().st_size > 0:
        return out
    url = f"{UD_EWT_BASE}/{fname}"
    print(f"[ud] downloading {url}", flush=True)
    urllib.request.urlretrieve(url, out)
    print(f"[ud] saved {out} ({out.stat().st_size // 1024} KB)", flush=True)
    return out


_WH_LEMMAS = {"what", "who", "which", "where", "when", "why", "how", "whose", "whom"}
_NEG_FORMS = {"not", "n't", "no", "never"}


def _strip_subtype(dep: str) -> str:
    return dep.split(":", 1)[0]


def parse_conllu(path: Path) -> Iterator[UDSentence]:
    """Yield sentences. Multiword tokens (id with '-') and empty nodes ('.')
    are skipped — only base tokens are kept."""
    tokens: list[UDToken] = []
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                if tokens:
                    yield UDSentence(tokens=tokens)
                    tokens = []
                continue
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 8:
                continue
            tok_id = parts[0]
            if "-" in tok_id or "." in tok_id:
                continue
            form_lower = parts[1].lower()
            lemma_lower = parts[2].lower()
            upos = parts[3]
            head = int(parts[6]) if parts[6].isdigit() else 0
            dep = _strip_subtype(parts[7])
            tokens.append(UDToken(
                upos=upos,
                dep=dep,
                head=head,
                is_q_marker=lemma_lower in _WH_LEMMAS or form_lower in _WH_LEMMAS,
                is_neg=lemma_lower in _NEG_FORMS or form_lower in _NEG_FORMS,
            ))
    if tokens:
        yield UDSentence(tokens=tokens)


def classify(sent: UDSentence) -> str:
    """Heuristic kind: QUESTION / REFUTE / TEACH."""
    last = sent.tokens[-1] if sent.tokens else None
    if last and last.upos == "PUNCT" and any(t.upos == "PUNCT" for t in sent.tokens):
        # crude: any "?" lemma surfaced via FORM check would be cleaner, but
        # we discarded forms. Use is_q_marker presence as the question signal.
        pass
    if any(t.is_q_marker for t in sent.tokens):
        return "QUESTION"
    if any(t.is_neg for t in sent.tokens):
        return "REFUTE"
    return "TEACH"


def to_input_tokens(sent: UDSentence, max_tokens: int = 32) -> list[str]:
    """Flatten a sentence into a (DEPREL, UPOS) interleaved tag stream."""
    out: list[str] = []
    for t in sent.tokens[:max_tokens]:
        out.append(t.dep)
        out.append(t.upos)
    return out


def to_target_tokens(kind: str) -> list[str]:
    if kind == "TEACH":
        return ["TEACH", "[SUBJ]", "[VERB]", "[OBJ]"]
    if kind == "REFUTE":
        return ["REFUTE", "[SUBJ]", "[NEG]", "[VERB]", "[OBJ]"]
    if kind == "QUESTION":
        return ["QUESTION", "TOOL:brain_value", "[CONCEPT]", "[TYPE]"]
    raise ValueError(kind)


def load_examples(
    split: str = "train",
    cache_dir: Path = DEFAULT_CACHE,
    max_tokens: int = 32,
) -> list[tuple[list[str], list[str], str]]:
    """Returns list of (input_tokens, target_tokens, kind)."""
    path = ensure_split(split, cache_dir)
    out: list[tuple[list[str], list[str], str]] = []
    for sent in parse_conllu(path):
        if not sent.tokens:
            continue
        kind = classify(sent)
        inp = to_input_tokens(sent, max_tokens=max_tokens)
        tgt = to_target_tokens(kind)
        out.append((inp, tgt, kind))
    return out
