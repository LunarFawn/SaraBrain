"""Universal Dependencies ingestion for the Grammar Cortex.

Downloads CoNLL-U treebanks on first use and parses sentences into
delexicalized (UPOS, DEPREL) token streams. Word forms are discarded —
the cortex must learn from structure alone, per v024.

Multiple English treebanks are supported (EWT, GUM, LinES, ParTUT, Atis,
ESL) — they share the same UPOS + UD relation vocabulary, so they can be
mixed without changing the model.
"""
from __future__ import annotations

import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

# treebank key -> (UD repo name, file slug)
TREEBANKS = {
    "ewt":    ("UD_English-EWT",    "en_ewt"),
    "gum":    ("UD_English-GUM",    "en_gum"),
    "lines":  ("UD_English-LinES",  "en_lines"),
    "partut": ("UD_English-ParTUT", "en_partut"),
    "atis":   ("UD_English-Atis",   "en_atis"),
    "esl":    ("UD_English-ESL",    "en_esl"),
}
ENGLISH_ALL = list(TREEBANKS.keys())

DEFAULT_CACHE_ROOT = Path("data/ud")
DEFAULT_CACHE = DEFAULT_CACHE_ROOT / "en_ewt"  # back-compat alias


@dataclass
class UDToken:
    upos: str
    dep: str
    head: int
    is_q_marker: bool
    is_neg: bool


@dataclass
class UDSentence:
    tokens: list[UDToken]


def _treebank_dir(treebank: str, cache_root: Path) -> Path:
    return cache_root / TREEBANKS[treebank][1]


def ensure_split(
    treebank: str = "ewt",
    split: str = "train",
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> Path:
    if treebank not in TREEBANKS:
        raise ValueError(f"unknown treebank: {treebank}")
    if split not in ("train", "dev", "test"):
        raise ValueError(f"unknown split: {split}")
    repo, slug = TREEBANKS[treebank]
    cache_dir = _treebank_dir(treebank, cache_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{slug}-ud-{split}.conllu"
    out = cache_dir / fname
    if out.exists() and out.stat().st_size > 0:
        return out
    url = f"https://raw.githubusercontent.com/UniversalDependencies/{repo}/master/{fname}"
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


def iter_sentences(
    treebanks: list[str],
    split: str = "train",
    cache_root: Path = DEFAULT_CACHE_ROOT,
) -> Iterator[UDSentence]:
    """Yield sentences across multiple treebanks. Treebanks missing the
    requested split are skipped with a warning."""
    for tb in treebanks:
        try:
            path = ensure_split(tb, split, cache_root)
        except Exception as e:
            print(f"[ud] skip {tb}/{split}: {e}", flush=True)
            continue
        yield from parse_conllu(path)


def to_input_tokens(sent: UDSentence, max_tokens: int = 32) -> list[str]:
    """Flatten a sentence into a (DEPREL, UPOS) interleaved tag stream."""
    out: list[str] = []
    for t in sent.tokens[:max_tokens]:
        out.append(t.dep)
        out.append(t.upos)
    return out
