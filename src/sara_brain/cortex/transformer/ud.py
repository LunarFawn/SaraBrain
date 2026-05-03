"""Universal Dependencies ingestion for the Grammar Cortex.

Downloads CoNLL-U treebanks on first use and parses sentences into
delexicalized (UPOS, DEPREL) token streams. Word forms are discarded
by default (per v024) — the cortex L1 layer learns from structure
alone.

L2 layers (per-language overlays — see v028/v029/v030) opt into
function-word lexicalization via `to_input_tokens(...,
lexicalize_function_words=True, function_word_set=...)`. With the
flag on, tokens whose lowercased form is in the supplied set emit
the literal form instead of their UPOS tag; everything else stays
delexicalized. The structural skeleton (DEPREL half of each pair)
is unchanged either way.

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
    form: str = ""
    """Lowercased surface form. Populated by `parse_conllu`. Used only
    when `to_input_tokens(..., lexicalize_function_words=True)` is
    requested by an L2 caller; ignored on the L1 (delexicalized) path.
    Defaults to empty so existing constructors stay valid."""


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
                form=form_lower,
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


def to_input_tokens(
    sent: UDSentence,
    max_tokens: int = 32,
    lexicalize_function_words: bool = False,
    function_word_set: frozenset[str] | None = None,
) -> list[str]:
    """Flatten a sentence into a (DEPREL, UPOS-or-form) interleaved
    tag stream.

    Default (L1 path): emits `dep upos` per token — pure structure, no
    surface forms. Identical to the v024-era behaviour.

    Lexicalized (L2 path): when `lexicalize_function_words=True` and a
    `function_word_set` is provided, each token emits `dep` followed
    by:
      - the literal lowercased `form` if it is in the set, or
      - the `upos` tag otherwise.

    The dep half is unchanged either way, so the structural skeleton
    L1 was trained on stays intact when L2 reads the same corpus."""
    lex = lexicalize_function_words and function_word_set is not None
    out: list[str] = []
    for t in sent.tokens[:max_tokens]:
        out.append(t.dep)
        if lex and t.form in function_word_set:
            out.append(t.form)
        else:
            out.append(t.upos)
    return out
