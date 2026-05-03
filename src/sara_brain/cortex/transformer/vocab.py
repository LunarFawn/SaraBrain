from __future__ import annotations

SPECIAL = ["<pad>", "<bos>", "<eos>", "<sep>", "<unk>"]

# 17 Universal POS tags
UPOS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
    "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
]

# 37 Universal Dependency relations (canonical UD v2 set, no language-specific subtypes)
UD_DEPS = [
    "acl", "advcl", "advmod", "amod", "appos", "aux", "case", "cc", "ccomp",
    "clf", "compound", "conj", "cop", "csubj", "dep", "det", "discourse",
    "dislocated", "expl", "fixed", "flat", "goeswith", "iobj", "list", "mark",
    "nmod", "nsubj", "nummod", "obj", "obl", "orphan", "parataxis", "punct",
    "reparandum", "root", "vocative", "xcomp",
]

# Slot-fill tokens used by router and synthesizer roles.
SLOTS = [
    "[CONCEPT]", "[TYPE]", "[TOOL]", "[VALUE]", "[ARG]",
    "[SUBJ]", "[OBJ]", "[VERB]", "[NEG]",
]

# Mode and action tokens (router role)
MODES = ["<router>", "<synth>"]
ACTIONS = [
    "TEACH", "REFUTE", "QUESTION",
    "TOOL:brain_value", "TOOL:brain_list", "TOOL:brain_path",
]


def build_vocab() -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for tok in SPECIAL + MODES + UPOS + UD_DEPS + SLOTS + ACTIONS:
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


VOCAB = build_vocab()
TOK2ID = {t: i for i, t in enumerate(VOCAB)}
ID2TOK = {i: t for t, i in TOK2ID.items()}

PAD_ID = TOK2ID["<pad>"]
BOS_ID = TOK2ID["<bos>"]
EOS_ID = TOK2ID["<eos>"]
SEP_ID = TOK2ID["<sep>"]
UNK_ID = TOK2ID["<unk>"]


def encode(tokens: list[str]) -> list[int]:
    return [TOK2ID.get(t, UNK_ID) for t in tokens]


def decode(ids: list[int]) -> list[str]:
    return [ID2TOK.get(i, "<unk>") for i in ids]


VOCAB_SIZE = len(VOCAB)
