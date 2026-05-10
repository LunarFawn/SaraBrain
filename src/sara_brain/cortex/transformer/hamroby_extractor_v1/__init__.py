"""HamRoby Extractor v1 — grammar-feature transformer for triple extraction.

Sibling of HamRoby Router v1. Same design philosophy: model holds form
(grammar), substrate holds meaning (Sara). The extractor takes a
sentence, identifies word-level (subject, relation, object) span
positions from grammatical role features, and the decoder reads
whole-word substrings off a parallel "conveyor belt" — verbatim, no
fragmentation possible.

Architectural commitment:
  - Input: per-word grammatical features (POS, dependency label,
    head offset, optional function-word ID for closed-class words).
  - The model never embeds open-class content words. "molecular
    snare" rides the conveyor belt unembedded; the encoder only sees
    "this position is NOUN with dep=nsubj."
  - Output: word-level BIO span tags. Decoder slices the original
    word array by index. Atomic by construction — subword
    fragmentation is impossible because subwords don't exist here.

Replaces v2/. v2's BPE-tokenized encoder was a paradigm misalignment;
content-aware subword embeddings undermined the form/meaning split
that Sara's substrate is built on.
"""
