# HamRoby Extractor v1 — grammar-feature transformer for triple extraction

**Date:** 2026-05-09
**Module:** `src/sara_brain/cortex/transformer/hamroby_extractor_v1/`
**Status:** built, validated on synthetic, real-prose-trained checkpoint pending

## Why this exists

Sara's substrate stores whole-label neurons (e.g. "molecular snare", "creed 2", "the paper") with the original source clause preserved as provenance on each path. Standard NLP extractors don't naturally produce that:

- **BPE-based transformers** (BERT, GPT, the v2 we tried first) tokenize at sub-word level. A word like "delves" splits into pieces like `▁del` + `ves`. When the head tags spans, it can mark only one piece as part of a span, leaving fragments like `'del'` or `'ight'` (from `highlights`) as the predicted output. That can't be what lands in Sara's substrate.
- **Content-aware embeddings** mean the model statistically memorizes domain words during training. To preserve Sara's content-orthogonality property (the model has provably zero exposure to user's papers), the model must not embed open-class content at all.

The v2 BPE-based extraction head hit 86.9% triple_em on synthetic but 0% on real prose: the fragments it produced ("del", "ight", "datum to") couldn't possibly match author-curated triples or, worse, couldn't even be searched against the substrate. The architectural mismatch was structural, not solvable by more training.

`hamroby_extractor_v1` is the realignment: a transformer whose input contains only grammatical metadata, never open-class content. Surface text rides a parallel "conveyor belt" array used only at decode time.

## Architecture

Inference flow:

```
sentence: "The molecular snare binds to its target."
   │
   ▼
spaCy parse  (already used by the rule stub)
   │
   ▼     (per-word feature tuples)
   word_0  "The"        POS=DET    dep=det      head=+2
   word_1  "molecular"  POS=ADJ    dep=amod     head=+1
   word_2  "snare"      POS=NOUN   dep=nsubj    head=+1
   word_3  "binds"      POS=VERB   dep=ROOT     head=0
   word_4  "to"         POS=ADP    dep=case     head=-1
   word_5  "its"        POS=PRON   dep=nmod     head=+1
   word_6  "target"     POS=NOUN   dep=obl      head=-2
   │
   ▼  ┌────────────────────────────────────────────┐
      │  Grammar encoder sees (POS, dep, offset,   │ ←── never sees "molecular",
      │  funcword_id) per word.                    │     "snare", "target".
      │                                            │     Open-class words are
      │  Bidirectional transformer blocks.         │     not in the input vocab.
      └────────────────────────────────────────────┘
   │
   ▼
word-level BIO tags:  O B-S I-S B-R I-R B-O I-O
   │
   ▼  ┌──────────────────────────┐
      │ Decoder slices the       │
      │ original word array      │ ←── this array is the conveyor belt.
      │ by predicted indices.    │     Surface text passed through unchanged.
      └──────────────────────────┘
   │
   ▼
ExtractedTriple(
  subject  = "molecular snare",
  relation = "binds to",
  object   = "its target",
)
   │
   ▼
brain.teach_triple(subject, relation, object,
                   source_text=<full sentence>)
```

### Input vocabulary

The encoder's input space is a closed grammatical vocabulary. ~250 tokens total:

| stream | tokens | source |
|---|---|---|
| POS tags | ~19 (17 UPOS + PAD + UNK) | Universal POS, same as HamRoby v1's `vocab.UPOS` |
| Dependency labels | ~39 (37 UD deprels + PAD + UNK) | Universal Dependencies v2, mapped from spaCy's ClearNLP-style via the existing `CLEARNLP_TO_UD` table |
| Head offset bins | ~24 (PAD, FAR_LEFT, -10..+10, FAR_RIGHT) | binned dependency-head relative position |
| Function words | ~101 (99 closed-class English + NONE + PAD) | reused from `vocab_en.ENGLISH_FUNCTION_WORDS` |

Open-class content words (nouns, verbs, adjectives that carry domain meaning) get **no slot** in the input vocabulary. They are embedded as `(POS, dep, head_offset, NONE)` — only their grammatical role is visible to the model. Their surface text rides a parallel `words: list[str]` array (the conveyor belt) used only at decode time.

### Encoder body

Standard BERT-style bidirectional transformer:

- Pre-LN multi-head attention + feed-forward blocks
- Sinusoidal-style learned position embeddings
- Padding-aware key padding masks
- Configs: `tiny` (128d / 2 layers / 313K params), `base` (256d / 4 layers / ~970K), `large` (384d / 6 layers / ~5M)

Internal mechanics are conventional. The distinctive design lives in the input embedding layer (four parallel feature embedding tables → concat → linear projection to `d_model`) and the output decoder, not in the attention math.

### Output: word-level BIO tagging

Per-word classification over 7 tags:

```
O      = outside any span
B-S, I-S = subject span
B-R, I-R = relation span
B-O, I-O = object span
```

One tag per word — never per subword, because subwords don't exist in this design. The decoder uses lenient BIO interpretation (orphan I-tags open spans) and emits one ExtractedTriple per object span (handles list-object patterns: "X verbed A, B, and C" → three triples sharing the same subject and relation).

### Decoder

Trivial. Reads the BIO tag sequence and slices `parsed.words[start:end]` by index. Output spans are guaranteed to be:

- **Atomic at word boundaries** — sliced from the actual word array, never reconstructed from subwords.
- **Verbatim** — the model never saw or modified the surface text. It rode the conveyor belt unchanged.
- **In source order** — span indices are word positions in the original sentence.

## File layout

```
src/sara_brain/cortex/transformer/hamroby_extractor_v1/
├── __init__.py             — module docstring, design philosophy
├── vocab.py                — closed grammar vocabulary + BIO tags
├── feature_extractor.py    — spaCy → ParsedSentence; whitespace-first tokenizer
├── model.py                — ExtractorConfig + GrammarEncoder
├── extraction_head.py      — ExtractionHead (word-level BIO classifier)
├── decoder.py              — decode(parsed, tags) → list[ExtractedTriple] (lenient BIO, multi-triple)
├── synthetic_features.py   — synthetic_pairs.Pair → grammar-feature training examples
├── delexicalizer.py        — real-prose content substitution (consistent word→nonsense map)
├── ud_triple_extractor.py  — gold-UD-tree → canonical (s, r, o) triples
├── real_prose_pairs.py     — UD walker emitting delexicalized real-prose Pairs
└── train.py                — supervised training loop (with --real-prose-max-sentences)
```

Reuses from elsewhere in the codebase:

- `cortex.router_data.CLEARNLP_TO_UD` — spaCy → UD label normalization
- `cortex.transformer.vocab.UPOS`, `vocab.UD_DEPS` — base grammatical vocabulary
- `cortex.transformer.vocab_en.ENGLISH_FUNCTION_WORDS` — closed-class function words
- `cortex.transformer.ud` — UD treebank loader (six English treebanks already cached)
- `cortex.transformer.v2.synthetic_pairs` — synthetic (prose, triple) pair generator with article-attachment, verb-particle pool, numbered-NP modifiers, and weird-token (scientific notation) injection

### Tokenizer: whitespace-first rule

`feature_extractor.load_domain_nlp()` returns a spaCy nlp whose tokenizer follows a single rule (per the design intuition): **if there's no whitespace inside it, it's one word**. The only splitter is trailing sentence punctuation (`.,;:!?`). Internal characters — apostrophes, hyphens, slashes, underscores, equals signs, comparison operators — stay attached.

This handles the long tail of scientific notation without enumerating cases:

```
"5'3'"        → ["5'3'"]
"kdoff"       → ["kdoff"]
"K_d"         → ["K_d"]
"p<0.05"      → ["p<0.05"]
"37°C"        → ["37°C"]
"1.2nM"       → ["1.2nM"]
"path-of-thought" → ["path-of-thought"]
"Lee's"       → ["Lee's"]
"a/b"         → ["a/b"]
```

These tokens get spaCy-tagged with their best-guess POS (often `X` or `NUM`) and a dependency role. The encoder sees `(POS, dep, head_offset, funcword_id)` for each — content rides the conveyor belt unembedded.

## Training data

Two complementary streams, both content-orthogonal:

### Synthetic (templated nonsense)

Generated by `synthetic_pairs.generate_pairs(...)`:

- **Subjects and objects** — 1–2 nonsense words ("ekefu", "bortle dekubuju"), with ~60% getting a leading article and ~18% an optional trailing number (`ekefu 3`, `creed 2`-style).
- **~20% of NPs** get a "weird-token" swap — one word replaced with a scientific-notation pattern: RNA strand (`5'3'`, `13'7'`), kinetic constant (`K_d`, `k_off`), alphanumeric ID (`SSNG3`), concentration (`1.5mM`, `37°C`), key=value (`ph=7.4`), significance (`p<0.05`), hyphenated compound (`fold-signal-region`), slash-pair (`mg/mL`), capital acronym (`ATP12`).
- **Relations** — real English action verbs from `_ACTION_VERBS` (past-tense: `carried`, `walked_to`, `examined`...) plus a bare-stem verb-particle pool (`folds_into`, `relies_on`, `binds_to`, `looks_at`, ~24 pairs). Underscores render as spaces in prose; the canonical relation in the triple is the spaced form so the head learns multi-word B-R + I-R.
- **Modifiers** — closed manner-adverb pool (`quickly`, `slowly`, ...).
- **Templates** — simple, temporal-prefix, temporal-located, located-suffix, modified, modified-located, copular (`X is Y`), copular-indef (`X is a Y`), passive (`Y was VERBed by X`), list-object-two (`X verbed A and B`), list-object-three (`X verbed A, B, and C`), and combinations with qualifiers.

Content orthogonality is structural by construction. Canary check (`grep` for `molecular snare` / `creed 2` / `switch acceptance theory` / etc.) returns zero hits over the training corpus.

### Real-prose (delexicalized gold-UD)

Per `real_prose_pairs.generate_real_prose_pairs(...)`. Activated by the `--real-prose-max-sentences N` flag (or `REAL_PROSE=N` env var on the launcher script). Pipeline:

1. Walk a UD English treebank (EWT, GUM, LinES, ParTUT, Atis, ESL — already cached).
2. For each sentence, extract canonical `(subject, relation, object)` triples directly from the **gold dependency tree** (`ud_triple_extractor.extract_triples_from_ud`). Reads UD's human-curated `nsubj` / `obj` / `obl` / `cop` / `conj` annotations; doesn't run a parser. Handles compound subjects, copular constructions, conjunct splitting, and oblique objects.
3. Build a delexicalization map (`delexicalizer.DelexMapping`) that grows across the whole corpus: same surface word always maps to the same nonsense substitute; closed-class function words pass through verbatim.
4. Apply the map to the prose AND each triple part. Re-find char spans of the substituted triple in the substituted prose.
5. Emit `synthetic_pairs.Pair`-shaped records.

This gives the head **real syntactic distribution** (PP-modified subjects, conjunctions, gerund phrases, anaphora, parentheticals) at the actual frequencies they occur in real English, with **gold-quality structural labels**, while keeping content-orthogonality intact (the model never sees the original surface words).

200 UD sentences yield ~280 training pairs (~7.5× the pair count we got from running the rule-based stub on the same input). Scaling: `REAL_PROSE=10000` produces ~14k real-prose pairs, mixed alongside synthetic pairs in the same training pool.

## Validation results

Smoke and tuning runs on a 3070 (8 GB):

| config | params | training | synthetic eval (token / triple_em) | wall-clock |
|---|---|---|---|---|
| `tiny` | 313K | 200 steps × batch 16 | 95.1% / 82.4% | 3.3s |
| `base` (initial) | 970K | 5000 steps × batch 32 | 96.2% / 85.3% | ~3 min |
| `base` (latest, gold-UD mix) | 970K | 15000 steps + 10k UD sentences | 95.3% / 82.6% | ~12 min |

Synthetic eval numbers are stable around 95% / 83% across iterations — the absolute number stops moving once the eval set itself starts including the same harder patterns the training does.

### Real-prose behavior (trained `base`, never seen during training)

After the latest training round (synthetic + 10k UD sentences, gold-UD-derived labels):

```
"The 5'3' static stem provides stability."  → ("The 5'3' static stem", 'provides', 'stability')
"Bruce Lee created Jeet Kune Do."           → ('Bruce Lee', 'created', 'Jeet Kune')   ← lost "Do"
"The molecular snare binds to its target."  → ('The molecular snare', 'binds to', 'its target')
"RNA aptamers fold into hairpins."          → ('RNA aptamers', 'fold into', 'hairpins')
"Creed 2 builds on Creed 1."                → ('Creed 2', 'builds on', 'Creed 1')
"The reaction occurs at 37°C."              → ('The reaction', 'occurs at', '37°C')
"K_d equals 5nM."                           → ('K_d', 'equals', '5nM')
"SSNG3 has high fold change."               → ('SSNG3', 'high', 'fold change')   ← relation broken
"Noticing a limitation shows the way."      → ('Noticing a limitation', 'shows', 'the way')
"Jennifer Pearl wrote the JKD paper."       → ('Jennifer Pearl', 'wrote', 'the JKD paper')
```

**No fragments anywhere.** Every output is a whole-word atomic span. The "del" / "ight" failure mode of the BPE-based v2 is structurally impossible.

Persistent failures (historical — pre-aug9; all resolved as of 2026-05-10):

```
"Marker theory predicts kdoff with p<0.05." → garbled boundaries    (fixed in aug4)
"K_d for the binding is 1.2nM."             → 2 wrong triples       (fixed in aug3)
"DNA and RNA share base pairing."           → no triple              (fixed in aug9 via spaCy cascade)
"Cluster analysis groups proteins by similarity." → garbled         (fixed in aug9)
"She bought apples and oranges."            → 1-of-2 conjuncts only (fixed in aug8 via multi-object Pair labels)
"k_off during the wash is 0.04 per second." → spurious '+second'    (fixed in aug9 via `obl`-when-no-`obj` rule)
```

Current state (aug9): on a 58-sentence extended battery covering K_d-style intj-pp, compound subjects, conjoined subjects/objects, weird tokens, gerund subjects, proper nouns with capitalized auxiliaries, plain SVO, particle verbs, copular, intransitives, pronoun subjects — **57/58 produce sensible triples** and the remaining 1 (`Strong proteins persist.`, true intransitive) correctly returns NO TRIPLE. Effectively 58/58.

Key architectural changes in aug9:
- **Multi-object Pair labels**: a single training example carries B-O labels for ALL conjuncts, replacing the per-conjunct-Pair structure that trained competing signals.
- **Gold UD features**: real-prose pairs carry pre-computed `ParsedSentence` from gold UD annotations instead of re-running spaCy on delexicalized prose.
- **`obl`-when-no-`obj` rule** in `ud_triple_extractor.py`: matches the rule stub — oblique modifiers are objects only when the verb has no direct object.
- **spaCy cascade** (sm primary + trf fallback): catches degenerate parses (all-caps acronym subjects like "DNA and RNA share...") where sm's POS classifier can't find a verb.

### Iteration trajectory

Each tuning round netted small but real wins on real prose:

| round | synthetic data | real-prose | net real-prose change |
|---|---|---|---|
| 1 | base templates (simple/temporal/located/modified/copular) | none | baseline |
| 2 | + determiner attachment (~60% of NPs get `the`/`a`/`an`) | none | full subjects with articles ("the molecular snare" instead of "snare") |
| 3 | + verb-particle pool (`folds_into`, `relies_on`...) + numbered NP (`X 2`, `Y 7`) | none | `Creed 2 builds on Creed 1`, `RNA aptamers fold into hairpins` clean |
| 4 | + weird-token generator (8 scientific-notation patterns + ~20% NP swap) | none | `K_d equals 5nM`, `37°C` clean; minor regressions on simple cases |
| 5 | (held) | rule-stub on UD (delexicalized) | ~flat — labels inherited stub's mistakes |
| 6 | (held) | gold-UD-tree → triples (delexicalized) | `5'3' static stem` fixed; partial improvement on `K_d for the binding` and `SSNG3` |

Diminishing returns are clear after round 4. Persistent failures remain across multiple iterations — those are architectural-distribution mismatches (spaCy at inference vs. UD gold at training) rather than data-recipe gaps.

## What's borrowed, what's distinctive

**Borrowed (standard NLP):**

- Transformer encoder block (multi-head attention + FFN, pre-LN). Identical to BERT-style.
- BIO span tagging output. Classic NER design.
- POS + dependency features as input. Used by older NLP systems (BiLSTM-CRF parsers, Stanford NER).
- Closed-class grammar vocabulary. HamRoby v1's design choice; this module reuses the exact 175-token vocabulary it built.
- spaCy as the parser providing POS+dep+head features.

**Distinctive (uncommon combination, motivated by Sara's substrate):**

- **Structural content orthogonality.** Mainstream NLP claims about generalization are statistical: "we held this data out, the metric on it is X." Ours is structural: the input vocabulary has no slot for domain content, so memorization is impossible by construction. Verifiable by inspecting the vocabulary tables, not by holdout statistics.
- **Conveyor-belt decoder pattern.** Surface text rides outside the model. The model performs pure positional inference; the decoder reads spans verbatim from a parallel array. Older grammar-only systems sometimes did this; transformer-based extractors usually fold content into the model. Combining the two is uncommon.
- **Atomic word output by construction.** Because the smallest unit the model sees is a word and the decoder slices a word array, there is no mechanism by which output can be sub-word. Subword fragmentation isn't avoided by training — it's structurally impossible.
- **Synthetic content-orthogonal training data.** Nonsense entities + closed-class English glue + closed verb pool. Content orthogonality is enforced at data-generation time, not just at architecture time.

This is **not** a new transformer architecture in the textbook sense. The internal math is conventional. What's new is the I/O design and the data philosophy — a coherent pattern for using transformers in a way that fits Sara's "form vs meaning" architectural commitment without introducing the failure modes mainstream NLP extractors carry.

## Usage

### Training

```
.venv/bin/python -m sara_brain.cortex.transformer.hamroby_extractor_v1.train \
  --out src/sara_brain/cortex/checkpoints/hamroby_extractor_v1.pt \
  --size base \
  --steps 15000 \
  --scenes 30000 \
  --batch-size 32 \
  --max-seq 64 \
  --lr 5e-4
```

In tmux for long runs:

```
tmux new -s hamroby-extract '.venv/bin/python -m sara_brain.cortex.transformer.hamroby_extractor_v1.train --out src/sara_brain/cortex/checkpoints/hamroby_extractor_v1.pt --size base --steps 15000 --scenes 30000 --batch-size 32 --max-seq 64 --lr 5e-4 2>&1 | tee /tmp/hamroby_extract_train.log'
```

### Inference (Python)

```python
import torch, spacy
from sara_brain.cortex.transformer.hamroby_extractor_v1.feature_extractor import parse_sentence
from sara_brain.cortex.transformer.hamroby_extractor_v1.model import ExtractorConfig, GrammarEncoder
from sara_brain.cortex.transformer.hamroby_extractor_v1.extraction_head import ExtractionHead
from sara_brain.cortex.transformer.hamroby_extractor_v1.decoder import decode

raw = torch.load("src/sara_brain/cortex/checkpoints/hamroby_extractor_v1.pt",
                 map_location="cpu", weights_only=False)
cfg = ExtractorConfig(**raw["encoder_cfg"])
encoder = GrammarEncoder(cfg)
head = ExtractionHead(encoder)
head.load_state_dict(raw["head_state"])
head.eval()
nlp = spacy.load("en_core_web_sm")

def extract(text: str):
    ps = parse_sentence(text, nlp)
    pos, dep, off, fw = zip(*ps.feature_ids)
    p = torch.tensor([list(pos)]); d = torch.tensor([list(dep)])
    o = torch.tensor([list(off)]); f = torch.tensor([list(fw)])
    tags = head.predict_tags(p, d, o, f)[0].tolist()[: len(ps.words)]
    return decode(ps, tags)

for t in extract("RNA aptamers fold into hairpins."):
    print(t.subject, "|", t.relation, "|", t.object)
```

### Wiring into the ingest pipeline

Open work item: replace the rule-based `extractor_rules.extract_triples` call in
`src/sara_reader/cli_teach_book.py` and `src/sara_brain/mcp_server.py`
(`brain_ingest(extractor="grammar")` path) with a thin adapter that loads the
trained head and exposes the same `extract_triples(clause, nlp) → list[Triple]`
interface. The rule-based stub stays as a fallback when the trained head
returns no decode.

## Open questions / future work

- **Routing head.** The previous CortexRouterV2 (BPE-based) was removed alongside v2's neural pieces. A new content-free routing head on top of `GrammarEncoder` would restore the `--cortex-router-v2` capability without re-introducing BPE; the `RouterHead` design from v2 is reusable, just retrained against grammar-feature input.
- **Real-prose evaluation harness.** The v2 eval (`v2/eval_real_prose.py`) compared against author-curated teach-script triples, which mismatched surface extraction. A new harness is needed that measures the head against held-out *surface* triples — either hand-labeled or from a public benchmark like WebNLG.
- **Multi-clause input.** Currently the head expects a single clause; long compound sentences need to be split via `EnhancedParser._split_compound` upstream (the ingest pipeline already does this).
- **Domain transfer evaluation.** Once trained, run on a Wikipedia article in an unrelated domain to confirm the structural orthogonality holds — extraction quality should be the same on a topic the model has never seen because the model has no representation of any topic.

## Related

- `docs/v025_hamlinllm_status.md` — original HamRoby v1 router design
- `docs/v026_hamroby_name.md` — naming + form/meaning split
- `docs/v028_multi_layer_cortex_architecture.md` — L1/L2 vocabulary architecture
- `docs/v029_vocab_en_plan.md` — function-word vocabulary motivation
- `src/sara_brain/cortex/transformer/v2/__init__.py` — deprecation note for the BPE-based predecessor
