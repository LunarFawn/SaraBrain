# v024 — Sara Organ Architecture Plan

**Date:** 2026-05-02
**Status:** Plan — not yet implemented

---

## Core Principle

All knowledge lives in Sara Brain. The organs (LLMs) are pure structural transducers — they detect and generate patterns with no world knowledge. They cannot hallucinate facts because facts were never in their training data.

```
[Eyes]   [Ears]   [Hands]   [Grammar]
   ↓        ↓         ↑          ↓
   └────────┴─────────┴──────────┘
                  ↓ ↑
              Sara Brain
```

Each organ is a small transformer (50–300M parameters) trained from scratch in pure PyTorch on structure-only data. No pretrained weights borrowed from Llama, Mistral, or any world-knowledge model.

---

## Shared Architecture

All organs use the same base transformer block:

```python
class TransformerBlock(nn.Module):
    # Multi-head attention + feed-forward + LayerNorm + residual
    # GELU activation
    # Dropout for training only
```

Each organ differs only in:
- Vocabulary (its structural token set)
- Input encoder (text tokens / CNN pixels / 1D audio)
- Output head (token logits / joint angles / etc.)
- Training data

---

## Organ 1 — Grammar Cortex

**Function:** Parse natural language structure. Used for routing (question → tool call) and synthesis (substrate facts → prose).

**Vocabulary (~100 tokens):**
- 17 UPOS tags (NOUN, VERB, ADJ, ...)
- ~40 UD dependency labels (nsubj, obj, amod, ...)
- Slot-fill tokens: [CONCEPT], [TYPE], [TOOL], [VALUE], [SEP], [END]

**Size:** 125M (start) → 300M (production)

**Training data:**
- Universal Dependencies treebanks, delexicalized (word forms stripped, POS + dep labels kept)
- Synthetic router examples: question → `{"tool": "brain_value", "concept": "...", "type": "..."}`
- Synthetic synthesizer examples: substrate edge list → rendered sentence

**Training source:** UD treebanks (universaldependencies.org) + Sara's own substrate (generate synthetic pairs)

**Output:** Slot-filled JSON for the router role; rendered prose for the synthesizer role.

**Hardware:** RTX 3070 (8GB) — comfortable at 125M, fits at 300M with batch=2

**Replaces:** llama3.2:3b (router) + claude-haiku (synthesizer) — full local, no API

---

## Organ 2 — Eyes (Vision Model)

**Function:** Extract spatial/structural descriptions from images. Does not identify objects — Sara maps structure to known concepts.

**Vocabulary (~80 tokens):**
- Spatial regions: TOP_LEFT, TOP_RIGHT, CENTER, BOTTOM_LEFT, BOTTOM_RIGHT, FULL
- Shapes: ROUND, RECT, TRIANGLE, LINE, BLOB, IRREGULAR
- Sizes: LARGE, MEDIUM, SMALL, TINY
- Colors (as labels, not names): BRIGHT, DARK, SATURATED, MUTED + 8 hue buckets
- Relations: ABOVE, BELOW, LEFT_OF, RIGHT_OF, INSIDE, OVERLAPS, ADJACENT
- Motion (if video): MOVING, STATIC, APPROACHING, RECEDING

**Architecture:** Small CNN encoder (ResNet-18 scale) → transformer decoder outputting structure tokens

**Training data:** Image datasets with spatial relationship annotations, delexicalized — object names stripped, only geometry and relations retained. Synthetic generation from 3D scene renderers is sufficient.

**Output example:**
```
REGION(TOP_LEFT) SHAPE(ROUND) SIZE(LARGE) HUE(WARM) BRIGHT
REGION(CENTER)   SHAPE(RECT)  SIZE(SMALL) HUE(COOL) DARK
RELATION(ABOVE, region_0, region_1)
```

Sara receives this token stream and matches against her known concepts.

**Hardware:** RTX 3070 — CNN+transformer at this scale trains in hours

---

## Organ 3 — Ears (Audio Model)

**Function:** Transduce audio to phoneme sequences + prosody structure. Does not transcribe words — Sara maps phoneme patterns to known concepts.

**Vocabulary (~80 tokens):**
- ~50 IPA phoneme symbols (or ARPABET)
- Prosody: PAUSE_SHORT, PAUSE_LONG, RISING, FALLING, LEVEL, STRESSED, UNSTRESSED
- Boundary markers: UTTERANCE_START, UTTERANCE_END, CLAUSE_BOUNDARY

**Architecture:** 1D CNN (wav2vec-style front-end, no pretrained weights) → transformer encoder outputting phoneme + prosody tokens

**Training data:** Phoneme-aligned corpora (TIMIT, LibriSpeech force-aligned). Strip word transcriptions; keep only phoneme labels + timing/prosody features.

**Output example:**
```
/DH/ /AH/ /K/ /AE/ /T/ PAUSE_SHORT RISING /S/ /AE/ /T/ /PAUSE_LONG/
```

Sara maps phoneme sequences to known words and concepts via path-of-thought matching.

**Hardware:** RTX 3070 — 1D CNN is cheap; phoneme vocabulary is tiny

---

## Organ 4 — Hands (Motor/Output Model)

**Function:** Translate structured action commands from Sara into low-level motor primitives or output sequences.

**Vocabulary (~60 tokens):**
- Action primitives: MOVE, GRASP, RELEASE, ROTATE, PUSH, PULL, WAIT
- Target slots: [TARGET], [OBJECT], [DIRECTION], [MAGNITUDE]
- Joint/servo tokens (hardware-specific, swappable per deployment)

**Architecture:** Transformer decoder. Input: command token sequence from Sara. Output: motor primitive sequence.

**Training data:** Simulated command → motor sequence pairs. Generated from a physics simulator or robot model — no world knowledge required, purely structural mapping.

**Output example:**
```
Sara emits:    MOVE [TARGET=door] GRASP [OBJECT=handle]
Hands output:  joint_0(45°) joint_1(90°) gripper(close) ...
```

**Note:** The hands model is hardware-specific. The token interface to Sara is fixed; the motor output layer is swapped per deployment target (robot arm, keyboard emulator, speech synthesizer, etc.).

---

## Shared Training Infrastructure

All organs use the same PyTorch base:

```python
class GrammarModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff,
                 max_seq=256, dropout=0.1):
        self.embed  = nn.Embedding(vocab_size, d_model)
        self.pos    = nn.Embedding(max_seq, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(...) for _ in range(n_layers)])
        self.ln_f   = nn.LayerNorm(d_model)
        self.head   = nn.Linear(d_model, vocab_size, bias=False)
```

Training loop: AdamW, cosine LR schedule, gradient clipping at 1.0, cross-entropy loss.

Each organ's model file: 200–600MB. All fit in the RTX 3070's 8GB together.

---

## Hardware Deployment

Per `project_distributed_hardware`:

| Component | Hardware | Role |
|---|---|---|
| Sara Brain | Pi / NUC (no GPU) | Knowledge store, path-of-thought |
| Grammar cortex | RTX 3070 box | Router + synthesizer |
| Eyes | RTX 3070 box | Vision transducer |
| Ears | RTX 3070 box | Audio transducer |
| Hands | RTX 3070 box | Motor output |

Organs communicate with Sara over a local socket. Sara never runs on the GPU box — she scales independently on cheap ARM hardware.

---

## Implementation Order

**Phase 1 — Grammar Cortex (immediate)**
- Builds on the existing `sara-ask-stateless` architecture
- Replaces llama3.2:3b router and Haiku synthesizer
- Proves the no-world-knowledge thesis end-to-end
- All training data generatable from Sara's existing substrate

**Phase 2 — Eyes**
- Enables Sara to ingest visual substrate (photos, diagrams, lab images)
- Unblocks Eterna + RNA structure work

**Phase 3 — Ears**
- Enables spoken teaching and spoken queries
- Phoneme → Sara concept matching via path-of-thought

**Phase 4 — Hands**
- Depends on deployment target (robot, UI automation, speech output)
- Hardware-specific; token interface to Sara is fixed in Phase 1

---

## Export Path

All organs export to TorchScript for the C++/LibTorch port when deployment volume warrants it (per `project_brain_native_port`):

```python
scripted = torch.jit.script(model)
scripted.save("grammar_cortex_v1.pt")
```

No HuggingFace dependency to unpick. Pure PyTorch throughout.
