"""HamRobySum inference — RESEARCH ARTIFACT (not part of the production
two-layer architecture).

**Architectural status:** HamRobySum-EN was an experiment in rendering
substrate edges as prose using a small (~30M-param) slot-based LLM.
It is **not part of the two-layer architecture from Pearl 2026a**
("LLM as cortex, Sara Brain as hippocampus" — §7.3) — it adds a third
"language production" layer that the papers explicitly leave to the
consumer cortex (Pearl 2026 rev8 §2.4: *"a reader LLM receives not a
single 'best' answer but a structured neighborhood of related triples.
The reader must do its own selection."*).

**Why this code is preserved:** the empirical finding from v035–v048.1
is that small renderers (30M params, slot-based, synthetic training)
cannot reliably replace a frontier cortex for prose generation. Each
new substrate shape produced new failure modes (homogeneous-cluster
mangling, qualifier-pattern collapse, discourse-slot leakage, etc.)
which required a growing pile of inference-side patches. That patching
pile *is the empirical evidence* for the two-layer architecture —
the third layer protests by misbehaving. Preserving the artifact and
reframing it as a counterexample makes the result citable.

**How to access:** `chat.py --format prose --use-hamrobysum`.
Default chat REPL paths (`--format raw`, `--format prose` without
`--use-hamrobysum`) do not load this module's model.

**Production architecture (paper-aligned):**
- Hippocampus (Sara): SQLite path graph, structured triples, MCP-served.
- Cortex (frontier LLM, frozen): receives `--format raw` triples or
  MCP output, does selection + synthesis + prose.

See `docs/v050_two_layer_realignment.md` for the full architectural
reasoning. See `docs/v035_generic_slot_hamrobysum.md` through
`docs/v048_1_richer_training_data.md` for the research artifact's
development history.

────────────────────────────────────────────────────────────────────

Original module description (preserved for research-mode users):

Loads a synth checkpoint (which embeds the brain-extended vocab),
formats an edge cluster into the same `<facts>...<prose>` prefix the
trainer saw, and decodes greedily (or with sampling) to `</prose>`.
The decoded prose tokens are detokenized back to a string.

Falls back gracefully:
- If no synth ckpt is loaded → caller should use `synthesizer.render_edges`
- If the facts prefix exceeds `max_seq` → truncates edges (oldest first)
  with a warning printed once
- If decoding never emits `</prose>` within `max_new_tokens` → returns
  what was generated up to that point

Usage:
    .venv/bin/python -m sara_brain.cortex.transformer.inference_synth \\
        --ckpt src/sara_brain/cortex/checkpoints/hamroby_sum_synth_pairs_002000.pt \\
        --brain /tmp/sara_demo.db \\
        --topic "ribosome"
"""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import torch

from .model import GrammarConfig, GrammarModel
from .synth_data import (
    Edge, _tokenize_text, build_pred_mapping, build_slot_mapping,
    cluster_by_subject, load_substrate_edges, SynthExample,
)
from .vocab_synth import (
    BOS_ID,
    EDGE_SEP_ID,
    END_PROSE_ID,
    EOS_ID,
    FACTS_ID,
    ID2TOK_SYNTH,
    OBJ_ID,
    PAD_ID,
    PRED_ID,
    PROSE_ID,
    REFUTED_ID,
    ATTR_ID,
    SUBJ_ID,
    SYNTH_PRED_SLOT_SET,
    SYNTH_SLOT_ID_SET,
    SYNTH_SLOTS,
    TOK2ID_SYNTH,
    UNK_ID,
    VOCAB_SIZE_SYNTH,
)


_DETOKENIZE_NO_LEAD_SPACE = {".", ",", ";", ":", "?", "!", "'s", "n't", ")"}
_DETOKENIZE_NO_TRAIL_SPACE = {"(",}


# v039 slice 2 — article post-processor.
# Matches an article (`a` or `an`, any case) followed by whitespace
# and the first letter of the next word. We only fix obvious vowel
# agreement mismatches; we never insert or remove articles. The
# model's emission decides "should there be an article here"; we
# only fix `a apple` -> `an apple` and `an cat` -> `a cat`.
_ARTICLE_FIX_RE = re.compile(r"\b(a|an|A|An)\s+(\w)")


def _fix_articles(text: str) -> str:
    """Swap `a` ↔ `an` based on vowel-onset of the following word.
    Conservative: never inserts or removes articles, only fixes
    vowel-agreement mismatches in already-emitted output."""
    def _swap(m):
        article = m.group(1)
        next_letter = m.group(2)
        is_vowel = next_letter.lower() in "aeiou"
        is_an = article.lower() == "an"
        if is_vowel and not is_an:
            # `a apple` -> `an apple`. Preserve original case of `a`.
            new_article = "An" if article[0].isupper() else "an"
            return f"{new_article} {next_letter}"
        if not is_vowel and is_an:
            # `an cat` -> `a cat`. Preserve original case.
            new_article = "A" if article[0].isupper() else "a"
            return f"{new_article} {next_letter}"
        return m.group(0)  # already correct
    return _ARTICLE_FIX_RE.sub(_swap, text)


def _detokenize(tokens: list[str]) -> str:
    """Turn a list of prose tokens back into a readable string. Glues
    punctuation to the previous word; capitalizes the first letter."""
    if not tokens:
        return ""
    out: list[str] = []
    prev_no_trail = False
    for tok in tokens:
        if not out:
            out.append(tok)
        elif tok in _DETOKENIZE_NO_LEAD_SPACE:
            out.append(tok)
        elif prev_no_trail:
            out.append(tok)
        else:
            out.append(" ")
            out.append(tok)
        prev_no_trail = tok in _DETOKENIZE_NO_TRAIL_SPACE
    text = "".join(out)
    if text:
        text = text[0].upper() + text[1:]
    return text


def load_synth_checkpoint(
    path: Path, device: torch.device,
) -> GrammarModel:
    """Load a v035 generic synth checkpoint. Vocab is the fixed
    VOCAB_SYNTH — no per-checkpoint vocab loading."""
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = GrammarConfig(**ck["config"])
    if cfg.vocab_size != VOCAB_SIZE_SYNTH:
        raise SystemExit(
            f"checkpoint vocab_size={cfg.vocab_size} but "
            f"VOCAB_SIZE_SYNTH={VOCAB_SIZE_SYNTH}; this is not a "
            f"v035-generic synth checkpoint. Older per-brain ckpts "
            f"are deprecated — retrain with v035 train_synth.py."
        )
    model = GrammarModel(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(
        f"[load] {path.name}  step={ck.get('step')}  "
        f"loss={ck.get('loss', float('nan')):.4f}  "
        f"dev_loss={ck.get('dev_loss', float('nan')):.4f}  "
        f"vocab={cfg.vocab_size}  flavor={ck.get('vocab_flavor', 'unknown')}",
        flush=True,
    )
    return model


def load_vocab_brain(path: Path) -> dict[str, list[str]]:
    """Open a vocab brain (per v040 — a Sara brain.db whose neurons
    are relation names mapped to English phrases via the
    `english_form` segment relation). Returns
    `{relation_name: [phrase, ...]}` — every form for each relation.

    v043: a relation may carry multiple english_form segments (taught
    via `/teach-vocab`); inference rotates among them at decode time
    via `_expand_pred_slots`. Order is segment-insertion order via
    SQLite's `s.id ASC`."""
    import sqlite3
    conn = sqlite3.connect(str(path))
    rows = conn.execute(
        "SELECT n_src.label, n_tgt.label "
        "FROM segments s "
        "JOIN neurons n_src ON s.source_id = n_src.id "
        "JOIN neurons n_tgt ON s.target_id = n_tgt.id "
        "WHERE s.relation = 'english_form' "
        "ORDER BY n_src.label, s.id"
    ).fetchall()
    conn.close()
    lookup: dict[str, list[str]] = {}
    for relation, phrase in rows:
        lookup.setdefault(relation, []).append(phrase)
    return lookup


def _facts_prefix_with_slots(
    edges: list[Edge],
    slot_mapping: dict[str, str],
    pred_mapping: dict[str, str] | None = None,
) -> list[int]:
    """Build the facts prefix + leading <prose> marker. Substitutes
    each substrate string with its content slot token AND each
    relation name with its predicate slot token (v040). Predicates
    not in `pred_mapping` fall back to literal-encoding via
    `relation.replace("_", " ")`."""
    inv = {v.lower(): k for k, v in slot_mapping.items()}
    pred_mapping = pred_mapping or {}

    def _slot_or_encode(s: str) -> list[int]:
        s_norm = s.strip()
        if s_norm.lower() in inv:
            return [TOK2ID_SYNTH[inv[s_norm.lower()]]]
        return [TOK2ID_SYNTH.get(t, UNK_ID) for t in _tokenize_text(s_norm)]

    ids: list[int] = [BOS_ID, FACTS_ID]
    for e in edges:
        ids.append(SUBJ_ID)
        ids.extend(_slot_or_encode(e.src))
        ids.append(PRED_ID)
        if e.rel in pred_mapping:
            ids.append(TOK2ID_SYNTH[pred_mapping[e.rel]])
        else:
            for t in _tokenize_text(e.rel.replace("_", " ")):
                ids.append(TOK2ID_SYNTH.get(t, UNK_ID))
        ids.append(OBJ_ID)
        ids.extend(_slot_or_encode(e.tgt))
        if e.refuted:
            ids.append(REFUTED_ID)
        if e.target_was_attribute:
            ids.append(ATTR_ID)
        ids.append(EDGE_SEP_ID)
    ids.append(PROSE_ID)
    return ids


def _expand_slots(prose_tokens: list[str], slot_mapping: dict[str, str]) -> list[str]:
    """Replace each `<Cn>` token with its substrate string. Other
    tokens pass through unchanged."""
    out: list[str] = []
    for tok in prose_tokens:
        if tok in slot_mapping:
            out.append(slot_mapping[tok])
        else:
            out.append(tok)
    return out


# v048 — discourse-slot expansion. The training corpus uses `<R0>`...
# `<R3>` as discourse-connective slots; at inference time we map them
# to a fixed pool of English connectives, deterministic-by-index so
# repeated renders of the same training pattern stay stable.
_DISCOURSE_POOL: tuple[str, ...] = (
    "however", "therefore", "meanwhile", "furthermore",
)


def _expand_discourse_slots(prose_tokens: list[str]) -> list[str]:
    """Map `<R0>`...`<R3>` -> connective word. Tokens outside that
    range pass through. Defensive when the slot index is unexpected."""
    out: list[str] = []
    for tok in prose_tokens:
        if tok.startswith("<R") and tok.endswith(">"):
            try:
                idx = int(tok[2:-1])
            except ValueError:
                out.append(tok)
                continue
            if 0 <= idx < len(_DISCOURSE_POOL):
                out.append(_DISCOURSE_POOL[idx])
            else:
                out.append(_DISCOURSE_POOL[idx % len(_DISCOURSE_POOL)])
            continue
        out.append(tok)
    return out


def _combine_same_subject_slotted(prose_tokens: list[str]) -> list[str]:
    """v044 — post-decode combining for same-subject runs.

    Operates on slot-format prose tokens BEFORE slot expansion. Groups
    ADJACENT sentences (delimited by `.`) that share a leading subject
    (everything before the first `<Pn>` token) and joins their
    predicates with Oxford-comma clauses.

    Example:
        ['<C0>', '<P0>', '<C1>', '.',
         '<C0>', '<P1>', '<C2>', '.',
         '<C0>', '<P2>', '<C3>', '.']
      ->
        ['<C0>', '<P0>', '<C1>', ',',
         '<P1>', '<C2>', ',', 'and',
         '<P2>', '<C3>', '.']

    Adjacent-only: never reorders sentences across non-adjacent
    positions. Preserves the model's emission order, which encodes
    attribute-flag and topic priority.

    Defensive: sentences without a `<Pn>` (or with an empty subject)
    pass through as standalone with no combining attempt."""
    if not prose_tokens:
        return prose_tokens

    # Split into sentences at '.' tokens.
    sentences: list[list[str]] = []
    current: list[str] = []
    for tok in prose_tokens:
        if tok == ".":
            if current:
                sentences.append(current)
                current = []
        else:
            current.append(tok)
    if current:
        sentences.append(current)
    if not sentences:
        return prose_tokens

    # Identify (subject, predicate) for each sentence.
    parsed: list[tuple[tuple[str, ...], list[str]]] = []
    for sent in sentences:
        # Find first <Pn> slot — everything before is subject.
        split_at = None
        for i, t in enumerate(sent):
            if t in SYNTH_PRED_SLOT_SET:
                split_at = i
                break
        if split_at is None or split_at == 0:
            # No <Pn> or empty subject — treat as standalone.
            parsed.append((tuple(sent), []))
        else:
            parsed.append((tuple(sent[:split_at]), sent[split_at:]))

    # Group adjacent same-subject sentences.
    # Each group: (subject_tokens, list_of_predicates_or_None).
    # None marker = standalone (cannot combine).
    groups: list[tuple[tuple[str, ...], list[list[str]] | None]] = []
    for subj, pred in parsed:
        if not pred:
            groups.append((subj, None))
            continue
        if (groups
                and groups[-1][1] is not None
                and groups[-1][0] == subj):
            groups[-1][1].append(pred)
        else:
            groups.append((subj, [pred]))

    # Emit combined token stream.
    out: list[str] = []
    for subj, preds in groups:
        out.extend(subj)
        if preds is None:
            # Standalone — no predicates, but we still terminate the
            # sentence with a period so detokenization is consistent.
            out.append(".")
            continue
        if len(preds) == 1:
            out.extend(preds[0])
        elif len(preds) == 2:
            out.extend(preds[0])
            out.append("and")
            out.extend(preds[1])
        else:
            # 3+: Oxford-comma list.
            for i, pred in enumerate(preds):
                out.extend(pred)
                if i == len(preds) - 2:
                    out.append(",")
                    out.append("and")
                elif i < len(preds) - 1:
                    out.append(",")
        out.append(".")
    return out


def _expand_pred_slots(
    prose_tokens: list[str],
    pred_mapping: dict[str, str],
    vocab_lookup: dict[str, list[str]],
) -> list[str]:
    """Replace each `<Pn>` token with an English phrase from the
    vocab brain (or fall back to `relation.replace("_", " ")` if the
    relation isn't in the vocab brain). v040 base; v043 adds
    multi-form rotation.

    When a relation has multiple english_form segments
    (`/teach-vocab` adds alternates), the N-th emission of `<Pn>`
    in the prose picks `forms[N % len(forms)]` — deterministic
    round-robin per slot. Same cluster + same checkpoint always
    produces the same expansion.

    Defensive: if the model emits a `<Pn>` token whose index isn't
    in `pred_mapping` (e.g. single-relation cluster only allocated
    `<P0>` but the model overgeneralized to `<P1>`), fall back to
    the FIRST allocated relation."""
    if not pred_mapping:
        return prose_tokens
    # Reverse: <Pn> -> relation name -> english phrase(s)
    reverse = {slot: rel for rel, slot in pred_mapping.items()}
    default_relation = next(iter(pred_mapping)) if pred_mapping else None
    # Per-slot emission counter for round-robin form selection.
    emission_count: dict[str, int] = {}

    out: list[str] = []
    for tok in prose_tokens:
        if tok in reverse:
            relation = reverse[tok]
        elif tok in SYNTH_PRED_SLOT_SET and default_relation is not None:
            relation = default_relation
        else:
            out.append(tok)
            continue
        forms = vocab_lookup.get(relation)
        if forms:
            idx = emission_count.get(tok, 0)
            phrase = forms[idx % len(forms)]
            emission_count[tok] = idx + 1
        else:
            phrase = relation.replace("_", " ")
        out.append(phrase)
    return out


def _apply_repetition_penalty(
    logits: torch.Tensor,
    recent_ids: list[int],
    penalty: float,
) -> None:
    """In-place: divide positive logits / multiply negative logits by
    `penalty` for any token id in `recent_ids`. HuggingFace-style.
    `penalty>1.0` discourages repeats; `penalty=1.0` is a no-op."""
    if penalty == 1.0 or not recent_ids:
        return
    for tid in set(recent_ids):
        v = logits[tid].item()
        logits[tid] = v / penalty if v > 0 else v * penalty


def _would_close_repeating_ngram(
    out_ids: list[int], candidate: int, n: int,
) -> bool:
    """True when appending `candidate` to `out_ids` would form an
    n-gram that already appeared earlier in `out_ids`. n=0 disables."""
    if n <= 0 or len(out_ids) < n - 1:
        return False
    new_ngram = tuple(out_ids[-(n - 1):]) + (candidate,) if n > 1 else (candidate,)
    # Look for new_ngram anywhere in out_ids (treating out_ids as one
    # window of past emissions).
    for i in range(0, len(out_ids) - n + 1):
        if tuple(out_ids[i:i + n]) == new_ngram:
            return True
    return False


@torch.no_grad()
def synthesize_cluster(
    model: GrammarModel,
    edges: list[Edge],
    device: torch.device,
    max_new_tokens: int = 80,
    temperature: float = 0.0,
    top_k: int = 0,
    rng: random.Random | None = None,
    repetition_penalty: float = 1.0,
    repetition_window: int = 32,
    no_repeat_ngram_size: int = 0,
    vocab_lookup: dict[str, str] | None = None,
    max_cluster_size: int = 8,
) -> str:
    """Render `edges` as prose. v040: dual slot pipeline.

    Builds per-cluster mappings for content (`<C0>`...`<C31>`) AND
    predicates (`<P0>`...`<P15>`). Formats facts prefix with both
    slot types. Decodes prose. Expands `<Pn>` via `vocab_lookup`
    (relation -> English phrase), then expands `<Cn>` via the
    cluster's content mapping, then detokenizes, then runs the v039
    article post-processor.

    `vocab_lookup` defaults to `{}`, in which case predicates fall
    back to `relation.replace("_", " ")` (same as the unknown-relation
    path).

    v046: when `len(edges) > max_cluster_size`, split into chunks of
    `max_cluster_size` and render each chunk independently, then join
    with `" "`. The model degenerates on big clusters (training
    distribution was 1-8 edges; 20+ edge clusters loop and emit
    bare slot expansions). Chunking keeps each render in-distribution.
    Set `max_cluster_size=0` to disable."""
    if not edges:
        return ""
    if max_cluster_size > 0 and len(edges) > max_cluster_size:
        parts: list[str] = []
        for i in range(0, len(edges), max_cluster_size):
            chunk = edges[i:i + max_cluster_size]
            parts.append(synthesize_cluster(
                model, chunk, device,
                max_new_tokens=max_new_tokens,
                temperature=temperature, top_k=top_k, rng=rng,
                repetition_penalty=repetition_penalty,
                repetition_window=repetition_window,
                no_repeat_ngram_size=no_repeat_ngram_size,
                vocab_lookup=vocab_lookup,
                max_cluster_size=0,
            ))
        return " ".join(p for p in parts if p)
    rng = rng or random.Random(0)
    max_seq = model.cfg.max_seq
    vocab_lookup = vocab_lookup or {}

    # Build per-cluster mappings from a pseudo-example.
    pseudo = SynthExample(edges=list(edges), prose="", subject="")
    slot_mapping = build_slot_mapping(pseudo)
    pred_mapping = build_pred_mapping(pseudo)

    # Build facts prefix; truncate edges if it overflows.
    prefix = _facts_prefix_with_slots(edges, slot_mapping, pred_mapping)
    truncated = 0
    while len(prefix) >= max_seq - 4 and len(edges) > 1:
        edges = edges[1:]   # drop oldest first
        pseudo = SynthExample(edges=list(edges), prose="", subject="")
        slot_mapping = build_slot_mapping(pseudo)
        pred_mapping = build_pred_mapping(pseudo)
        prefix = _facts_prefix_with_slots(edges, slot_mapping, pred_mapping)
        truncated += 1
    if truncated:
        print(f"[synth] truncated {truncated} edges to fit max_seq={max_seq}",
              flush=True)

    ids = list(prefix)
    out_ids: list[int] = []
    for _ in range(max_new_tokens):
        if len(ids) >= max_seq:
            break
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits, _ = model(x)
        next_logits = logits[0, -1].clone()

        # Apply repetition penalty over the prose tail only — the facts
        # prefix is conditioning, repeats there are expected.
        recent = out_ids[-repetition_window:] if repetition_window else out_ids
        _apply_repetition_penalty(next_logits, recent, repetition_penalty)

        if temperature > 0.0:
            next_logits = next_logits / temperature
            if top_k > 0:
                v, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < v[-1]] = -float("inf")
            probs = torch.softmax(next_logits.float(), dim=-1).cpu().numpy()
            nxt = rng.choices(range(len(probs)), weights=probs.tolist(), k=1)[0]
        else:
            nxt = int(next_logits.argmax().item())

        # n-gram veto: try up to a few alternates if the top pick would
        # close a repeating n-gram. After that, give up and emit anyway
        # to avoid an infinite loop.
        if no_repeat_ngram_size > 0 and _would_close_repeating_ngram(
            out_ids, nxt, no_repeat_ngram_size,
        ):
            sorted_ids = next_logits.argsort(descending=True).tolist()
            for alt in sorted_ids[:8]:
                if not _would_close_repeating_ngram(
                    out_ids, alt, no_repeat_ngram_size,
                ):
                    nxt = alt
                    break

        ids.append(nxt)
        if nxt == END_PROSE_ID or nxt == EOS_ID:
            break
        out_ids.append(nxt)

    # Strip structural delimiters that leaked into output.
    structural_ids = {
        FACTS_ID, PROSE_ID, END_PROSE_ID, SUBJ_ID, PRED_ID, OBJ_ID,
        EDGE_SEP_ID, REFUTED_ID, ATTR_ID, BOS_ID, EOS_ID, PAD_ID,
    }
    prose_tokens = [
        ID2TOK_SYNTH[i] for i in out_ids if i not in structural_ids
    ]
    # v044: combine adjacent same-subject sentences into Oxford-comma
    # clauses BEFORE slot expansion (subjects are still <Cn> tokens).
    prose_tokens = _combine_same_subject_slotted(prose_tokens)
    # v040: expand predicate slots first (relation -> English phrase
    # via vocab brain lookup), then content slots (substrate strings).
    expanded = _expand_pred_slots(prose_tokens, pred_mapping, vocab_lookup)
    expanded = _expand_slots(expanded, slot_mapping)
    # v048: discourse-slot expansion (<R0>..<R3> -> however/therefore/...).
    expanded = _expand_discourse_slots(expanded)
    text = _detokenize(expanded)
    # v039 slice 2: a/an vowel-onset agreement on slot-expanded prose.
    return _fix_articles(text)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True,
                   help="Path to a hamroby_sum_*.pt checkpoint")
    p.add_argument("--brain", type=Path, required=True,
                   help="brain.db whose edges to synthesize from. The vocab "
                        "should match what the ckpt was trained on.")
    p.add_argument("--topic", type=str, default=None,
                   help="If set, only render the cluster whose subject "
                        "matches the topic (substring, case-insensitive)")
    p.add_argument("--n", type=int, default=5,
                   help="Number of clusters to render (when no --topic given)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 = greedy, >0 = sampling")
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--repetition-penalty", type=float, default=1.0,
                   help="Divide logits of recently-emitted tokens by this. "
                        "1.0 = no penalty (v0 behavior). 1.2 is a good "
                        "starting value to break out of repetition loops.")
    p.add_argument("--no-repeat-ngram-size", type=int, default=0,
                   help="Ban candidates that would close an n-gram already "
                        "present in the prose tail. 0 = disabled. 3 is "
                        "common.")
    p.add_argument("--repetition-window", type=int, default=32,
                   help="How many recent prose tokens the repetition "
                        "penalty considers.")
    p.add_argument("--max-cluster-size", type=int, default=8,
                   help="v046: split clusters above this size into "
                        "chunks and render each separately. 0 disables "
                        "chunking (pre-v046 behavior).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--vocab-brain", type=Path,
        default=Path("src/sara_brain/cortex/vocab/vocab_en.db"),
        help="Vocab brain (per v040): a Sara brain.db mapping relation "
             "names to English phrases. Default ships at "
             "src/sara_brain/cortex/vocab/vocab_en.db.",
    )
    args = p.parse_args()

    device = torch.device(args.device)
    rng = random.Random(args.seed)
    model = load_synth_checkpoint(args.ckpt, device)
    if args.vocab_brain.exists():
        vocab_lookup = load_vocab_brain(args.vocab_brain)
        print(f"[vocab] loaded {len(vocab_lookup)} relation -> english_form mappings "
              f"from {args.vocab_brain}", flush=True)
    else:
        vocab_lookup = {}
        print(f"[vocab] vocab brain not found at {args.vocab_brain}; predicates "
              f"will fall back to relation.replace('_',' ')", flush=True)

    edges = load_substrate_edges(args.brain)
    clusters = cluster_by_subject(edges)
    print(f"loaded {len(edges)} edges, {len(clusters)} clusters from {args.brain}")

    if args.topic:
        topic_l = args.topic.lower()
        matching = [(s, c) for s, c in clusters.items() if topic_l in s.lower()]
        if not matching:
            print(f"no cluster matches topic {args.topic!r}")
            return
        for subject, cluster in matching[:args.n]:
            print()
            print(f"=== {subject!r}  ({len(cluster)} edges) ===")
            for e in cluster[:8]:
                print(f"   {e.src} --[{e.rel}]--> {e.tgt}{' [attr]' if e.target_was_attribute else ''}")
            if len(cluster) > 8:
                print(f"   ... +{len(cluster) - 8} more")
            prose = synthesize_cluster(
                model, cluster, device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_k=args.top_k, rng=rng,
                repetition_penalty=args.repetition_penalty,
                repetition_window=args.repetition_window,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                vocab_lookup=vocab_lookup,
                max_cluster_size=args.max_cluster_size,
            )
            print(f"   PROSE: {prose}")
    else:
        items = list(clusters.items())
        rng.shuffle(items)
        for subject, cluster in items[:args.n]:
            print()
            print(f"=== {subject!r}  ({len(cluster)} edges) ===")
            for e in cluster[:6]:
                print(f"   {e.src} --[{e.rel}]--> {e.tgt}{' [attr]' if e.target_was_attribute else ''}")
            if len(cluster) > 6:
                print(f"   ... +{len(cluster) - 6} more")
            prose = synthesize_cluster(
                model, cluster, device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_k=args.top_k, rng=rng,
                repetition_penalty=args.repetition_penalty,
                repetition_window=args.repetition_window,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                vocab_lookup=vocab_lookup,
                max_cluster_size=args.max_cluster_size,
            )
            print(f"   PROSE: {prose}")


if __name__ == "__main__":
    main()
