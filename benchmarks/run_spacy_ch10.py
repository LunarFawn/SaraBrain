#!/usr/bin/env python3
"""Compartmentalized exam — spaCy cortex (no LLM).

Same pipeline as run_compartment_exam.py: route → regional activation →
cortex picks answer. Difference: cortex is spaCy + lemma/SVO/temporal
scoring, not Ollama. Zero ollama calls, zero Mac memory risk.

Usage:
    .venv/bin/python benchmarks/run_spacy_ch10.py \\
        --db claude_taught.db \\
        --questions benchmarks/ch10_test_questions.json
"""
from __future__ import annotations

import argparse
import json
import re
import time

import spacy

from sara_brain.core.brain import Brain
from sara_brain.core.short_term import ShortTerm
from sara_brain.core.recognizer import Recognizer
from sara_brain.storage.neuron_repo import NeuronRepo
from sara_brain.storage.segment_repo import SegmentRepo


TEMPORAL_PREPS = {"during", "after", "before", "while", "until",
                  "since", "when", "as"}

_WORD_RE = re.compile(r"[a-z0-9]+")


def _singularize_fallback(word: str) -> str | None:
    """Return a singularized form when spaCy's lemma keeps the plural
    (e.g. 'snrnps' → 'snrnp'). Used as a fallback for acronym plurals
    and domain terms spaCy doesn't know. Returns None if no change
    would be produced."""
    if len(word) < 4:
        return None
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "xes", "zes", "ses")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return None


def path_words(label: str, nlp=None) -> set[str]:
    """Break a neuron label into lemmas (or words if no nlp given).
    Attribute suffixes ('cell cycle_attribute') are flattened.
    Lemmatizing here lets path-word matching line up with spaCy's
    lemmatized choice lemmas (so 'stabilizing' and 'stabilize' match).
    Also adds singularized fallbacks so 'snrnps' lines up with 'snrnp'."""
    cleaned = label.lower().replace("_attribute", "")
    out: set[str] = set()
    if nlp is None:
        for w in _WORD_RE.findall(cleaned):
            out.add(w)
            s = _singularize_fallback(w)
            if s:
                out.add(s)
        return out
    for tok in nlp(cleaned):
        if tok.is_punct or tok.is_space:
            continue
        lemma = tok.lemma_.lower().strip()
        if lemma and _WORD_RE.fullmatch(lemma):
            out.add(lemma)
            s = _singularize_fallback(lemma)
            if s:
                out.add(s)
    return out


def load_region_paths(brain, region: str, nlp=None) -> list[dict]:
    """Load all paths in a region, each as a bag of words across all steps.

    The bag represents the property-path for a concept: a path that
    combines property phrase + concept label across its segments is
    collapsed into the set of content words that lie on it together.
    Path co-occurrence of question + choice words is the signal.
    """
    rows = brain.conn.execute(
        f"""
        SELECT ps.path_id,
               ns.label AS source_label,
               nt.label AS target_label,
               p.source_text,
               p.terminus_id,
               nt.id AS target_id
        FROM {region}_path_steps ps
        JOIN {region}_segments s ON s.id = ps.segment_id
        JOIN {region}_neurons ns ON ns.id = s.source_id
        JOIN {region}_neurons nt ON nt.id = s.target_id
        JOIN {region}_paths p ON p.id = ps.path_id
        ORDER BY ps.path_id, ps.step_order
        """
    ).fetchall()

    paths: dict[int, dict] = {}
    for pid, sl, tl, src, term_id, tgt_id in rows:
        entry = paths.setdefault(
            pid,
            {"words": set(), "source_text": src,
             "terminus_words": set()},
        )
        entry["words"].update(path_words(sl, nlp))
        entry["words"].update(path_words(tl, nlp))
        # Track words that identify the path's terminus (the concept).
        if tgt_id == term_id:
            entry["terminus_words"].update(path_words(tl, nlp))
    return list(paths.values())
CAUSAL_LEMMAS = {"because", "cause", "therefore", "thus", "result",
                 "produce", "lead"}

# Generic stopwords filtered before using lemmas as seeds / score tokens.
STOP = {
    "be", "have", "do", "the", "a", "an", "this", "that", "these", "those",
    "it", "its", "they", "their", "them", "some", "any", "all", "of", "in",
    "on", "at", "to", "for", "with", "by", "from", "and", "or", "but", "not",
    "as", "so", "if", "then", "than", "also", "only", "most", "more", "such",
    "which", "what", "who", "when", "where", "why", "how", "following",
    "example", "correct", "true", "false", "whose", "whom",
}


def content_lemmas(doc) -> list[str]:
    """Lowercased content-word lemmas from a spaCy doc, stopwords removed.

    Short tokens (len < 3) are normally dropped as noise, but a short
    token that is the ENTIRE sentence (e.g. an answer choice of just 'S'
    or '48') is kept — it's the only signal the choice has.

    A singularized fallback is emitted alongside the lemma so plural
    acronyms like 'snRNPs' → 'snrnps' line up with Sara's neuron
    'snrnp'."""
    out: list[str] = []
    seen: set[str] = set()

    def _add(lemma: str) -> None:
        if lemma and lemma not in seen:
            seen.add(lemma)
            out.append(lemma)
            s = _singularize_fallback(lemma)
            if s:
                _add(s)

    # Count content-worthy tokens to detect very short choices.
    content_tokens = [
        t for t in doc
        if not t.is_punct and not t.is_space
        and t.pos_ in {"NOUN", "PROPN", "VERB", "ADJ", "NUM", "X", "SYM"}
    ]
    is_very_short = len(content_tokens) <= 1

    for tok in doc:
        if tok.is_punct or tok.is_space:
            continue
        if tok.pos_ not in {"NOUN", "PROPN", "VERB", "ADJ", "NUM", "X", "SYM"}:
            continue
        lemma = tok.lemma_.lower().strip()
        if lemma in STOP:
            continue
        # Normally filter short tokens as noise. Exception: when the
        # entire choice is a short token (e.g. "S", "48"), keep it.
        if len(lemma) < 3 and not is_very_short:
            continue
        _add(lemma)

    # Fallback: spaCy's POS tagger is unreliable on short isolated phrases
    # like "Prophase I" (tagged INTJ+PRON instead of PROPN+PROPN). When the
    # POS-filtered pass yields nothing, fall back to raw words minus
    # stopwords — correctness over elegance on short choices.
    if not out:
        for w in _WORD_RE.findall(doc.text.lower()):
            if w in STOP:
                continue
            if len(w) < 3 and not is_very_short:
                continue
            _add(w)
    return out


def svo_triples(doc) -> list[tuple[str, str, str]]:
    triples = []
    for tok in doc:
        if tok.pos_ not in {"VERB", "AUX"}:
            continue
        subj = next((c.lemma_.lower() for c in tok.children
                     if c.dep_ in {"nsubj", "nsubjpass"}), None)
        obj = next((c.lemma_.lower() for c in tok.children
                    if c.dep_ in {"dobj", "attr", "pobj", "acomp"}), None)
        if subj or obj:
            triples.append((subj or "?", tok.lemma_.lower(), obj or "?"))
    return triples


def temporal_signal(doc) -> dict:
    preps = {t.text.lower() for t in doc if t.text.lower() in TEMPORAL_PREPS}
    causals = {t.lemma_.lower() for t in doc if t.lemma_.lower() in CAUSAL_LEMMAS}
    return {"preps": preps, "causals": causals}


# ── Region selection ──

def select_regions(lemmas: list[str], brain: Brain,
                   regions: list[str], top_k: int = 3) -> list[str]:
    """Score each region by how many of the given lemmas match neurons
    in that region, and return the top-k scorers. The lemmas are
    expected to be question + choice lemmas combined — the correct
    region for an MC question may be driven by the choices as much
    as the question stem."""
    scored = []
    for region in regions:
        nr = NeuronRepo(brain.conn, prefix=region)
        hits = 0
        for w in lemmas:
            if nr.get_by_label(w) is not None:
                hits += 1
                continue
            # Singularize fallback for plural/lemma mismatches.
            s = _singularize_fallback(w)
            if s and nr.get_by_label(s) is not None:
                hits += 1
        if hits > 0:
            scored.append((region, hits))
    scored.sort(key=lambda x: -x[1])
    return [r for r, _ in scored[:top_k]] if scored else regions[:1]


# ── Regional activation ──

def get_regional_activation(brain: Brain, regions: list[str],
                            seeds: list[str]) -> dict[str, float]:
    activation: dict[str, float] = {}
    for region in regions:
        nr = NeuronRepo(brain.conn, prefix=region)
        sr = SegmentRepo(brain.conn, prefix=region)
        recognizer = Recognizer(nr, sr, max_depth=3, min_strength=0.1)
        st = ShortTerm(
            event_id=f"exam-{time.time():.6f}",
            event_type="exam",
        )
        recognizer.propagate_echo(
            seeds, st, max_rounds=2, min_strength=0.1, exact_only=True,
        )
        for nid, weight in st.convergence_map.items():
            n = nr.get_by_id(nid)
            if n and len(n.label) >= 3:
                label = n.label.lower()
                activation[label] = activation.get(label, 0) + weight
    return activation


# ── Path-based scorer (apple = red ∧ round ∧ juicy) ──

def score_choice_by_path(q_lemmas: set[str], c_lemmas: set[str],
                         region_paths: list[list[dict]],
                         q_temporal: dict,
                         choice_doc) -> tuple[float, dict]:
    """Rank by path co-occurrence, not activation magnitude.

    A choice wins when its lemmas and the question's lemmas land on the
    same path(s) in Sara's graph together. The tighter the overlap per
    path, the stronger the signal. This is apple = red ∧ round ∧ juicy:
    a property-set scores only where those properties converge on a
    single concept's path, not when they scatter across the region.
    """
    path_score = 0.0
    terminus_bonus = 0.0
    convergent_paths: list[dict] = []

    for paths in region_paths:
        for p in paths:
            labels = p["words"]
            q_hits = q_lemmas & labels
            c_hits = c_lemmas & labels
            if q_hits and c_hits:
                # Tightness: min(q_overlap, c_overlap) — both must be present
                # in strength for the path to count as a convergence.
                tight = min(len(q_hits), len(c_hits))
                path_score += tight
                # Extra bonus if the choice's lemmas name the path's terminus
                # (the concept the path describes). This is "this choice IS
                # the answer", not "this choice mentions words on the path".
                if c_lemmas & p["terminus_words"]:
                    terminus_bonus += 1.0
                convergent_paths.append({
                    "source": p["source_text"],
                    "q_hits": sorted(q_hits),
                    "c_hits": sorted(c_hits),
                })

    # Temporal/causal fit — kept light, operates on spaCy parse only.
    c_temp = temporal_signal(choice_doc)
    temp_score = 0.0
    if q_temporal["preps"] and c_temp["preps"]:
        temp_score += 0.5 * len(q_temporal["preps"] & c_temp["preps"])
    if q_temporal["causals"] and c_temp["causals"]:
        temp_score += 0.5 * len(q_temporal["causals"] & c_temp["causals"])

    total = path_score + terminus_bonus + temp_score
    return total, {
        "path": round(path_score, 2),
        "terminus": round(terminus_bonus, 2),
        "temp": round(temp_score, 2),
        "convergent_paths": convergent_paths[:3],  # top-3 for log
    }


# ── spaCy cortex: score each choice (volume mode) ──

def score_choice(q_doc, choice_doc, activation: dict[str, float],
                 q_temporal: dict) -> tuple[float, dict]:
    """Score a single choice using spaCy + activation overlap."""
    q_lemmas = set(content_lemmas(q_doc))
    c_lemmas = set(content_lemmas(choice_doc))

    # 1. Activation overlap: choice lemmas present in Sara's activation.
    act_score = 0.0
    hits = []
    for lemma in c_lemmas:
        if lemma in activation:
            act_score += activation[lemma]
            hits.append(lemma)

    # 2. SVO alignment: verbs in choice whose subject/object appears
    #    in the question's lemmas (choice "continues the thought").
    svo_score = 0.0
    for subj, verb, obj in svo_triples(choice_doc):
        if subj in q_lemmas or obj in q_lemmas:
            svo_score += 1.0

    # 3. Temporal fit: if question has temporal/causal signal,
    #    reward choices that do too.
    temp_score = 0.0
    c_temp = temporal_signal(choice_doc)
    if q_temporal["preps"] and c_temp["preps"]:
        temp_score += 0.5 * len(q_temporal["preps"] & c_temp["preps"])
    if q_temporal["causals"] and c_temp["causals"]:
        temp_score += 0.5 * len(q_temporal["causals"] & c_temp["causals"])

    # 4. Light shared-with-question penalty: if the choice is basically
    #    echoing the question's own words, that's not informative.
    shared_with_q = q_lemmas & c_lemmas
    overlap_penalty = 0.1 * len(shared_with_q)

    total = act_score + svo_score + temp_score - overlap_penalty
    details = {
        "act": round(act_score, 2),
        "svo": round(svo_score, 2),
        "temp": round(temp_score, 2),
        "penalty": round(overlap_penalty, 2),
        "hits": hits,
    }
    return total, details


# ── Teaching-gap report ──

def report_teaching_gaps(results: list[dict], region_paths_cache: dict,
                          regions: list[str],
                          markdown_out: str | None = None) -> None:
    """For each wrong/abstained question, print what Sara is missing.

    The report answers three questions per failure:
      1. Which question lemmas does Sara know at all? (any path mentions them)
      2. Which correct-choice lemmas does Sara know?
      3. Does a path exist that mentions BOTH a question lemma AND a correct-
         choice lemma? If no, that's the teaching gap — the link between
         question-subject and correct answer isn't in the brain.
    """
    # Union of all path word-bags across all regions, keyed by lemma.
    all_path_words: dict[str, int] = {}
    for paths in region_paths_cache.values():
        for p in paths:
            for w in p["words"]:
                all_path_words[w] = all_path_words.get(w, 0) + 1

    def lemma_known(lemma: str) -> bool:
        return lemma in all_path_words

    def find_linking_paths(q_lemmas: list[str], c_lemmas: list[str]) -> list[dict]:
        """Paths that contain ≥1 question lemma AND ≥1 correct-choice lemma."""
        q_set = set(q_lemmas)
        c_set = set(c_lemmas)
        links = []
        for paths in region_paths_cache.values():
            for p in paths:
                q_hit = q_set & p["words"]
                c_hit = c_set & p["words"]
                if q_hit and c_hit:
                    links.append({
                        "source": p["source_text"],
                        "q": sorted(q_hit),
                        "c": sorted(c_hit),
                    })
        return links

    failures = [r for r in results if r["outcome"] != "correct"]
    if not failures:
        print("\n  (no gaps — all questions answered correctly)")
        return

    print(f"\n  {'='*60}")
    print(f"  TEACHING GAPS — {len(failures)} question(s)")
    print(f"  {'='*60}\n")

    gap_summary: list[dict] = []

    for r in failures:
        qid = r["id"]
        q_lemmas = r["q_lemmas"]
        c_lemmas = r["correct_lemmas"]
        q_known = [l for l in q_lemmas if lemma_known(l)]
        q_unknown = [l for l in q_lemmas if not lemma_known(l)]
        c_known = [l for l in c_lemmas if lemma_known(l)]
        c_unknown = [l for l in c_lemmas if not lemma_known(l)]
        links = find_linking_paths(q_lemmas, c_lemmas)

        print(f"  Q{qid}  [{r['outcome']}]  (picked {r['pick']}, "
              f"correct {r['correct']})")
        print(f"    Q: {r['question'][:100]}")
        print(f"    A: {r['correct_text'][:100]}")
        print(f"    Q-lemmas known:   {q_known}")
        if q_unknown:
            print(f"    Q-lemmas UNKNOWN: {q_unknown}  ← concepts Sara has never seen")
        print(f"    A-lemmas known:   {c_known}")
        if c_unknown:
            print(f"    A-lemmas UNKNOWN: {c_unknown}  ← correct-answer terms Sara has never seen")
        if links:
            print(f"    Linking paths ({len(links)}):")
            for link in links[:2]:
                print(f"      • {link['source'][:80]}")
                print(f"        q={link['q']} c={link['c']}")
        else:
            print(f"    NO LINKING PATH — Sara has no fact connecting "
                  f"question subject to correct answer")
        print()

        # Classify the gap. Priority is based on the CORRECT ANSWER's terms
        # being known — question-side unknowns are often generic verbs like
        # "know", "follow", "statement" and don't reflect real knowledge gaps.
        if c_unknown:
            gap_kind = "vocab_gap"  # correct-answer terms Sara has never seen
        elif not links:
            gap_kind = "relation_gap"  # terms known, no fact connects them
        else:
            gap_kind = "distinction_gap"  # linking path exists but ambiguous
        gap_summary.append({
            "id": qid,
            "kind": gap_kind,
            "outcome": r["outcome"],
            "question": r["question"],
            "correct": r["correct_text"],
            "q_unknown": q_unknown,
            "c_unknown": c_unknown,
            "has_linking_path": bool(links),
        })

    # Summary by gap kind
    by_kind: dict[str, int] = {}
    for g in gap_summary:
        by_kind[g["kind"]] = by_kind.get(g["kind"], 0) + 1
    print(f"  {'-'*60}")
    print(f"  GAPS BY KIND:")
    for kind in ("vocab_gap", "relation_gap", "distinction_gap"):
        if kind in by_kind:
            print(f"    {kind:20s} {by_kind[kind]:2d}")
    print(f"  {'-'*60}")
    print(f"  Teach priority:")
    print(f"    1. vocab_gap     — words Sara has never seen; any fact "
          f"mentioning them adds ground truth")
    print(f"    2. relation_gap  — words known, but no fact connects "
          f"question subject ↔ correct answer")
    print(f"    3. distinction_gap — linking fact exists but doesn't "
          f"discriminate among choices; teach a finer fact")
    print()

    if markdown_out:
        _write_gap_markdown(gap_summary, by_kind, markdown_out)
        print(f"  Wrote teaching-gap markdown: {markdown_out}\n")


def _write_gap_markdown(gap_summary: list[dict], by_kind: dict[str, int],
                         path: str) -> None:
    """Write a teaching-gap report as a markdown checklist grouped by kind."""
    KIND_ORDER = ("relation_gap", "distinction_gap", "vocab_gap")
    KIND_NOTES = {
        "vocab_gap": ("Sara has never seen one or more key terms in the "
                      "correct answer. Teach a definition for each unknown term."),
        "relation_gap": ("Sara knows all the words, but no fact in her graph "
                         "connects the question's subject to the correct "
                         "answer. Teach the connecting fact."),
        "distinction_gap": ("A linking fact exists, but it doesn't "
                            "discriminate between the MC choices. Teach a "
                            "finer-grained fact that distinguishes them."),
    }

    lines: list[str] = []
    lines.append("# Ch10 Teaching Gaps")
    lines.append("")
    lines.append(f"Auto-generated from `run_spacy_ch10.py --report-gaps`. "
                 f"Each entry is a question Sara got wrong or abstained on, "
                 f"classified by the kind of teaching it needs.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Kind | Count | Teach priority |")
    lines.append("|---|---|---|")
    priority = {"relation_gap": 1, "distinction_gap": 2, "vocab_gap": 3}
    for kind in KIND_ORDER:
        if kind in by_kind:
            lines.append(f"| {kind} | {by_kind[kind]} | {priority[kind]} |")
    lines.append("")
    lines.append("Priority ordering: relation > distinction > vocab. "
                 "Relation-gap fixes reuse vocabulary Sara already has; "
                 "vocab-gap fixes require teaching new terms from scratch "
                 "and are cheaper when paired with a fact that uses them.")
    lines.append("")

    for kind in KIND_ORDER:
        entries = [g for g in gap_summary if g["kind"] == kind]
        if not entries:
            continue
        lines.append(f"## {kind}  ({len(entries)})")
        lines.append("")
        lines.append(f"*{KIND_NOTES[kind]}*")
        lines.append("")
        for g in entries:
            qid = g["id"]
            lines.append(f"### Q{qid}  ({g['outcome']})")
            lines.append("")
            lines.append(f"- **Question:** {g['question']}")
            lines.append(f"- **Correct answer:** {g['correct']}")
            if g["q_unknown"]:
                lines.append(f"- **Unknown in question:** "
                             f"`{', '.join(g['q_unknown'])}`")
            if g["c_unknown"]:
                lines.append(f"- **Unknown in answer:** "
                             f"`{', '.join(g['c_unknown'])}`")
            lines.append(f"- **Linking path exists:** "
                         f"{'yes' if g['has_linking_path'] else 'no'}")
            lines.append(f"- [ ] Teach: _(fact to add)_")
            lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


# ── Main ──

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", default="benchmarks/spacy_ch10_results.json")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--abstain-threshold", type=float, default=1.0,
                    help="If top choice score < this, abstain instead of guessing.")
    ap.add_argument("--tie-margin", type=float, default=0.05,
                    help="If (top - runner_up) / top < this, abstain. "
                         "Default 0.05 = 5%%.")
    ap.add_argument("--report-gaps", action="store_true",
                    help="After the exam, print a per-question teaching-gap "
                         "report for wrong and abstained answers.")
    ap.add_argument("--mode", choices=["volume", "path"], default="path",
                    help="Scoring mode. 'volume' = sum activation magnitudes "
                         "(old LLM-style). 'path' = property-path convergence "
                         "(apple = red ∧ round ∧ juicy). Default: path.")
    args = ap.parse_args()

    with open(args.questions) as f:
        questions = json.load(f)
    if args.limit > 0:
        questions = questions[:args.limit]

    brain = Brain(args.db)

    meta_path = args.db + ".regions.json"
    try:
        with open(meta_path) as f:
            regions = json.load(f)["regions"]
    except FileNotFoundError:
        regions = [r["name"] for r in brain.db.list_regions()]

    print(f"\n  spaCy Compartmentalized Exam")
    print(f"  Mode: {args.mode}")
    print(f"  Brain: {args.db}")
    print(f"  Regions: {', '.join(regions)}")
    print(f"  Questions: {len(questions)}\n")

    nlp = spacy.load("en_core_web_sm")

    # Preload per-region path bags for path-mode scoring (cheap: < 200 paths total).
    region_paths_cache: dict[str, list[dict]] = {}
    if args.mode == "path":
        for region in regions:
            region_paths_cache[region] = load_region_paths(brain, region, nlp)
    labels = ["A", "B", "C", "D"]
    correct = 0
    wrong = 0
    abstained = 0
    bench_start = time.time()
    results = []

    for qi, q in enumerate(questions):
        q_start = time.time()
        qid = q["id"]

        q_doc = nlp(q["question"])
        q_lemmas = content_lemmas(q_doc)
        q_temporal = temporal_signal(q_doc)

        # Route using question + all choices combined — the correct region
        # for an MC question is often driven by choice vocabulary.
        routing_lemmas = list(q_lemmas)
        for choice in q["choices"]:
            routing_lemmas.extend(content_lemmas(nlp(choice)))
        selected = select_regions(routing_lemmas, brain, regions)

        choice_scores = []
        per_choice_detail = []
        q_lemmas_set = set(q_lemmas)

        for i, choice in enumerate(q["choices"]):
            c_doc = nlp(choice)
            c_lemmas = content_lemmas(c_doc)
            c_lemmas_set = set(c_lemmas)

            if args.mode == "path":
                selected_path_bags = [region_paths_cache[r] for r in selected]
                score, detail = score_choice_by_path(
                    q_lemmas_set, c_lemmas_set,
                    selected_path_bags, q_temporal, c_doc,
                )
                per_choice_detail.append({
                    "letter": labels[i],
                    "text": choice,
                    "score": round(score, 2),
                    **detail,
                })
            else:
                seeds = q_lemmas + c_lemmas
                activation = get_regional_activation(brain, selected, seeds)
                score, detail = score_choice(q_doc, c_doc, activation, q_temporal)
                per_choice_detail.append({
                    "letter": labels[i],
                    "text": choice,
                    "score": round(score, 2),
                    **detail,
                    "top_activation": sorted(activation.items(),
                                              key=lambda x: -x[1])[:5],
                })
            choice_scores.append(score)

        best_idx = max(range(len(choice_scores)), key=lambda i: choice_scores[i])
        best_score = choice_scores[best_idx]
        # Runner-up = highest score among the other choices
        runner_up = max(
            (s for i, s in enumerate(choice_scores) if i != best_idx),
            default=0.0,
        )
        if best_score > 0:
            gap = (best_score - runner_up) / best_score
        else:
            gap = 0.0
        correct_letter = labels[q["answer_idx"]]

        if best_score < args.abstain_threshold:
            answer = "-"
            outcome = "abstain"
            abstained += 1
            status = "○"
        elif gap < args.tie_margin:
            answer = "-"
            outcome = "tie"
            abstained += 1
            status = "≈"
        else:
            answer = labels[best_idx]
            if answer == correct_letter:
                correct += 1
                outcome = "correct"
                status = "✓"
            else:
                wrong += 1
                outcome = "wrong"
                status = "✗"

        is_correct = outcome == "correct"
        elapsed = time.time() - q_start
        answered = correct + wrong
        accuracy = (correct / answered * 100) if answered else 0.0
        coverage = (answered / (qi + 1)) * 100

        print(f"  [{qi+1}/{len(questions)}] Q{qid}: {status} "
              f"pick={answer} correct={correct_letter} "
              f"score={best_score:.1f} gap={gap*100:.0f}% regions={selected} — "
              f"acc={accuracy:.1f}% cov={coverage:.1f}% ({elapsed:.1f}s)",
              flush=True)

        if args.verbose or not is_correct:
            for d in per_choice_detail:
                marker = "←CORRECT" if d["letter"] == correct_letter else ""
                pick = "←PICK" if d["letter"] == answer else ""
                if args.mode == "path":
                    conv = d.get("convergent_paths", [])
                    print(f"      {d['letter']}. score={d['score']:.2f} "
                          f"path={d['path']} terminus={d['terminus']} "
                          f"temp={d['temp']} "
                          f"paths={len(conv)} {marker}{pick}")
                    for p in conv:
                        print(f"         • {p['source'][:80]}  "
                              f"q={p['q_hits']} c={p['c_hits']}")
                else:
                    print(f"      {d['letter']}. score={d['score']:.2f} "
                          f"act={d['act']} svo={d['svo']} temp={d['temp']} "
                          f"pen={d['penalty']} hits={d['hits'][:4]} "
                          f"{marker}{pick}")

        results.append({
            "id": qid,
            "question": q["question"],
            "correct": correct_letter,
            "correct_text": q["choices"][q["answer_idx"]],
            "pick": answer,
            "outcome": outcome,
            "best_score": round(best_score, 2),
            "gap_pct": round(gap * 100, 1),
            "regions": selected,
            "q_lemmas": q_lemmas,
            "q_temporal": {k: list(v) for k, v in q_temporal.items()},
            "correct_lemmas": content_lemmas(nlp(q["choices"][q["answer_idx"]])),
            "choices": per_choice_detail,
        })

    total_time = time.time() - bench_start
    total = len(questions)
    answered = correct + wrong
    accuracy = (correct / answered * 100) if answered else 0.0
    coverage = (answered / total * 100) if total else 0.0
    print(f"\n  {'='*60}")
    print(f"  spaCy cortex (no LLM)")
    print(f"    correct  : {correct}/{answered}  ({accuracy:.1f}% of answered)")
    print(f"    wrong    : {wrong}")
    print(f"    abstained: {abstained}/{total}  ({100-coverage:.1f}%)")
    print(f"    coverage : {coverage:.1f}%")
    print(f"    threshold: {args.abstain_threshold}")
    print(f"    time     : {total_time:.1f}s  ({total_time/total:.2f}s per Q)")
    print(f"  {'='*60}")

    with open(args.output, "w") as f:
        json.dump({
            "accuracy_of_answered": accuracy / 100,
            "coverage": coverage / 100,
            "correct": correct,
            "wrong": wrong,
            "abstained": abstained,
            "total": total,
            "seconds": total_time,
            "abstain_threshold": args.abstain_threshold,
            "results": results,
        }, f, indent=2)
    print(f"\n  Wrote {args.output}")

    if args.report_gaps:
        # Report-gaps needs region_paths_cache; populate if volume mode skipped it.
        if not region_paths_cache:
            for region in regions:
                region_paths_cache[region] = load_region_paths(brain, region, nlp)
        markdown_out = args.output.rsplit(".", 1)[0] + "_gaps.md"
        report_teaching_gaps(results, region_paths_cache, regions,
                             markdown_out=markdown_out)

    brain.close()


if __name__ == "__main__":
    main()
