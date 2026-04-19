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
             "terminus_words": set(),
             # Polarity: True when the path is a refutation rather than
             # an affirmation. Detected from the source text so that
             # pre-Level-1 teachings (negations stored as affirmations
             # with 'not' stripped) are still recognised. Newly-taught
             # negations also land here correctly because the parser
             # preserves the original source_text verbatim.
             "negated": detect_polarity(src)},
        )
        entry["words"].update(path_words(sl, nlp))
        entry["words"].update(path_words(tl, nlp))
        # Track words that identify the path's terminus (the concept).
        if tgt_id == term_id:
            entry["terminus_words"].update(path_words(tl, nlp))

    # Augment word-bags with the full source_text lemmas. Relation
    # verbs live on segment edges, not on neurons, so reading just
    # from neuron labels misses content like "lacking" in the path
    # "Turner syndrome is caused by lacking an X chromosome". The
    # source_text is the ground truth of what was taught — use its
    # content lemmas as the authoritative word-bag.
    for p in paths.values():
        src = p.get("source_text") or ""
        if src and nlp is not None:
            for tok in nlp(src):
                if tok.is_punct or tok.is_space:
                    continue
                if tok.pos_ not in {
                    "NOUN", "PROPN", "VERB", "ADJ", "NUM", "ADP"
                }:
                    # ADP (prepositions) kept because "by", "of" etc.
                    # don't matter here but we strip them via STOP check.
                    continue
                lemma = tok.lemma_.lower().strip()
                if not lemma or lemma in STOP:
                    continue
                if _WORD_RE.fullmatch(lemma):
                    p["words"].add(lemma)
                    s = _singularize_fallback(lemma)
                    if s:
                        p["words"].add(s)
    return list(paths.values())
CAUSAL_LEMMAS = {"because", "cause", "therefore", "thus", "result",
                 "produce", "lead"}

# ── Negation-word lexicon ─────────────────────────────────────────────
# Words that signal polarity. Used to detect whether the question is
# asking for a FALSE statement, whether a choice denies something, and
# whether a taught path refutes rather than affirms.
_NEG_TOKENS = {
    "not", "never", "no", "neither", "nor", "cannot",
    "except", "excluding", "excludes",
    "false", "incorrect", "wrong",
}
# Strong question-level markers — require capitalisation or a multi-word
# phrase so we don't mis-fire on an incidental lowercase 'not' inside a
# clause. Questions that want a FALSE answer almost always signal it
# strongly ("Which is NOT true", "all of the following EXCEPT", etc.).
_QUESTION_NEG_RE = re.compile(
    r"\bNOT\b|\bEXCEPT\b|\bEXCLUDING\b|\bfalse\b|\bincorrect\b"
    r"|\bwrong\b|\bnone of\b|\bleast likely\b"
)


def detect_polarity(text: str, question_mode: bool = False) -> bool:
    """Return True when text is negated.

    question_mode=True: match strong question-level markers only
        (capitalised NOT/EXCEPT, phrases). Avoids false-positives from
        incidental lowercase 'not' inside clauses.
    question_mode=False (default): for a fact's source_text or for a
        choice's text. Any negation token or "n't" contraction anywhere
        inverts polarity.
    """
    if not text:
        return False
    if question_mode:
        return bool(_QUESTION_NEG_RE.search(text))
    low = text.lower()
    if "n't" in low:
        return True
    for w in _WORD_RE.findall(low):
        if w in _NEG_TOKENS:
            return True
    return False


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


def extract_topic(doc) -> set[str]:
    """Extract the question's topic concept(s) — multi-subject aware.

    The topic is what the question is ASKING ABOUT. For property-match
    scoring we need the *set* of concepts the question targets so we can
    filter to paths that describe any of them. Real-world MC questions
    routinely carry the subject matter inside prep phrases or noun
    modifiers, not just in the grammatical subject.

    Strategies are all applied (no early-return) and accumulate into a
    single set, capped at a reasonable ceiling to avoid inflation:

      1. 'about X' preposition chains.
      2. nsubj / nsubjpass content nouns + their compound/amod chains.
      3. Prep-phrase objects hanging off any nsubj (with/of/regarding
         X — where the real subject matter sits in medical questions
         like "Females with Turner's syndrome").
      4. Conjunctions anchored to already-added topics ("X and Y").
      5. Noun-chunk fallback when 1-4 found nothing substantive.

    Meta-reference nouns (statement, following, option, ...) are
    excluded — they're question-framing words, not topics.
    """
    out: set[str] = set()

    def _add(lemma: str) -> None:
        lemma = lemma.lower().strip()
        if not lemma or lemma in _META_TERMS:
            return
        if len(out) >= _MAX_TOPICS:
            return
        out.add(lemma)
        s = _singularize_fallback(lemma)
        if s and s not in _META_TERMS:
            out.add(s)

    def _add_np_chain(tok) -> None:
        """Add a noun and its compound/amod/poss modifier chain.

        Short-circuits when the head noun is a meta-term (statement,
        following, etc.) — modifiers of a meta-term are also question-
        framing and should not contaminate the topic set.
        """
        head_lemma = tok.lemma_.lower().strip()
        if head_lemma in _META_TERMS:
            return
        if tok.pos_ in {"NOUN", "PROPN"}:
            _add(tok.lemma_)
        for child in tok.children:
            if child.dep_ in {"compound", "amod", "poss"} and child.pos_ in {
                "NOUN", "PROPN", "ADJ"}:
                _add(child.lemma_)

    # 1. 'about X' preposition → pobj child
    for tok in doc:
        if tok.text.lower() == "about" and tok.dep_ == "prep":
            for child in tok.children:
                if child.dep_ == "pobj" and child.pos_ in {"NOUN", "PROPN"}:
                    _add_np_chain(child)

    # 2. Root verb's nsubj + 3. its prep-phrase objects
    for tok in doc:
        if tok.dep_ != "ROOT":
            continue
        for child in tok.children:
            if child.dep_ in {"nsubj", "nsubjpass"} and child.pos_ in {
                "NOUN", "PROPN"}:
                _add_np_chain(child)
                # Prep phrases MODIFYING the nsubj (e.g. "with Turner's
                # syndrome") — walk prep → pobj → NP chain.
                for prep in child.children:
                    if prep.dep_ == "prep":
                        for pobj in prep.children:
                            if pobj.dep_ == "pobj" and pobj.pos_ in {
                                "NOUN", "PROPN"}:
                                _add_np_chain(pobj)

    # 4. Conjunctions anchored to any token already added as a topic.
    for tok in doc:
        if tok.dep_ == "conj" and tok.pos_ in {"NOUN", "PROPN"}:
            head_lemma = tok.head.lemma_.lower().strip()
            if head_lemma in out:
                _add_np_chain(tok)

    # 5. Noun-chunk fallback — when 1-4 produced nothing (or only
    #    meta-terms got filtered out), iterate doc.noun_chunks and add
    #    the head noun of each chunk. Provides topics for questions
    #    whose grammatical subject is a meta-reference noun like
    #    "following", "statement", "information".
    if not out:
        for chunk in doc.noun_chunks:
            _add_np_chain(chunk.root)

    return out


# Meta-reference nouns that frame questions without being topics.
# When nsubj or a noun chunk surfaces one of these, we discard it so
# the topic set doesn't get contaminated by question-framing words.
_META_TERMS = frozenset({
    "statement", "following", "option", "answer", "choice",
    "information", "example", "case", "situation", "item", "scenario",
})

# Cap on topics accumulated from a single question. Six is enough for
# any plausible multi-subject question; beyond that we're adding noise
# that dilutes the property-match signal (every added topic removes
# one more word from the property phrase during scoring).
_MAX_TOPICS = 6


def score_choice_by_property(q_topic: set[str], c_lemmas: set[str],
                             region_paths: list[list[dict]],
                             choice_doc, q_temporal: dict,
                             q_neg: bool = False,
                             choice_text: str = "",
                             ) -> tuple[float, dict]:
    """Topic-filtered property match, MAX-over-paths.

    Defeats vocabulary echo by:
      1. Filtering to paths whose subject matter is the question's topic
         (either terminus_words or path words intersect the topic).
      2. Removing topic words from the path's word-bag so the 'content'
         of the path is what remains — the property phrase.
      3. Counting choice lemma coverage of that property content.
      4. Taking the MAX across topic-relevant paths rather than the sum,
         so a choice cannot win by lightly touching many paths.

    Returns (score, detail_dict). Detail includes the best-matching path's
    source_text so the verbose output is inspectable. A score of 0 with
    zero topic-relevant paths signals the caller to fall back to path-mode.
    """
    c_neg = detect_polarity(choice_text)
    best_score = 0.0
    best_path_info: dict | None = None
    topic_relevant_count = 0
    polarity_skips = 0

    for paths in region_paths:
        for p in paths:
            words = p["words"]
            term_words = p.get("terminus_words", set())
            # Topic-relevant: path's terminus names the topic, OR the
            # topic is present anywhere in the path's word-bag.
            if not (q_topic & term_words) and not (q_topic & words):
                continue
            topic_relevant_count += 1

            # Polarity alignment: a match counts iff (c_neg XOR p_neg)
            # == q_neg. Intuition:
            #   - Positive Q + affirming path + affirming choice → OK
            #   - Positive Q + refuting path + affirming choice → SKIP
            #     (choice asserts what the graph refutes)
            #   - Negative Q ("which is NOT true?") + affirming path +
            #     affirming choice → SKIP (Q wants a refutable claim)
            #   - Negative Q + affirming path + denying choice → OK
            #   - Negative Q + refuting path + affirming choice → OK
            #     (the choice says what the graph says is false — it IS
            #     the false statement the question wants)
            p_neg = p.get("negated", False)
            if (c_neg != p_neg) != q_neg:
                polarity_skips += 1
                continue

            property_words = words - q_topic
            hits = c_lemmas & property_words
            score = float(len(hits))
            if score > best_score:
                best_score = score
                best_path_info = {
                    "source": p["source_text"],
                    "property_hits": sorted(hits),
                    "terminus_words": sorted(term_words),
                    "path_negated": p_neg,
                    "choice_negated": c_neg,
                }

    # Temporal/causal fit — kept as a small side-signal for parity with
    # path mode; never enough to outweigh a property match.
    c_temp = temporal_signal(choice_doc)
    temp_score = 0.0
    if q_temporal["preps"] and c_temp["preps"]:
        temp_score += 0.5 * len(q_temporal["preps"] & c_temp["preps"])
    if q_temporal["causals"] and c_temp["causals"]:
        temp_score += 0.5 * len(q_temporal["causals"] & c_temp["causals"])

    total = best_score + temp_score
    detail = {
        "property_score": round(best_score, 2),
        "temp": round(temp_score, 2),
        "topic_relevant_paths": topic_relevant_count,
        "polarity_skips": polarity_skips,
        "c_neg": c_neg,
        "q_neg": q_neg,
        "best_path": best_path_info,
    }
    return total, detail


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
                         choice_doc,
                         q_neg: bool = False,
                         choice_text: str = "",
                         apply_polarity: bool = True) -> tuple[float, dict]:
    """Rank by path co-occurrence, not activation magnitude.

    A choice wins when its lemmas and the question's lemmas land on the
    same path(s) in Sara's graph together. The tighter the overlap per
    path, the stronger the signal. This is apple = red ∧ round ∧ juicy:
    a property-set scores only where those properties converge on a
    single concept's path, not when they scatter across the region.

    Polarity alignment: (c_neg XOR p_neg) must equal q_neg for the
    convergence to count. See score_choice_by_property for the
    reasoning.
    """
    c_neg = detect_polarity(choice_text)
    path_score = 0.0
    terminus_bonus = 0.0
    polarity_skips = 0
    convergent_paths: list[dict] = []

    for paths in region_paths:
        for p in paths:
            labels = p["words"]
            q_hits = q_lemmas & labels
            c_hits = c_lemmas & labels
            if q_hits and c_hits:
                p_neg = p.get("negated", False)
                if apply_polarity and (c_neg != p_neg) != q_neg:
                    polarity_skips += 1
                    continue
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
                    "path_negated": p_neg,
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
        "polarity_skips": polarity_skips,
        "c_neg": c_neg,
        "q_neg": q_neg,
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


# ── Math pass: apply taught operations when the question has numbers ──

def math_pass(question_text: str, choices: list[str],
              selected_regions: list[str], brain: Brain
              ) -> tuple[set[float], dict[int, str]]:
    """Compute candidate numeric answers for the question by applying
    every operation-tagged segment in the selected regions to every
    number extracted from the question. Match results against MC
    choices.

    Returns:
        candidates: set of computed numeric results (for logging).
        choice_matches: {choice_idx: computed_value_that_matched_it}.
    """
    from sara_brain.core.math import (
        NumberExtractor, tag_to_operation, MathCompute,
    )

    numbers = NumberExtractor().extract(question_text)
    if not numbers:
        return set(), {}

    compute = MathCompute()
    candidates: set[float] = set()

    for region in selected_regions:
        seg_repo = SegmentRepo(brain.conn, prefix=region)
        for seg in seg_repo.list_all():
            if not seg.operation_tag:
                continue
            op = tag_to_operation(seg.operation_tag)
            if op is None:
                continue
            for val in numbers.values():
                try:
                    candidates.add(compute.apply(op, val))
                except (NotImplementedError, ZeroDivisionError, ValueError):
                    continue

    if not candidates:
        return candidates, {}

    # Match computed numbers against integer/float tokens in each choice.
    _num_in_choice_re = re.compile(r"-?\d+(?:\.\d+)?")
    choice_matches: dict[int, str] = {}
    for idx, choice in enumerate(choices):
        choice_nums = set()
        for m in _num_in_choice_re.finditer(choice):
            try:
                choice_nums.add(float(m.group()))
            except ValueError:
                pass
        if not choice_nums:
            continue
        for ans in candidates:
            # Integer equality (96.0 == 96) is handled by float compare.
            if ans in choice_nums:
                choice_matches[idx] = f"computed {ans} from {question_text[:40]}…"
                break
    return candidates, choice_matches


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
    ap.add_argument("--mode", choices=["volume", "path", "property"],
                    default="property",
                    help="Scoring mode. 'volume' = sum activation magnitudes "
                         "(old LLM-style). 'path' = property-path "
                         "convergence (apple = red ∧ round ∧ juicy). "
                         "'property' = topic-filtered property match, "
                         "max-over-paths (defeats vocabulary echo; falls "
                         "through to path-mode when topic has no matching "
                         "paths). Default: property.")
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

    # Preload per-region path bags for path/property-mode scoring
    # (cheap: < 200 paths total across all regions).
    region_paths_cache: dict[str, list[dict]] = {}
    if args.mode in {"path", "property"}:
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
        q_topic = extract_topic(q_doc) if args.mode == "property" else set()
        q_neg = detect_polarity(q["question"], question_mode=True)

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

            selected_path_bags = (
                [region_paths_cache[r] for r in selected]
                if args.mode in {"path", "property"} else []
            )
            if args.mode == "property":
                # Try property first; detail + score computed once. We
                # may still fall through later if the whole choice-set
                # scored zero (handled after the per-choice loop).
                if q_topic:
                    score, detail = score_choice_by_property(
                        q_topic, c_lemmas_set, selected_path_bags,
                        c_doc, q_temporal,
                        q_neg=q_neg, choice_text=choice,
                    )
                    scored_by = "property"
                    if detail["topic_relevant_paths"] == 0:
                        # No topic-matching paths at all — use path-mode now.
                        # Polarity alignment is applied ONLY when the
                        # question is negative (EXCEPT/NOT). For positive
                        # questions, loose co-occurrence shouldn't be
                        # gated by incidental negation words in paths.
                        p_score, p_detail = score_choice_by_path(
                            q_lemmas_set, c_lemmas_set,
                            selected_path_bags, q_temporal, c_doc,
                            q_neg=q_neg, choice_text=choice,
                            apply_polarity=q_neg,
                        )
                        score = p_score
                        detail = {**p_detail, "fallback": "no-topic-paths"}
                        scored_by = "path-fallback"
                else:
                    # No topic extractable — fall back to path-mode.
                    score, detail = score_choice_by_path(
                        q_lemmas_set, c_lemmas_set,
                        selected_path_bags, q_temporal, c_doc,
                        q_neg=q_neg, choice_text=choice,
                        apply_polarity=q_neg,
                    )
                    detail = {**detail, "fallback": "no-topic"}
                    scored_by = "path-fallback"
                per_choice_detail.append({
                    "letter": labels[i],
                    "text": choice,
                    "score": round(score, 2),
                    "scored_by": scored_by,
                    **detail,
                })
            elif args.mode == "path":
                score, detail = score_choice_by_path(
                    q_lemmas_set, c_lemmas_set,
                    selected_path_bags, q_temporal, c_doc,
                    q_neg=q_neg, choice_text=choice,
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

        # Property-mode whole-question fallback: if topic filtering produced
        # all-zero scores across every choice, the topic filter was too
        # restrictive for this question. Fall back to path-mode so we get
        # some signal — and if path-mode also scores all zero, the abstain
        # logic below handles it honestly.
        if (args.mode == "property"
                and max(choice_scores) == 0
                and any(d.get("scored_by") == "property"
                        for d in per_choice_detail)):
            choice_scores = []
            per_choice_detail = []
            for i, choice in enumerate(q["choices"]):
                c_doc = nlp(choice)
                c_lemmas_set = set(content_lemmas(c_doc))
                # All-zero fallback: polarity is applied ONLY for
                # explicitly-negative questions (same rule as the
                # per-choice path-fallback).
                score, detail = score_choice_by_path(
                    q_lemmas_set, c_lemmas_set,
                    [region_paths_cache[r] for r in selected],
                    q_temporal, c_doc,
                    q_neg=q_neg, choice_text=choice,
                    apply_polarity=q_neg,
                )
                choice_scores.append(score)
                per_choice_detail.append({
                    "letter": labels[i],
                    "text": choice,
                    "score": round(score, 2),
                    "scored_by": "path-fallback-all-zero",
                    **detail,
                })

        # Math pass: if the question contains numbers and an
        # operation-tagged segment in the selected regions can produce
        # a numeric answer that uniquely matches one MC choice, boost
        # that choice so the scoring pipeline picks it. The STM/math
        # module is where the math actually runs; the scorer is just
        # the gate that funnels the boost in.
        math_candidates, math_matches = math_pass(
            q["question"], q["choices"], selected, brain,
        )
        math_note: str | None = None
        if len(math_matches) == 1:
            (mi,) = math_matches.keys()
            # Override with a dominant score so abstain+tie-margin logic
            # picks this choice cleanly.
            prior = choice_scores[mi]
            choice_scores[mi] = max(prior, 1000.0)
            per_choice_detail[mi]["score"] = round(choice_scores[mi], 2)
            per_choice_detail[mi]["math_match"] = math_matches[mi]
            math_note = f"math→{labels[mi]} candidates={sorted(math_candidates)}"

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
            if args.mode == "property":
                print(f"      topic={sorted(q_topic) if q_topic else '(none)'}")
            for d in per_choice_detail:
                marker = "←CORRECT" if d["letter"] == correct_letter else ""
                pick = "←PICK" if d["letter"] == answer else ""
                if args.mode == "property" and d.get("scored_by") == "property":
                    bp = d.get("best_path")
                    print(f"      {d['letter']}. score={d['score']:.2f} "
                          f"prop={d['property_score']} temp={d['temp']} "
                          f"topic_paths={d['topic_relevant_paths']} "
                          f"{marker}{pick}")
                    if bp:
                        print(f"         ↳ {bp['source'][:80]}  "
                              f"property_hits={bp['property_hits']}")
                elif args.mode == "property" and d.get("scored_by", "").startswith("path-fallback"):
                    conv = d.get("convergent_paths", [])
                    print(f"      {d['letter']}. score={d['score']:.2f} "
                          f"[{d.get('scored_by','path-fallback')}] "
                          f"paths={len(conv)} {marker}{pick}")
                    for p in conv:
                        print(f"         • {p['source'][:80]}  "
                              f"q={p['q_hits']} c={p['c_hits']}")
                elif args.mode == "path":
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
