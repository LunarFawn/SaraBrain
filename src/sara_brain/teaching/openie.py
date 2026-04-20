"""Rule-based OpenIE over spaCy dep parse, with purpose-filter sensors.

One extraction pass produces (subject, relation, object) triples from
source-token spans only. Five sensors classify each triple:
definition, process, causation, temporal, datetime. A triple matched
by multiple sensors carries higher witness consensus. Nothing is
generated — every triple is copied spans from the source sentence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Triple:
    subject: str
    relation: str
    obj: str
    sensors: list[str] = field(default_factory=list)
    engines: list[str] = field(default_factory=list)


_BE_LEMMAS = {"be"}
_COP_REL_WORDS = frozenset({
    "is", "are", "was", "were", "be", "been", "being",
    "means", "defines",
})
_CAUSAL_REL_WORDS = frozenset({
    "cause", "causes", "caused",
    "lead", "leads", "led",
    "result", "results", "resulted",
    "trigger", "triggers", "triggered",
    "prevent", "prevents", "prevented",
    "require", "requires", "required",
    "drive", "drives", "driven",
})
_TEMPORAL_REL_WORDS = frozenset({
    "before", "after", "during", "while", "until", "since",
    "precede", "precedes", "preceded",
    "follow", "follows", "followed",
    "begin", "begins", "began",
    "end", "ends", "ended",
    "occur", "occurs", "occurred",
})
_DATE_RE = re.compile(
    r"\b("
    r"\d{4}"                                    # year
    r"|\d{1,3}\s*(?:million|billion)\s*years?"  # deep-time
    r"|(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{1,2}?,?\s*\d{2,4}?"
    r"|(?:Cambrian|Ordovician|Silurian|Devonian|Carboniferous|Permian|"
    r"Triassic|Jurassic|Cretaceous|Paleogene|Neogene|Quaternary|"
    r"Archean|Proterozoic|Phanerozoic)"
    r"|\d+\s*(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)"
    r")\b",
    re.IGNORECASE,
)


def _span_text(tokens) -> str:
    return " ".join(t.text for t in tokens).strip()


def _collect_subtree(tok) -> list:
    """Return tokens in the subtree rooted at tok, in sentence order."""
    return sorted(list(tok.subtree), key=lambda t: t.i)


def _walk_conj(head):
    """Yield head + all conjuncts (head, head's conj, conj's conj, ...)."""
    yield head
    for c in head.children:
        if c.dep_ == "conj":
            yield from _walk_conj(c)


def extract_triples_broad(doc) -> list[Triple]:
    """Broad engine — aggressive recall.

    Expands conjunctions across both subject and object, pulls
    prep-phrase objects, permits any verb/aux as a relation. More
    triples per sentence; more likely to produce redundant or
    slightly noisy extractions.
    """
    triples: list[Triple] = []

    for sent in doc.sents:
        for verb in sent:
            if verb.pos_ not in {"VERB", "AUX"}:
                continue

            # Subjects (nsubj / nsubjpass, expanded across conjuncts)
            subjs = [c for c in verb.children if c.dep_ in {"nsubj",
                                                             "nsubjpass"}]
            if not subjs:
                continue

            # Relation = verb itself (plus immediate prt particles and aux)
            rel_tokens = [verb]
            for c in verb.children:
                if c.dep_ in {"prt", "aux", "auxpass", "neg"}:
                    rel_tokens.append(c)
            rel_tokens.sort(key=lambda t: t.i)
            rel_text = _span_text(rel_tokens)

            # Objects: direct objects, attributes, prep objects
            obj_heads: list = []
            for c in verb.children:
                if c.dep_ in {"dobj", "attr", "acomp", "oprd"}:
                    obj_heads.append(c)
                elif c.dep_ == "prep":
                    # Pull prep + pobj subtree as an extended object
                    for pobj in c.children:
                        if pobj.dep_ == "pobj":
                            obj_heads.append((c, pobj))

            # Emit one triple per (subject_conjunct, object_conjunct)
            for subj_head in subjs:
                for subj in _walk_conj(subj_head):
                    subj_text = _span_text(_collect_subtree(subj))
                    for oh in obj_heads:
                        if isinstance(oh, tuple):
                            prep, pobj = oh
                            for pobj_head in _walk_conj(pobj):
                                obj_text = _span_text(
                                    [prep] + _collect_subtree(pobj_head)
                                )
                                triples.append(Triple(
                                    subject=subj_text,
                                    relation=rel_text,
                                    obj=obj_text,
                                ))
                        else:
                            for obj_head in _walk_conj(oh):
                                obj_text = _span_text(
                                    _collect_subtree(obj_head)
                                )
                                triples.append(Triple(
                                    subject=subj_text,
                                    relation=rel_text,
                                    obj=obj_text,
                                ))

    return _dedup(triples)


def extract_triples_strict(doc) -> list[Triple]:
    """Strict engine — high precision, lower recall.

    Only direct objects / attributes / complements are emitted (no
    prep-phrase objects). No conjunction expansion — head subject
    and head object only. Fewer triples, each more "core."
    """
    triples: list[Triple] = []

    for sent in doc.sents:
        for verb in sent:
            if verb.pos_ not in {"VERB", "AUX"}:
                continue
            subjs = [c for c in verb.children if c.dep_ in {"nsubj",
                                                             "nsubjpass"}]
            if not subjs:
                continue

            rel_tokens = [verb]
            for c in verb.children:
                if c.dep_ in {"prt", "auxpass", "neg"}:
                    rel_tokens.append(c)
            rel_tokens.sort(key=lambda t: t.i)
            rel_text = _span_text(rel_tokens)

            # Strict objects: only direct object / attribute / complement
            obj_heads = [
                c for c in verb.children
                if c.dep_ in {"dobj", "attr", "acomp", "oprd"}
            ]

            # Head subject only (no conjunct expansion)
            for subj in subjs:
                subj_text = _span_text(_collect_subtree(subj))
                for obj_head in obj_heads:
                    obj_text = _span_text(_collect_subtree(obj_head))
                    triples.append(Triple(
                        subject=subj_text,
                        relation=rel_text,
                        obj=obj_text,
                    ))

    return _dedup(triples)


def _dedup(triples: list[Triple]) -> list[Triple]:
    seen = set()
    out: list[Triple] = []
    for t in triples:
        key = (t.subject.lower(), t.relation.lower(), t.obj.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


ENGINES = {
    "broad": extract_triples_broad,
    "strict": extract_triples_strict,
}


def run_all_engines(doc) -> list[Triple]:
    """Run every engine, merge, annotate each triple with which engines
    produced it. Witness count = len(triple.engines).
    """
    by_key: dict[tuple, Triple] = {}
    for name, fn in ENGINES.items():
        for t in fn(doc):
            key = (t.subject.lower(), t.relation.lower(), t.obj.lower())
            if key in by_key:
                if name not in by_key[key].engines:
                    by_key[key].engines.append(name)
            else:
                t.engines = [name]
                by_key[key] = t
    return list(by_key.values())


def classify(triple: Triple, source_text: str) -> list[str]:
    """Return list of sensor names that classified this triple.

    A triple can be tagged by more than one sensor — that's the
    witness-consensus signal. No sensor generates content; each is a
    pattern test over existing triple spans.
    """
    sensors: list[str] = []
    rel_words = {w.lower() for w in triple.relation.split()}
    rel_lower = triple.relation.lower()
    subj_obj_text = f"{triple.subject} {triple.obj}".lower()

    # Definition: relation is a copula or defining verb
    if rel_words & _COP_REL_WORDS or rel_lower.strip() in _COP_REL_WORDS:
        sensors.append("definition")

    # Causation: relation contains a causal verb
    if rel_words & _CAUSAL_REL_WORDS:
        sensors.append("causation")

    # Temporal: relation contains a temporal word
    if rel_words & _TEMPORAL_REL_WORDS:
        sensors.append("temporal")

    # Datetime: subject or object contains a date/period/duration token
    if _DATE_RE.search(triple.subject) or _DATE_RE.search(triple.obj):
        sensors.append("datetime")

    # Process: relation is a transitive verb NOT already classified
    # above. This is the catch-all for active-voice action verbs.
    if not sensors and rel_words:
        if any(not w.isalpha() or len(w) >= 3 for w in rel_words):
            sensors.append("process")

    return sensors


def extract_and_classify(sentence: str, nlp) -> list[Triple]:
    """Top-level: parse → run every engine → classify each triple.

    Each triple carries `engines` (which extractors produced it) and
    `sensors` (which purpose-filters matched it). Witness counts on
    both dimensions are available to downstream consumers.
    """
    doc = nlp(sentence)
    triples = run_all_engines(doc)
    for t in triples:
        t.sensors = classify(t, sentence)
    return triples
