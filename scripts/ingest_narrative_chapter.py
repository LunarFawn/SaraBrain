"""v047 slice B.1 — narrative-chapter ingestion (assisted, manual review).

Walks a chapter of plain-text prose paragraph-by-paragraph and emits
a *draft* TSV of (subject, action, object, location, time, source_text)
event candidates plus (subject, relation, object) triple candidates.
The user reviews and edits the TSV before any data lands in the
brain — this script never writes to the substrate.

Workflow:
  1. extract:   ingest_narrative_chapter.py extract --chapter 1
                  -> writes /tmp/chapter_1_draft.tsv
  2. user opens the TSV, edits/deletes rows, fills in missing fields
  3. apply:     ingest_narrative_chapter.py apply --tsv /tmp/chapter_1_draft.tsv
                  -> writes events + triples to brain.db via brain_teach_event
                     and Brain.teach_triple

Heuristics (intentionally rough — the human is the editor):
  - Character names from a curated characters.txt — only paragraphs
    containing one of those names get scanned.
  - Action verbs from a small in-script pool (extensible).
  - Location markers: "at <Loc>", "in the <Loc>", "to <Loc>" where
    Loc is in the curated locations.txt.
  - Time markers: regex for "after <duration>", "as <event>",
    "later", "the next day", "<HH:MM>", etc.
  - Dialogue: `"<text>" said <character>` or `<character> said "..."`.

Curated-name files keep extraction precise; auto-detected proper
nouns produced too many false positives ("She", "It", "Hello") in
draft testing.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import sqlite3
import time as _time

from sara_brain.core.brain import Brain
from sara_reader.event_tools import teach_event


# v047 B.2 follow-up: dialogue triples bypass Brain.teach_triple's
# chain-learning. teach_triple decomposes multi-word objects into
# is_part_of edges from each constituent token — useful for short-
# phrase facts but pure noise for dialogue ("Smith is part of hello
# engineer smith this is..."). Direct SQLite writes avoid that
# decomposition and keep the dialogue as a flat triple.

def _direct_teach(brain: Brain, subj: str, rel: str, obj: str) -> None:
    """Add a (subj, rel, obj) edge via direct SQLite, no chain
    learning. Both endpoints become neurons of type 'concept'.
    Mirrors the bypass pattern in scripts/build_vocab_brain_en.py."""
    conn = brain.conn
    def ensure(label: str) -> int:
        row = conn.execute("SELECT id FROM neurons WHERE label=?", (label,)).fetchone()
        if row:
            return row[0]
        cur = conn.execute(
            "INSERT INTO neurons (label, neuron_type, created_at) VALUES (?,?,?)",
            (label, "concept", _time.time()),
        )
        return cur.lastrowid
    s_id = ensure(subj)
    o_id = ensure(obj)
    conn.execute(
        "INSERT OR IGNORE INTO segments "
        "(source_id, target_id, relation, strength, created_at) "
        "VALUES (?,?,?,?,?)",
        (s_id, o_id, rel, 1.0, _time.time()),
    )
    conn.commit()


# ── Section splitter ─────────────────────────────────────────────────

_SECTION_RE = re.compile(
    r"^(Prologue|Chapter\s+\d+|Epilogue)(?::|\b).*$",
    re.IGNORECASE | re.MULTILINE,
)


def split_chapters(text: str) -> dict[str, str]:
    """Split novella text into {section_label: section_body}.
    Section starts at the first matching Prologue/Chapter/Epilogue
    line and runs until the next match (or EOF)."""
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return {"all": text}
    out: dict[str, str] = {}
    for i, m in enumerate(matches):
        label = m.group(0).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out[label] = text[start:end].strip()
    return out


# ── Heuristic extraction ─────────────────────────────────────────────

# A small, deliberately-narrow action verb pool. The user augments
# during review when verbs from the chapter aren't here.
_DEFAULT_ACTION_VERBS = (
    "entered", "left", "walked", "ran", "arrived", "departed",
    "sat", "stood", "looked", "saw", "heard", "spoke", "said",
    "asked", "answered", "replied", "responded",
    "remembered", "forgot", "noticed", "realized",
    "opened", "closed", "locked", "unlocked", "engaged",
    "carried", "found", "lost", "began", "started", "finished",
    "approached", "joined", "followed", "watched", "observed",
    "examined", "greeted", "thanked", "warned", "told",
    "boarded", "exited", "fired", "evaded",
    "built", "designed", "tested", "removed", "installed",
    "called", "dialed", "messaged", "received", "transmitted",
    "scrapped", "decommissioned", "activated", "deactivated",
)

# Match either ASCII or curly quotes: " " “ ” ' '
_QUOTE_OPEN = r'["“‘]'
_QUOTE_CLOSE = r'["”’]'
_QUOTED = rf'{_QUOTE_OPEN}(?P<quote>[^"“”‘’]+){_QUOTE_CLOSE}'

_DIALOGUE_RES = [
    # "...quoted text..." said <character>
    re.compile(
        rf'{_QUOTED}\s*(?:[,.])?\s*'
        r'(?:said|asked|replied|stated|stammered|continued|added|pleaded)\s+'
        r'(?P<char>[A-Z][a-zA-Z]+)'
    ),
    # <character> said/asked "..."
    re.compile(
        r'(?P<char>[A-Z][a-zA-Z]+)\s+'
        r'(?:said|asked|replied|stated|stammered|continued|added|pleaded)'
        rf'[,]?\s*{_QUOTED}'
    ),
    # <character> + speech tag without quotes (e.g., 'Smith mumbled')
    re.compile(r'(?P<char>[A-Z][a-zA-Z]+)\s+(?:mumbled|whispered|shouted|sighed|nodded)\s+'),
]

_TIME_PATTERNS = [
    re.compile(r'\b(after\s+(?:a\s+(?:few|couple)\s+)?(?:minutes?|seconds?|hours?|days?|moments?))\b', re.I),
    re.compile(r'\b(the\s+(?:next|following)\s+(?:day|morning|night|hour))\b', re.I),
    re.compile(r'\b(later\s+that\s+(?:day|night|morning))\b', re.I),
    re.compile(r'\b(at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b', re.I),
    re.compile(r'\b(suddenly|finally|immediately|moments?\s+later)\b', re.I),
    re.compile(r'\b(once\s+\w+\s+(?:was|had|were))\b', re.I),
]


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]


def _split_paragraphs(text: str) -> list[str]:
    """Split on blank lines; trim leading/trailing whitespace."""
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _find_first(needles: list[str], haystack: str) -> str | None:
    """Return the first needle (case-insensitive whole-word) found
    in haystack, or None."""
    h = haystack.lower()
    for n in needles:
        if not n:
            continue
        n_l = n.lower()
        # Word-bounded match for single-word names; substring for multi-word.
        if " " in n_l:
            if n_l in h:
                return n
        else:
            if re.search(r"\b" + re.escape(n_l) + r"\b", h):
                return n
    return None


def _find_all_chars_in(chars: list[str], paragraph: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for c in chars:
        c_l = c.lower()
        if c_l in seen:
            continue
        if " " in c_l:
            if c_l in paragraph.lower():
                found.append(c)
                seen.add(c_l)
        else:
            if re.search(r"\b" + re.escape(c_l) + r"\b", paragraph.lower()):
                found.append(c)
                seen.add(c_l)
    return found


def _extract_action_for_char(
    paragraph: str, character: str, verbs: list[str],
) -> tuple[str, str] | None:
    """Look for `<char> <verb> <rest>` patterns. Returns (verb, rest)
    or None. Rest is truncated at the first sentence terminator."""
    char_l = re.escape(character)
    for v in verbs:
        # `Character verbed ...`
        pat = re.compile(
            rf"\b{char_l}\b\s+(?P<verb>{re.escape(v)})\s+(?P<rest>[^.!?]*)",
            re.IGNORECASE,
        )
        m = pat.search(paragraph)
        if m:
            rest = m.group("rest").strip()
            return (v, rest)
    return None


def _find_location(rest: str, locations: list[str]) -> str | None:
    """Look for a location mention inside the action's tail."""
    for loc in locations:
        loc_l = loc.lower()
        for prefix in ("at ", "in ", "in the ", "to ", "to the ", "into "):
            if (prefix + loc_l) in rest.lower():
                return loc
    return None


def _find_time(paragraph: str) -> str | None:
    for pat in _TIME_PATTERNS:
        m = pat.search(paragraph)
        if m:
            return m.group(1)
    return None


def _split_into_object(rest: str) -> str:
    """Pull a noun-phrase-like chunk from the rest. v0 heuristic:
    take up to the first comma or end of clause, strip trailing
    function words. The user will fix bad extractions during review."""
    chunk = re.split(r"[,;]", rest, maxsplit=1)[0].strip()
    # Drop trailing prepositional clauses for cleaner objects.
    chunk = re.sub(
        r"\s+(at|in|on|to|for|with|by|from|into|toward)\s+.*$",
        "",
        chunk,
        flags=re.IGNORECASE,
    )
    return chunk


def extract_events_from_paragraph(
    paragraph: str,
    paragraph_n: int,
    characters: list[str],
    locations: list[str],
    verbs: list[str],
) -> list[dict]:
    """Return list of draft event dicts found in this paragraph.
    Each dict has fields: paragraph_n, subject, action, object,
    location, time, source_text."""
    out: list[dict] = []
    chars_in_para = _find_all_chars_in(characters, paragraph)
    if not chars_in_para:
        return out
    time_marker = _find_time(paragraph) or ""
    for char in chars_in_para:
        match = _extract_action_for_char(paragraph, char, verbs)
        if not match:
            continue
        verb, rest = match
        location = _find_location(rest, locations) or ""
        obj = _split_into_object(rest)
        out.append({
            "paragraph_n": paragraph_n,
            "kind": "event",
            "subject": char.lower(),
            "action": verb.lower(),
            "object": obj.lower(),
            "location": location.lower(),
            "time": time_marker.lower(),
            "source_text": paragraph[:200] + ("..." if len(paragraph) > 200 else ""),
        })
    return out


def extract_dialogue_from_paragraph(
    paragraph: str, paragraph_n: int,
) -> list[dict]:
    """Detect 'X said "..."' patterns and return triple-shaped rows
    (kind=triple) with relation='said' and object as the quote."""
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for pat in _DIALOGUE_RES:
        for m in pat.finditer(paragraph):
            char = m.groupdict().get("char", "").strip()
            quote = m.groupdict().get("quote", "").strip()
            if not char or not quote:
                continue
            key = (char, quote[:60])
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "paragraph_n": paragraph_n,
                "kind": "dialogue",
                "subject": char.lower(),
                "action": "said",
                "object": quote[:120],
                "location": "",
                "time": "",
                "source_text": paragraph[:200] + ("..." if len(paragraph) > 200 else ""),
            })
    return out


# ── Extract pass ─────────────────────────────────────────────────────


def cmd_extract(args: argparse.Namespace) -> int:
    text = Path(args.input).read_text()
    chapters = split_chapters(text)
    if args.chapter:
        target = None
        for label in chapters:
            if str(args.chapter).lower() in label.lower():
                target = label
                break
        if target is None:
            print(f"chapter {args.chapter!r} not found in {sorted(chapters)}")
            return 1
        body = chapters[target]
        print(f"# extracting from: {target}")
    else:
        body = text
        print("# extracting from: entire file")

    characters = _read_lines(Path(args.characters))
    locations = _read_lines(Path(args.locations))
    verbs = list(_DEFAULT_ACTION_VERBS)
    if args.verbs and Path(args.verbs).exists():
        verbs = list(_DEFAULT_ACTION_VERBS) + _read_lines(Path(args.verbs))

    if not characters:
        print(
            f"no characters in {args.characters}; create the file with one "
            f"name per line and re-run."
        )
        return 1

    paragraphs = _split_paragraphs(body)
    rows: list[dict] = []
    for i, p in enumerate(paragraphs):
        rows.extend(extract_events_from_paragraph(p, i, characters, locations, verbs))
        rows.extend(extract_dialogue_from_paragraph(p, i))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "keep", "paragraph_n", "kind", "subject", "action",
                "object", "location", "time", "source_text",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for row in rows:
            row_out = {"keep": "1", **row}
            writer.writerow(row_out)
    print(f"wrote {len(rows)} draft rows to {out_path}")
    print(
        "review: open the TSV, set keep=0 on garbage rows, fix subject/"
        "action/object/location/time fields, then run apply."
    )
    return 0


# ── Apply pass ───────────────────────────────────────────────────────


def cmd_apply(args: argparse.Namespace) -> int:
    tsv_path = Path(args.tsv)
    if not tsv_path.exists():
        print(f"tsv not found: {tsv_path}")
        return 1
    db_path = Path(args.brain)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    brain = Brain(str(db_path))

    n_events = 0
    n_triples = 0
    n_skipped = 0
    with tsv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("keep", "1").strip() != "1":
                n_skipped += 1
                continue
            subj = (row.get("subject") or "").strip()
            act = (row.get("action") or "").strip()
            obj = (row.get("object") or "").strip()
            loc = (row.get("location") or "").strip() or None
            time = (row.get("time") or "").strip() or None
            kind = (row.get("kind") or "event").strip().lower()
            if not subj or not act:
                n_skipped += 1
                continue
            if kind == "dialogue":
                # Dialogue lands as a flat triple via direct SQLite —
                # bypasses chain learning so the quoted text doesn't
                # get decomposed into 'X is part of <quote>' noise.
                try:
                    _direct_teach(brain, subj, act, obj or "?")
                    n_triples += 1
                except Exception as e:
                    print(f"  direct dialogue insert failed for "
                          f"{subj!r} {act!r} {obj!r}: {e}")
                    n_skipped += 1
                continue
            # Event row: bundle into a reified event node.
            try:
                teach_event(
                    brain,
                    subject=subj,
                    action=act,
                    obj=obj or None,
                    location=loc,
                    start_time=time,
                )
                n_events += 1
            except Exception as e:
                print(f"  teach_event failed for {subj!r} {act!r}: {e}")
                n_skipped += 1
    print(
        f"applied: {n_events} events, {n_triples} triples; "
        f"skipped: {n_skipped}; brain: {db_path}"
    )
    return 0


# ── CLI ──────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_extract = sub.add_parser("extract", help="generate a draft TSV")
    p_extract.add_argument("--input", required=True, help="path to chapter text")
    p_extract.add_argument("--chapter", default=None,
                           help="section label substring (e.g. 'Chapter 1') "
                                "or omit to scan the whole file")
    p_extract.add_argument("--characters", default="narrative/characters.txt",
                           help="curated character names, one per line")
    p_extract.add_argument("--locations", default="narrative/locations.txt",
                           help="curated location names, one per line")
    p_extract.add_argument("--verbs", default="narrative/verbs.txt",
                           help="extra action verbs (optional)")
    p_extract.add_argument("--out", required=True,
                           help="output TSV path")

    p_apply = sub.add_parser("apply", help="write reviewed TSV to brain.db")
    p_apply.add_argument("--tsv", required=True, help="path to reviewed TSV")
    p_apply.add_argument("--brain", required=True, help="output brain.db path")

    args = ap.parse_args()
    if args.cmd == "extract":
        return cmd_extract(args)
    if args.cmd == "apply":
        return cmd_apply(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
