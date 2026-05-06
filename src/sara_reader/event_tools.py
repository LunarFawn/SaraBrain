"""v047 — reified event tools for sara_reader.

Events are nodes (`neuron_type='event'`) bundled with binding edges:
  event:<n>  --[event_subject]-->   <subject>
  event:<n>  --[event_action]-->    <action>
  event:<n>  --[event_object]-->    <object>          (optional)
  event:<n>  --[event_location]-->  <location>        (optional)
  event:<n>  --[event_start]-->     <ISO-8601 string> (optional)
  event:<n>  --[event_end]-->       <ISO-8601 string> (optional)

This solves the multi-valued binding problem of binary triplets:
"Bob is at cafe 3-5pm Tuesday" stays bound because the event node
holds the location and time pair together.

Reification is also nesting — once `event:bob_at_cafe_t1` is a node,
you can say `(event:bob_at_cafe_t1, observed_by, alice)`.

Tools provided here:
- `brain_teach_event` — create an event node + binding edges in one
  call. Bypasses Brain.teach_triple (the chain-learning machinery's
  `_attribute` convention is wrong for atomic event facts).
- `brain_query_event_at` — find events where a subject participates
  and `event_start <= timestamp <= event_end`.
- `brain_query_events` — list all events involving a subject in
  chronological order.

These tools mutate the brain — they're WRITE tools, unlike the
read-only retrieval tools in `tools.py`. Caller is responsible for
authorisation.
"""
from __future__ import annotations

import sqlite3
import time
from typing import Any

from sara_brain.core.brain import Brain


_EVENT_PREFIX = "event:"
_EVENT_BINDING_RELATIONS: tuple[str, ...] = (
    "event_subject", "event_action", "event_object",
    "event_location", "event_start", "event_end", "event_modifier",
)


# ── Helpers ──────────────────────────────────────────────────────────


def _ensure_neuron(
    conn: sqlite3.Connection, label: str, neuron_type: str = "concept",
) -> int:
    """Find or create a neuron, return its id. Bypasses Brain's
    chain-learning so event-binding edges stay flat."""
    row = conn.execute("SELECT id FROM neurons WHERE label=?", (label,)).fetchone()
    if row is not None:
        return row[0]
    cur = conn.execute(
        "INSERT INTO neurons (label, neuron_type, created_at) VALUES (?,?,?)",
        (label, neuron_type, time.time()),
    )
    return cur.lastrowid


def _add_segment(
    conn: sqlite3.Connection, src_id: int, rel: str, tgt_id: int,
) -> None:
    """Add a segment, idempotent via the UNIQUE constraint."""
    conn.execute(
        "INSERT OR IGNORE INTO segments "
        "(source_id, target_id, relation, strength, created_at) "
        "VALUES (?,?,?,?,?)",
        (src_id, tgt_id, rel, 1.0, time.time()),
    )


def _next_event_label(conn: sqlite3.Connection, subject: str, action: str) -> str:
    """Pick a fresh event label of the form `event:<subj>_<act>_<n>`.
    Walks existing labels with that prefix; n is the next free index."""
    base = f"{_EVENT_PREFIX}{subject}_{action}_"
    rows = conn.execute(
        "SELECT label FROM neurons WHERE label LIKE ?", (base + "%",),
    ).fetchall()
    used: set[int] = set()
    for (label,) in rows:
        suffix = label[len(base):]
        if suffix.isdigit():
            used.add(int(suffix))
    n = 0
    while n in used:
        n += 1
    return f"{base}{n}"


def _normalize(label: str) -> str:
    return label.strip().lower()


# ── Tools ────────────────────────────────────────────────────────────


def teach_event(
    brain: Brain,
    subject: str,
    action: str,
    obj: str | None = None,
    location: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    modifier: str | None = None,
) -> str:
    """Create an event node + binding edges in one operation.

    Returns the event-node label so the caller can reference it in
    subsequent operations (nested triplets, observation edges, etc.)."""
    subject_n = _normalize(subject)
    action_n = _normalize(action)
    if not subject_n or not action_n:
        return "ERROR: subject and action are required."

    conn = brain.conn
    event_label = _next_event_label(conn, subject_n, action_n)
    event_id = _ensure_neuron(conn, event_label, neuron_type="event")
    subject_id = _ensure_neuron(conn, subject_n)
    action_id = _ensure_neuron(conn, action_n)

    _add_segment(conn, event_id, "event_subject", subject_id)
    _add_segment(conn, event_id, "event_action", action_id)

    bindings_added = 2
    if obj:
        obj_id = _ensure_neuron(conn, _normalize(obj))
        _add_segment(conn, event_id, "event_object", obj_id)
        bindings_added += 1
    if location:
        loc_id = _ensure_neuron(conn, _normalize(location))
        _add_segment(conn, event_id, "event_location", loc_id)
        bindings_added += 1
    if start_time:
        start_id = _ensure_neuron(conn, start_time.strip(), neuron_type="timestamp")
        _add_segment(conn, event_id, "event_start", start_id)
        bindings_added += 1
    if end_time:
        end_id = _ensure_neuron(conn, end_time.strip(), neuron_type="timestamp")
        _add_segment(conn, event_id, "event_end", end_id)
        bindings_added += 1
    if modifier:
        mod_id = _ensure_neuron(conn, _normalize(modifier))
        _add_segment(conn, event_id, "event_modifier", mod_id)
        bindings_added += 1
    conn.commit()
    return f"taught event {event_label!r} ({bindings_added} bindings)."


def _fetch_event_bindings(
    conn: sqlite3.Connection, event_label: str,
) -> dict[str, str]:
    """Pull all binding edges for an event node, return as
    {relation: target_label}."""
    rows = conn.execute(
        """SELECT s.relation, n2.label
           FROM segments s
           JOIN neurons n1 ON s.source_id = n1.id
           JOIN neurons n2 ON s.target_id = n2.id
           WHERE n1.label = ?
             AND s.relation IN ({})""".format(
            ",".join(["?"] * len(_EVENT_BINDING_RELATIONS))
        ),
        (event_label, *_EVENT_BINDING_RELATIONS),
    ).fetchall()
    return {rel: tgt for rel, tgt in rows}


def _format_event(event_label: str, bindings: dict[str, str]) -> str:
    """Render an event's bindings as one readable line."""
    subj = bindings.get("event_subject", "?")
    act = bindings.get("event_action", "?")
    obj = bindings.get("event_object")
    loc = bindings.get("event_location")
    start = bindings.get("event_start")
    end = bindings.get("event_end")
    mod = bindings.get("event_modifier")
    parts = [subj]
    if mod:
        parts.append(mod)
    parts.append(act.replace("_", " "))
    if obj:
        parts.append(obj)
    # Skip 'at <loc>' when object and location share a head/tail
    # token to avoid 'closed hatch at docking hatch' style noise.
    used_at_for_loc = False
    if loc:
        obj_l = (obj or "").lower()
        loc_l = loc.lower()
        obj_tail = obj_l.split()[-1] if obj_l else ""
        loc_head = loc_l.split()[0] if loc_l else ""
        if not (obj_tail and (obj_tail == loc_head or obj_tail in loc_l or loc_l in obj_l)):
            parts.append(f"at {loc}")
            used_at_for_loc = True
    time_prep = "on" if used_at_for_loc else "at"
    if start and end:
        parts.append(f"from {start} to {end}")
    elif start:
        parts.append(f"{time_prep} {start}")
    return f"  - {' '.join(parts)}  [{event_label}]"


def query_events(brain: Brain, subject: str) -> str:
    """List all events that involve `subject` (as event_subject),
    chronologically by event_start (events without a start_time
    sort last)."""
    subject_n = _normalize(subject)
    conn = brain.conn
    rows = conn.execute(
        """SELECT n1.label
           FROM segments s
           JOIN neurons n1 ON s.source_id = n1.id
           JOIN neurons n2 ON s.target_id = n2.id
           WHERE s.relation = 'event_subject' AND n2.label = ?""",
        (subject_n,),
    ).fetchall()
    if not rows:
        return f"No events found involving {subject!r}."
    events: list[tuple[str, dict[str, str]]] = []
    for (event_label,) in rows:
        events.append((event_label, _fetch_event_bindings(conn, event_label)))
    events.sort(key=lambda x: x[1].get("event_start", "~"))
    lines = [f"Events involving {subject!r}:"]
    for label, b in events:
        lines.append(_format_event(label, b))
    return "\n".join(lines)


def query_event_at(brain: Brain, subject: str, timestamp: str) -> str:
    """Find events where `subject` participates and the timestamp
    falls in `[event_start, event_end]`. Timestamps are compared as
    strings — ISO-8601 is the recommended format because it's
    lexicographically chronological."""
    subject_n = _normalize(subject)
    ts = timestamp.strip()
    conn = brain.conn
    rows = conn.execute(
        """SELECT n1.label
           FROM segments s
           JOIN neurons n1 ON s.source_id = n1.id
           JOIN neurons n2 ON s.target_id = n2.id
           WHERE s.relation = 'event_subject' AND n2.label = ?""",
        (subject_n,),
    ).fetchall()
    if not rows:
        return f"No events found for {subject!r}."
    matched: list[tuple[str, dict[str, str]]] = []
    for (event_label,) in rows:
        bindings = _fetch_event_bindings(conn, event_label)
        start = bindings.get("event_start")
        end = bindings.get("event_end")
        # Both bounds present: classic interval-inclusion match.
        # Only one bound (or only start with no end): treat as POINT-
        # in-time — exact-string match. Open-ended "from start onward"
        # was the original behaviour but it over-matched on named-
        # time-label events ("t4_at_helix" matched t1/t2/t3 too).
        # Callers who want range semantics should provide both bounds.
        if start and end:
            if start <= ts <= end:
                matched.append((event_label, bindings))
        elif start and not end:
            if start == ts:
                matched.append((event_label, bindings))
        elif end and not start:
            if end == ts:
                matched.append((event_label, bindings))
    if not matched:
        return (
            f"No active events for {subject!r} at {ts!r}. "
            f"Honest miss — DO NOT invent a location."
        )
    lines = [f"Events for {subject!r} at {ts!r}:"]
    for label, b in matched:
        lines.append(_format_event(label, b))
    return "\n".join(lines)


def is_event_node(brain: Brain, label: str) -> bool:
    """Quick check: does this label name an event node?"""
    if not label:
        return False
    label_n = _normalize(label)
    if not label_n.startswith(_EVENT_PREFIX):
        return False
    row = brain.conn.execute(
        "SELECT neuron_type FROM neurons WHERE label = ?", (label_n,),
    ).fetchone()
    return row is not None and row[0] == "event"


def render_event_neuron(brain: Brain, label: str) -> str:
    """Return a readable rendering of one event node. Used by the
    brain_explore extension when the seed is an event."""
    bindings = _fetch_event_bindings(brain.conn, _normalize(label))
    if not bindings:
        return f"Event {label!r} has no bindings (likely not an event node)."
    return f"Event {label!r}:\n{_format_event(label, bindings)}"


# ── Tool registry definitions (registered by tools.py) ────────────────


def _exec_brain_teach_event(brain: Brain, args: dict) -> str:
    return teach_event(
        brain,
        subject=args["subject"],
        action=args["action"],
        obj=args.get("object"),
        location=args.get("location"),
        start_time=args.get("start_time"),
        end_time=args.get("end_time"),
        modifier=args.get("modifier"),
    )


def _exec_brain_query_events(brain: Brain, args: dict) -> str:
    return query_events(brain, args["subject"])


def _exec_brain_query_event_at(brain: Brain, args: dict) -> str:
    return query_event_at(brain, args["subject"], args["timestamp"])


EVENT_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "brain_teach_event": {
        "description": (
            "Create a reified event node bundling a multi-valued fact "
            "(subject + action + optional object/location/time bounds/"
            "modifier). Use when a fact has more than two arguments — "
            "e.g. 'Bob was at the cafe from 3 to 5pm Tuesday'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "action": {"type": "string"},
                "object": {"type": "string"},
                "location": {"type": "string"},
                "start_time": {
                    "type": "string",
                    "description": "ISO-8601 preferred, but any string works "
                                   "(comparisons are lexicographic).",
                },
                "end_time": {"type": "string"},
                "modifier": {"type": "string"},
            },
            "required": ["subject", "action"],
        },
        "executor": _exec_brain_teach_event,
    },
    "brain_query_events": {
        "description": (
            "List all events involving `subject` chronologically. Use "
            "when the question is 'what has subject been doing' rather "
            "than 'where is subject right now'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
            },
            "required": ["subject"],
        },
        "executor": _exec_brain_query_events,
    },
    "brain_query_event_at": {
        "description": (
            "Find events where `subject` participates and `timestamp` "
            "falls in [event_start, event_end]. Use for 'where is X "
            "now' or 'where was X at time T' questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "timestamp": {"type": "string"},
            },
            "required": ["subject", "timestamp"],
        },
        "executor": _exec_brain_query_event_at,
    },
}


__all__ = [
    "teach_event", "query_events", "query_event_at",
    "is_event_node", "render_event_neuron",
    "EVENT_TOOL_SCHEMAS",
]
