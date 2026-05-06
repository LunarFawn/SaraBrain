"""v049 — code knowledge tools for sara_reader.

Functions are reified nodes in the substrate (`neuron_type='function'`)
bundled with binding edges:

  function:<module>.<name>   neuron_type='function'
    --[has_signature]-->     <signature string>
    --[returns]-->           <type label>
    --[takes_param]-->       parameter:<name>
    --[raises]-->            <exception type>
    --[defined_in]-->        <file path>
    --[calls]-->             function:<other>
    --[uses_library]-->      <library label>
    --[has_docstring]-->     <docstring string>
    --[has_example]-->       example:<id>

Parameter sub-nodes carry type info:
  parameter:<func>.<name>    neuron_type='parameter'
    --[has_type]-->          <type label>
    --[is_optional]-->       'true' / 'false'
    --[has_default]-->       <default value or 'none'>

This is the same reification pattern v047 uses for events. The
convention generalises: any multi-valued fact (event, function,
recipe, protocol) becomes one canonical node + binding edges.

Tools provided:
- teach_function — create a function node + bindings.
- teach_parameter — attach parameter info to a function.
- query_function — render the full function info as one bundled
  rendering (signature + params + returns + docstring + calls).
- query_callers / query_callees — caller-callee navigation.
- query_by_returns / query_by_param — find functions by type
  signature.

These mutate the brain. Caller is responsible for authorisation.
"""
from __future__ import annotations

import sqlite3
import time
from typing import Any

from sara_brain.core.brain import Brain


_FUNCTION_PREFIX = "function:"
_PARAMETER_PREFIX = "parameter:"
_EXAMPLE_PREFIX = "example:"

_FUNCTION_BINDING_RELATIONS: tuple[str, ...] = (
    "has_signature", "returns", "takes_param", "raises",
    "defined_in", "calls", "uses_library", "has_docstring",
    "has_example",
)
_PARAMETER_BINDING_RELATIONS: tuple[str, ...] = (
    "has_type", "is_optional", "has_default", "describes",
)


# ── Helpers ──────────────────────────────────────────────────────────


def _ensure_neuron(
    conn: sqlite3.Connection, label: str, neuron_type: str = "concept",
) -> int:
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
    conn.execute(
        "INSERT OR IGNORE INTO segments "
        "(source_id, target_id, relation, strength, created_at) "
        "VALUES (?,?,?,?,?)",
        (src_id, tgt_id, rel, 1.0, time.time()),
    )


def _normalize(label: str) -> str:
    return label.strip().lower()


def _function_label(name: str, module: str | None = None) -> str:
    """Return the canonical event-style label for a function. With
    module: `function:<module>.<name>`. Without: `function:<name>`."""
    name_n = _normalize(name)
    if module:
        return f"{_FUNCTION_PREFIX}{_normalize(module)}.{name_n}"
    return f"{_FUNCTION_PREFIX}{name_n}"


def _parameter_label(func_name: str, param_name: str) -> str:
    """`parameter:<func_name>.<param_name>`."""
    return f"{_PARAMETER_PREFIX}{_normalize(func_name)}.{_normalize(param_name)}"


# ── Write tools ──────────────────────────────────────────────────────


def teach_function(
    brain: Brain,
    name: str,
    signature: str | None = None,
    returns: str | None = None,
    raises: list[str] | None = None,
    defined_in: str | None = None,
    calls: list[str] | None = None,
    uses_library: list[str] | None = None,
    docstring: str | None = None,
    module: str | None = None,
) -> str:
    """Create a function node + binding edges. Returns the function
    label so callers can reference it (e.g. when teaching params)."""
    if not name:
        return "ERROR: name is required."
    func_label = _function_label(name, module)
    conn = brain.conn
    func_id = _ensure_neuron(conn, func_label, neuron_type="function")

    bindings_added = 0
    if signature:
        sig_id = _ensure_neuron(conn, signature.strip())
        _add_segment(conn, func_id, "has_signature", sig_id)
        bindings_added += 1
    if returns:
        ret_id = _ensure_neuron(conn, _normalize(returns))
        _add_segment(conn, func_id, "returns", ret_id)
        bindings_added += 1
    if defined_in:
        loc_id = _ensure_neuron(conn, defined_in.strip())
        _add_segment(conn, func_id, "defined_in", loc_id)
        bindings_added += 1
    if docstring:
        doc_id = _ensure_neuron(conn, docstring.strip())
        _add_segment(conn, func_id, "has_docstring", doc_id)
        bindings_added += 1
    for exc in (raises or []):
        exc_id = _ensure_neuron(conn, _normalize(exc))
        _add_segment(conn, func_id, "raises", exc_id)
        bindings_added += 1
    for callee in (calls or []):
        callee_label = (
            callee if callee.startswith(_FUNCTION_PREFIX)
            else _function_label(callee)
        )
        callee_id = _ensure_neuron(conn, callee_label, neuron_type="function")
        _add_segment(conn, func_id, "calls", callee_id)
        bindings_added += 1
    for lib in (uses_library or []):
        lib_id = _ensure_neuron(conn, _normalize(lib))
        _add_segment(conn, func_id, "uses_library", lib_id)
        bindings_added += 1
    conn.commit()
    return f"taught function {func_label!r} ({bindings_added} bindings)."


def teach_parameter(
    brain: Brain,
    func_name: str,
    param_name: str,
    type_label: str | None = None,
    optional: bool = False,
    default: str | None = None,
    describes: str | None = None,
    module: str | None = None,
) -> str:
    """Attach a parameter to a function. Creates the parameter node
    and the takes_param edge from the function."""
    if not func_name or not param_name:
        return "ERROR: func_name and param_name are required."
    func_label = _function_label(func_name, module)
    param_label = _parameter_label(func_name, param_name)
    conn = brain.conn
    func_id = _ensure_neuron(conn, func_label, neuron_type="function")
    param_id = _ensure_neuron(conn, param_label, neuron_type="parameter")
    _add_segment(conn, func_id, "takes_param", param_id)
    bindings_added = 1
    if type_label:
        type_id = _ensure_neuron(conn, _normalize(type_label))
        _add_segment(conn, param_id, "has_type", type_id)
        bindings_added += 1
    flag_id = _ensure_neuron(conn, "true" if optional else "false")
    _add_segment(conn, param_id, "is_optional", flag_id)
    bindings_added += 1
    if default is not None:
        default_id = _ensure_neuron(conn, default.strip() or "none")
        _add_segment(conn, param_id, "has_default", default_id)
        bindings_added += 1
    if describes:
        desc_id = _ensure_neuron(conn, describes.strip())
        _add_segment(conn, param_id, "describes", desc_id)
        bindings_added += 1
    conn.commit()
    return f"taught parameter {param_label!r} ({bindings_added} bindings)."


# ── Read tools ───────────────────────────────────────────────────────


def _fetch_function_bindings(
    conn: sqlite3.Connection, func_label: str,
) -> dict[str, list[str]]:
    """Return {relation: [target_labels]} for a function node. Some
    relations are list-shaped (raises, calls, takes_param,
    uses_library, has_example) so we always return a list."""
    rows = conn.execute(
        """SELECT s.relation, n2.label
           FROM segments s
           JOIN neurons n1 ON s.source_id = n1.id
           JOIN neurons n2 ON s.target_id = n2.id
           WHERE n1.label = ?
             AND s.relation IN ({})""".format(
            ",".join(["?"] * len(_FUNCTION_BINDING_RELATIONS))
        ),
        (func_label, *_FUNCTION_BINDING_RELATIONS),
    ).fetchall()
    out: dict[str, list[str]] = {}
    for rel, tgt in rows:
        out.setdefault(rel, []).append(tgt)
    return out


def _fetch_parameter_bindings(
    conn: sqlite3.Connection, param_label: str,
) -> dict[str, str]:
    rows = conn.execute(
        """SELECT s.relation, n2.label
           FROM segments s
           JOIN neurons n1 ON s.source_id = n1.id
           JOIN neurons n2 ON s.target_id = n2.id
           WHERE n1.label = ?
             AND s.relation IN ({})""".format(
            ",".join(["?"] * len(_PARAMETER_BINDING_RELATIONS))
        ),
        (param_label, *_PARAMETER_BINDING_RELATIONS),
    ).fetchall()
    return {rel: tgt for rel, tgt in rows}


def _format_function(
    conn: sqlite3.Connection, func_label: str, bindings: dict[str, list[str]],
) -> str:
    """Render a function's bindings as a structured block suitable
    for direct LLM consumption."""
    sig = bindings.get("has_signature", [None])[0]
    returns = bindings.get("returns", [None])[0]
    defined_in = bindings.get("defined_in", [None])[0]
    docstring = bindings.get("has_docstring", [None])[0]
    raises = bindings.get("raises", [])
    calls = bindings.get("calls", [])
    libs = bindings.get("uses_library", [])
    params = bindings.get("takes_param", [])

    short = func_label.removeprefix(_FUNCTION_PREFIX)
    out: list[str] = [f"function: {short}"]
    if sig:
        out.append(f"signature: {sig}")
    if returns:
        out.append(f"returns: {returns}")
    if defined_in:
        out.append(f"defined in: {defined_in}")
    if libs:
        out.append(f"uses: {', '.join(libs)}")
    if calls:
        callees = [c.removeprefix(_FUNCTION_PREFIX) for c in calls]
        out.append(f"calls: {', '.join(callees)}")
    if raises:
        out.append(f"raises: {', '.join(raises)}")
    if params:
        out.append("parameters:")
        for p in params:
            p_b = _fetch_parameter_bindings(conn, p)
            short_p = p.removeprefix(_PARAMETER_PREFIX).split(".", 1)[-1]
            type_label = p_b.get("has_type", "?")
            optional = p_b.get("is_optional") == "true"
            default = p_b.get("has_default")
            desc = p_b.get("describes")
            line = f"  - {short_p} ({type_label})"
            if optional:
                line += " [optional]"
            if default and default != "none":
                line += f" [default={default}]"
            if desc:
                line += f": {desc}"
            out.append(line)
    if docstring:
        out.append("docstring:")
        for line in docstring.splitlines():
            out.append(f"  {line}")
    return "\n".join(out)


def query_function(brain: Brain, name: str, module: str | None = None) -> str:
    """Render the full function info as one bundled block. Use
    this when answering 'what does X do' / 'how do I call X' style
    questions for an LLM coding context."""
    func_label = _function_label(name, module)
    bindings = _fetch_function_bindings(brain.conn, func_label)
    if not bindings:
        return (
            f"No function {name!r} found in the brain. "
            f"DO NOT invent a signature — confirm the name with the "
            f"user or use brain_did_you_mean."
        )
    return _format_function(brain.conn, func_label, bindings)


def query_callers(brain: Brain, name: str, module: str | None = None) -> str:
    """Return functions that call this one. For 'who uses X?' / 'is
    X safe to refactor?' questions."""
    func_label = _function_label(name, module)
    rows = brain.conn.execute(
        """SELECT n1.label
           FROM segments s
           JOIN neurons n1 ON s.source_id = n1.id
           JOIN neurons n2 ON s.target_id = n2.id
           WHERE s.relation = 'calls' AND n2.label = ?""",
        (func_label,),
    ).fetchall()
    if not rows:
        return f"No callers of {name!r} in the brain."
    short = func_label.removeprefix(_FUNCTION_PREFIX)
    lines = [f"Callers of {short}:"]
    for (caller,) in rows:
        lines.append(f"  - {caller.removeprefix(_FUNCTION_PREFIX)}")
    return "\n".join(lines)


def query_callees(brain: Brain, name: str, module: str | None = None) -> str:
    """Return functions this one calls. For 'what does X depend on?'
    questions."""
    func_label = _function_label(name, module)
    rows = brain.conn.execute(
        """SELECT n2.label
           FROM segments s
           JOIN neurons n1 ON s.source_id = n1.id
           JOIN neurons n2 ON s.target_id = n2.id
           WHERE s.relation = 'calls' AND n1.label = ?""",
        (func_label,),
    ).fetchall()
    if not rows:
        return f"{name!r} calls no other tracked functions."
    short = func_label.removeprefix(_FUNCTION_PREFIX)
    lines = [f"{short} calls:"]
    for (callee,) in rows:
        lines.append(f"  - {callee.removeprefix(_FUNCTION_PREFIX)}")
    return "\n".join(lines)


def query_by_returns(brain: Brain, type_label: str) -> str:
    """List functions whose return type matches. For 'find me a
    function that returns X' questions."""
    target = _normalize(type_label)
    rows = brain.conn.execute(
        """SELECT n1.label
           FROM segments s
           JOIN neurons n1 ON s.source_id = n1.id
           JOIN neurons n2 ON s.target_id = n2.id
           WHERE s.relation = 'returns' AND n2.label = ?""",
        (target,),
    ).fetchall()
    if not rows:
        return f"No functions in the brain return {target!r}."
    lines = [f"Functions returning {target}:"]
    for (label,) in rows:
        lines.append(f"  - {label.removeprefix(_FUNCTION_PREFIX)}")
    return "\n".join(lines)


def query_by_param(brain: Brain, type_label: str) -> str:
    """List functions taking a parameter of the given type."""
    target = _normalize(type_label)
    rows = brain.conn.execute(
        """SELECT DISTINCT n1.label
           FROM segments s_takes
           JOIN neurons n1 ON s_takes.source_id = n1.id
           JOIN neurons n_param ON s_takes.target_id = n_param.id
           JOIN segments s_type ON s_type.source_id = n_param.id
           JOIN neurons n_type ON s_type.target_id = n_type.id
           WHERE s_takes.relation = 'takes_param'
             AND s_type.relation = 'has_type'
             AND n_type.label = ?""",
        (target,),
    ).fetchall()
    if not rows:
        return f"No functions in the brain take a parameter of type {target!r}."
    lines = [f"Functions taking a parameter of type {target}:"]
    for (label,) in rows:
        lines.append(f"  - {label.removeprefix(_FUNCTION_PREFIX)}")
    return "\n".join(lines)


def is_function_node(brain: Brain, label: str) -> bool:
    if not label:
        return False
    label_n = _normalize(label)
    if not label_n.startswith(_FUNCTION_PREFIX):
        return False
    row = brain.conn.execute(
        "SELECT neuron_type FROM neurons WHERE label = ?", (label_n,),
    ).fetchone()
    return row is not None and row[0] == "function"


# ── Tool registry definitions ────────────────────────────────────────


def _exec_brain_query_function(brain: Brain, args: dict) -> str:
    return query_function(brain, args["name"], args.get("module"))


def _exec_brain_query_callers(brain: Brain, args: dict) -> str:
    return query_callers(brain, args["name"], args.get("module"))


def _exec_brain_query_callees(brain: Brain, args: dict) -> str:
    return query_callees(brain, args["name"], args.get("module"))


def _exec_brain_query_by_returns(brain: Brain, args: dict) -> str:
    return query_by_returns(brain, args["type"])


def _exec_brain_query_by_param(brain: Brain, args: dict) -> str:
    return query_by_param(brain, args["type"])


CODE_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "brain_query_function": {
        "description": (
            "Return the full info for a function: signature, returns, "
            "parameters with types, calls, raises, docstring. Use when "
            "you need grounded info to write code that calls X."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "module": {"type": "string"},
            },
            "required": ["name"],
        },
        "executor": _exec_brain_query_function,
    },
    "brain_query_callers": {
        "description": (
            "Return functions that call X. For refactor-safety / "
            "blast-radius questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "module": {"type": "string"},
            },
            "required": ["name"],
        },
        "executor": _exec_brain_query_callers,
    },
    "brain_query_callees": {
        "description": "Return functions that X calls. For dependency questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "module": {"type": "string"},
            },
            "required": ["name"],
        },
        "executor": _exec_brain_query_callees,
    },
    "brain_query_by_returns": {
        "description": "List functions whose return type matches X.",
        "parameters": {
            "type": "object",
            "properties": {"type": {"type": "string"}},
            "required": ["type"],
        },
        "executor": _exec_brain_query_by_returns,
    },
    "brain_query_by_param": {
        "description": "List functions taking a parameter of type X.",
        "parameters": {
            "type": "object",
            "properties": {"type": {"type": "string"}},
            "required": ["type"],
        },
        "executor": _exec_brain_query_by_param,
    },
}


__all__ = [
    "teach_function", "teach_parameter",
    "query_function", "query_callers", "query_callees",
    "query_by_returns", "query_by_param",
    "is_function_node",
    "CODE_TOOL_SCHEMAS",
]
