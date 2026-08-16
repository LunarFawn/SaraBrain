"""v049 slice B — ingest a coding guide / source module into a code
knowledge brain.

Two extraction backends:

  python-ast: parse a `.py` file with `ast`, extract function
              definitions, signatures, decorators, calls within the
              body, and the docstring. Most accurate path for any
              Python source.

  markdown:   regex-scrape `def name(...)` / `function name(...)` /
              `class.method(...)` patterns plus the following
              paragraph as docstring. Good for API doc files.

Workflow (mirrors ingest_narrative_chapter.py):

  extract:  ingest_coding_guide.py extract \\
              --backend python-ast \\
              --input src/sara_reader/event_tools.py \\
              --out /tmp/event_tools_draft.tsv

  # user reviews/edits the TSV — fixes types the AST didn't infer,
  # adds docstrings the source omitted, removes private helpers, etc.

  apply:    ingest_coding_guide.py apply \\
              --tsv /tmp/event_tools_draft.tsv \\
              --brain /tmp/code_kb.db

The TSV shape is the same as the narrative ingestion — uses kind=
function, kind=parameter, kind=triple — so apply paths converge.

Quality is intentionally rough on extract; manual review is the
expected step. The AST gets ~80% of Python source right out of the
box; the remaining 20% is docstring quality and example curation.
"""
from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path

from sara_brain.core.brain import Brain
from sara_reader.code_tools import teach_function, teach_parameter


# ── Python AST backend ───────────────────────────────────────────────


def _extract_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Build a readable signature string from an AST FunctionDef."""
    args = []
    posonly = getattr(node.args, "posonlyargs", []) or []
    for a in posonly:
        args.append(_arg_to_str(a))
    if posonly:
        args.append("/")
    for a in node.args.args:
        args.append(_arg_to_str(a))
    if node.args.vararg:
        args.append(f"*{node.args.vararg.arg}")
    elif node.args.kwonlyargs:
        args.append("*")
    for a in node.args.kwonlyargs:
        args.append(_arg_to_str(a))
    if node.args.kwarg:
        args.append(f"**{node.args.kwarg.arg}")
    sig = f"{node.name}({', '.join(args)})"
    if node.returns is not None:
        sig += f" -> {ast.unparse(node.returns)}"
    return sig


def _arg_to_str(a: ast.arg) -> str:
    out = a.arg
    if a.annotation is not None:
        out += f": {ast.unparse(a.annotation)}"
    return out


def _extract_returns_type(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    if node.returns is None:
        return None
    return ast.unparse(node.returns)


def _extract_calls(node: ast.AST) -> list[str]:
    """Pull bare-name and attribute-tail callee names from a function
    body. Misses `getattr(x, 'foo')()` style indirection — acceptable
    for v0."""
    out: list[str] = []
    seen: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            name = None
            if isinstance(f, ast.Name):
                name = f.id
            elif isinstance(f, ast.Attribute):
                name = f.attr
            if name and name not in seen:
                seen.add(name)
                out.append(name)
    return out


def _extract_docstring(node: ast.AST) -> str | None:
    return ast.get_docstring(node)


def _walk_python_ast(path: Path) -> list[dict]:
    """Return a list of draft TSV rows from a Python file."""
    src = path.read_text()
    tree = ast.parse(src, filename=str(path))
    rows: list[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_"):
            # Skip private helpers by default; user can re-add via TSV review.
            continue
        sig = _extract_signature(node)
        ret = _extract_returns_type(node)
        doc = _extract_docstring(node)
        calls = _extract_calls(node)
        rows.append({
            "keep": "1",
            "kind": "function",
            "name": node.name,
            "signature": sig,
            "returns": ret or "",
            "defined_in": str(path),
            "calls": ",".join(c for c in calls if c != node.name),
            "docstring": (doc or "").replace("\t", " ").replace("\n", " "),
        })
        for a in node.args.args:
            rows.append({
                "keep": "1",
                "kind": "parameter",
                "func_name": node.name,
                "param_name": a.arg,
                "type": ast.unparse(a.annotation) if a.annotation else "",
                "default": "",
                "describes": "",
            })
    return rows


# ── Markdown backend ─────────────────────────────────────────────────


_MD_DEF_RE = re.compile(
    r"^(?:def|function|fn)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"\s*\((?P<args>[^)]*)\)\s*"
    r"(?:->\s*(?P<ret>[^:\n]+))?:?\s*$",
    re.MULTILINE,
)


def _walk_markdown(path: Path) -> list[dict]:
    """Regex-scrape def-style function declarations from a markdown
    or doc file. Picks up the line and the paragraph after it as
    docstring."""
    text = path.read_text()
    rows: list[dict] = []
    for m in _MD_DEF_RE.finditer(text):
        name = m.group("name")
        args_raw = m.group("args").strip()
        ret = (m.group("ret") or "").strip()
        sig_args = ", ".join(a.strip() for a in args_raw.split(",") if a.strip())
        sig = f"{name}({sig_args})" + (f" -> {ret}" if ret else "")
        # Docstring = next paragraph after the def line.
        end = m.end()
        next_blank = text.find("\n\n", end)
        if next_blank == -1:
            next_blank = len(text)
        doc = text[end:next_blank].strip()
        rows.append({
            "keep": "1",
            "kind": "function",
            "name": name,
            "signature": sig,
            "returns": ret,
            "defined_in": str(path),
            "calls": "",
            "docstring": doc.replace("\t", " ").replace("\n", " "),
        })
        # Each comma-separated argument becomes a parameter row.
        for a in args_raw.split(","):
            a = a.strip()
            if not a or a in ("/", "*"):
                continue
            if a.startswith("**") or a.startswith("*"):
                # Skip *args/**kwargs as they'd need special handling.
                continue
            if ":" in a:
                pname, _, ptype = a.partition(":")
                pname = pname.strip()
                ptype = ptype.split("=")[0].strip()
            else:
                pname = a.split("=")[0].strip()
                ptype = ""
            rows.append({
                "keep": "1",
                "kind": "parameter",
                "func_name": name,
                "param_name": pname,
                "type": ptype,
                "default": "",
                "describes": "",
            })
    return rows


# ── Extract / apply commands ─────────────────────────────────────────


_FIELDNAMES = [
    "keep", "kind", "name", "signature", "returns", "defined_in",
    "calls", "docstring",
    # parameter rows reuse some columns and add these:
    "func_name", "param_name", "type", "default", "describes",
]


def cmd_extract(args: argparse.Namespace) -> int:
    in_path = Path(args.input)
    if not in_path.exists():
        print(f"input not found: {in_path}")
        return 1
    if args.backend == "python-ast":
        rows = _walk_python_ast(in_path)
    elif args.backend == "markdown":
        rows = _walk_markdown(in_path)
    else:
        print(f"unknown backend: {args.backend}")
        return 1
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=_FIELDNAMES, delimiter="\t", extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    n_func = sum(1 for r in rows if r["kind"] == "function")
    n_param = sum(1 for r in rows if r["kind"] == "parameter")
    print(f"wrote {len(rows)} draft rows to {out_path}  "
          f"({n_func} functions, {n_param} parameters)")
    print("review: open the TSV, set keep=0 on rows to drop, fix "
          "types/docstrings/describes, then run apply.")
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    tsv_path = Path(args.tsv)
    if not tsv_path.exists():
        print(f"tsv not found: {tsv_path}")
        return 1
    db_path = Path(args.brain)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    brain = Brain(str(db_path))

    n_func = 0
    n_param = 0
    n_skipped = 0
    with tsv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("keep", "1").strip() != "1":
                n_skipped += 1
                continue
            kind = (row.get("kind") or "").strip().lower()
            if kind == "function":
                name = (row.get("name") or "").strip()
                if not name:
                    n_skipped += 1
                    continue
                calls = [c.strip() for c in (row.get("calls") or "").split(",") if c.strip()]
                try:
                    teach_function(
                        brain, name=name,
                        signature=(row.get("signature") or None),
                        returns=(row.get("returns") or None) or None,
                        defined_in=(row.get("defined_in") or None) or None,
                        calls=calls or None,
                        docstring=(row.get("docstring") or None) or None,
                    )
                    n_func += 1
                except Exception as e:
                    print(f"  teach_function failed for {name!r}: {e}")
                    n_skipped += 1
                continue
            if kind == "parameter":
                func_name = (row.get("func_name") or "").strip()
                param_name = (row.get("param_name") or "").strip()
                if not func_name or not param_name:
                    n_skipped += 1
                    continue
                try:
                    teach_parameter(
                        brain, func_name=func_name, param_name=param_name,
                        type_label=(row.get("type") or None) or None,
                        default=(row.get("default") or None) or None,
                        describes=(row.get("describes") or None) or None,
                    )
                    n_param += 1
                except Exception as e:
                    print(f"  teach_parameter failed for "
                          f"{func_name}.{param_name}: {e}")
                    n_skipped += 1
                continue
            n_skipped += 1
    print(f"applied: {n_func} functions, {n_param} parameters; "
          f"skipped: {n_skipped}; brain: {db_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_extract = sub.add_parser("extract", help="produce draft TSV")
    p_extract.add_argument("--backend", choices=("python-ast", "markdown"),
                           default="python-ast")
    p_extract.add_argument("--input", required=True, help="path to source file")
    p_extract.add_argument("--out", required=True, help="output TSV path")

    p_apply = sub.add_parser("apply", help="write reviewed TSV to brain.db")
    p_apply.add_argument("--tsv", required=True)
    p_apply.add_argument("--brain", required=True)

    args = ap.parse_args()
    if args.cmd == "extract":
        return cmd_extract(args)
    if args.cmd == "apply":
        return cmd_apply(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
