"""Math module — arithmetic primitives for Sara Brain.

Mirrors the shape of core/temporal.py: a dedicated module that owns
one cognitive domain. Everything arithmetic lives here.

Architecture (see project_math_primitives memory):
  • Six ALU-level primitives (add, subtract, multiply, divide, abs,
    negate) — implemented directly in Python using native operators.
  • Every other operation — halve, double, fractions, percentages,
    probability, mean, variance, standard deviation, etc. — is TAUGHT
    as a composition of the six primitives. Taught compositions live
    in Sara's graph (see benchmarks/curriculum_math.txt) and execute
    by walking the graph's stated definition.

The primitives use Python's intrinsic operators; they are fast and
correct. Do not reimplement multiplication as repeated addition, per
Jennifer's design call — the performance cost would be absurd and
the primitive boundary matches every hardware ALU.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..innate.primitives import OPERATION


@dataclass
class Operation:
    """A single arithmetic operation to apply to a numeric value.

    kind:    A member of the OPERATION primitive layer (add, subtract,
             multiply, divide, abs, negate) OR a taught composition name
             (halve, double, percentage, probability, mean, variance, ...)
             that is defined by a graph path in Sara's brain.
    operand: The second argument for binary ops (e.g. 0.5 for "divide
             by 2 to halve"). None for unary ops (abs, negate) or when
             the operand is supplied at compute time.
    """
    kind: str
    operand: float | None = None


class MathCompute:
    """Arithmetic execution for Sara.

    Six intrinsic primitives call Python operators directly. All other
    operations dispatch through :meth:`execute`, which walks the graph's
    taught definition (implemented in a later layer — for now, execute()
    raises NotImplementedError when asked for a non-primitive op).
    """

    # ── The six ALU primitives ────────────────────────────────────────

    def add(self, a: float, b: float) -> float:
        return a + b

    def subtract(self, a: float, b: float) -> float:
        return a - b

    def multiply(self, a: float, b: float) -> float:
        return a * b

    def divide(self, a: float, b: float) -> float:
        # Let Python raise ZeroDivisionError naturally; the caller
        # (typically STM during a query) is expected to abstain rather
        # than swallow the exception.
        return a / b

    def abs_(self, a: float) -> float:
        return abs(a)

    def negate(self, a: float) -> float:
        return -a

    # ── Dispatch ──────────────────────────────────────────────────────

    _PRIMITIVE_BINARY = {"add", "subtract", "multiply", "divide"}
    _PRIMITIVE_UNARY = {"abs", "negate"}

    def apply(self, op: Operation, value: float,
              operand: float | None = None) -> float:
        """Apply a single Operation to a value.

        For binary primitives, `operand` (from the Operation or passed
        explicitly) is the second argument. For unary primitives, it
        is ignored. For non-primitive (taught) operations, this method
        delegates to :meth:`execute`.
        """
        kind = op.kind
        rhs = operand if operand is not None else op.operand

        if kind in self._PRIMITIVE_BINARY:
            if rhs is None:
                raise ValueError(f"operation {kind!r} requires an operand")
            if kind == "add":
                return self.add(value, rhs)
            if kind == "subtract":
                return self.subtract(value, rhs)
            if kind == "multiply":
                return self.multiply(value, rhs)
            if kind == "divide":
                return self.divide(value, rhs)
        if kind in self._PRIMITIVE_UNARY:
            if kind == "abs":
                return self.abs_(value)
            if kind == "negate":
                return self.negate(value)

        # Not a primitive — must be a taught composition.
        return self.execute(kind, [value] if rhs is None else [value, rhs])

    def execute(self, op_name: str, inputs: list[float]) -> float:
        """Execute a taught composite operation.

        For Level 5, a small built-in table of composite-to-primitive
        reductions lets Sara apply the most common taught operations
        (halve, double, percentage) without requiring the graph-walker
        to be live yet. The graph-walker is a future expansion: it will
        traverse Sara's taught definition (e.g., the path for
        "halving a means dividing a by two") and reduce to primitive
        calls automatically. Until then, any composite not in this
        table raises NotImplementedError so the gap is explicit.
        """
        if op_name == "halve" and len(inputs) == 1:
            return self.divide(inputs[0], 2.0)
        if op_name == "double" and len(inputs) == 1:
            return self.multiply(inputs[0], 2.0)
        if op_name == "percentage" and len(inputs) == 2:
            # N percent of X = N / 100 * X
            return self.multiply(self.divide(inputs[0], 100.0), inputs[1])
        if op_name == "fraction" and len(inputs) == 2:
            return self.divide(inputs[0], inputs[1])
        if op_name == "ratio" and len(inputs) == 2:
            return self.divide(inputs[0], inputs[1])
        if op_name == "mean" and len(inputs) > 0:
            total = 0.0
            for x in inputs:
                total = self.add(total, x)
            return self.divide(total, float(len(inputs)))
        if op_name == "probability" and len(inputs) == 2:
            return self.divide(inputs[0], inputs[1])
        raise NotImplementedError(
            f"operation {op_name!r} is not a primitive and has no built-in "
            f"composite reduction. Teach it via curriculum_math.txt or "
            f"add its reduction to MathCompute.execute()."
        )


# ── Resolver: detect arithmetic phrases in text → Operation ──────────

_HALF_RE = re.compile(
    r"\b(by half|halve[sd]?|cut in half|reduced by half|reduces.*by half)\b",
    re.IGNORECASE,
)
_DOUBLE_RE = re.compile(
    r"\b(doubles?|twofold|two-fold|2-fold|two times)\b",
    re.IGNORECASE,
)
_N_FOLD_RE = re.compile(
    r"\b(\d+)[- ]?fold\b|\b(\d+)\s+times\b",
    re.IGNORECASE,
)
_INCREASE_BY_RE = re.compile(
    r"\bincreas(?:es|ed)?\s+by\s+(-?\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)
_DECREASE_BY_RE = re.compile(
    r"\bdecreas(?:es|ed)?\s+by\s+(-?\d+(?:\.\d+)?)\b"
    r"|\breduce[sd]?\s+by\s+(-?\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)
_DIVIDES_BY_RE = re.compile(
    r"\bdivide[sd]?\s+by\s+(-?\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)
_MULTIPLIED_BY_RE = re.compile(
    r"\bmultiplied\s+by\s+(-?\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)


class MathResolver:
    """Extract an arithmetic Operation from a taught statement.

    Called once per taught sentence at parse time. Returns None when
    the sentence carries no detectable arithmetic phrase, which is
    the vast majority of descriptive biological facts.
    """

    def resolve(self, text: str) -> Operation | None:
        if not text:
            return None

        # Order matters: "reduces chromosome number by half" must match
        # the half pattern before the generic "reduces by N" regex.
        if _HALF_RE.search(text):
            return Operation(kind="multiply", operand=0.5)
        if _DOUBLE_RE.search(text):
            return Operation(kind="multiply", operand=2.0)

        m = _N_FOLD_RE.search(text)
        if m:
            n = m.group(1) or m.group(2)
            try:
                return Operation(kind="multiply", operand=float(n))
            except ValueError:
                pass

        m = _INCREASE_BY_RE.search(text)
        if m:
            try:
                return Operation(kind="add", operand=float(m.group(1)))
            except ValueError:
                pass

        m = _DECREASE_BY_RE.search(text)
        if m:
            raw = m.group(1) or m.group(2)
            try:
                return Operation(kind="subtract", operand=float(raw))
            except ValueError:
                pass

        m = _DIVIDES_BY_RE.search(text)
        if m:
            try:
                return Operation(kind="divide", operand=float(m.group(1)))
            except ValueError:
                pass

        m = _MULTIPLIED_BY_RE.search(text)
        if m:
            try:
                return Operation(kind="multiply", operand=float(m.group(1)))
            except ValueError:
                pass

        return None


def operation_to_tag(op: Operation) -> str:
    """Encode an Operation as the segment.operation_tag string.

    Format: "kind:operand" — operand may be "none" for unary ops.
    """
    if op.operand is None:
        return f"{op.kind}:none"
    return f"{op.kind}:{op.operand}"


def tag_to_operation(tag: str) -> Operation | None:
    """Decode a segment.operation_tag back into an Operation.

    Returns None for empty / malformed tags so callers can abstain
    gracefully rather than crash.
    """
    if not tag or ":" not in tag:
        return None
    kind, _, operand_s = tag.partition(":")
    kind = kind.strip().lower()
    if not kind:
        return None
    if operand_s.strip().lower() in ("", "none"):
        return Operation(kind=kind, operand=None)
    try:
        return Operation(kind=kind, operand=float(operand_s))
    except ValueError:
        return None


# ── Linker: attach an Operation to a segment via repo ────────────────

class MathLinker:
    """Writes an operation_tag onto a segment when a math-aware
    statement is taught. Given the learner's newly-created segment
    id and the Operation extracted from the statement, links them.
    """

    def __init__(self, segment_repo) -> None:
        self.segment_repo = segment_repo

    def link(self, segment_id: int, op: Operation) -> None:
        tag = operation_to_tag(op)
        self.segment_repo.set_operation_tag(segment_id, tag)


# ── NumberExtractor: pull numeric values out of question text ────────

# Biology-specific numeric patterns live alongside the generic
# integer pattern. These cover the shapes seen in Ch10 questions.
_NUM_PATTERNS = [
    # 2n = 96, n = 48, 2n=96
    (re.compile(r"\b(\d+)\s*n\s*=\s*(-?\d+(?:\.\d+)?)\b", re.IGNORECASE),
     lambda m: (f"{m.group(1)}n", float(m.group(2)))),
    # N chromosomes / N cells / N genes  → tag by the trailing noun
    (re.compile(r"\b(-?\d+(?:\.\d+)?)\s+(chromosome|cell|gene|allele)s?\b",
                re.IGNORECASE),
     lambda m: (m.group(2).lower(), float(m.group(1)))),
]


class NumberExtractor:
    """Extract question-side numeric values.

    Returns a dict of {tag: value} so the caller can match the number
    to the right concept. For "Imagine an organism whose 2n = 96",
    returns {"2n": 96.0}.
    """

    def extract(self, text: str) -> dict[str, float]:
        if not text:
            return {}
        out: dict[str, float] = {}
        for pat, fn in _NUM_PATTERNS:
            for m in pat.finditer(text):
                tag, val = fn(m)
                out[tag] = val
        # Fallback: bare integers in the text (kept for last-resort
        # matches against MC choice text).
        for m in re.finditer(r"\b(-?\d+(?:\.\d+)?)\b", text):
            out.setdefault(f"_n{m.start()}", float(m.group(1)))
        return out
