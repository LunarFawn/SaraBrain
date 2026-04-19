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

from dataclasses import dataclass


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
        """Execute a taught composite operation by walking the graph.

        In this first cut the graph-walk is not yet wired up. When a
        later layer adds composition lookup, this method will follow
        the taught definition of op_name, reducing to primitive calls.
        For now it signals the gap explicitly so tests can cover both
        the primitives-only path and the not-yet-taught path.
        """
        raise NotImplementedError(
            f"operation {op_name!r} is not a primitive and has no taught "
            f"definition wired up yet. Teach it via curriculum_math.txt "
            f"once the composition layer is live."
        )
