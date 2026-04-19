"""Tests for Sara's math module — the six ALU primitives.

These cover only the primitive layer. Taught composites (halve,
probability, mean, etc.) get tested when the composition-executor
layer is wired up.
"""
from __future__ import annotations

import pytest

from sara_brain.core.math import MathCompute, Operation
from sara_brain.innate.primitives import (
    OPERATION, get_operation, is_operation,
)


# ── Primitive layer registration ──────────────────────────────────────

class TestOperationPrimitiveLayer:
    def test_layer_contains_exactly_six_primitives(self):
        assert OPERATION == frozenset({
            "add", "subtract", "multiply", "divide", "abs", "negate",
        })

    def test_get_operation_accessor_returns_layer(self):
        assert get_operation() == OPERATION

    def test_is_operation_positive_cases(self):
        for name in ("add", "subtract", "multiply", "divide", "abs", "negate"):
            assert is_operation(name), f"{name} should be recognised"

    def test_is_operation_rejects_composites(self):
        # Taught compositions must NOT be in the primitive layer —
        # if one slips in, the architectural invariant is broken.
        for name in ("halve", "double", "percentage", "probability",
                     "mean", "median", "variance", "stddev"):
            assert not is_operation(name), (
                f"{name!r} must be a taught composition, not a primitive"
            )

    def test_is_operation_handles_case_and_whitespace(self):
        assert is_operation(" ADD ")
        assert is_operation("Multiply")


# ── The six primitives ────────────────────────────────────────────────

class TestArithmeticPrimitives:
    def setup_method(self):
        self.m = MathCompute()

    def test_add(self):
        assert self.m.add(2, 3) == 5
        assert self.m.add(-1, 1) == 0
        assert self.m.add(0.1, 0.2) == pytest.approx(0.3)

    def test_subtract(self):
        assert self.m.subtract(5, 3) == 2
        assert self.m.subtract(0, 5) == -5
        assert self.m.subtract(2.5, 0.5) == 2.0

    def test_multiply(self):
        # Jennifer's call: multiply is a native-Python primitive.
        # No loops of additions. This test enforces that.
        assert self.m.multiply(4, 3) == 12
        assert self.m.multiply(96, 0.5) == 48.0  # the Q95 case
        assert self.m.multiply(-2, 5) == -10
        assert self.m.multiply(0, 999_999) == 0

    def test_divide(self):
        assert self.m.divide(10, 2) == 5.0
        assert self.m.divide(96, 2) == 48.0  # halving by division
        assert self.m.divide(1, 4) == 0.25

    def test_divide_by_zero_raises(self):
        # Let Python's exception propagate; the caller (STM/scorer) is
        # expected to abstain rather than hide the error.
        with pytest.raises(ZeroDivisionError):
            self.m.divide(5, 0)

    def test_abs(self):
        assert self.m.abs_(3) == 3
        assert self.m.abs_(-3) == 3
        assert self.m.abs_(0) == 0
        assert self.m.abs_(-2.5) == 2.5

    def test_negate(self):
        assert self.m.negate(5) == -5
        assert self.m.negate(-5) == 5
        assert self.m.negate(0) == 0
        assert self.m.negate(0.25) == -0.25


# ── apply() dispatch ──────────────────────────────────────────────────

class TestApplyDispatch:
    def setup_method(self):
        self.m = MathCompute()

    def test_apply_add_uses_operation_operand(self):
        op = Operation(kind="add", operand=10)
        assert self.m.apply(op, 5) == 15

    def test_apply_multiply_halving_case(self):
        # "meiosis reduces chromosome number by half" → multiply(x, 0.5)
        op = Operation(kind="multiply", operand=0.5)
        assert self.m.apply(op, 96) == 48.0

    def test_apply_explicit_operand_overrides_op_operand(self):
        op = Operation(kind="multiply", operand=999)
        assert self.m.apply(op, 3, operand=4) == 12

    def test_apply_binary_without_operand_raises(self):
        op = Operation(kind="subtract", operand=None)
        with pytest.raises(ValueError):
            self.m.apply(op, 5)

    def test_apply_unary_abs_ignores_operand(self):
        op = Operation(kind="abs", operand=None)
        assert self.m.apply(op, -7) == 7

    def test_apply_unary_negate_ignores_operand(self):
        op = Operation(kind="negate", operand=None)
        assert self.m.apply(op, 3) == -3

    def test_apply_halve_composite_reduction(self):
        # halve is a built-in composite reduction: divide by 2.
        op = Operation(kind="halve", operand=None)
        assert self.m.apply(op, 96) == 48.0

    def test_apply_double_composite_reduction(self):
        op = Operation(kind="double", operand=None)
        assert self.m.apply(op, 46) == 92.0

    def test_apply_truly_unknown_op_raises_notimplemented(self):
        # Operations with no primitive mapping AND no built-in composite
        # reduction must signal the gap explicitly so it's obvious where
        # to add the taught definition.
        op = Operation(kind="transmogrify", operand=None)
        with pytest.raises(NotImplementedError) as excinfo:
            self.m.apply(op, 7)
        assert "transmogrify" in str(excinfo.value)


# ── Resolver, Linker, NumberExtractor ─────────────────────────────────

class TestMathResolver:
    def setup_method(self):
        from sara_brain.core.math import MathResolver
        self.r = MathResolver()

    def test_resolves_by_half(self):
        op = self.r.resolve("meiosis reduces chromosome number by half")
        assert op is not None
        assert op.kind == "multiply"
        assert op.operand == 0.5

    def test_resolves_doubles(self):
        op = self.r.resolve("DNA doubles during S phase")
        assert op is not None
        assert op.kind == "multiply"
        assert op.operand == 2.0

    def test_resolves_n_fold(self):
        op = self.r.resolve("expression increased threefold")
        # "threefold" isn't parsed as integer; try 3-fold
        op = self.r.resolve("expression increased 3-fold")
        assert op is not None
        assert op.kind == "multiply"
        assert op.operand == 3.0

    def test_resolves_increases_by(self):
        op = self.r.resolve("the count increases by 5")
        assert op is not None
        assert op.kind == "add"
        assert op.operand == 5.0

    def test_returns_none_for_descriptive_text(self):
        assert self.r.resolve("apples are red") is None
        assert self.r.resolve("") is None
        assert self.r.resolve("meiosis is sexual reproduction") is None


class TestTagEncoding:
    def test_encode_decode_roundtrip_binary(self):
        from sara_brain.core.math import (
            Operation, operation_to_tag, tag_to_operation,
        )
        op = Operation(kind="multiply", operand=0.5)
        tag = operation_to_tag(op)
        assert tag == "multiply:0.5"
        round_tripped = tag_to_operation(tag)
        assert round_tripped.kind == "multiply"
        assert round_tripped.operand == 0.5

    def test_encode_decode_roundtrip_unary(self):
        from sara_brain.core.math import (
            Operation, operation_to_tag, tag_to_operation,
        )
        op = Operation(kind="abs", operand=None)
        tag = operation_to_tag(op)
        assert tag == "abs:none"
        round_tripped = tag_to_operation(tag)
        assert round_tripped.kind == "abs"
        assert round_tripped.operand is None

    def test_decode_returns_none_for_malformed(self):
        from sara_brain.core.math import tag_to_operation
        assert tag_to_operation("") is None
        assert tag_to_operation("nocolon") is None
        assert tag_to_operation("multiply:not_a_number") is None


class TestNumberExtractor:
    def setup_method(self):
        from sara_brain.core.math import NumberExtractor
        self.n = NumberExtractor()

    def test_extract_2n_equals(self):
        r = self.n.extract("Imagine an organism whose 2n = 96")
        assert r["2n"] == 96.0

    def test_extract_bare_integer(self):
        r = self.n.extract("How many chromosomes are in the cell?")
        # No bare ints in this sentence — should yield empty-ish
        assert all(isinstance(v, float) for v in r.values())

    def test_extract_N_chromosomes(self):
        r = self.n.extract("Human somatic cells have 46 chromosomes")
        assert r.get("chromosome") == 46.0

    def test_empty_text(self):
        assert self.n.extract("") == {}
