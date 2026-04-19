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

    def test_apply_unknown_op_raises_notimplemented(self):
        # Taught composites exist but have no graph-walker yet in Level 1.
        # Calling apply() on one should signal the gap clearly.
        op = Operation(kind="halve", operand=None)
        with pytest.raises(NotImplementedError) as excinfo:
            self.m.apply(op, 96)
        # Error message must mention the composition gap — future sessions
        # should know exactly where to hook the graph-walker in.
        assert "halve" in str(excinfo.value)
        assert "taught" in str(excinfo.value).lower()
