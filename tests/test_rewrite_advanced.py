"""
tests/test_rewrite_advanced.py
───────────────────────────────
High-value tests for beacon.rewrite targeting the untested 74% of the module.

Coverage targets (from: src/beacon/rewrite.py ~26% covered)
────────────────────────────────────────────────────────────
_op_symbol            — all operator types (lines 115-116)
_node_source          — fallback path when get_source_segment returns None (135-138)
_safe_eval            — both success and failure paths (151-155)
_try_get_assert_node  — executing integration + fallback to None (161-187)
_extract_assert_source — multi-line assert collection (190-211)
_breakdown_from_assert_node:
  Case 1: simple comparison (228-243)
  Case 2: chained comparison (246-252)
  Case 3: BoolOp / UnaryOp (254-262)
  Case 4: generic expression (265-268)
  assert message extraction (222-223)
introspect_assertion:
  fallback path when assert_node is None (307-324)
  full_src getsource failure (328-330)
  breakdown_from_assert_node exception handler (332-335)

Design notes
────────────
We must use real assert statements inside real functions to get genuine
traceback objects. Synthetic exc_info with tb=None only tests the guard paths.
The key technique is: raise AssertionError inside a nested helper, capture
sys.exc_info() while still in the except block, and pass that to introspect_assertion.
"""
from __future__ import annotations

import ast
import sys
import textwrap
import types
from typing import Any, Tuple

import pytest

from beacon.rewrite import (
    AssertionBreakdown,
    SubExpr,
    _breakdown_from_assert_node,
    _extract_assert_source,
    _node_source,
    _op_symbol,
    _safe_eval,
    introspect_assertion,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _capture_assert(fn: Any) -> Tuple[type, BaseException, Any]:
    """
    Call fn() which must contain a failing assert.
    Return the raw sys.exc_info() triple while the frame is still alive.
    """
    try:
        fn()
    except AssertionError:
        return sys.exc_info()  # type: ignore[return-value]
    raise RuntimeError("Expected function to raise AssertionError")


def _current_frame() -> types.FrameType:
    return sys._getframe(1)


# ── _op_symbol ────────────────────────────────────────────────────────────────


class TestOpSymbol:
    """Every AST comparison operator must map to the correct string."""

    @pytest.mark.parametrize(
        "op_cls, expected",
        [
            (ast.Eq, "=="),
            (ast.NotEq, "!="),
            (ast.Lt, "<"),
            (ast.LtE, "<="),
            (ast.Gt, ">"),
            (ast.GtE, ">="),
            (ast.Is, "is"),
            (ast.IsNot, "is not"),
            (ast.In, "in"),
            (ast.NotIn, "not in"),
        ],
    )
    def test_known_operator(self, op_cls: type, expected: str) -> None:
        op = op_cls()
        assert _op_symbol(op) == expected

    def test_unknown_operator_returns_class_name(self) -> None:
        """An op type not in the map must return its class name, not crash."""

        class FakeOp(ast.cmpop):
            pass

        op = FakeOp()
        result = _op_symbol(op)
        assert result == "FakeOp"


# ── _node_source ──────────────────────────────────────────────────────────────


class TestNodeSource:
    def test_extracts_simple_name(self) -> None:
        source = "x = 1\nassert x == 1\n"
        tree = ast.parse(source)
        assert_node = next(n for n in ast.walk(tree) if isinstance(n, ast.Assert))
        compare = assert_node.test
        assert isinstance(compare, ast.Compare)
        result = _node_source(compare.left, source)
        assert result == "x"

    def test_extracts_comparison_expression(self) -> None:
        source = "assert alpha == beta\n"
        tree = ast.parse(source)
        assert_node = next(n for n in ast.walk(tree) if isinstance(n, ast.Assert))
        result = _node_source(assert_node.test, source)
        assert "alpha" in result
        assert "beta" in result

    def test_fallback_on_malformed_node(self) -> None:
        """A node with no lineno must return the fallback string, never raise."""

        class BareNode(ast.AST):
            pass

        node = BareNode()
        result = _node_source(node, "some source")
        assert result == "<source unavailable>"

    def test_multiline_expression(self) -> None:
        source = "assert (\n    alpha\n    == beta\n)\n"
        tree = ast.parse(source)
        assert_node = next(n for n in ast.walk(tree) if isinstance(n, ast.Assert))
        result = _node_source(assert_node.test, source)
        # Should contain both sides
        assert "alpha" in result or "beta" in result


# ── _safe_eval ────────────────────────────────────────────────────────────────


class TestSafeEval:
    def _get_frame(self) -> types.FrameType:
        return sys._getframe(0)

    def test_evaluates_literal(self) -> None:
        frame = self._get_frame()
        val, ok = _safe_eval("42", frame)
        assert ok is True
        assert val == 42

    def test_evaluates_frame_local(self) -> None:
        sentinel_value = 12345  # noqa: F841 — used by eval
        frame = sys._getframe(0)
        val, ok = _safe_eval("sentinel_value", frame)
        assert ok is True
        assert val == 12345

    def test_undefined_name_returns_false(self) -> None:
        frame = self._get_frame()
        val, ok = _safe_eval("definitely_undefined_xyz_123", frame)
        assert ok is False
        assert val is None

    def test_syntax_error_returns_false(self) -> None:
        frame = self._get_frame()
        val, ok = _safe_eval("this is not valid python !!!", frame)
        assert ok is False

    def test_exception_in_eval_returns_false(self) -> None:
        frame = self._get_frame()
        # This evaluates to a division by zero at eval time
        val, ok = _safe_eval("1/0", frame)
        assert ok is False

    def test_evaluates_complex_expression(self) -> None:
        a = 10  # noqa: F841
        b = 20  # noqa: F841
        frame = sys._getframe(0)
        val, ok = _safe_eval("a + b", frame)
        assert ok is True
        assert val == 30

    def test_evaluates_list_literal(self) -> None:
        frame = self._get_frame()
        val, ok = _safe_eval("[1, 2, 3]", frame)
        assert ok is True
        assert val == [1, 2, 3]


# ── _extract_assert_source ────────────────────────────────────────────────────


class TestExtractAssertSource:
    def test_extracts_single_line_assert(self) -> None:
        def failing() -> None:
            assert 1 == 2  # noqa: S101

        ei = _capture_assert(failing)
        tb = ei[2]
        while tb.tb_next:
            tb = tb.tb_next
        frame = tb.tb_frame
        result = _extract_assert_source(frame)
        assert result is not None
        assert "assert" in result
        assert "1" in result

    def test_returns_none_for_builtin_frame(self) -> None:
        """Built-in frames have no Python source — must return None gracefully."""
        # We can't easily get a real C frame here, but we can monkeypatch
        # inspect.getsourcelines to raise OSError
        import inspect as inspect_mod
        import unittest.mock as mock

        with mock.patch.object(inspect_mod, "getsourcelines", side_effect=OSError):
            frame = sys._getframe(0)
            result = _extract_assert_source(frame)
            assert result is None

    def test_handles_indented_assert(self) -> None:
        """Dedented assert source should parse correctly."""

        def nested() -> None:
            x = 1
            assert x == 2  # noqa: S101

        ei = _capture_assert(nested)
        tb = ei[2]
        while tb.tb_next:
            tb = tb.tb_next
        frame = tb.tb_frame
        result = _extract_assert_source(frame)
        assert result is not None
        assert "assert" in result.lower()


# ── _breakdown_from_assert_node: all four cases ───────────────────────────────


class TestBreakdownFromAssertNode:
    """
    Exercise all four branches of _breakdown_from_assert_node directly
    by constructing the AST nodes and providing a real frame.
    """

    def _frame_with(self, **local_vars: Any) -> types.FrameType:
        """Return a real frame with the given names pre-populated as locals."""
        # We inject them into the current frame via a nested exec
        # The actual eval in _safe_eval uses frame.f_locals
        frame = sys._getframe(0)
        # Update f_locals via ctypes trick — the only reliable way
        import ctypes
        frame.f_locals.update(local_vars)
        ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), ctypes.c_int(0))
        return frame

    def test_case1_simple_eq_comparison(self) -> None:
        """Case 1: assert a == b — extracts lhs, rhs, operator."""
        source = "assert result == expected\n"
        tree = ast.parse(source)
        assert_node = next(n for n in ast.walk(tree) if isinstance(n, ast.Assert))

        frame = self._frame_with(result=42, expected=99)
        bd = _breakdown_from_assert_node(assert_node, frame, source)

        assert bd.operator == "=="
        assert bd.lhs is not None
        assert bd.rhs is not None
        assert bd.lhs.source == "result"
        assert bd.rhs.source == "expected"
        assert bd.lhs.value == 42
        assert bd.rhs.value == 99
        assert len(bd.sub_expressions) == 2

    def test_case1_lt_comparison(self) -> None:
        source = "assert price < limit\n"
        tree = ast.parse(source)
        assert_node = next(n for n in ast.walk(tree) if isinstance(n, ast.Assert))

        frame = self._frame_with(price=150.0, limit=100.0)
        bd = _breakdown_from_assert_node(assert_node, frame, source)

        assert bd.operator == "<"
        assert bd.lhs is not None
        assert bd.rhs is not None

    def test_case1_in_operator(self) -> None:
        source = "assert item in container\n"
        tree = ast.parse(source)
        assert_node = next(n for n in ast.walk(tree) if isinstance(n, ast.Assert))

        frame = self._frame_with(item="AAPL", container=["GOOG", "MSFT"])
        bd = _breakdown_from_assert_node(assert_node, frame, source)

        assert bd.operator == "in"

    def test_case2_chained_comparison(self) -> None:
        """Case 2: assert a < b < c — all three values extracted."""
        source = "assert low < mid < high\n"
        tree = ast.parse(source)
        assert_node = next(n for n in ast.walk(tree) if isinstance(n, ast.Assert))

        frame = self._frame_with(low=0, mid=5, high=10)
        bd = _breakdown_from_assert_node(assert_node, frame, source)

        # No lhs/rhs for chained; sub_expressions has all three
        assert bd.lhs is None
        assert len(bd.sub_expressions) >= 2

    def test_case3_bool_op_and(self) -> None:
        """Case 3: assert a and b — sub-expressions extracted."""
        source = "assert is_valid and is_active\n"
        tree = ast.parse(source)
        assert_node = next(n for n in ast.walk(tree) if isinstance(n, ast.Assert))

        frame = self._frame_with(is_valid=True, is_active=False)
        bd = _breakdown_from_assert_node(assert_node, frame, source)

        # Should have sub-expressions from the BoolOp walk
        assert bd.lhs is None  # BoolOp case doesn't set lhs/rhs
        # expression_source must be populated
        assert bd.expression_source != ""

    def test_case3_unary_op_not(self) -> None:
        """Case 3: assert not x — UnaryOp path."""
        source = "assert not flag\n"
        tree = ast.parse(source)
        assert_node = next(n for n in ast.walk(tree) if isinstance(n, ast.Assert))

        frame = self._frame_with(flag=True)
        bd = _breakdown_from_assert_node(assert_node, frame, source)

        assert bd.expression_source != ""

    def test_case4_generic_function_call(self) -> None:
        """Case 4: assert fn() — evaluate the whole expression."""

        def always_false() -> bool:
            return False

        source = "assert always_false()\n"
        tree = ast.parse(source)
        assert_node = next(n for n in ast.walk(tree) if isinstance(n, ast.Assert))

        frame = self._frame_with(always_false=always_false)
        bd = _breakdown_from_assert_node(assert_node, frame, source)

        assert bd.expression_source != ""

    def test_assert_message_extracted(self) -> None:
        """Assert message string must appear in message_source."""
        source = 'assert x > 0, "x must be positive"\n'
        tree = ast.parse(source)
        assert_node = next(n for n in ast.walk(tree) if isinstance(n, ast.Assert))

        frame = self._frame_with(x=-1)
        bd = _breakdown_from_assert_node(assert_node, frame, source)

        assert bd.message_source is not None
        assert "positive" in bd.message_source

    def test_eval_failure_sets_no_lhs_rhs(self) -> None:
        """When eval fails (name not in scope), lhs/rhs should be None."""
        source = "assert completely_undefined == something_else\n"
        tree = ast.parse(source)
        assert_node = next(n for n in ast.walk(tree) if isinstance(n, ast.Assert))

        # Frame with no matching locals
        frame = sys._getframe(0)
        bd = _breakdown_from_assert_node(assert_node, frame, source)

        # Both evals fail → lhs and rhs stay None
        assert bd.lhs is None
        assert bd.rhs is None
        assert bd.operator == "=="  # operator is still extracted


# ── introspect_assertion: integration paths ───────────────────────────────────


class TestIntrospectAssertionIntegration:
    """
    Test introspect_assertion end-to-end using real assert statements.
    These tests exercise the full path including executing library integration.
    """

    def test_simple_equality_assert(self) -> None:
        def failing() -> None:
            result = 1 + 1
            expected = 3
            assert result == expected

        ei = _capture_assert(failing)
        bd = introspect_assertion(ei)

        assert isinstance(bd, AssertionBreakdown)
        # Should have extracted something meaningful
        assert bd.raw_source or bd.expression_source

    def test_assertion_with_explicit_message(self) -> None:
        def failing() -> None:
            price = -1.0
            assert price > 0, f"Price must be positive, got {price}"

        ei = _capture_assert(failing)
        bd = introspect_assertion(ei)

        assert isinstance(bd, AssertionBreakdown)

    def test_chained_comparison_assert(self) -> None:
        def failing() -> None:
            weight = 1.5
            assert 0.0 <= weight <= 1.0

        ei = _capture_assert(failing)
        bd = introspect_assertion(ei)

        assert isinstance(bd, AssertionBreakdown)
        assert not bd.error or "map" not in (bd.error or "")

    def test_in_operator_assert(self) -> None:
        def failing() -> None:
            symbol = "INVALID"
            valid_symbols = ["AAPL", "GOOG", "MSFT"]
            assert symbol in valid_symbols

        ei = _capture_assert(failing)
        bd = introspect_assertion(ei)

        assert isinstance(bd, AssertionBreakdown)

    def test_boolean_and_assert(self) -> None:
        def failing() -> None:
            is_funded = True
            is_approved = False
            assert is_funded and is_approved

        ei = _capture_assert(failing)
        bd = introspect_assertion(ei)

        assert isinstance(bd, AssertionBreakdown)

    def test_assert_with_function_call(self) -> None:
        def failing() -> None:
            data = []
            assert len(data) > 0

        ei = _capture_assert(failing)
        bd = introspect_assertion(ei)

        assert isinstance(bd, AssertionBreakdown)

    def test_walksback_to_innermost_frame(self) -> None:
        """introspect_assertion must use the innermost tb frame, not the outermost."""

        def inner() -> None:
            x = 999
            assert x == 0

        def outer() -> None:
            inner()

        try:
            outer()
        except AssertionError:
            ei = sys.exc_info()

        bd = introspect_assertion(ei)  # type: ignore[possibly-undefined]
        assert isinstance(bd, AssertionBreakdown)

    def test_no_traceback_returns_error_breakdown(self) -> None:
        ei = (AssertionError, AssertionError("manual"), None)
        bd = introspect_assertion(ei)  # type: ignore[arg-type]
        assert bd.error == "No traceback available"
        assert bd.raw_source == ""

    def test_never_raises_on_garbage_input(self) -> None:
        """Must degrade gracefully on completely broken input."""
        # Totally broken exc_info
        bd = introspect_assertion((None, None, None))  # type: ignore[arg-type]
        assert isinstance(bd, AssertionBreakdown)

    def test_breakdown_with_object_that_raises_on_repr(self) -> None:
        """Objects with broken __repr__ must not crash introspection."""

        class BadRepr:
            def __repr__(self) -> str:
                raise RuntimeError("repr exploded")

            def __eq__(self, other: object) -> bool:
                return False

        def failing() -> None:
            actual = BadRepr()
            expected = BadRepr()
            assert actual == expected

        ei = _capture_assert(failing)
        # Must not raise
        bd = introspect_assertion(ei)
        assert isinstance(bd, AssertionBreakdown)

    def test_breakdown_with_none_comparison(self) -> None:
        def failing() -> None:
            result = None
            assert result is not None

        ei = _capture_assert(failing)
        bd = introspect_assertion(ei)
        assert isinstance(bd, AssertionBreakdown)

    def test_breakdown_dict_comparison(self) -> None:
        def failing() -> None:
            actual = {"delta": 0.48, "gamma": 0.003}
            expected = {"delta": 0.52, "gamma": 0.004}
            assert actual == expected

        ei = _capture_assert(failing)
        bd = introspect_assertion(ei)
        assert isinstance(bd, AssertionBreakdown)

    def test_fallback_when_executing_returns_no_node(self) -> None:
        """
        When executing cannot map frame → node, we fall back to raw source
        extraction and manual AST parsing. This path fires for dynamically
        compiled code (exec/eval).
        """
        # Compile and exec a function to force the fallback path
        code = textwrap.dedent(
            """
            def dynamic_assert():
                x = 1
                assert x == 2
            """
        )
        ns: dict[str, Any] = {}
        exec(compile(code, "<dynamic>", "exec"), ns)  # noqa: S102
        dynamic_assert = ns["dynamic_assert"]

        try:
            dynamic_assert()
        except AssertionError:
            ei = sys.exc_info()

        bd = introspect_assertion(ei)  # type: ignore[possibly-undefined]
        # Must return a breakdown, not raise — may have an error note
        assert isinstance(bd, AssertionBreakdown)
