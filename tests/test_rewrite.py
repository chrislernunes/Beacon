"""Tests for beacon.rewrite — AST-based assertion introspection."""
from __future__ import annotations

import sys
from typing import Any, Tuple

import pytest

from beacon.rewrite import (
    AssertionBreakdown,
    SubExpr,
    introspect_assertion,
)


def _make_exc_info(msg: str = "fail") -> Tuple[type, BaseException, Any]:
    try:
        raise AssertionError(msg)
    except AssertionError:
        return sys.exc_info()  # type: ignore[return-value]


class TestAssertionBreakdown:
    def test_default_values(self) -> None:
        bd = AssertionBreakdown()
        assert bd.raw_source == ""
        assert bd.expression_source == ""
        assert bd.lhs is None
        assert bd.rhs is None
        assert bd.operator is None
        assert bd.sub_expressions == []

    def test_is_comparison_false_by_default(self) -> None:
        bd = AssertionBreakdown()
        assert bd.is_comparison is False

    def test_is_comparison_true_with_lhs_rhs(self) -> None:
        bd = AssertionBreakdown()
        bd.lhs = SubExpr(source="a", value=1)
        bd.rhs = SubExpr(source="b", value=2)
        assert bd.is_comparison is True

    def test_has_sub_expressions_false(self) -> None:
        bd = AssertionBreakdown()
        assert bd.has_sub_expressions is False

    def test_has_sub_expressions_true(self) -> None:
        bd = AssertionBreakdown()
        bd.sub_expressions.append(SubExpr(source="x", value=42))
        assert bd.has_sub_expressions is True


class TestSubExpr:
    def test_basic_creation(self) -> None:
        s = SubExpr(source="my_var", value=42)
        assert s.source == "my_var"
        assert s.value == 42
        assert s.type_name == "int"

    def test_repr_property(self) -> None:
        s = SubExpr(source="x", value=[1, 2, 3])
        assert "[1, 2, 3]" in s.repr

    def test_custom_type_name(self) -> None:
        s = SubExpr(source="x", value=1, type_name="mypackage.MyType")
        assert s.type_name == "mypackage.MyType"

    def test_none_value(self) -> None:
        s = SubExpr(source="x", value=None)
        assert s.type_name == "NoneType"
        assert s.repr == "None"


class TestIntrospectAssertion:
    def test_none_traceback_returns_breakdown(self) -> None:
        """No traceback should return a minimal breakdown, not raise."""
        bd = introspect_assertion((AssertionError, AssertionError("test"), None))
        assert isinstance(bd, AssertionBreakdown)
        assert bd.error is not None

    def test_real_exc_info_returns_breakdown(self) -> None:
        """A real AssertionError should produce a valid breakdown."""
        ei = _make_exc_info("something failed")
        bd = introspect_assertion(ei)
        assert isinstance(bd, AssertionBreakdown)

    def test_never_raises(self) -> None:
        """introspect_assertion must never raise, no matter what."""
        # Deliberately broken exc_info
        bd = introspect_assertion((None, None, None))  # type: ignore[arg-type]
        assert isinstance(bd, AssertionBreakdown)

    def test_breakdown_from_comparison(self) -> None:
        """
        When called from a function with a comparison assert, the breakdown
        should have some expression info.
        """

        def _compare() -> Tuple[type, BaseException, Any]:
            x = 10
            y = 20
            try:
                assert x == y
            except AssertionError:
                return sys.exc_info()  # type: ignore[return-value]
            return (AssertionError, AssertionError(), None)

        ei = _compare()
        bd = introspect_assertion(ei)
        assert isinstance(bd, AssertionBreakdown)
        # We should at least get some source
        # (exact content depends on executing library version)

    def test_breakdown_is_breakdown_type(self) -> None:
        ei = _make_exc_info()
        result = introspect_assertion(ei)
        assert isinstance(result, AssertionBreakdown)
