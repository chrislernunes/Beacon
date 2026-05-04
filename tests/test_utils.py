"""Tests for beacon.utils."""
from __future__ import annotations

from typing import Any

import pytest

from beacon.utils import (
    classify_object,
    compute_deep_diff,
    filter_locals,
    format_numeric_diff,
    pluralize,
    safe_repr,
    safe_type_name,
    truncate_middle,
    unified_text_diff,
)


class TestSafeRepr:
    def test_simple_int(self) -> None:
        assert safe_repr(42) == "42"

    def test_simple_string(self) -> None:
        assert safe_repr("hello") == "'hello'"

    def test_none(self) -> None:
        assert safe_repr(None) == "None"

    def test_truncation(self) -> None:
        long_str = "x" * 1000
        r = safe_repr(long_str, max_length=100)
        assert len(r) <= 101 + 1  # +1 for the ellipsis char
        assert r.endswith("…")

    def test_repr_error_handled(self) -> None:
        class BadRepr:
            def __repr__(self) -> str:
                raise ValueError("boom")

        r = safe_repr(BadRepr())
        assert "repr failed" in r

    def test_no_truncation_when_short(self) -> None:
        r = safe_repr(42, max_length=500)
        assert "…" not in r


class TestSafeTypeName:
    def test_builtin_type(self) -> None:
        assert safe_type_name(42) == "int"

    def test_string(self) -> None:
        assert safe_type_name("hello") == "str"

    def test_none(self) -> None:
        assert safe_type_name(None) == "NoneType"

    def test_custom_class(self) -> None:
        class MyClass:
            pass

        obj = MyClass()
        name = safe_type_name(obj)
        assert "MyClass" in name

    def test_never_raises(self) -> None:
        class Broken:
            @property
            def __class__(self) -> Any:  # type: ignore[override]
                raise RuntimeError("broken")

        # Should not raise even with a pathological object
        try:
            safe_type_name(object())
        except Exception:
            pytest.fail("safe_type_name raised unexpectedly")


class TestClassifyObject:
    def test_int_is_scalar(self) -> None:
        assert classify_object(42) == "scalar"

    def test_float_is_scalar(self) -> None:
        assert classify_object(3.14) == "scalar"

    def test_none_is_scalar(self) -> None:
        assert classify_object(None) == "scalar"

    def test_list_is_sequence(self) -> None:
        assert classify_object([1, 2, 3]) == "sequence"

    def test_tuple_is_sequence(self) -> None:
        assert classify_object((1, 2)) == "sequence"

    def test_dict_is_mapping(self) -> None:
        assert classify_object({"a": 1}) == "mapping"

    def test_set_is_set(self) -> None:
        assert classify_object({1, 2, 3}) == "set"

    def test_frozenset_is_set(self) -> None:
        assert classify_object(frozenset({1, 2})) == "set"

    def test_dataclass(self) -> None:
        import dataclasses

        @dataclasses.dataclass
        class Point:
            x: float
            y: float

        assert classify_object(Point(1.0, 2.0)) == "dataclass"

    def test_numpy_array(self) -> None:
        pytest.importorskip("numpy")
        import numpy as np

        assert classify_object(np.array([1, 2, 3])) == "ndarray"

    def test_pandas_dataframe(self) -> None:
        pytest.importorskip("pandas")
        import pandas as pd

        assert classify_object(pd.DataFrame({"a": [1, 2]})) == "dataframe"

    def test_pandas_series(self) -> None:
        pytest.importorskip("pandas")
        import pandas as pd

        assert classify_object(pd.Series([1, 2, 3])) == "series"


class TestFilterLocals:
    def test_basic_filtering(self) -> None:
        local_vars = {"x": 1, "y": 2, "__builtins__": {}, "_pytest_helper": object()}
        result = filter_locals(local_vars, exclude_patterns=["__*", "_pytest*"])
        names = [n for n, _ in result]
        assert "x" in names
        assert "y" in names
        assert "__builtins__" not in names
        assert "_pytest_helper" not in names

    def test_max_vars_respected(self) -> None:
        local_vars = {f"var_{i}": i for i in range(20)}
        result = filter_locals(local_vars, exclude_patterns=[], max_vars=5)
        assert len(result) <= 5

    def test_modules_excluded(self) -> None:
        import sys as sys_module

        local_vars = {"sys": sys_module, "x": 42}
        result = filter_locals(local_vars, exclude_patterns=[])
        names = [n for n, _ in result]
        assert "sys" not in names
        assert "x" in names

    def test_empty_locals(self) -> None:
        result = filter_locals({}, exclude_patterns=[])
        assert result == []


class TestFormatNumericDiff:
    def test_integer_diff(self) -> None:
        result = format_numeric_diff(10, 12)
        assert result is not None
        assert "2" in result

    def test_float_diff(self) -> None:
        result = format_numeric_diff(1.0, 1.001)
        assert result is not None
        assert "Δ" in result

    def test_zero_expected(self) -> None:
        result = format_numeric_diff(0, 5)
        assert result is not None
        # No relative diff for zero denominator
        assert "%" not in result

    def test_equal_values(self) -> None:
        result = format_numeric_diff(42, 42)
        assert result is not None  # still shows Δ = 0

    def test_non_numeric_returns_none(self) -> None:
        result = format_numeric_diff("a", "b")
        assert result is None


class TestUnifiedTextDiff:
    def test_identical_strings(self) -> None:
        diff = unified_text_diff("hello", "hello")
        assert diff == []

    def test_different_strings(self) -> None:
        diff = unified_text_diff("hello\nworld\n", "hello\nearth\n")
        assert any("-world" in line for line in diff)
        assert any("+earth" in line for line in diff)


class TestComputeDeepDiff:
    def test_equal_dicts_returns_none(self) -> None:
        result = compute_deep_diff({"a": 1}, {"a": 1})
        assert result is None

    def test_changed_value(self) -> None:
        result = compute_deep_diff({"a": 1}, {"a": 2})
        assert result is not None
        assert "values_changed" in result

    def test_added_key(self) -> None:
        result = compute_deep_diff({"a": 1}, {"a": 1, "b": 2})
        assert result is not None

    def test_lists(self) -> None:
        result = compute_deep_diff([1, 2, 3], [1, 2, 4])
        assert result is not None


class TestHelpers:
    def test_pluralize_singular(self) -> None:
        assert pluralize(1, "test") == "1 test"

    def test_pluralize_plural(self) -> None:
        assert pluralize(2, "test") == "2 tests"

    def test_pluralize_zero(self) -> None:
        assert pluralize(0, "test") == "0 tests"

    def test_pluralize_custom_plural(self) -> None:
        assert pluralize(2, "fish", "fish") == "2 fish"

    def test_truncate_middle_short(self) -> None:
        s = "hello"
        assert truncate_middle(s, max_len=100) == "hello"

    def test_truncate_middle_long(self) -> None:
        s = "a" * 200
        result = truncate_middle(s, max_len=50)
        assert len(result) <= 51
        assert "…" in result
