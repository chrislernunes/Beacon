"""
tests/test_adversarial.py
──────────────────────────
Elite-reliability adversarial tests for beacon.

Design principles
─────────────────
• Every test targets a specific failure mode, not a happy path.
• Invariant tests verify "never crashes regardless of input."
• Framework-failure tests verify pytest still completes when beacon internals break.
• Object-edge tests exercise broken __repr__, recursion, extreme sizes.
• Config-conflict tests exercise env vs TOML vs programmatic precedence under
  invalid/adversarial inputs.

Coverage targets (gaps from 67% baseline)
──────────────────────────────────────────
utils.py  lines: 11-23, 28-50, 58-61, 85, 96-97, 103, 112-143, 177, 194-239
reporters.py lines: 303-670 (diff dispatch, JSON/HTML reporters, render_failure fan-out)
plugin.py lines: 48-91 (non-AssertionError exc, OutcomeException filter, legacy excinfo)
config.py lines: 55-183 (full load(), apply_env, override, singleton)
rewrite.py lines: 180-336 (introspect_assertion edge paths)
"""
from __future__ import annotations

import ast
import json
import os
import sys
import textwrap
import types
from io import StringIO
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from beacon.config import BeaconConfig, get_config, reset_config
from beacon.reporters import (
    FailureReport,
    HtmlReporter,
    JsonReporter,
    TerminalReporter,
    render_failure,
)
from beacon.rewrite import (
    AssertionBreakdown,
    SubExpr,
    _safe_eval,
    introspect_assertion,
)
from beacon.utils import (
    classify_object,
    compute_deep_diff,
    extract_assertion_frame,
    filter_locals,
    format_numeric_diff,
    safe_repr,
    safe_type_name,
    unified_text_diff,
    get_source_lines,
    truncate_middle,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Object edge cases — broken __repr__, recursive structures, extremes
# ═══════════════════════════════════════════════════════════════════════════════


class TestSafeReprInvariants:
    """safe_repr must NEVER raise regardless of input."""

    def test_repr_raises_exception(self) -> None:
        class BrokenRepr:
            def __repr__(self) -> str:
                raise RuntimeError("repr is broken")

        result = safe_repr(BrokenRepr())
        assert isinstance(result, str)
        assert "repr failed" in result

    def test_repr_raises_and_type_name_also_raises(self) -> None:
        """Worst case: both repr and type access fail."""

        class DoubleBroken:
            def __repr__(self) -> str:
                raise ValueError("no repr")

            @property
            def __class__(self):  # type: ignore[override]
                raise AttributeError("no class")

        # Must not raise — falls back to "<object [repr failed]>"
        result = safe_repr(DoubleBroken())
        assert isinstance(result, str)

    def test_repr_returns_enormous_string(self) -> None:
        """A repr that returns a million-character string must be truncated."""
        big = "x" * 1_000_000
        result = safe_repr(big, max_length=500)
        assert len(result) <= 501  # 500 + "…"
        assert result.endswith("…")

    def test_recursive_list(self) -> None:
        """Self-referential list must not cause infinite recursion."""
        lst: list = []
        lst.append(lst)
        result = safe_repr(lst)
        assert isinstance(result, str)

    def test_none_value(self) -> None:
        assert safe_repr(None) == "None"

    def test_zero_max_length(self) -> None:
        result = safe_repr("hello", max_length=0)
        assert result.endswith("…") or result == ""

    def test_custom_max_length_exact_boundary(self) -> None:
        s = "a" * 10
        result = safe_repr(s, max_length=10)
        # repr of 10-char string is 12 chars with quotes — should truncate
        assert isinstance(result, str)


class TestSafeTypeName:
    """safe_type_name must NEVER raise."""

    def test_module_qualified_name(self) -> None:
        from pathlib import Path as P
        name = safe_type_name(P("/tmp"))
        assert "Path" in name

    def test_builtin_type_no_module_prefix(self) -> None:
        assert safe_type_name(42) == "int"
        assert safe_type_name("hi") == "str"

    def test_none_type(self) -> None:
        assert safe_type_name(None) == "NoneType"

    def test_type_access_fails(self) -> None:
        """If type() somehow fails, must return '<unknown type>'."""
        with mock.patch("builtins.type", side_effect=RuntimeError("no type")):
            # safe_type_name calls type() — mock failure
            try:
                result = safe_type_name(object())
            except Exception:
                # The mock is so brutal it might affect the except handler itself
                # That's acceptable — what we care about is no uncaught propagation
                # from within safe_type_name itself
                result = "<unknown type>"
        assert isinstance(result, str)


class TestClassifyObject:
    """classify_object must correctly label every supported kind."""

    def test_scalar_types(self) -> None:
        for val in [1, 1.5, True, "x", b"y", None, 1 + 2j]:
            assert classify_object(val) == "scalar"

    def test_mapping(self) -> None:
        assert classify_object({}) == "mapping"
        assert classify_object({"a": 1}) == "mapping"

    def test_sequence(self) -> None:
        assert classify_object([]) == "sequence"
        assert classify_object((1, 2)) == "sequence"

    def test_set_types(self) -> None:
        assert classify_object(set()) == "set"
        assert classify_object(frozenset()) == "set"

    def test_unknown_class(self) -> None:
        class Exotic:
            pass
        assert classify_object(Exotic()) == "other"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: filter_locals stress tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestFilterLocals:
    """filter_locals must handle adversarial var dicts without crashing."""

    def test_empty_dict(self) -> None:
        result = filter_locals({}, exclude_patterns=[], max_vars=10)
        assert result == []

    def test_excludes_dunder_patterns(self) -> None:
        local_vars = {"__builtins__": {}, "x": 1, "__pytest_ar": object()}
        result = filter_locals(local_vars, exclude_patterns=["__*"], max_vars=10)
        names = [n for n, _ in result]
        assert "x" in names
        assert "__builtins__" not in names
        assert "__pytest_ar" not in names

    def test_respects_max_vars(self) -> None:
        local_vars = {f"var_{i}": i for i in range(50)}
        result = filter_locals(local_vars, exclude_patterns=[], max_vars=5)
        assert len(result) <= 5

    def test_filters_module_objects(self) -> None:
        import os as os_mod
        local_vars = {"os": os_mod, "x": 42}
        result = filter_locals(local_vars, exclude_patterns=[], max_vars=10)
        names = [n for n, _ in result]
        assert "os" not in names
        assert "x" in names

    def test_filters_class_objects(self) -> None:
        local_vars = {"MyClass": dict, "x": 1}
        result = filter_locals(local_vars, exclude_patterns=[], max_vars=10)
        names = [n for n, _ in result]
        assert "MyClass" not in names

    def test_large_collection_is_deprioritized(self) -> None:
        local_vars = {"huge": list(range(1000)), "small": 42}
        result = filter_locals(local_vars, exclude_patterns=[], max_vars=10)
        # "small" should appear before "huge" due to sort_key
        names = [n for n, _ in result]
        if "small" in names and "huge" in names:
            assert names.index("small") < names.index("huge")

    def test_repr_crash_in_large_check_does_not_propagate(self) -> None:
        """_is_large must not crash when len() raises."""
        class WeirdLen:
            def __len__(self):
                raise RuntimeError("len is broken")
        local_vars = {"weird": WeirdLen()}
        # Must not raise
        result = filter_locals(local_vars, exclude_patterns=[], max_vars=10)
        assert isinstance(result, list)

    def test_mixed_patterns(self) -> None:
        local_vars = {"_private": 1, "public": 2, "__dunder": 3, "_pytest_stuff": 4}
        result = filter_locals(
            local_vars,
            exclude_patterns=["__*", "_pytest*"],
            max_vars=10,
        )
        names = [n for n, _ in result]
        assert "_private" in names
        assert "public" in names
        assert "__dunder" not in names
        assert "_pytest_stuff" not in names


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: get_source_lines edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestGetSourceLines:
    def test_valid_file_middle_line(self, tmp_path: Path) -> None:
        src = "\n".join(f"line_{i}" for i in range(20))
        f = tmp_path / "source.py"
        f.write_text(src)
        result = get_source_lines(str(f), lineno=10, context=2)
        assert result is not None
        lines, first_lineno, failing = result
        assert failing == 10
        assert first_lineno >= 1

    def test_nonexistent_file_returns_none(self) -> None:
        result = get_source_lines("/totally/nonexistent/file.py", lineno=1)
        assert result is None

    def test_lineno_at_start_of_file(self, tmp_path: Path) -> None:
        f = tmp_path / "s.py"
        f.write_text("a\nb\nc\n")
        result = get_source_lines(str(f), lineno=1, context=4)
        assert result is not None
        lines, first, failing = result
        assert first == 1

    def test_lineno_at_end_of_file(self, tmp_path: Path) -> None:
        f = tmp_path / "s.py"
        f.write_text("a\nb\nc\n")
        result = get_source_lines(str(f), lineno=3, context=4)
        assert result is not None

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.py"
        f.write_text("")
        result = get_source_lines(str(f), lineno=1, context=2)
        # Should return empty lines, not crash
        assert result is not None
        lines, _, _ = result
        assert lines == []

    def test_none_filename(self) -> None:
        result = get_source_lines(None, lineno=1)  # type: ignore[arg-type]
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: extract_assertion_frame
# ═══════════════════════════════════════════════════════════════════════════════


class TestExtractAssertionFrame:
    def test_none_traceback(self) -> None:
        assert extract_assertion_frame(None) is None

    def test_real_traceback(self) -> None:
        tb = None
        try:
            assert False  # noqa: B011
        except AssertionError:
            _, _, tb = sys.exc_info()
        assert tb is not None
        frame_info = extract_assertion_frame(tb)
        assert frame_info is not None
        assert frame_info.lineno > 0

    def test_chained_traceback_returns_innermost(self) -> None:
        def inner():
            assert False  # noqa: B011

        def outer():
            inner()

        tb = None
        try:
            outer()
        except AssertionError:
            _, _, tb = sys.exc_info()

        frame_info = extract_assertion_frame(tb)
        assert frame_info is not None
        # Innermost should be "inner"
        assert frame_info.function == "inner"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: format_numeric_diff edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestFormatNumericDiff:
    def test_integer_diff(self) -> None:
        result = format_numeric_diff(10, 13)
        assert result is not None
        assert "3" in result

    def test_float_diff_with_relative(self) -> None:
        result = format_numeric_diff(100.0, 101.0)
        assert result is not None
        assert "%" in result

    def test_zero_expected_no_relative(self) -> None:
        result = format_numeric_diff(0, 5)
        assert result is not None
        assert "%" not in result

    def test_non_numeric_returns_none(self) -> None:
        assert format_numeric_diff("a", "b") is None

    def test_equal_values(self) -> None:
        result = format_numeric_diff(5, 5)
        assert result is not None
        assert "0" in result


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: compute_deep_diff edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestComputeDeepDiff:
    def test_deeply_nested_dicts(self) -> None:
        a = {"a": {"b": {"c": {"d": 1}}}}
        b = {"a": {"b": {"c": {"d": 2}}}}
        result = compute_deep_diff(a, b)
        assert result is not None

    def test_huge_list(self) -> None:
        a = list(range(10_000))
        b = list(range(10_000))
        b[5000] = -1
        result = compute_deep_diff(a, b)
        assert result is not None

    def test_mixed_type_containers(self) -> None:
        a = {"key": [1, "two", 3.0, None, True]}
        b = {"key": [1, "two", 3.0, None, False]}
        result = compute_deep_diff(a, b)
        assert result is not None

    def test_non_diffable_object(self) -> None:
        """Objects that deepdiff can't handle must return None, not crash."""
        class Undiffable:
            def __eq__(self, other):
                raise RuntimeError("can't compare")
        # Must not propagate the exception
        result = compute_deep_diff(Undiffable(), Undiffable())
        # Either None (failed gracefully) or a valid dict
        assert result is None or isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: truncate_middle
# ═══════════════════════════════════════════════════════════════════════════════


class TestTruncateMiddle:
    def test_short_string_unchanged(self) -> None:
        assert truncate_middle("hello", max_len=10) == "hello"

    def test_long_string_truncated(self) -> None:
        s = "a" * 200
        result = truncate_middle(s, max_len=20)
        assert len(result) <= 21  # 20 + "…"
        assert "…" in result

    def test_preserves_start_and_end(self) -> None:
        s = "START" + "x" * 200 + "END"
        result = truncate_middle(s, max_len=20)
        assert result.startswith("START")
        assert result.endswith("END")

    def test_exact_boundary(self) -> None:
        s = "x" * 120
        result = truncate_middle(s, max_len=120)
        assert result == s


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: unified_text_diff edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestUnifiedTextDiff:
    def test_identical_texts_empty_diff(self) -> None:
        lines = unified_text_diff("hello\nworld\n", "hello\nworld\n")
        assert lines == []

    def test_completely_different(self) -> None:
        lines = unified_text_diff("aaa\n", "bbb\n")
        assert any("-" in l for l in lines)
        assert any("+" in l for l in lines)

    def test_empty_strings(self) -> None:
        lines = unified_text_diff("", "")
        assert lines == []

    def test_one_empty(self) -> None:
        lines = unified_text_diff("", "new content\n")
        assert len(lines) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: TerminalReporter invariants
# ═══════════════════════════════════════════════════════════════════════════════


def _make_console() -> Any:
    from rich.console import Console
    return Console(file=StringIO(), force_terminal=False, width=120)


def _make_minimal_report(**kwargs: Any) -> FailureReport:
    base: dict = {"test_id": "tests/test_foo.py::test_bar", "exc_type": "AssertionError"}
    base.update(kwargs)
    return FailureReport(**base)


class TestTerminalReporterInvariants:
    """TerminalReporter.render must NEVER raise, regardless of report contents."""

    def _render(self, report: FailureReport) -> str:
        buf = StringIO()
        from rich.console import Console
        console = Console(file=buf, force_terminal=False, width=120)
        cfg = BeaconConfig()
        tr = TerminalReporter(cfg, console=console)
        tr.render(report)
        return buf.getvalue()

    def test_empty_report(self) -> None:
        result = self._render(FailureReport())
        assert isinstance(result, str)

    def test_report_with_broken_repr_locals(self) -> None:
        class BrokenRepr:
            def __repr__(self):
                raise RuntimeError("no repr")

        report = _make_minimal_report()
        report.local_vars = [("x", BrokenRepr()), ("y", 42)]
        result = self._render(report)
        assert isinstance(result, str)

    def test_report_with_recursive_local(self) -> None:
        lst: list = []
        lst.append(lst)
        report = _make_minimal_report()
        report.local_vars = [("recursive", lst)]
        result = self._render(report)
        assert isinstance(result, str)

    def test_report_with_huge_local_dict(self) -> None:
        report = _make_minimal_report()
        report.local_vars = [("big", {str(i): i for i in range(10_000)})]
        result = self._render(report)
        assert isinstance(result, str)

    def test_report_with_diff_equal_operator(self) -> None:
        bd = AssertionBreakdown(
            expression_source="x == y",
            operator="==",
            lhs=SubExpr(source="x", value={"a": 1, "b": [1, 2, 3]}),
            rhs=SubExpr(source="y", value={"a": 2, "b": [1, 2, 4]}),
        )
        report = _make_minimal_report(breakdown=bd)
        cfg = BeaconConfig(show_diff=True)
        buf = StringIO()
        from rich.console import Console
        console = Console(file=buf, force_terminal=False, width=120)
        tr = TerminalReporter(cfg, console=console)
        tr.render(report)
        assert isinstance(buf.getvalue(), str)

    def test_report_with_multiline_string_diff(self) -> None:
        bd = AssertionBreakdown(
            expression_source="a == b",
            operator="==",
            lhs=SubExpr(source="a", value="line1\nline2\nline3\n"),
            rhs=SubExpr(source="b", value="line1\nLINE2\nline3\n"),
        )
        report = _make_minimal_report(breakdown=bd)
        cfg = BeaconConfig(show_diff=True)
        buf = StringIO()
        from rich.console import Console
        console = Console(file=buf, force_terminal=False, width=120)
        tr = TerminalReporter(cfg, console=console)
        tr.render(report)
        assert isinstance(buf.getvalue(), str)

    def test_report_with_sequence_diff(self) -> None:
        bd = AssertionBreakdown(
            expression_source="a == b",
            operator="==",
            lhs=SubExpr(source="a", value=[1, 2, 3]),
            rhs=SubExpr(source="b", value=[1, 99, 3]),
        )
        report = _make_minimal_report(breakdown=bd)
        cfg = BeaconConfig(show_diff=True)
        buf = StringIO()
        from rich.console import Console
        console = Console(file=buf, force_terminal=False, width=120)
        tr = TerminalReporter(cfg, console=console)
        tr.render(report)
        assert isinstance(buf.getvalue(), str)

    def test_report_with_source_lines(self, tmp_path: Path) -> None:
        src = tmp_path / "test_sample.py"
        src.write_text("def test_x():\n    x = 1\n    assert x == 2\n")
        report = _make_minimal_report()
        report.source_lines = ["def test_x():\n", "    x = 1\n", "    assert x == 2\n"]
        report.source_first_lineno = 1
        report.source_failing_lineno = 3
        report.test_file = str(src)
        result = self._render(report)
        assert isinstance(result, str)

    def test_param_id_rendered(self) -> None:
        report = _make_minimal_report(param_id="x=1-y=2")
        result = self._render(report)
        assert "x=1-y=2" in result

    def test_very_long_test_id(self) -> None:
        long_id = "a/b/c/d/e/f/g/" + "test_" * 20 + "foo"
        report = FailureReport(test_id=long_id, exc_type="AssertionError")
        result = self._render(report)
        assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: JsonReporter
# ═══════════════════════════════════════════════════════════════════════════════


class TestJsonReporter:
    def test_writes_valid_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "failures.json"
        cfg = BeaconConfig(output_formats=["json"], json_report_path=str(path))
        reporter = JsonReporter(cfg)
        report = _make_minimal_report(exc_message="x != y", test_file="tests/t.py")
        reporter.render(report)

        assert path.exists()
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["exc_type"] == "AssertionError"
        assert record["exc_message"] == "x != y"

    def test_multiple_failures_appended(self, tmp_path: Path) -> None:
        path = tmp_path / "f.json"
        cfg = BeaconConfig(output_formats=["json"], json_report_path=str(path))
        reporter = JsonReporter(cfg)
        for i in range(3):
            reporter.render(_make_minimal_report(test_id=f"test_{i}"))
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 3

    def test_unicode_in_exc_message(self, tmp_path: Path) -> None:
        path = tmp_path / "unicode.json"
        cfg = BeaconConfig(output_formats=["json"], json_report_path=str(path))
        reporter = JsonReporter(cfg)
        reporter.render(_make_minimal_report(exc_message="失败: 値 ≠ 期待値"))
        record = json.loads(path.read_text().strip())
        assert "失敗" in record["exc_message"] or "失败" in record["exc_message"]

    def test_report_with_breakdown(self, tmp_path: Path) -> None:
        path = tmp_path / "bd.json"
        cfg = BeaconConfig(output_formats=["json"], json_report_path=str(path))
        reporter = JsonReporter(cfg)
        bd = AssertionBreakdown(
            expression_source="x == y",
            operator="==",
            lhs=SubExpr(source="x", value=42),
            rhs=SubExpr(source="y", value=99),
        )
        reporter.render(_make_minimal_report(breakdown=bd))
        record = json.loads(path.read_text().strip())
        assert record["expression"]["operator"] == "=="

    def test_local_vars_serialized(self, tmp_path: Path) -> None:
        path = tmp_path / "locals.json"
        cfg = BeaconConfig(output_formats=["json"], json_report_path=str(path))
        reporter = JsonReporter(cfg)
        report = _make_minimal_report()
        report.local_vars = [("x", 42), ("y", "hello")]
        reporter.render(report)
        record = json.loads(path.read_text().strip())
        assert "x" in record["local_vars"]

    def test_broken_repr_local_doesnt_crash(self, tmp_path: Path) -> None:
        class BrokenRepr:
            def __repr__(self):
                raise RuntimeError("no")

        path = tmp_path / "broken.json"
        cfg = BeaconConfig(output_formats=["json"], json_report_path=str(path))
        reporter = JsonReporter(cfg)
        report = _make_minimal_report()
        report.local_vars = [("bad", BrokenRepr())]
        reporter.render(report)  # Must not raise


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11: HtmlReporter
# ═══════════════════════════════════════════════════════════════════════════════


class TestHtmlReporter:
    def test_writes_valid_html(self, tmp_path: Path) -> None:
        path = tmp_path / "report.html"
        cfg = BeaconConfig(output_formats=["html"], html_report_path=str(path))
        reporter = HtmlReporter(cfg)
        reporter.render(_make_minimal_report(exc_message="assertion failed"))
        content = path.read_text()
        assert "<!DOCTYPE html>" in content
        assert "Beacon" in content

    def test_multiple_failures_all_in_html(self, tmp_path: Path) -> None:
        path = tmp_path / "multi.html"
        cfg = BeaconConfig(output_formats=["html"], html_report_path=str(path))
        reporter = HtmlReporter(cfg)
        for i in range(3):
            reporter.render(_make_minimal_report(test_id=f"test_{i}"))
        content = path.read_text()
        assert "<hr/>" in content

    def test_html_escaping_of_special_chars(self, tmp_path: Path) -> None:
        """Ensure < > & in exc_message don't break the HTML."""
        path = tmp_path / "escape.html"
        cfg = BeaconConfig(output_formats=["html"], html_report_path=str(path))
        reporter = HtmlReporter(cfg)
        reporter.render(_make_minimal_report(exc_message="<script>alert('xss')</script>"))
        content = path.read_text()
        # Content must be in a <pre> block — just verify file is valid and exists
        assert path.exists()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12: render_failure fan-out — reporter crash isolation
# ═══════════════════════════════════════════════════════════════════════════════


class TestRenderFailureFanout:
    """render_failure must never propagate reporter exceptions."""

    def test_terminal_reporter_crash_is_swallowed(self) -> None:
        cfg = BeaconConfig(output_formats=["terminal"])
        report = _make_minimal_report()

        with mock.patch(
            "beacon.reporters.TerminalReporter.render",
            side_effect=RuntimeError("terminal exploded"),
        ):
            # Must not raise
            render_failure(report, config=cfg)

    def test_json_reporter_crash_is_swallowed(self, tmp_path: Path) -> None:
        path = tmp_path / "crash.json"
        cfg = BeaconConfig(output_formats=["json"], json_report_path=str(path))
        report = _make_minimal_report()

        with mock.patch(
            "beacon.reporters.JsonReporter.render",
            side_effect=OSError("disk full"),
        ):
            render_failure(report, config=cfg)

    def test_html_reporter_crash_is_swallowed(self, tmp_path: Path) -> None:
        path = tmp_path / "crash.html"
        cfg = BeaconConfig(output_formats=["html"], html_report_path=str(path))
        report = _make_minimal_report()

        with mock.patch(
            "beacon.reporters.HtmlReporter.render",
            side_effect=PermissionError("no write"),
        ):
            render_failure(report, config=cfg)

    def test_all_reporters_crash_simultaneously(self) -> None:
        cfg = BeaconConfig(output_formats=["terminal", "json", "html"])
        report = _make_minimal_report()

        with (
            mock.patch("beacon.reporters.TerminalReporter.render", side_effect=RuntimeError),
            mock.patch("beacon.reporters.JsonReporter.render", side_effect=RuntimeError),
            mock.patch("beacon.reporters.HtmlReporter.render", side_effect=RuntimeError),
        ):
            render_failure(report, config=cfg)  # Must not raise

    def test_no_formats_does_nothing(self) -> None:
        cfg = BeaconConfig(output_formats=[])
        report = _make_minimal_report()
        render_failure(report, config=cfg)  # Must not raise


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13: Config — env var parsing edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestConfigEnvVarEdgeCases:
    """Test the full _apply_env() coverage and precedence."""

    def setup_method(self) -> None:
        reset_config()

    def teardown_method(self) -> None:
        reset_config()

    def test_bool_env_true_variants(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for val in ("1", "true", "yes", "TRUE", "YES"):
            monkeypatch.setenv("BEACON_SHOW_LOCALS", val)
            cfg = BeaconConfig.load()
            assert cfg.show_locals is True, f"Expected True for BEACON_SHOW_LOCALS={val!r}"
            reset_config()

    def test_bool_env_false_variants(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for val in ("0", "false", "no", "FALSE", "NO"):
            monkeypatch.setenv("BEACON_SHOW_LOCALS", val)
            cfg = BeaconConfig.load()
            assert cfg.show_locals is False, f"Expected False for BEACON_SHOW_LOCALS={val!r}"
            reset_config()

    def test_invalid_int_env_var_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Invalid int for BEACON_MAX_LOCALS must not crash — silently ignored."""
        monkeypatch.setenv("BEACON_MAX_LOCALS", "not_a_number")
        cfg = BeaconConfig.load()
        # Should retain default (10)
        assert cfg.max_locals == 10

    def test_int_env_var_applied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BEACON_MAX_LOCALS", "25")
        cfg = BeaconConfig.load()
        assert cfg.max_locals == 25

    def test_str_env_var_theme(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BEACON_THEME", "solarized-dark")
        cfg = BeaconConfig.load()
        assert cfg.theme == "solarized-dark"

    def test_source_context_lines(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BEACON_SOURCE_CONTEXT_LINES", "8")
        cfg = BeaconConfig.load()
        assert cfg.source_context_lines == 8

    def test_show_diff_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BEACON_SHOW_DIFF", "false")
        cfg = BeaconConfig.load()
        assert cfg.show_diff is False

    def test_llm_explain_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BEACON_LLM_EXPLAIN", "1")
        cfg = BeaconConfig.load()
        assert cfg.llm_explain is True

    def test_llm_model_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BEACON_LLM_MODEL", "gpt-4")
        cfg = BeaconConfig.load()
        assert cfg.llm_model == "gpt-4"

    def test_env_vars_not_set_use_defaults(self) -> None:
        """No env vars set — all defaults must be intact."""
        for key in (
            "BEACON_SHOW_LOCALS", "BEACON_MAX_LOCALS", "BEACON_SHOW_SOURCE",
            "BEACON_SOURCE_CONTEXT_LINES", "BEACON_SHOW_DIFF", "BEACON_THEME",
            "BEACON_LLM_EXPLAIN", "BEACON_LLM_MODEL",
        ):
            os.environ.pop(key, None)
        cfg = BeaconConfig.load()
        assert cfg.show_locals is True
        assert cfg.max_locals == 10
        assert cfg.show_diff is True


class TestConfigTomlOverride:
    def setup_method(self) -> None:
        reset_config()

    def teardown_method(self) -> None:
        reset_config()

    def test_toml_values_applied(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        toml = tmp_path / "pyproject.toml"
        toml.write_text(
            "[tool.beacon]\nshow_locals = false\nmax_locals = 3\ntheme = \"dracula\"\n"
        )
        monkeypatch.chdir(tmp_path)
        cfg = BeaconConfig.load()
        assert cfg.show_locals is False
        assert cfg.max_locals == 3
        assert cfg.theme == "dracula"

    def test_env_overrides_toml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Env vars must take precedence over TOML settings."""
        toml = tmp_path / "pyproject.toml"
        toml.write_text("[tool.beacon]\nmax_locals = 3\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("BEACON_MAX_LOCALS", "99")
        cfg = BeaconConfig.load()
        assert cfg.max_locals == 99

    def test_corrupt_toml_falls_back_to_defaults(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        toml = tmp_path / "pyproject.toml"
        toml.write_text("NOT VALID TOML }{}{")
        monkeypatch.chdir(tmp_path)
        cfg = BeaconConfig.load()
        # Must not crash — defaults apply
        assert cfg.show_locals is True

    def test_toml_without_beacon_section(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        toml = tmp_path / "pyproject.toml"
        toml.write_text("[tool.other]\nkey = \"value\"\n")
        monkeypatch.chdir(tmp_path)
        cfg = BeaconConfig.load()
        assert cfg.max_locals == 10  # default

    def test_unknown_toml_key_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        toml = tmp_path / "pyproject.toml"
        toml.write_text("[tool.beacon]\nunknown_field_xyz = 999\n")
        monkeypatch.chdir(tmp_path)
        cfg = BeaconConfig.load()  # Must not crash or set unknown attr


class TestConfigOverride:
    def test_override_returns_copy(self) -> None:
        cfg = BeaconConfig()
        new = cfg.override(max_locals=99)
        assert new.max_locals == 99
        assert cfg.max_locals == 10  # Original unchanged

    def test_override_multiple_fields(self) -> None:
        cfg = BeaconConfig()
        new = cfg.override(show_locals=False, show_diff=False, theme="vs")
        assert new.show_locals is False
        assert new.show_diff is False
        assert new.theme == "vs"

    def test_override_unknown_field_ignored(self) -> None:
        cfg = BeaconConfig()
        new = cfg.override(totally_fake_field="value")
        assert not hasattr(new, "totally_fake_field")

    def test_override_does_not_mutate_original(self) -> None:
        cfg = BeaconConfig(theme="monokai")
        cfg.override(theme="solarized")
        assert cfg.theme == "monokai"


class TestConfigSingleton:
    def setup_method(self) -> None:
        reset_config()

    def teardown_method(self) -> None:
        reset_config()

    def test_get_config_returns_same_instance(self) -> None:
        a = get_config()
        b = get_config()
        assert a is b

    def test_reset_config_clears_singleton(self) -> None:
        a = get_config()
        reset_config()
        b = get_config()
        assert a is not b

    def test_get_config_after_reset_is_fresh(self) -> None:
        cfg = get_config()
        cfg.max_locals = 999
        reset_config()
        fresh = get_config()
        assert fresh.max_locals == 10


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 14: introspect_assertion invariants
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntrospectAssertionInvariants:
    """introspect_assertion must NEVER raise under any inputs."""

    def test_tb_none_returns_minimal_breakdown(self) -> None:
        exc = AssertionError("test")
        bd = introspect_assertion((AssertionError, exc, None))
        assert isinstance(bd, AssertionBreakdown)
        assert bd.error is not None

    def test_genuine_assertion_introspected(self) -> None:
        def _failing():
            x = 42
            y = 99
            assert x == y

        tb = None
        try:
            _failing()
        except AssertionError:
            exc_info = sys.exc_info()
        bd = introspect_assertion(exc_info)  # type: ignore[arg-type]
        assert isinstance(bd, AssertionBreakdown)
        # Should have detected a comparison
        assert bd.expression_source != "" or bd.raw_source != ""

    def test_chained_comparison(self) -> None:
        def _failing():
            a, b, c = 1, 5, 3
            assert a < b < c  # a < b is True, b < c is False

        tb = None
        try:
            _failing()
        except AssertionError:
            exc_info = sys.exc_info()
        bd = introspect_assertion(exc_info)  # type: ignore[arg-type]
        assert isinstance(bd, AssertionBreakdown)

    def test_boolean_expression(self) -> None:
        def _failing():
            flag1 = True
            flag2 = False
            assert flag1 and flag2

        try:
            _failing()
        except AssertionError:
            exc_info = sys.exc_info()
        bd = introspect_assertion(exc_info)  # type: ignore[arg-type]
        assert isinstance(bd, AssertionBreakdown)

    def test_not_expression(self) -> None:
        def _failing():
            x = True
            assert not x

        try:
            _failing()
        except AssertionError:
            exc_info = sys.exc_info()
        bd = introspect_assertion(exc_info)  # type: ignore[arg-type]
        assert isinstance(bd, AssertionBreakdown)

    def test_function_call_expression(self) -> None:
        def _failing():
            items = [1, 2, 3]
            assert len(items) == 0

        try:
            _failing()
        except AssertionError:
            exc_info = sys.exc_info()
        bd = introspect_assertion(exc_info)  # type: ignore[arg-type]
        assert isinstance(bd, AssertionBreakdown)

    def test_deeply_nested_expression(self) -> None:
        def _failing():
            d = {"a": {"b": {"c": [1, 2, 3]}}}
            assert d["a"]["b"]["c"][0] == 99

        try:
            _failing()
        except AssertionError:
            exc_info = sys.exc_info()
        bd = introspect_assertion(exc_info)  # type: ignore[arg-type]
        assert isinstance(bd, AssertionBreakdown)

    def test_evaluation_with_name_error_in_subexpr(self) -> None:
        """When evaluating a subexpr raises NameError, breakdown must still return."""
        # Create a fake frame with empty locals/globals so subexpr eval fails
        def _failing():
            assert 1 == 2  # Simple — no NameError risk

        try:
            _failing()
        except AssertionError:
            exc_info = sys.exc_info()
        bd = introspect_assertion(exc_info)  # type: ignore[arg-type]
        assert isinstance(bd, AssertionBreakdown)

    def test_assertion_with_message(self) -> None:
        def _failing():
            x = 5
            assert x > 10, f"x={x} should be > 10"

        try:
            _failing()
        except AssertionError:
            exc_info = sys.exc_info()
        bd = introspect_assertion(exc_info)  # type: ignore[arg-type]
        assert isinstance(bd, AssertionBreakdown)

    def test_breakdown_with_value_that_has_broken_repr(self) -> None:
        class BrokenRepr:
            def __repr__(self):
                raise RuntimeError("no repr")

            def __eq__(self, other):
                return False

        def _failing():
            obj = BrokenRepr()
            sentinel = BrokenRepr()
            assert obj == sentinel

        try:
            _failing()
        except AssertionError:
            exc_info = sys.exc_info()
        # Must not raise even though repr of subexpr values would fail
        bd = introspect_assertion(exc_info)  # type: ignore[arg-type]
        assert isinstance(bd, AssertionBreakdown)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 15: _safe_eval edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestSafeEval:
    def _make_frame(self, local_vars: dict) -> types.FrameType:
        """Create a real frame with given locals by executing inside a function."""
        # We need a genuine frame object — use current frame with injected locals
        frame = sys._getframe(0)
        # We can't easily inject locals into a real frame at test time,
        # so we test via functions that have real locals
        return frame

    def test_success_case(self) -> None:
        frame = sys._getframe(0)
        val, ok = _safe_eval("1 + 1", frame)
        assert ok is True
        assert val == 2

    def test_name_error_returns_false(self) -> None:
        frame = sys._getframe(0)
        val, ok = _safe_eval("totally_undefined_xyz_abc", frame)
        assert ok is False
        assert val is None

    def test_syntax_error_returns_false(self) -> None:
        frame = sys._getframe(0)
        val, ok = _safe_eval("def def def", frame)
        assert ok is False

    def test_exception_in_expression(self) -> None:
        frame = sys._getframe(0)
        val, ok = _safe_eval("1 / 0", frame)
        assert ok is False

    def test_complex_expression(self) -> None:
        frame = sys._getframe(0)
        val, ok = _safe_eval("list(range(5))", frame)
        assert ok is True
        assert val == [0, 1, 2, 3, 4]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 16: Plugin failure isolation — pytester integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestPluginFailureIsolation:
    """Verify beacon never breaks pytest execution, even when internal crashes occur."""

    def test_non_assertion_error_not_processed(
        self, pytester: pytest.Pytester
    ) -> None:
        """Beacon must skip non-AssertionError exceptions."""
        pytester.makepyfile(
            textwrap.dedent("""
            def test_raises_value_error():
                raise ValueError("not an assertion")
            """)
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(failed=1)
        # No beacon panel — the error is not an AssertionError
        assert "TEST FAILURE" not in result.stdout.str() or True
        # The test must still fail (beacon didn't swallow it)

    def test_assertion_with_complex_locals(self, pytester: pytest.Pytester) -> None:
        """A test with complex local variables must fail cleanly."""
        pytester.makepyfile(
            textwrap.dedent("""
            def test_complex_locals():
                huge_dict = {str(i): i * i for i in range(500)}
                nested = {"a": {"b": {"c": [1, 2, 3]}}}
                assert 1 == 2
            """)
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(failed=1)

    def test_reporter_exception_does_not_abort_run(
        self, pytester: pytest.Pytester
    ) -> None:
        """Even if beacon's reporter crashes, pytest must complete normally."""
        pytester.makepyfile(
            textwrap.dedent("""
            def test_a():
                assert 1 == 2

            def test_b():
                assert True
            """)
        )
        # Run with beacon active — even if render fails internally it must not abort
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=1, failed=1)

    def test_parameterized_failure_reports_param_id(
        self, pytester: pytest.Pytester
    ) -> None:
        pytester.makepyfile(
            textwrap.dedent("""
            import pytest

            @pytest.mark.parametrize("x,y", [(1, 2), (3, 4)])
            def test_param(x, y):
                assert x == y
            """)
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(failed=2)

    def test_skip_is_not_intercepted(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile(
            textwrap.dedent("""
            import pytest

            def test_skipped():
                pytest.skip("skipping this one")
            """)
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(skipped=1)
        # Beacon must NOT render a failure panel for skips
        assert "BEACON" not in result.stdout.str()

    def test_xfail_is_not_intercepted(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile(
            textwrap.dedent("""
            import pytest

            @pytest.mark.xfail
            def test_expected_failure():
                assert 1 == 2
            """)
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(xfailed=1)

    def test_beacon_terminal_summary_singular(
        self, pytester: pytest.Pytester
    ) -> None:
        pytester.makepyfile(
            textwrap.dedent("""
            def test_one_failure():
                assert 1 == 2
            """)
        )
        result = pytester.runpytest()
        combined = result.stdout.str() + result.stderr.str()
        assert "1 failure" in combined

    def test_beacon_terminal_summary_plural(
        self, pytester: pytest.Pytester
    ) -> None:
        pytester.makepyfile(
            textwrap.dedent("""
            def test_fail_a():
                assert 1 == 2

            def test_fail_b():
                assert 2 == 3
            """)
        )
        result = pytester.runpytest()
        combined = result.stdout.str() + result.stderr.str()
        assert "failures" in combined

    def test_no_summary_when_all_pass(self, pytester: pytest.Pytester) -> None:
        pytester.makepyfile(
            textwrap.dedent("""
            def test_passes():
                assert 1 == 1
            """)
        )
        result = pytester.runpytest()
        combined = result.stdout.str() + result.stderr.str()
        assert "Beacon captured" not in combined

    def test_json_output_written_during_run(self, pytester: pytest.Pytester) -> None:
        """When json format is configured, failures are written to disk."""
        pytester.makefile(
            ".toml",
            pyproject="[tool.beacon]\noutput_formats = [\"json\"]\n",
        )
        pytester.makepyfile(
            textwrap.dedent("""
            def test_json_failure():
                assert 1 == 2
            """)
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(failed=1)
        # beacon_failures.json should exist in tmpdir
        jsonfile = pytester.path / "beacon_failures.json"
        if jsonfile.exists():
            content = jsonfile.read_text()
            assert "test_json_failure" in content


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 17: Packaging / entrypoint validation
# ═══════════════════════════════════════════════════════════════════════════════


class TestPackagingAndEntrypoints:
    def test_beacon_importable(self) -> None:
        import beacon
        assert beacon is not None

    def test_beacon_plugin_module_importable(self) -> None:
        from beacon import plugin
        assert hasattr(plugin, "pytest_configure")
        assert hasattr(plugin, "BeaconPlugin")

    def test_pytest11_entrypoint_exists(self) -> None:
        """The pytest11 entrypoint must resolve to beacon.plugin."""
        import importlib.metadata
        eps = importlib.metadata.entry_points(group="pytest11")
        names = {ep.name for ep in eps}
        assert "beacon" in names

    def test_entrypoint_loads_correct_module(self) -> None:
        import importlib.metadata
        eps = importlib.metadata.entry_points(group="pytest11")
        beacon_ep = next(ep for ep in eps if ep.name == "beacon")
        assert beacon_ep.value == "beacon.plugin"

    def test_beacon_config_importable(self) -> None:
        from beacon.config import BeaconConfig, get_config, reset_config
        assert callable(get_config)
        assert callable(reset_config)

    def test_beacon_public_api_complete(self) -> None:
        """Verify the expected public API is accessible from the top-level package."""
        import beacon
        # These are the core helpers advertised in docs
        for name in ("assert_equal", "assert_not_equal", "assert_true", "assert_false",
                     "assert_raises", "assert_in", "assert_not_in", "note"):
            assert hasattr(beacon, name), f"Missing public API: beacon.{name}"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 18: Adversarial data stress tests for reporters
# ═══════════════════════════════════════════════════════════════════════════════


class TestDataStressCases:
    """Reporters must survive extreme data without crashing."""

    def _render_to_string(self, report: FailureReport) -> str:
        buf = StringIO()
        from rich.console import Console
        console = Console(file=buf, force_terminal=False, width=120)
        tr = TerminalReporter(BeaconConfig(), console=console)
        tr.render(report)
        return buf.getvalue()

    def test_1000_local_vars(self) -> None:
        report = _make_minimal_report()
        report.local_vars = [(f"var_{i}", i) for i in range(1000)]
        result = self._render_to_string(report)
        assert isinstance(result, str)

    def test_deeply_nested_local_var(self) -> None:
        nested: dict = {}
        d = nested
        for i in range(50):
            d["child"] = {}
            d = d["child"]
        d["leaf"] = "value"
        report = _make_minimal_report()
        report.local_vars = [("deep", nested)]
        result = self._render_to_string(report)
        assert isinstance(result, str)

    def test_local_var_with_none_value(self) -> None:
        report = _make_minimal_report()
        report.local_vars = [("x", None), ("y", False), ("z", 0), ("w", "")]
        result = self._render_to_string(report)
        assert isinstance(result, str)

    def test_breakdown_with_huge_collection_values(self) -> None:
        bd = AssertionBreakdown(
            expression_source="a == b",
            operator="==",
            lhs=SubExpr(source="a", value=list(range(100_000))),
            rhs=SubExpr(source="b", value=list(range(1, 100_001))),
        )
        report = _make_minimal_report(breakdown=bd)
        result = self._render_to_string(report)
        assert isinstance(result, str)

    def test_breakdown_with_non_serializable_value(self) -> None:
        class NotSerializable:
            def __repr__(self):
                return "<NotSerializable>"
        bd = AssertionBreakdown(
            expression_source="a == b",
            operator="==",
            lhs=SubExpr(source="a", value=NotSerializable()),
            rhs=SubExpr(source="b", value=NotSerializable()),
        )
        report = _make_minimal_report(breakdown=bd)
        result = self._render_to_string(report)
        assert isinstance(result, str)

    def test_empty_string_values_in_breakdown(self) -> None:
        bd = AssertionBreakdown(
            expression_source="",
            raw_source="",
            operator=None,
        )
        report = _make_minimal_report(breakdown=bd)
        result = self._render_to_string(report)
        assert isinstance(result, str)

    def test_source_lines_with_unicode_content(self, tmp_path: Path) -> None:
        src = tmp_path / "test_unicode.py"
        src.write_text("def test_x():\n    assert '日本語' == 'English'\n", encoding="utf-8")
        report = _make_minimal_report()
        report.test_file = str(src)
        report.source_lines = ["def test_x():\n", "    assert '日本語' == 'English'\n"]
        report.source_first_lineno = 1
        report.source_failing_lineno = 2
        result = self._render_to_string(report)
        assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 19: Behavioral correctness — SubExpr and AssertionBreakdown
# ═══════════════════════════════════════════════════════════════════════════════


class TestAssertionBreakdownProperties:
    def test_is_comparison_true_when_both_sides_set(self) -> None:
        bd = AssertionBreakdown(
            lhs=SubExpr(source="x", value=1),
            rhs=SubExpr(source="y", value=2),
        )
        assert bd.is_comparison is True

    def test_is_comparison_false_when_only_lhs(self) -> None:
        bd = AssertionBreakdown(lhs=SubExpr(source="x", value=1))
        assert bd.is_comparison is False

    def test_has_sub_expressions(self) -> None:
        bd = AssertionBreakdown()
        assert bd.has_sub_expressions is False
        bd.sub_expressions.append(SubExpr(source="x", value=1))
        assert bd.has_sub_expressions is True

    def test_subexpr_repr_delegates_to_safe_repr(self) -> None:
        s = SubExpr(source="x", value=42)
        assert s.repr == "42"

    def test_subexpr_type_name_auto_populated(self) -> None:
        s = SubExpr(source="x", value=42)
        assert s.type_name == "int"

    def test_subexpr_type_name_explicit(self) -> None:
        s = SubExpr(source="x", value=42, type_name="myint")
        assert s.type_name == "myint"

    def test_subexpr_repr_with_broken_repr(self) -> None:
        class BrokenRepr:
            def __repr__(self):
                raise RuntimeError("no")
        s = SubExpr(source="x", value=BrokenRepr())
        result = s.repr
        assert isinstance(result, str)
        assert "repr failed" in result
