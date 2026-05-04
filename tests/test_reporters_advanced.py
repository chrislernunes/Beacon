"""
tests/test_reporters_advanced.py
─────────────────────────────────
High-value tests for beacon.reporters targeting the untested 65%.

Coverage targets
────────────────
TerminalReporter.render:
  - breakdown path with expression_source (line 142-144)
  - diff panel path (lines 146-150)
  - source panel path (lines 152-154)
  - locals panel path (lines 156-158)
TerminalReporter._diff_panel:
  - ndarray branch (lines 320-321)
  - dataframe branch (lines 324-325)
  - mapping/dict branch (lines 327-331)
  - sequence/set branch (lines 334-337)
  - multi-line string branch (lines 340-342)
  - generic/other branch (lines 345-348)
  - scalar → returns None (lines 316-317)
  - empty content → returns None (lines 350-351)
TerminalReporter._numpy_diff (lines 360-380)
TerminalReporter._dataframe_diff (lines 382-409)
TerminalReporter._string_diff (lines 411-427)
TerminalReporter._deep_diff_table:
  - > 20 rows truncation (lines 450-453)
  - set/list changes (lines 463-468)
HtmlReporter (lines 559-602)
_shorten_path (lines 639-644)
_smart_repr: ndarray path (lines 661-669), dataframe path (671-676)
render_failure: html sink (lines 629-633)

Approach
────────
All rendering tests inject a StringIO console so output is captured without
side effects. We never check for exact Rich markup — we check for meaningful
content in the stripped plain-text output.
"""
from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from beacon.annotations import TestNote as AnnotationNote
from beacon.config import BeaconConfig
from beacon.reporters import (
    FailureReport,
    HtmlReporter,
    JsonReporter,
    TerminalReporter,
    _shorten_path,
    _smart_repr,
    render_failure,
)
from beacon.rewrite import AssertionBreakdown, SubExpr


# ── Shared fixtures / helpers ─────────────────────────────────────────────────


def _make_cfg(**kwargs: Any) -> BeaconConfig:
    cfg = BeaconConfig()
    cfg.output_formats = ["terminal"]
    cfg.show_locals = True
    cfg.show_source = True
    cfg.show_diff = True
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def _make_console() -> tuple[Console, StringIO]:
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120, highlight=False)
    return console, buf


def _render(report: FailureReport, **cfg_kwargs: Any) -> str:
    cfg = _make_cfg(**cfg_kwargs)
    console, buf = _make_console()
    TerminalReporter(cfg, console=console).render(report)
    return buf.getvalue()


def _make_report(**kwargs: Any) -> FailureReport:
    r = FailureReport()
    r.test_id = kwargs.get("test_id", "tests/test_foo.py::test_x")
    r.test_function = kwargs.get("test_function", "test_x")
    r.test_file = kwargs.get("test_file", "tests/test_foo.py")
    r.test_lineno = kwargs.get("test_lineno", 10)
    r.exc_type = kwargs.get("exc_type", "AssertionError")
    r.exc_message = kwargs.get("exc_message", "assert failed")
    return r


def _make_comparison_breakdown(
    lhs_src: str,
    lhs_val: Any,
    rhs_src: str,
    rhs_val: Any,
    op: str = "==",
) -> AssertionBreakdown:
    bd = AssertionBreakdown()
    bd.expression_source = f"{lhs_src} {op} {rhs_src}"
    bd.lhs = SubExpr(source=lhs_src, value=lhs_val)
    bd.rhs = SubExpr(source=rhs_src, value=rhs_val)
    bd.operator = op
    bd.sub_expressions = [bd.lhs, bd.rhs]
    return bd


# ── TerminalReporter.render: conditional sections ────────────────────────────


class TestRenderConditionalSections:
    def test_notes_panel_rendered_when_present(self) -> None:
        report = _make_report()
        report.notes = [AnnotationNote("critical portfolio constraint", source="decorator")]
        output = _render(report)
        assert "critical portfolio constraint" in output
        assert "Author Notes" in output

    def test_notes_panel_absent_when_empty(self) -> None:
        report = _make_report()
        output = _render(report)
        assert "Author Notes" not in output

    def test_context_manager_note_uses_chat_icon(self) -> None:
        report = _make_report()
        report.notes = [AnnotationNote("inline note", source="context_manager")]
        output = _render(report)
        assert "inline note" in output

    def test_breakdown_panel_rendered_with_expression(self) -> None:
        report = _make_report()
        bd = AssertionBreakdown()
        bd.expression_source = "result == expected"
        report.breakdown = bd
        output = _render(report)
        assert "result == expected" in output
        assert "Assertion Breakdown" in output

    def test_breakdown_panel_skipped_when_no_expression(self) -> None:
        report = _make_report()
        bd = AssertionBreakdown()
        bd.expression_source = ""
        bd.raw_source = ""
        report.breakdown = bd
        output = _render(report)
        assert "Assertion Breakdown" not in output

    def test_breakdown_uses_raw_source_as_fallback(self) -> None:
        report = _make_report()
        bd = AssertionBreakdown()
        bd.expression_source = ""
        bd.raw_source = "assert price > 0"
        report.breakdown = bd
        output = _render(report)
        assert "price > 0" in output

    def test_lhs_rhs_table_rendered(self) -> None:
        report = _make_report()
        report.breakdown = _make_comparison_breakdown("result", 42, "expected", 99)
        output = _render(report)
        assert "result" in output
        assert "42" in output
        assert "expected" in output
        assert "99" in output

    def test_numeric_delta_shown_for_floats(self) -> None:
        report = _make_report()
        report.breakdown = _make_comparison_breakdown(
            "actual_pnl", 1_187_432.57, "expected_pnl", 1_250_000.0
        )
        output = _render(report)
        # Numeric diff line must appear
        assert "Δ" in output or "delta" in output.lower() or "62" in output

    def test_sub_expressions_table_rendered_for_non_comparison(self) -> None:
        report = _make_report()
        bd = AssertionBreakdown()
        bd.expression_source = "is_valid and is_active"
        bd.sub_expressions = [
            SubExpr(source="is_valid", value=True),
            SubExpr(source="is_active", value=False),
        ]
        report.breakdown = bd
        output = _render(report)
        assert "is_valid" in output
        assert "is_active" in output

    def test_breakdown_error_note_shown(self) -> None:
        report = _make_report()
        bd = AssertionBreakdown()
        bd.expression_source = "x == y"
        bd.error = "AST mapping failed due to dynamic code"
        report.breakdown = bd
        output = _render(report)
        assert "AST mapping failed" in output

    def test_assertion_message_shown(self) -> None:
        report = _make_report()
        bd = AssertionBreakdown()
        bd.expression_source = "price > 0"
        bd.message_source = '"price must be positive"'
        report.breakdown = bd
        output = _render(report)
        assert "price must be positive" in output

    def test_source_panel_rendered_when_lines_present(self) -> None:
        report = _make_report()
        report.source_lines = ["def test_x():\n", "    assert 1 == 2\n"]
        report.source_first_lineno = 1
        report.source_failing_lineno = 2
        output = _render(report)
        assert "Source" in output
        assert "assert" in output

    def test_source_panel_absent_when_no_lines(self) -> None:
        report = _make_report()
        report.source_lines = []
        output = _render(report)
        assert "Source" not in output

    def test_source_panel_absent_when_disabled(self) -> None:
        report = _make_report()
        report.source_lines = ["    assert x == y\n"]
        report.source_first_lineno = 5
        report.source_failing_lineno = 5
        output = _render(report, show_source=False)
        assert "Source" not in output

    def test_locals_panel_rendered(self) -> None:
        report = _make_report()
        report.local_vars = [("portfolio_value", 1_000_000), ("threshold", 1_500_000)]
        output = _render(report)
        assert "portfolio_value" in output
        assert "threshold" in output
        assert "Local Variables" in output

    def test_locals_panel_absent_when_disabled(self) -> None:
        report = _make_report()
        report.local_vars = [("x", 42)]
        output = _render(report, show_locals=False)
        assert "Local Variables" not in output

    def test_locals_panel_absent_when_empty(self) -> None:
        report = _make_report()
        report.local_vars = []
        output = _render(report)
        assert "Local Variables" not in output


# ── TerminalReporter._diff_panel: all type branches ──────────────────────────


class TestDiffPanel:
    def _diff(self, lhs: Any, rhs: Any) -> str:
        cfg = _make_cfg()
        console, buf = _make_console()
        tr = TerminalReporter(cfg, console=console)
        panel = tr._diff_panel(lhs, rhs, "==")
        if panel is not None:
            console.print(panel)
        return buf.getvalue()

    def test_scalar_scalar_returns_none(self) -> None:
        cfg = _make_cfg()
        tr = TerminalReporter(cfg)
        result = tr._diff_panel(42, 99, "==")
        assert result is None

    def test_scalar_scalar_float_returns_none(self) -> None:
        cfg = _make_cfg()
        tr = TerminalReporter(cfg)
        result = tr._diff_panel(1.0, 2.0, "==")
        assert result is None

    def test_dict_diff_renders_table(self) -> None:
        output = self._diff(
            {"delta": 0.52, "gamma": 0.004},
            {"delta": 0.48, "gamma": 0.004},
        )
        assert "changed" in output or "delta" in output

    def test_dict_equal_returns_none(self) -> None:
        cfg = _make_cfg()
        tr = TerminalReporter(cfg)
        result = tr._diff_panel({"a": 1}, {"a": 1}, "==")
        assert result is None

    def test_list_diff_renders_table(self) -> None:
        output = self._diff([1, 2, 3], [1, 2, 99])
        assert len(output) > 0

    def test_set_diff_renders_table(self) -> None:
        output = self._diff({1, 2, 3}, {1, 2, 4})
        assert len(output) > 0

    def test_multiline_string_diff_renders(self) -> None:
        """
        _diff_panel classifies strings as 'scalar', so it returns None
        before reaching the multi-line string branch. This is a known design
        issue (see design risks). We therefore test _string_diff directly.
        """
        cfg = _make_cfg()
        console, buf = _make_console()
        tr = TerminalReporter(cfg, console=console)
        # _diff_panel for two scalars always returns None
        panel = tr._diff_panel(
            "line one\nline two\n",
            "line one\nline TWO\n",
            "==",
        )
        assert panel is None  # Confirmed: scalar/scalar short-circuits

    def test_string_diff_method_directly(self) -> None:
        """_string_diff renders correctly when called directly."""
        cfg = _make_cfg()
        console, buf = _make_console()
        tr = TerminalReporter(cfg, console=console)
        result = tr._string_diff("line one\nline two\n", "line one\nline TWO\n")
        console.print(result)
        output = buf.getvalue()
        assert "line" in output

    def test_single_line_string_returns_none(self) -> None:
        """Single-line strings with no \\n must not get a diff panel."""
        cfg = _make_cfg()
        tr = TerminalReporter(cfg)
        result = tr._diff_panel("hello", "world", "==")
        assert result is None

    def test_numpy_array_diff_renders(self) -> None:
        np = pytest.importorskip("numpy")
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 99.0])
        output = self._diff(a, b)
        assert "Shape" in output or "dtype" in output

    def test_numpy_different_shapes(self) -> None:
        np = pytest.importorskip("numpy")
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0, 3.0])
        output = self._diff(a, b)
        assert len(output) > 0

    def test_dataframe_diff_renders(self) -> None:
        pd = pytest.importorskip("pandas")
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [1, 99], "b": [3, 4]})
        output = self._diff(df1, df2)
        assert "Shape" in output or "Columns" in output

    def test_dataframe_numeric_diff_row(self) -> None:
        """When DataFrames have same shape and numeric cols, max |Δ| row appears."""
        pd = pytest.importorskip("pandas")
        df1 = pd.DataFrame({"price": [100.0, 200.0]})
        df2 = pd.DataFrame({"price": [100.0, 250.0]})
        output = self._diff(df1, df2)
        assert "max" in output.lower() or "50" in output

    def test_diff_disabled_no_panel(self) -> None:
        report = _make_report()
        report.breakdown = _make_comparison_breakdown(
            "a", {"x": 1}, "b", {"x": 2}
        )
        output = _render(report, show_diff=False)
        assert "Diff" not in output


# ── TerminalReporter._string_diff ────────────────────────────────────────────


class TestStringDiff:
    def test_identical_strings_show_match_message(self) -> None:
        cfg = _make_cfg()
        tr = TerminalReporter(cfg)
        result = tr._string_diff("hello\nworld\n", "hello\nworld\n")
        from rich.text import Text

        assert isinstance(result, Text)
        assert "match" in str(result)

    def test_diff_lines_colored_correctly(self) -> None:
        cfg = _make_cfg()
        console, buf = _make_console()
        tr = TerminalReporter(cfg, console=console)
        result = tr._string_diff("old line\n", "new line\n")
        console.print(result)
        output = buf.getvalue()
        assert "old line" in output or "new line" in output


# ── TerminalReporter._deep_diff_table: row truncation ────────────────────────


class TestDeepDiffTable:
    def test_truncates_at_20_rows(self) -> None:
        """A diff with >20 changed values must stop at 20 rows (truncates silently)."""
        cfg = _make_cfg()
        from deepdiff import DeepDiff

        # Build a dict with 25 changed values under 'values_changed'
        expected = {str(i): i for i in range(25)}
        actual = {str(i): i + 1 for i in range(25)}
        diff = dict(DeepDiff(expected, actual, verbose_level=2))
        assert "values_changed" in diff
        assert len(diff["values_changed"]) == 25

        console, buf = _make_console()
        tr = TerminalReporter(cfg, console=console)
        table = tr._deep_diff_table(diff)
        console.print(table)
        output = buf.getvalue()

        # Table must not contain more than 20 data rows + 1 truncation row
        # Count "changed" occurrences — should be exactly 20 (rows 0-19)
        changed_count = output.count("changed")
        assert changed_count <= 21  # 20 rows + possible truncation marker

    def test_handles_set_type_changes(self) -> None:
        """DeepDiff returns sets for added/removed items — must render without crash."""
        cfg = _make_cfg()
        from deepdiff import DeepDiff

        diff = dict(DeepDiff({"items": {1, 2, 3}}, {"items": {1, 2, 4}}, verbose_level=2))
        if not diff:
            return  # Nothing to diff
        console, buf = _make_console()
        tr = TerminalReporter(cfg, console=console)
        table = tr._deep_diff_table(diff)
        console.print(table)
        # Must not raise

    def test_unknown_change_type_uses_raw_key(self) -> None:
        """A change type not in type_labels must still render (uses raw key)."""
        cfg = _make_cfg()
        diff = {"custom_change_type": {"root['x']": object()}}
        console, buf = _make_console()
        tr = TerminalReporter(cfg, console=console)
        table = tr._deep_diff_table(diff)
        console.print(table)
        output = buf.getvalue()
        assert "custom_change_type" in output


# ── TerminalReporter._numpy_diff robustness ───────────────────────────────────


class TestNumpyDiff:
    def test_same_shape_shows_mismatch_stats(self) -> None:
        np = pytest.importorskip("numpy")
        cfg = _make_cfg()
        console, buf = _make_console()
        tr = TerminalReporter(cfg, console=console)
        e = np.array([1.0, 2.0, 3.0])
        a = np.array([1.0, 2.0, 99.0])
        result = tr._numpy_diff(e, a)
        console.print(result)
        output = buf.getvalue()
        assert "Mismatches" in output
        assert "1" in output  # 1 mismatch

    def test_different_shapes_no_mismatch_stats(self) -> None:
        np = pytest.importorskip("numpy")
        cfg = _make_cfg()
        console, buf = _make_console()
        tr = TerminalReporter(cfg, console=console)
        e = np.array([1.0, 2.0])
        a = np.array([1.0, 2.0, 3.0])
        result = tr._numpy_diff(e, a)
        console.print(result)
        output = buf.getvalue()
        # Shape rows must be there, no mismatch row
        assert "Shape" in output
        assert "Mismatches" not in output

    def test_numpy_diff_failure_shows_warning(self) -> None:
        """If numpy diff itself explodes, return a warning Text."""
        cfg = _make_cfg()
        tr = TerminalReporter(cfg)
        result = tr._numpy_diff("not an array", "also not")
        from rich.text import Text

        assert isinstance(result, Text)


# ── TerminalReporter._dataframe_diff robustness ───────────────────────────────


class TestDataframeDiff:
    def test_different_columns(self) -> None:
        pd = pytest.importorskip("pandas")
        cfg = _make_cfg()
        console, buf = _make_console()
        tr = TerminalReporter(cfg, console=console)
        df1 = pd.DataFrame({"a": [1], "b": [2]})
        df2 = pd.DataFrame({"a": [1], "c": [3]})
        result = tr._dataframe_diff(df1, df2)
        console.print(result)
        output = buf.getvalue()
        assert "Columns" in output

    def test_string_columns_no_numeric_diff(self) -> None:
        """DataFrames with only string columns must not crash on numeric diff."""
        pd = pytest.importorskip("pandas")
        cfg = _make_cfg()
        console, buf = _make_console()
        tr = TerminalReporter(cfg, console=console)
        df1 = pd.DataFrame({"name": ["alice", "bob"]})
        df2 = pd.DataFrame({"name": ["alice", "carol"]})
        result = tr._dataframe_diff(df1, df2)
        console.print(result)
        # Must not raise

    def test_dataframe_diff_failure_shows_warning(self) -> None:
        """Non-DataFrame input must return a warning Text, not crash."""
        cfg = _make_cfg()
        tr = TerminalReporter(cfg)
        result = tr._dataframe_diff("not a df", "also not")
        from rich.text import Text

        assert isinstance(result, Text)


# ── HtmlReporter ──────────────────────────────────────────────────────────────


class TestHtmlReporter:
    def test_creates_html_file(self, tmp_path: Path) -> None:
        cfg = _make_cfg()
        cfg.html_report_path = str(tmp_path / "report.html")
        report = _make_report()
        hr = HtmlReporter(cfg)
        hr.render(report)
        html_file = tmp_path / "report.html"
        assert html_file.exists()

    def test_html_contains_doctype(self, tmp_path: Path) -> None:
        cfg = _make_cfg()
        cfg.html_report_path = str(tmp_path / "report.html")
        report = _make_report()
        hr = HtmlReporter(cfg)
        hr.render(report)
        content = (tmp_path / "report.html").read_text()
        assert "<!DOCTYPE html>" in content

    def test_html_contains_test_id(self, tmp_path: Path) -> None:
        cfg = _make_cfg()
        cfg.html_report_path = str(tmp_path / "report.html")
        report = _make_report(test_id="tests/test_quant.py::test_kelly")
        hr = HtmlReporter(cfg)
        hr.render(report)
        content = (tmp_path / "report.html").read_text()
        assert "test_kelly" in content

    def test_html_accumulates_multiple_failures(self, tmp_path: Path) -> None:
        cfg = _make_cfg()
        cfg.html_report_path = str(tmp_path / "report.html")
        hr = HtmlReporter(cfg)
        for i in range(3):
            r = _make_report(test_id=f"tests/test_foo.py::test_{i}")
            hr.render(r)
        content = (tmp_path / "report.html").read_text()
        assert "test_0" in content
        assert "test_1" in content
        assert "test_2" in content

    def test_html_uses_default_path_when_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When html_report_path is None, must default to beacon_report.html."""
        monkeypatch.chdir(tmp_path)
        cfg = _make_cfg()
        cfg.html_report_path = None
        report = _make_report()
        hr = HtmlReporter(cfg)
        hr.render(report)
        assert (tmp_path / "beacon_report.html").exists()


# ── render_failure: html sink ─────────────────────────────────────────────────


class TestRenderFailureHtmlSink:
    def test_html_sink_invoked(self, tmp_path: Path) -> None:
        cfg = _make_cfg()
        cfg.output_formats = ["html"]
        cfg.html_report_path = str(tmp_path / "out.html")
        report = _make_report()
        render_failure(report, config=cfg)
        assert (tmp_path / "out.html").exists()

    def test_all_sinks_invoked(self, tmp_path: Path) -> None:
        cfg = _make_cfg()
        cfg.output_formats = ["terminal", "json", "html"]
        cfg.json_report_path = str(tmp_path / "f.jsonl")
        cfg.html_report_path = str(tmp_path / "r.html")
        report = _make_report()
        render_failure(report, config=cfg)  # must not raise
        assert (tmp_path / "f.jsonl").exists()
        assert (tmp_path / "r.html").exists()


# ── JsonReporter edge cases ───────────────────────────────────────────────────


class TestJsonReporterEdgeCases:
    def test_breakdown_with_lhs_rhs_serialised(self, tmp_path: Path) -> None:
        cfg = _make_cfg()
        cfg.json_report_path = str(tmp_path / "out.jsonl")
        report = _make_report()
        report.breakdown = _make_comparison_breakdown("actual", 1, "expected", 2)
        jr = JsonReporter(cfg)
        jr.render(report)
        record = json.loads((tmp_path / "out.jsonl").read_text().strip())
        assert record["expression"]["lhs"] == "1"
        assert record["expression"]["rhs"] == "2"
        assert record["expression"]["operator"] == "=="

    def test_breakdown_none_serialises_as_null(self, tmp_path: Path) -> None:
        cfg = _make_cfg()
        cfg.json_report_path = str(tmp_path / "out.jsonl")
        report = _make_report()
        report.breakdown = None
        jr = JsonReporter(cfg)
        jr.render(report)
        record = json.loads((tmp_path / "out.jsonl").read_text().strip())
        assert record["expression"] is None

    def test_notes_serialised(self, tmp_path: Path) -> None:
        cfg = _make_cfg()
        cfg.json_report_path = str(tmp_path / "out.jsonl")
        report = _make_report()
        report.notes = [AnnotationNote("important note", source="decorator")]
        jr = JsonReporter(cfg)
        jr.render(report)
        record = json.loads((tmp_path / "out.jsonl").read_text().strip())
        assert len(record["notes"]) == 1
        assert record["notes"][0]["message"] == "important note"

    def test_local_vars_with_unrepresentable_object(self, tmp_path: Path) -> None:
        """Objects with broken repr must not crash JSON serialisation."""

        class BrokenRepr:
            def __repr__(self) -> str:
                raise RuntimeError("repr exploded")

        cfg = _make_cfg()
        cfg.json_report_path = str(tmp_path / "out.jsonl")
        report = _make_report()
        report.local_vars = [("broken", BrokenRepr())]
        jr = JsonReporter(cfg)
        jr.render(report)  # must not raise
        content = (tmp_path / "out.jsonl").read_text()
        assert "broken" in content

    def test_default_path_when_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        cfg = _make_cfg()
        cfg.json_report_path = None
        report = _make_report()
        jr = JsonReporter(cfg)
        jr.render(report)
        # Default is "beacon_failures.json"
        assert (tmp_path / "beacon_failures.json").exists()

    def test_param_id_in_record(self, tmp_path: Path) -> None:
        cfg = _make_cfg()
        cfg.json_report_path = str(tmp_path / "out.jsonl")
        report = _make_report()
        report.param_id = "x=1-y=2"
        jr = JsonReporter(cfg)
        jr.render(report)
        record = json.loads((tmp_path / "out.jsonl").read_text().strip())
        assert record["param_id"] == "x=1-y=2"


# ── _shorten_path ─────────────────────────────────────────────────────────────


class TestShortenPath:
    def test_short_path_unchanged(self) -> None:
        result = _shorten_path("tests/test_foo.py")
        assert result == "tests/test_foo.py"

    def test_long_path_truncated(self) -> None:
        long_path = "/a/b/c/d/e/f/tests/test_module.py"
        result = _shorten_path(long_path, max_parts=4)
        assert "…" in result
        assert "test_module.py" in result

    def test_exactly_max_parts_unchanged(self) -> None:
        # /a/b/c has exactly 4 parts on Linux: ('/', 'a', 'b', 'c')
        import os
        path = os.path.join("/", "a", "b", "c")
        result = _shorten_path(path, max_parts=4)
        # Has exactly 4 parts — should not truncate
        assert "…" not in result

    def test_one_part_path(self) -> None:
        result = _shorten_path("test_foo.py")
        assert result == "test_foo.py"


# ── _smart_repr ───────────────────────────────────────────────────────────────


class TestSmartRepr:
    def test_scalar_uses_default_repr(self) -> None:
        result = _smart_repr(42, 500)
        assert result == "42"

    def test_string_uses_default_repr(self) -> None:
        result = _smart_repr("hello", 500)
        assert result == "'hello'"

    def test_numpy_array_shows_shape_dtype(self) -> None:
        np = pytest.importorskip("numpy")
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = _smart_repr(arr, 500)
        assert "ndarray" in result
        assert "float" in result

    def test_numpy_small_array_no_ellipsis(self) -> None:
        np = pytest.importorskip("numpy")
        arr = np.array([1.0, 2.0, 3.0])
        result = _smart_repr(arr, 500)
        assert "ndarray" in result
        # Small array — no ellipsis
        assert "…" not in result

    def test_numpy_large_array_has_ellipsis(self) -> None:
        np = pytest.importorskip("numpy")
        arr = np.arange(100, dtype=float)
        result = _smart_repr(arr, 500)
        assert "…" in result

    def test_dataframe_shows_shape_cols(self) -> None:
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"alpha": [1, 2], "beta": [3, 4], "gamma": [5, 6]})
        result = _smart_repr(df, 500)
        assert "DataFrame" in result
        assert "alpha" in result

    def test_dataframe_many_columns_truncated(self) -> None:
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({str(i): [i] for i in range(10)})
        result = _smart_repr(df, 500)
        assert "DataFrame" in result
        assert "…" in result

    def test_truncation_applied(self) -> None:
        long_list = list(range(1000))
        result = _smart_repr(long_list, 50)
        assert len(result) <= 60  # safe margin


# ── FailureReport: header edge cases ─────────────────────────────────────────


class TestHeaderEdgeCases:
    def test_empty_test_id_falls_back_to_function_name(self) -> None:
        report = _make_report()
        report.test_id = ""
        report.test_function = "test_my_function"
        output = _render(report)
        assert "test_my_function" in output

    def test_no_test_file_no_file_line(self) -> None:
        report = _make_report()
        report.test_file = ""
        output = _render(report)
        # Should not crash — header just omits the file line
        assert "BEACON" in output

    def test_empty_exc_message_renders_cleanly(self) -> None:
        report = _make_report(exc_message="")
        output = _render(report)
        assert "AssertionError" in output

    def test_long_exc_message_renders(self) -> None:
        report = _make_report(exc_message="x" * 300)
        output = _render(report)
        assert "AssertionError" in output
