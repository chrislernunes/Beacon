"""Tests for beacon.reporters."""
from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

import beacon
from beacon.config import BeaconConfig
from beacon.reporters import (
    FailureReport,
    JsonReporter,
    TerminalReporter,
    render_failure,
)


def _silent_cfg() -> BeaconConfig:
    c = BeaconConfig()
    c.output_formats = []
    return c


def _make_report(**kwargs: Any) -> FailureReport:
    r = FailureReport()
    r.test_id = kwargs.get("test_id", "tests/test_foo.py::test_bar")
    r.test_function = kwargs.get("test_function", "test_bar")
    r.test_file = kwargs.get("test_file", "tests/test_foo.py")
    r.test_lineno = kwargs.get("test_lineno", 42)
    r.exc_type = kwargs.get("exc_type", "AssertionError")
    r.exc_message = kwargs.get("exc_message", "1 != 2")
    return r


class TestFailureReport:
    def test_default_values(self) -> None:
        r = FailureReport()
        assert r.test_id == ""
        assert r.exc_type == ""
        assert r.local_vars == []
        assert r.notes == []
        assert r.source_lines == []
        assert r.breakdown is None

    def test_captured_at_is_set(self) -> None:
        r = FailureReport()
        assert r.captured_at.endswith("Z")
        assert "T" in r.captured_at


class TestTerminalReporter:
    def _make_console(self) -> Console:
        buf = StringIO()
        return Console(file=buf, force_terminal=False, width=120)

    def _render_to_string(self, report: FailureReport, cfg: BeaconConfig) -> str:
        buf = StringIO()
        console = Console(file=buf, force_terminal=False, width=120)
        tr = TerminalReporter(cfg, console=console)
        tr.render(report)
        return buf.getvalue()

    def test_renders_without_error(self) -> None:
        cfg = BeaconConfig()
        cfg.output_formats = ["terminal"]
        report = _make_report()
        # Should not raise
        self._render_to_string(report, cfg)

    def test_test_id_in_output(self) -> None:
        cfg = BeaconConfig()
        report = _make_report(test_id="tests/test_foo.py::test_bar")
        output = self._render_to_string(report, cfg)
        assert "test_bar" in output

    def test_exc_type_in_output(self) -> None:
        cfg = BeaconConfig()
        report = _make_report(exc_type="ValueError")
        output = self._render_to_string(report, cfg)
        assert "ValueError" in output

    def test_exc_message_in_output(self) -> None:
        cfg = BeaconConfig()
        report = _make_report(exc_message="something went wrong")
        output = self._render_to_string(report, cfg)
        assert "something went wrong" in output

    def test_notes_in_output(self) -> None:
        from beacon.annotations import TestNote

        cfg = BeaconConfig()
        report = _make_report()
        report.notes = [TestNote("This is an important test", source="decorator")]
        output = self._render_to_string(report, cfg)
        assert "This is an important test" in output

    def test_local_vars_in_output(self) -> None:
        cfg = BeaconConfig()
        report = _make_report()
        report.local_vars = [("alpha", 42), ("beta", "hello")]
        output = self._render_to_string(report, cfg)
        assert "alpha" in output
        assert "beta" in output

    def test_param_id_in_output(self) -> None:
        cfg = BeaconConfig()
        report = _make_report()
        report.param_id = "x=1-y=2"
        output = self._render_to_string(report, cfg)
        assert "x=1-y=2" in output

    def test_no_locals_when_disabled(self) -> None:
        cfg = BeaconConfig()
        cfg.show_locals = False
        report = _make_report()
        report.local_vars = [("secret_var", 999)]
        output = self._render_to_string(report, cfg)
        assert "secret_var" not in output

    def test_renders_with_breakdown(self) -> None:
        from beacon.rewrite import AssertionBreakdown, SubExpr

        cfg = BeaconConfig()
        report = _make_report()
        bd = AssertionBreakdown()
        bd.expression_source = "result == expected"
        bd.lhs = SubExpr(source="result", value=1)
        bd.rhs = SubExpr(source="expected", value=2)
        bd.operator = "=="
        report.breakdown = bd
        output = self._render_to_string(report, cfg)
        assert "result" in output


class TestJsonReporter:
    def test_writes_jsonl(self, tmp_path: Path) -> None:
        cfg = BeaconConfig()
        cfg.json_report_path = str(tmp_path / "failures.jsonl")
        report = _make_report()

        jr = JsonReporter(cfg)
        jr.render(report)

        lines = (tmp_path / "failures.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["test_id"] == "tests/test_foo.py::test_bar"
        assert record["exc_type"] == "AssertionError"

    def test_appends_multiple_failures(self, tmp_path: Path) -> None:
        cfg = BeaconConfig()
        cfg.json_report_path = str(tmp_path / "failures.jsonl")
        jr = JsonReporter(cfg)

        for i in range(3):
            r = _make_report(test_id=f"test_{i}")
            jr.render(r)

        lines = (tmp_path / "failures.jsonl").read_text().strip().splitlines()
        assert len(lines) == 3

    def test_json_contains_local_vars(self, tmp_path: Path) -> None:
        cfg = BeaconConfig()
        cfg.json_report_path = str(tmp_path / "failures.jsonl")
        report = _make_report()
        report.local_vars = [("x", 42)]

        jr = JsonReporter(cfg)
        jr.render(report)

        record = json.loads((tmp_path / "failures.jsonl").read_text().strip())
        assert "x" in record["local_vars"]


class TestRenderFailure:
    def test_no_output_when_empty_formats(self) -> None:
        """render_failure with no output formats should not raise."""
        cfg = BeaconConfig()
        cfg.output_formats = []
        report = _make_report()
        render_failure(report, config=cfg)  # should not raise or print anything

    def test_reporter_crash_doesnt_propagate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If a reporter crashes, render_failure should swallow the error."""
        cfg = BeaconConfig()
        cfg.output_formats = ["terminal"]

        # Monkeypatch TerminalReporter.render to raise
        from beacon import reporters

        def bad_render(self: Any, report: Any) -> None:
            raise RuntimeError("reporter crashed")

        monkeypatch.setattr(reporters.TerminalReporter, "render", bad_render)
        report = _make_report()
        render_failure(report, config=cfg)  # must not propagate RuntimeError
