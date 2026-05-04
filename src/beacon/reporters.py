"""
beacon.reporters
────────────────
All output rendering lives here.

The main entry point is ``render_failure()``, which accepts a ``FailureReport``
and dispatches to every configured output format.

Architecture
────────────
  FailureReport            — plain dataclass assembled by core.py
  TerminalReporter         — renders to the Rich console (the main show)
  JsonReporter             — writes structured JSON to a file
  HtmlReporter             — renders a self-contained HTML file
  render_failure(report)   — fan-out to all active reporters
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .annotations import TestNote
from .config import BeaconConfig, get_config
from .rewrite import AssertionBreakdown
from .utils import (
    classify_object,
    compute_deep_diff,
    format_numeric_diff,
    safe_repr,
    safe_type_name,
    unified_text_diff,
)

# ── Palette ───────────────────────────────────────────────────────────────────

_C_HEADER = "bold cyan"
_C_FAIL = "bold red"
_C_PASS = "bold green"
_C_WARN = "bold yellow"
_C_DIM = "dim"
_C_OP = "bold magenta"
_C_VAR_NAME = "bold blue"
_C_VAR_TYPE = "dim cyan"
_C_NOTE = "bold yellow"
_C_DIFF_ADD = "green"
_C_DIFF_REM = "red"
_C_DIFF_HDR = "cyan"


# ── FailureReport ─────────────────────────────────────────────────────────────


@dataclass
class FailureReport:
    """
    Self-contained record of a single test failure.

    Assembled by ``core.py`` and consumed by the reporters.
    All fields are optional so the reporter degrades gracefully.
    """

    # Identity
    test_id: str = ""
    test_function: str = ""
    test_file: str = ""
    test_lineno: int = 0

    # The exception
    exc_type: str = ""
    exc_message: str = ""

    # AST breakdown
    breakdown: Optional[AssertionBreakdown] = None

    # Source context
    source_lines: List[str] = field(default_factory=list)
    source_first_lineno: int = 0
    source_failing_lineno: int = 0

    # Local variables
    local_vars: List[tuple[str, Any]] = field(default_factory=list)

    # Author notes
    notes: List[TestNote] = field(default_factory=list)

    # Timestamp
    captured_at: str = field(
        default_factory=lambda: __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
    ).isoformat(timespec="seconds").replace("+00:00", "Z")
    )

    # Parameterization info
    param_id: Optional[str] = None


# ── TerminalReporter ──────────────────────────────────────────────────────────


class TerminalReporter:
    """Renders a FailureReport to the terminal using Rich."""

    def __init__(self, config: BeaconConfig, console: Optional[Console] = None) -> None:
        self.config = config
        self.console = console or Console(
            stderr=True,
            highlight=True,
            force_terminal=True,
        )

    # ── Top-level render ──────────────────────────────────────────────────────

    def render(self, report: FailureReport) -> None:
        c = self.console
        c.print()
        c.print(self._header(report))
        c.print()

        if report.notes:
            c.print(self._notes_panel(report.notes))
            c.print()

        if report.breakdown:
            bd = report.breakdown
            if bd.expression_source or bd.raw_source:
                c.print(self._assertion_panel(bd))
                c.print()

            if self.config.show_diff and bd.lhs is not None and bd.rhs is not None:
                diff_panel = self._diff_panel(bd.lhs.value, bd.rhs.value, bd.operator)
                if diff_panel is not None:
                    c.print(diff_panel)
                    c.print()

        if self.config.show_source and report.source_lines:
            c.print(self._source_panel(report))
            c.print()

        if self.config.show_locals and report.local_vars:
            c.print(self._locals_panel(report.local_vars))
            c.print()

    # ── Header ────────────────────────────────────────────────────────────────

    def _header(self, report: FailureReport) -> Panel:
        title_text = Text()
        title_text.append("  BEACON ", style="bold white on red")
        title_text.append("  TEST FAILURE", style=_C_FAIL)

        lines: List[Any] = [title_text]
        lines.append(Text())

        id_text = Text()
        id_text.append("  Test: ", style=_C_DIM)
        id_text.append(report.test_id or report.test_function, style="bold white")
        lines.append(id_text)

        if report.test_file:
            loc_text = Text()
            loc_text.append("  File: ", style=_C_DIM)
            short_file = _shorten_path(report.test_file)
            loc_text.append(f"{short_file}:{report.test_lineno}", style="cyan")
            lines.append(loc_text)

        if report.param_id:
            param_text = Text()
            param_text.append("Param: ", style=_C_DIM)
            param_text.append(report.param_id, style="bold yellow")
            lines.append(param_text)

        lines.append(Text())

        exc_text = Text()
        exc_text.append("  Error: ", style=_C_DIM)
        exc_text.append(report.exc_type, style=_C_FAIL)
        if report.exc_message:
            exc_text.append(f"  {report.exc_message}", style="white")
        lines.append(exc_text)

        return Panel(
            Group(*lines),
            border_style="red",
            padding=(0, 1),
        )

    # ── Notes ─────────────────────────────────────────────────────────────────

    def _notes_panel(self, notes: List[TestNote]) -> Panel:
        content: List[Any] = []
        for note in notes:
            t = Text()
            icon = "📝" if note.source == "decorator" else "💬"
            t.append(f"{icon} ", style="")
            t.append(note.message, style=_C_NOTE)
            content.append(t)
        return Panel(
            Group(*content),
            title="[bold yellow]Author Notes[/]",
            border_style="yellow",
            padding=(0, 1),
        )

    # ── Assertion breakdown ───────────────────────────────────────────────────

    def _assertion_panel(self, bd: AssertionBreakdown) -> Panel:
        content: List[Any] = []

        # Expression
        expr_src = bd.expression_source or bd.raw_source
        if expr_src:
            expr_text = Text()
            expr_text.append("  assert ", style="bold dim")
            expr_text.append(expr_src, style="bold white")
            content.append(expr_text)
            content.append(Text())

        # LHS  OP  RHS breakdown
        if bd.is_comparison and bd.lhs and bd.rhs:
            tbl = Table(show_header=False, box=None, padding=(0, 2))
            tbl.add_column("side", style=_C_DIM, no_wrap=True)
            tbl.add_column("src", style=_C_VAR_NAME)
            tbl.add_column("eq", style=_C_OP)
            tbl.add_column("val", style="white")
            tbl.add_column("type", style=_C_VAR_TYPE)

            tbl.add_row(
                "left ",
                bd.lhs.source,
                "=",
                _truncated_repr(bd.lhs.value),
                f"({bd.lhs.type_name})",
            )
            if bd.operator:
                op_row = Text(f"  {bd.operator}", style=_C_OP)
                content.append(op_row)
            tbl.add_row(
                "right",
                bd.rhs.source,
                "=",
                _truncated_repr(bd.rhs.value),
                f"({bd.rhs.type_name})",
            )
            content.append(tbl)
            content.append(Text())

            # Numeric diff
            num_diff = format_numeric_diff(bd.lhs.value, bd.rhs.value)
            if num_diff:
                nd = Text()
                nd.append("  Δ  ", style=_C_WARN)
                nd.append(num_diff, style="white")
                content.append(nd)
                content.append(Text())

        # Sub-expressions (non-comparison case)
        elif bd.has_sub_expressions:
            tbl = Table(show_header=True, box=box.SIMPLE, padding=(0, 1))
            tbl.add_column("Expression", style=_C_VAR_NAME)
            tbl.add_column("Value", style="white")
            tbl.add_column("Type", style=_C_VAR_TYPE)
            for sub in bd.sub_expressions:
                tbl.add_row(sub.source, _truncated_repr(sub.value), sub.type_name)
            content.append(tbl)

        # Assert message
        if bd.message_source:
            msg_text = Text()
            msg_text.append("  Message: ", style=_C_DIM)
            msg_text.append(bd.message_source, style="italic white")
            content.append(msg_text)

        if bd.error:
            err_text = Text()
            err_text.append(f"  ⚠ Note: {bd.error}", style=_C_WARN)
            content.append(err_text)

        return Panel(
            Group(*content),
            title="[bold cyan]Assertion Breakdown[/]",
            border_style="cyan",
            padding=(0, 1),
        )

    # ── Diff ──────────────────────────────────────────────────────────────────

    def _diff_panel(
        self,
        expected: Any,
        actual: Any,
        operator: Optional[str],
    ) -> Optional[Panel]:
        """Generate the richest possible diff for the two values."""
        kind_exp = classify_object(expected)
        kind_act = classify_object(actual)

        content: List[Any] = []

        # ── Numeric scalar ────────────────────────────────────────────────────
        if kind_exp == "scalar" and kind_act == "scalar":
            return None  # already shown in breakdown table

        # ── numpy array ───────────────────────────────────────────────────────
        if kind_exp == "ndarray" or kind_act == "ndarray":
            content.append(self._numpy_diff(expected, actual))

        # ── pandas DataFrame ──────────────────────────────────────────────────
        elif kind_exp == "dataframe" or kind_act == "dataframe":
            content.append(self._dataframe_diff(expected, actual))

        # ── Mapping / dict ────────────────────────────────────────────────────
        elif kind_exp in ("mapping",) or kind_act in ("mapping",):
            diff = compute_deep_diff(expected, actual)
            if diff:
                content.append(self._deep_diff_table(diff))

        # ── Sequence / set ────────────────────────────────────────────────────
        elif kind_exp in ("sequence", "set") or kind_act in ("sequence", "set"):
            diff = compute_deep_diff(expected, actual)
            if diff:
                content.append(self._deep_diff_table(diff))

        # ── Multi-line strings ────────────────────────────────────────────────
        elif isinstance(expected, str) and isinstance(actual, str):
            if "\n" in expected or "\n" in actual:
                content.append(self._string_diff(expected, actual))

        # ── Generic: DeepDiff ─────────────────────────────────────────────────
        else:
            diff = compute_deep_diff(expected, actual)
            if diff:
                content.append(self._deep_diff_table(diff))

        if not content:
            return None

        return Panel(
            Group(*content),
            title="[bold magenta]Diff[/]",
            border_style="magenta",
            padding=(0, 1),
        )

    def _numpy_diff(self, expected: Any, actual: Any) -> Any:
        try:
            import numpy as np

            e = np.asarray(expected)
            a = np.asarray(actual)
            tbl = Table(show_header=True, box=box.SIMPLE_HEAVY, padding=(0, 1))
            tbl.add_column("Metric", style=_C_DIM)
            tbl.add_column("Expected", style=_C_PASS)
            tbl.add_column("Actual", style=_C_FAIL)
            tbl.add_row("Shape", str(e.shape), str(a.shape))
            tbl.add_row("dtype", str(e.dtype), str(a.dtype))
            if e.shape == a.shape:
                diff = np.abs(a.astype(float) - e.astype(float))
                tbl.add_row("max |Δ|", "—", f"{float(np.max(diff)):.6g}")
                tbl.add_row("mean |Δ|", "—", f"{float(np.mean(diff)):.6g}")
                mismatches = int(np.sum(a != e))
                tbl.add_row("Mismatches", "—", str(mismatches))
            return tbl
        except Exception as e:  # noqa: BLE001
            return Text(f"[numpy diff failed: {e}]", style=_C_WARN)

    def _dataframe_diff(self, expected: Any, actual: Any) -> Any:
        try:
            import pandas as pd

            tbl = Table(show_header=True, box=box.SIMPLE_HEAVY, padding=(0, 1))
            tbl.add_column("Metric", style=_C_DIM)
            tbl.add_column("Expected", style=_C_PASS)
            tbl.add_column("Actual", style=_C_FAIL)
            tbl.add_row("Shape", str(expected.shape), str(actual.shape))
            tbl.add_row("Columns", str(list(expected.columns)), str(list(actual.columns)))

            if set(expected.columns) == set(actual.columns) and expected.shape == actual.shape:
                try:
                    numeric_exp = expected.select_dtypes(include="number")
                    numeric_act = actual.select_dtypes(include="number")
                    if not numeric_exp.empty:
                        diff = (numeric_act - numeric_exp).abs()
                        worst_col = diff.max().idxmax()
                        tbl.add_row(
                            f"max |Δ| (col: {worst_col})",
                            "—",
                            f"{diff.max().max():.6g}",
                        )
                except Exception:  # noqa: BLE001
                    pass
            return tbl
        except Exception as e:  # noqa: BLE001
            return Text(f"[DataFrame diff failed: {e}]", style=_C_WARN)

    def _string_diff(self, expected: str, actual: str) -> Any:
        diff_lines = unified_text_diff(expected, actual)
        if not diff_lines:
            return Text("Strings match (unified diff is empty)", style=_C_DIM)
        text = Text()
        for line in diff_lines:
            if line.startswith("+++") or line.startswith("---"):
                text.append(line, style=_C_DIFF_HDR)
            elif line.startswith("+"):
                text.append(line, style=_C_DIFF_ADD)
            elif line.startswith("-"):
                text.append(line, style=_C_DIFF_REM)
            elif line.startswith("@@"):
                text.append(line, style=_C_DIFF_HDR)
            else:
                text.append(line, style=_C_DIM)
        return text

    def _deep_diff_table(self, diff: Dict[str, Any]) -> Any:
        tbl = Table(show_header=True, box=box.SIMPLE, padding=(0, 1))
        tbl.add_column("Change Type", style=_C_WARN, no_wrap=True)
        tbl.add_column("Path / Key", style=_C_VAR_NAME)
        tbl.add_column("Expected", style=_C_PASS)
        tbl.add_column("Actual", style=_C_FAIL)

        max_rows = 20
        row_count = 0

        type_labels = {
            "values_changed": "changed",
            "dictionary_item_added": "added",
            "dictionary_item_removed": "removed",
            "iterable_item_added": "added",
            "iterable_item_removed": "removed",
            "type_changes": "type changed",
            "attribute_added": "attr added",
            "attribute_removed": "attr removed",
        }

        for change_type, changes in diff.items():
            if row_count >= max_rows:
                tbl.add_row("…", f"({len(diff)} total change types)", "", "")
                break
            label = type_labels.get(change_type, change_type)
            if isinstance(changes, dict):
                for path, detail in changes.items():
                    if row_count >= max_rows:
                        break
                    old_val = safe_repr(getattr(detail, "t1", "—"))
                    new_val = safe_repr(getattr(detail, "t2", "—"))
                    tbl.add_row(label, str(path), old_val[:60], new_val[:60])
                    row_count += 1
            elif isinstance(changes, (set, list)):
                for item in changes:
                    if row_count >= max_rows:
                        break
                    tbl.add_row(label, str(item), "", "")
                    row_count += 1
        return tbl

    # ── Source ────────────────────────────────────────────────────────────────

    def _source_panel(self, report: FailureReport) -> Panel:
        code = "".join(report.source_lines)
        # Figure out highlight line relative to Syntax widget (1-based within snippet)
        highlight_line = report.source_failing_lineno - report.source_first_lineno + 1

        syntax = Syntax(
            code,
            "python",
            theme=self.config.theme,
            line_numbers=True,
            start_line=report.source_first_lineno,
            highlight_lines={report.source_failing_lineno},
            word_wrap=False,
        )
        return Panel(
            syntax,
            title=f"[bold cyan]Source  [dim]{_shorten_path(report.test_file)}[/][/]",
            border_style="cyan",
            padding=(0, 0),
        )

    # ── Locals ────────────────────────────────────────────────────────────────

    def _locals_panel(self, local_vars: List[tuple[str, Any]]) -> Panel:
        tbl = Table(show_header=True, box=box.SIMPLE_HEAVY, padding=(0, 1))
        tbl.add_column("Variable", style=_C_VAR_NAME, no_wrap=True)
        tbl.add_column("Type", style=_C_VAR_TYPE, no_wrap=True)
        tbl.add_column("Value", style="white", overflow="fold")

        for name, value in local_vars:
            type_name = safe_type_name(value)
            repr_str = _smart_repr(value, self.config.max_repr_length)
            tbl.add_row(name, type_name, repr_str)

        return Panel(
            tbl,
            title="[bold cyan]Local Variables[/]",
            border_style="blue",
            padding=(0, 0),
        )


# ── JsonReporter ──────────────────────────────────────────────────────────────


class JsonReporter:
    """Appends a structured JSON record to a file for each failure."""

    def __init__(self, config: BeaconConfig) -> None:
        self.config = config
        self.path = Path(config.json_report_path or "beacon_failures.json")

    def render(self, report: FailureReport) -> None:
        record = {
            "test_id": report.test_id,
            "test_file": report.test_file,
            "test_lineno": report.test_lineno,
            "exc_type": report.exc_type,
            "exc_message": report.exc_message,
            "captured_at": report.captured_at,
            "param_id": report.param_id,
            "notes": [{"message": n.message, "source": n.source} for n in report.notes],
            "expression": (
                {
                    "source": report.breakdown.expression_source,
                    "lhs": safe_repr(report.breakdown.lhs.value)
                    if report.breakdown.lhs
                    else None,
                    "rhs": safe_repr(report.breakdown.rhs.value)
                    if report.breakdown.rhs
                    else None,
                    "operator": report.breakdown.operator,
                }
                if report.breakdown
                else None
            ),
            "local_vars": {name: safe_repr(val) for name, val in report.local_vars},
        }
        # Append to JSONL file
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── HtmlReporter ──────────────────────────────────────────────────────────────


class HtmlReporter:
    """
    Renders a self-contained HTML failure report using Rich's HTML export.
    """

    def __init__(self, config: BeaconConfig) -> None:
        self.config = config
        self.path = Path(config.html_report_path or "beacon_report.html")
        self._records: List[str] = []

    def render(self, report: FailureReport) -> None:
        # Render to string via a string console, then embed in HTML
        from io import StringIO

        from rich.console import Console as RichConsole

        buf = StringIO()
        string_console = RichConsole(file=buf, force_terminal=False, width=120)
        tr = TerminalReporter(config=self.config, console=string_console)
        tr.render(report)
        self._records.append(buf.getvalue())
        self._flush()

    def _flush(self) -> None:
        sections = "\n<hr/>\n".join(
            f"<pre>{escape(r)}</pre>" for r in self._records
        )
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>Beacon Failure Report</title>
  <style>
    body {{ background: #1e1e2e; color: #cdd6f4; font-family: monospace; padding: 2rem; }}
    pre {{ background: #181825; padding: 1rem; border-radius: 6px; overflow-x: auto; }}
    hr {{ border-color: #45475a; }}
  </style>
</head>
<body>
<h1 style="color:#f38ba8">🔦 Beacon Failure Report</h1>
{sections}
</body>
</html>"""
        self.path.write_text(html, encoding="utf-8")


# ── Fan-out ───────────────────────────────────────────────────────────────────


def render_failure(report: FailureReport, config: Optional[BeaconConfig] = None) -> None:
    """
    Render *report* to all configured output sinks.

    This is the single public entry-point called by ``core.py``.
    """
    cfg = config or get_config()
    formats = cfg.output_formats

    if "terminal" in formats:
        try:
            TerminalReporter(cfg).render(report)
        except Exception:  # noqa: BLE001
            pass  # Never let the reporter break the test run

    if "json" in formats:
        try:
            JsonReporter(cfg).render(report)
        except Exception:  # noqa: BLE001
            pass

    if "html" in formats:
        try:
            HtmlReporter(cfg).render(report)
        except Exception:  # noqa: BLE001
            pass


# ── Helpers ───────────────────────────────────────────────────────────────────


def _shorten_path(path: str, max_parts: int = 4) -> str:
    """Return the last *max_parts* segments of *path*."""
    parts = Path(path).parts
    if len(parts) <= max_parts:
        return path
    return os.path.join("…", *parts[-max_parts:])


def _truncated_repr(value: Any, max_len: int = 120) -> str:
    r = safe_repr(value, max_len)
    return r


def _smart_repr(value: Any, max_len: int) -> str:
    """
    Produce a representation of *value* tuned for the locals table.

    • numpy arrays: shape + dtype + first few elements
    • DataFrames: shape + column names
    • Others: truncated repr
    """
    kind = classify_object(value)
    if kind == "ndarray":
        try:
            import numpy as np

            arr = value
            preview = np.array2string(arr.flat[:5] if arr.size > 5 else arr, precision=4)
            suffix = "…" if arr.size > 5 else ""
            return f"ndarray[{arr.shape}, {arr.dtype}] {preview}{suffix}"
        except Exception:  # noqa: BLE001
            pass
    if kind == "dataframe":
        try:
            cols = list(value.columns)[:6]
            suffix = "…" if len(value.columns) > 6 else ""
            return f"DataFrame[{value.shape}] cols={cols}{suffix}"
        except Exception:  # noqa: BLE001
            pass
    return safe_repr(value, max_len)
