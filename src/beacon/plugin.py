"""
beacon.plugin
─────────────
pytest plugin registration and hooks.

Beacon integrates at the pytest hook layer with **zero impact on passing tests**.

Hooks used
──────────
pytest_configure          — register the plugin, load config
pytest_runtest_logreport  — intercept failures before pytest's own formatter
pytest_runtest_makereport — capture exc_info while the frame is still live
pytest_collection_finish  — banner / summary
pytest_terminal_summary   — brief beacon status line

The plugin **does not suppress** pytest's native output — both run together.
Beacon adds a rich panel *before* pytest's standard traceback section.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, Generator, Optional

import pytest

from .config import BeaconConfig, get_config
from .core import capture_failure
from .reporters import FailureReport, render_failure

# ── Plugin class ──────────────────────────────────────────────────────────────


class BeaconPlugin:
    """
    The main pytest plugin object.

    Pytest discovers this via the ``pytest11`` entry-point in pyproject.toml.
    """

    def __init__(self, config: BeaconConfig) -> None:
        self.config = config
        # Maps node_id → FailureReport for the summary hook
        self._reports: Dict[str, FailureReport] = {}
        self._failure_count = 0

    # ── Configuration ─────────────────────────────────────────────────────────

    @pytest.hookimpl(trylast=True)
    def pytest_configure(self, config: pytest.Config) -> None:
        # Expose a reference to ourselves on the pytest config object
        # so other plugins/fixtures can access it
        if not hasattr(config, "_beacon"):
            config._beacon = self  # type: ignore[attr-defined]

    # ── Per-test failure capture ───────────────────────────────────────────────

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item: pytest.Item) -> Generator[None, None, None]:
        """
        Wrap the test call phase.

        We use hookwrapper to intercept the exception while the original
        frame (and thus local variables) are still alive on the stack.
        """
        outcome: Any = yield

        # Only act on actual failures.
        # pytest ≥ 7.4 / pluggy ≥ 1.3: outcome.excinfo is a raw
        # (type, value, traceback) tuple when present (not ExceptionInfo).
        exc_info = outcome.excinfo
        if exc_info is None:
            return

        # Support both the old ExceptionInfo object (pytest < 7.4) and the
        # raw tuple form (pytest ≥ 7.4 hookwrapper style).
        if isinstance(exc_info, tuple):
            exc_type, exc_value, tb = exc_info
        else:
            # Legacy: _pytest.python.ExceptionInfo-like object
            exc_type, exc_value, tb = exc_info.type, exc_info.value, exc_info.tb

        if exc_type is None:
            return

        # Never process pytest's own control-flow exceptions (skip, xfail, etc.)
        try:
            from _pytest.outcomes import OutcomeException  # type: ignore[import-untyped]
            if issubclass(exc_type, OutcomeException):
                return
        except ImportError:
            pass

        # Resolve test function
        test_fn = getattr(item, "function", None)

        # Parameterized ID (e.g. "[x=1-y=2]")
        param_id: Optional[str] = None
        if hasattr(item, "callspec"):
            param_id = item.callspec.id  # type: ignore[attr-defined]

        report = capture_failure(
            (exc_type, exc_value, tb),
            test_id=item.nodeid,
            test_function=test_fn,
            param_id=param_id,
            config=self.config,
        )

        render_failure(report, config=self.config)
        self._reports[item.nodeid] = report
        self._failure_count += 1

    # ── Terminal summary ───────────────────────────────────────────────────────

    def pytest_terminal_summary(
        self,
        terminalreporter: Any,
        exitstatus: int,
        config: pytest.Config,
    ) -> None:
        if self._failure_count == 0:
            return
        from rich.console import Console
        from rich.text import Text

        console = Console(stderr=True)
        t = Text()
        t.append("\n🔦 Beacon", style="bold cyan")
        t.append(f" captured {self._failure_count} failure", style="white")
        if self._failure_count != 1:
            t.append("s", style="white")
        t.append(" — see panels above for details.\n", style="dim")
        console.print(t)


# ── Plugin factory / registration ─────────────────────────────────────────────


def pytest_configure(config: pytest.Config) -> None:
    """
    Entry-point hook called by pytest on startup.

    This function is invoked because ``beacon.plugin`` is registered
    as a ``pytest11`` entry-point.
    """
    beacon_cfg = get_config()
    plugin = BeaconPlugin(beacon_cfg)
    config.pluginmanager.register(plugin, "beacon_plugin")


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def beacon_config() -> BeaconConfig:
    """
    Pytest fixture that exposes the BeaconConfig singleton.

    Can be overridden per-test::

        def test_something(beacon_config):
            beacon_config.show_locals = False
            ...
    """
    return get_config()


@pytest.fixture
def beacon_note() -> Any:
    """
    Pytest fixture for attaching inline notes from within a test body.

    Usage::

        def test_kelly(beacon_note):
            beacon_note("Kelly fraction must stay in [0, 1]")
            assert 0 <= kelly(mu=0.05, sigma=0.2) <= 1
    """
    from .annotations import push_annotation, pop_annotation, clear_annotations

    notes_pushed: list[str] = []

    def _note(message: str) -> None:
        push_annotation(message)
        notes_pushed.append(message)

    yield _note  # type: ignore[misc]

    # Clean up any notes left on the stack
    for _ in notes_pushed:
        pop_annotation()
