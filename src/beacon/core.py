"""
beacon.core
───────────
The central engine of Beacon.

Responsibilities
────────────────
1. ``capture_failure(exc_info, frame, ...)``
   Assemble a FailureReport from raw exception info and optional context.
   Called by the pytest plugin (plugin.py) and the rich assertion helpers below.

2. Rich assertion helpers
   ``assert_equal``, ``assert_not_equal``, ``assert_almost_equal``,
   ``assert_raises``, ``assert_true``, ``assert_false``, ``assert_in``,
   ``assert_not_in``, ``assert_is_none``, ``assert_is_not_none``.

   Each helper captures rich context on failure and surfaces it immediately —
   no waiting for the test runner to reformat.

3. ``assert_frame_equal``, ``assert_array_equal``
   Pandas- and NumPy-aware equivalents with structured diffs.

Design goals
────────────
• Zero overhead on passing tests.
• Every helper is composable and can be imported à-la-carte.
• The helpers work outside of pytest (plain unittest, scripts, notebooks).
"""
from __future__ import annotations

import inspect
import sys
import traceback
import types
from typing import Any, Callable, Optional, Tuple, Type, TypeVar, Union

from .annotations import collect_notes
from .config import BeaconConfig, get_config
from .reporters import FailureReport, render_failure
from .rewrite import introspect_assertion
from .utils import filter_locals, get_source_lines

E = TypeVar("E", bound=BaseException)

# ── FailureReport assembly ────────────────────────────────────────────────────


def capture_failure(
    exc_info: Tuple[Optional[type], Optional[BaseException], Any],
    *,
    calling_frame: Optional[types.FrameType] = None,
    test_id: str = "",
    test_function: Optional[Callable[..., Any]] = None,
    param_id: Optional[str] = None,
    config: Optional[BeaconConfig] = None,
) -> FailureReport:
    """
    Assemble a FailureReport from a raw exc_info triple.

    Parameters
    ----------
    exc_info:
        The (type, value, traceback) triple from sys.exc_info().
    calling_frame:
        The frame where the assertion lives. If None, inferred from traceback.
    test_id:
        pytest node ID (e.g. ``tests/test_math.py::test_add``).
    test_function:
        The actual test callable (used to collect @beacon.note annotations).
    param_id:
        Parameterized test ID suffix, if applicable.
    config:
        Override config; defaults to the global singleton.
    """
    cfg = config or get_config()
    exc_type, exc_value, tb = exc_info
    report = FailureReport()

    # ── Identity ──────────────────────────────────────────────────────────────
    report.test_id = test_id
    report.param_id = param_id
    report.test_function = test_function.__name__ if test_function else ""

    # ── Exception ─────────────────────────────────────────────────────────────
    report.exc_type = exc_type.__name__ if exc_type else "UnknownError"
    if exc_value is not None:
        msg = str(exc_value).strip()
        report.exc_message = msg if msg else ""

    # ── Resolve the innermost frame ───────────────────────────────────────────
    frame = calling_frame
    if frame is None and tb is not None:
        innermost = tb
        while innermost.tb_next is not None:
            innermost = innermost.tb_next
        frame = innermost.tb_frame

    if frame is not None:
        report.test_file = frame.f_code.co_filename
        report.test_lineno = frame.f_lineno

    # ── AST breakdown (only for AssertionError) ───────────────────────────────
    if exc_type is AssertionError and tb is not None:
        try:
            report.breakdown = introspect_assertion(exc_info)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            pass  # Never crash the reporter

    # ── Source snippet ────────────────────────────────────────────────────────
    if cfg.show_source and frame is not None:
        src = get_source_lines(
            report.test_file,
            report.test_lineno,
            context=cfg.source_context_lines,
        )
        if src is not None:
            lines, first_lineno, failing_lineno = src
            report.source_lines = lines
            report.source_first_lineno = first_lineno
            report.source_failing_lineno = failing_lineno

    # ── Local variables ───────────────────────────────────────────────────────
    if cfg.show_locals and frame is not None:
        raw_locals = {**frame.f_locals}
        report.local_vars = filter_locals(
            raw_locals,
            exclude_patterns=cfg.locals_exclude_patterns,
            max_vars=cfg.max_locals,
        )

    # ── Author notes ──────────────────────────────────────────────────────────
    report.notes = collect_notes(test_function)

    return report


# ── Internal: raise with beacon output ───────────────────────────────────────


def _beacon_raise(
    exc: AssertionError,
    config: Optional[BeaconConfig] = None,
) -> None:
    """
    Render the failure report for *exc* (which must be the current exception)
    and then re-raise.

    Called from the rich assertion helpers.
    """
    ei: Tuple[Optional[type], Optional[BaseException], Any] = sys.exc_info()
    # If sys.exc_info is empty (shouldn't happen, but be safe)
    if ei[0] is None:
        ei = (AssertionError, exc, None)

    cfg = config or get_config()

    # Walk up to the caller's frame (skip _beacon_raise + the helper itself)
    caller_frame: Optional[types.FrameType] = sys._getframe(2)

    report = capture_failure(
        ei,
        calling_frame=caller_frame,
        config=cfg,
    )
    render_failure(report, config=cfg)
    raise exc from None


# ── Rich assertion helpers ────────────────────────────────────────────────────


def assert_equal(
    actual: Any,
    expected: Any,
    msg: Optional[str] = None,
    *,
    config: Optional[BeaconConfig] = None,
) -> None:
    """
    Assert that ``actual == expected``, with a rich diff on failure.

    Parameters
    ----------
    actual:
        The value produced by the code under test.
    expected:
        The expected / reference value.
    msg:
        Optional human-readable message appended to the failure output.
    """
    if actual == expected:
        return
    label = msg or f"{actual!r} != {expected!r}"
    exc = AssertionError(label)
    try:
        raise exc
    except AssertionError:
        _beacon_raise(exc, config=config)


def assert_not_equal(
    actual: Any,
    expected: Any,
    msg: Optional[str] = None,
    *,
    config: Optional[BeaconConfig] = None,
) -> None:
    """Assert that ``actual != expected``."""
    if actual != expected:
        return
    label = msg or f"Expected values to differ, but both are {actual!r}"
    exc = AssertionError(label)
    try:
        raise exc
    except AssertionError:
        _beacon_raise(exc, config=config)


def assert_almost_equal(
    actual: float,
    expected: float,
    *,
    rtol: float = 1e-7,
    atol: float = 0.0,
    msg: Optional[str] = None,
    config: Optional[BeaconConfig] = None,
) -> None:
    """
    Assert that ``|actual - expected| <= atol + rtol * |expected|``.

    Uses the same convention as ``numpy.testing.assert_allclose``.
    """
    diff = abs(actual - expected)
    threshold = atol + rtol * abs(expected)
    if diff <= threshold:
        return
    label = msg or (
        f"{actual!r} is not close to {expected!r}  "
        f"(|Δ| = {diff:.6g}, threshold = {threshold:.6g})"
    )
    exc = AssertionError(label)
    try:
        raise exc
    except AssertionError:
        _beacon_raise(exc, config=config)


def assert_true(
    value: Any,
    msg: Optional[str] = None,
    *,
    config: Optional[BeaconConfig] = None,
) -> None:
    """Assert that ``bool(value)`` is True."""
    if value:
        return
    label = msg or f"Expected truthy value, got {value!r} ({type(value).__name__})"
    exc = AssertionError(label)
    try:
        raise exc
    except AssertionError:
        _beacon_raise(exc, config=config)


def assert_false(
    value: Any,
    msg: Optional[str] = None,
    *,
    config: Optional[BeaconConfig] = None,
) -> None:
    """Assert that ``bool(value)`` is False."""
    if not value:
        return
    label = msg or f"Expected falsy value, got {value!r} ({type(value).__name__})"
    exc = AssertionError(label)
    try:
        raise exc
    except AssertionError:
        _beacon_raise(exc, config=config)


def assert_in(
    member: Any,
    container: Any,
    msg: Optional[str] = None,
    *,
    config: Optional[BeaconConfig] = None,
) -> None:
    """Assert that ``member in container``."""
    if member in container:
        return
    label = msg or f"{member!r} not found in {type(container).__name__}"
    exc = AssertionError(label)
    try:
        raise exc
    except AssertionError:
        _beacon_raise(exc, config=config)


def assert_not_in(
    member: Any,
    container: Any,
    msg: Optional[str] = None,
    *,
    config: Optional[BeaconConfig] = None,
) -> None:
    """Assert that ``member not in container``."""
    if member not in container:
        return
    label = msg or f"{member!r} unexpectedly found in {type(container).__name__}"
    exc = AssertionError(label)
    try:
        raise exc
    except AssertionError:
        _beacon_raise(exc, config=config)


def assert_is_none(
    value: Any,
    msg: Optional[str] = None,
    *,
    config: Optional[BeaconConfig] = None,
) -> None:
    """Assert that ``value is None``."""
    if value is None:
        return
    label = msg or f"Expected None, got {value!r} ({type(value).__name__})"
    exc = AssertionError(label)
    try:
        raise exc
    except AssertionError:
        _beacon_raise(exc, config=config)


def assert_is_not_none(
    value: Any,
    msg: Optional[str] = None,
    *,
    config: Optional[BeaconConfig] = None,
) -> None:
    """Assert that ``value is not None``."""
    if value is not None:
        return
    label = msg or "Expected a non-None value, got None"
    exc = AssertionError(label)
    try:
        raise exc
    except AssertionError:
        _beacon_raise(exc, config=config)


def assert_raises(
    exc_type: Type[E],
    callable_: Callable[..., Any],
    /,
    *args: Any,
    msg: Optional[str] = None,
    config: Optional[BeaconConfig] = None,
    **kwargs: Any,
) -> E:
    """
    Assert that calling ``callable_(*args, **kwargs)`` raises ``exc_type``.

    Returns the caught exception so callers can inspect it further.

    ::

        exc = beacon.assert_raises(ValueError, int, "not-a-number")
        assert "invalid literal" in str(exc)
    """
    try:
        callable_(*args, **kwargs)
    except exc_type as caught:
        return caught  # type: ignore[return-value]
    except Exception as other:
        label = (
            msg
            or f"Expected {exc_type.__name__}, but got {type(other).__name__}: {other}"
        )
        exc = AssertionError(label)
        try:
            raise exc
        except AssertionError:
            _beacon_raise(exc, config=config)
            raise  # unreachable — satisfies type checker
    else:
        label = msg or f"Expected {exc_type.__name__} to be raised, but no exception was raised"
        exc = AssertionError(label)
        try:
            raise exc
        except AssertionError:
            _beacon_raise(exc, config=config)
            raise  # unreachable


# ── Pandas & NumPy ────────────────────────────────────────────────────────────


def assert_frame_equal(
    actual: Any,
    expected: Any,
    *,
    check_dtype: bool = True,
    check_index: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    msg: Optional[str] = None,
    config: Optional[BeaconConfig] = None,
) -> None:
    """
    Assert that two pandas DataFrames are equal within tolerances.

    Uses ``pandas.testing.assert_frame_equal`` under the hood but intercepts
    the failure to produce a Beacon-formatted diff.
    """
    try:
        import pandas as pd
        import pandas.testing as pdt
    except ImportError as e:
        raise RuntimeError(
            "assert_frame_equal requires pandas. Install it with: pip install pandas"
        ) from e

    try:
        pdt.assert_frame_equal(
            actual,
            expected,
            check_dtype=check_dtype,
            check_index_type=check_index,
            rtol=rtol,
            atol=atol,
        )
    except AssertionError as pandas_exc:
        label = msg or str(pandas_exc)
        exc = AssertionError(label)
        try:
            raise exc
        except AssertionError:
            _beacon_raise(exc, config=config)


def assert_array_equal(
    actual: Any,
    expected: Any,
    *,
    rtol: float = 1e-7,
    atol: float = 0.0,
    msg: Optional[str] = None,
    config: Optional[BeaconConfig] = None,
) -> None:
    """
    Assert that two numpy arrays are element-wise equal within tolerances.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise RuntimeError(
            "assert_array_equal requires numpy. Install it with: pip install numpy"
        ) from e

    try:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as np_exc:
        label = msg or str(np_exc)
        exc = AssertionError(label)
        try:
            raise exc
        except AssertionError:
            _beacon_raise(exc, config=config)
