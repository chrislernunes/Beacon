"""
beacon.annotations
──────────────────
Provides the @beacon.note decorator and the beacon.annotate() context manager,
allowing test authors to attach rich, human-readable context to assertions.

These annotations are captured at test collection time and surfaced in the
failure report alongside the assertion that failed.

Usage
-----
    import beacon

    @beacon.note("This tests that negative prices are rejected at the order router level.")
    def test_negative_price_rejected():
        order = Order(price=-1.0)
        beacon.assert_raises(ValueError, router.submit, order)


    def test_signal_calculation():
        with beacon.annotate("z-score should be bounded after winsorisation"):
            signal = compute_signal(raw_prices)
            assert -3.0 <= signal <= 3.0
"""
from __future__ import annotations

import functools
import threading
from contextlib import contextmanager
from typing import Any, Callable, Generator, List, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# ── Thread-local annotation stack ─────────────────────────────────────────────

_local = threading.local()


def _get_stack() -> List[str]:
    if not hasattr(_local, "annotations"):
        _local.annotations = []
    return _local.annotations  # type: ignore[no-any-return]


def push_annotation(note: str) -> None:
    """Push a note onto the current-thread annotation stack."""
    _get_stack().append(note)


def pop_annotation() -> Optional[str]:
    """Pop the topmost annotation, or return None if the stack is empty."""
    stack = _get_stack()
    return stack.pop() if stack else None


def get_annotations() -> List[str]:
    """Return a snapshot of the current annotation stack (LIFO order)."""
    return list(reversed(_get_stack()))


def clear_annotations() -> None:
    """Clear all annotations for the current thread."""
    _get_stack().clear()


# ── Public API ────────────────────────────────────────────────────────────────


def note(message: str) -> Callable[[F], F]:
    """
    Decorator that attaches a note to a test function.

    The note is injected as a ``_beacon_notes`` attribute on the function
    and is picked up by the Beacon pytest plugin at report time.

    Parameters
    ----------
    message:
        A human-readable description of *why* this test exists or what it
        is specifically verifying — think of it as the test's docstring
        surfaced directly in the failure report.

    Example
    -------
    ::

        @beacon.note("Ensures that the Kelly fraction is bounded to [0, 1].")
        def test_kelly_fraction_bounded():
            f = kelly_fraction(mu=0.05, sigma=0.20)
            assert 0.0 <= f <= 1.0
    """

    def decorator(fn: F) -> F:
        existing: List[str] = getattr(fn, "_beacon_notes", [])
        existing.append(message)
        fn._beacon_notes = existing  # type: ignore[attr-defined]

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        wrapper._beacon_notes = existing  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


@contextmanager
def annotate(message: str) -> Generator[None, None, None]:
    """
    Context manager that attaches an inline note to assertions within
    the ``with`` block.

    Automatically cleans up even if the block raises.

    Example
    -------
    ::

        def test_portfolio_greeks():
            with beacon.annotate("Delta must be in (-1, 1) for a vanilla option"):
                delta = option.delta()
                assert -1.0 < delta < 1.0
    """
    push_annotation(message)
    try:
        yield
    finally:
        pop_annotation()


class TestNote:
    """
    Structured container for a single author annotation.

    Attributes
    ----------
    message: str
        The annotation text.
    source: str
        Where the annotation came from: ``"decorator"`` or ``"context_manager"``.
    """

    __slots__ = ("message", "source")

    def __init__(self, message: str, source: str = "unknown") -> None:
        self.message = message
        self.source = source

    def __repr__(self) -> str:
        return f"TestNote({self.message!r}, source={self.source!r})"


def collect_notes(test_fn: Optional[Callable[..., Any]]) -> List[TestNote]:
    """
    Collect all annotations for *test_fn*: decorator notes + live stack notes.

    Called by the reporter at failure time.
    """
    notes: List[TestNote] = []

    # Notes from @beacon.note decorator
    if test_fn is not None:
        for msg in getattr(test_fn, "_beacon_notes", []):
            notes.append(TestNote(msg, source="decorator"))

    # Notes from beacon.annotate() context managers (still on stack)
    for msg in get_annotations():
        notes.append(TestNote(msg, source="context_manager"))

    return notes
