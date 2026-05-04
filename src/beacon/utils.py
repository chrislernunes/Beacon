"""
beacon.utils
────────────
Low-level, dependency-light utilities used throughout Beacon.

Design principles:
  • Every function is pure or near-pure (no side effects).
  • Handles arbitrary user objects gracefully — never raises.
  • Zero Rich imports here to keep the module lightweight.
"""
from __future__ import annotations

import fnmatch
import inspect
import sys
import textwrap
import traceback
from typing import Any, Dict, Iterator, List, Optional, Tuple

# ── Type detection helpers ────────────────────────────────────────────────────


def _is_numpy_array(obj: Any) -> bool:
    try:
        import numpy as np  # type: ignore[import-untyped]

        return isinstance(obj, np.ndarray)
    except ImportError:
        return False


def _is_pandas_df(obj: Any) -> bool:
    try:
        import pandas as pd  # type: ignore[import-untyped]

        return isinstance(obj, pd.DataFrame)
    except ImportError:
        return False


def _is_pandas_series(obj: Any) -> bool:
    try:
        import pandas as pd  # type: ignore[import-untyped]

        return isinstance(obj, pd.Series)
    except ImportError:
        return False


def _is_dataclass(obj: Any) -> bool:
    import dataclasses

    return dataclasses.is_dataclass(obj) and not isinstance(obj, type)


# ── Object classification ─────────────────────────────────────────────────────

ObjectKind = str  # "scalar" | "sequence" | "mapping" | "set" | "ndarray" | "dataframe" | "series" | "dataclass" | "other"


def classify_object(obj: Any) -> ObjectKind:
    """Return a human-readable kind label for *obj*."""
    if _is_numpy_array(obj):
        return "ndarray"
    if _is_pandas_df(obj):
        return "dataframe"
    if _is_pandas_series(obj):
        return "series"
    if _is_dataclass(obj):
        return "dataclass"
    if isinstance(obj, dict):
        return "mapping"
    if isinstance(obj, (list, tuple)):
        return "sequence"
    if isinstance(obj, (set, frozenset)):
        return "set"
    if isinstance(obj, (int, float, complex, bool, str, bytes, type(None))):
        return "scalar"
    return "other"


# ── Safe repr ────────────────────────────────────────────────────────────────


def safe_repr(obj: Any, max_length: int = 500) -> str:
    """
    Return a repr string for *obj*, guaranteed not to raise.

    Truncates at *max_length* characters and appends '…' if needed.
    """
    try:
        r = repr(obj)
    except Exception:  # noqa: BLE001
        try:
            r = f"<{type(obj).__name__} [repr failed]>"
        except Exception:  # noqa: BLE001
            r = "<object [repr failed]>"
    if len(r) > max_length:
        r = r[:max_length] + "…"
    return r


def safe_type_name(obj: Any) -> str:
    """Return the qualified type name of *obj*, never raises."""
    try:
        cls = type(obj)
        mod = cls.__module__
        name = cls.__qualname__
        if mod and mod not in ("builtins", "__main__"):
            return f"{mod}.{name}"
        return name
    except Exception:  # noqa: BLE001
        return "<unknown type>"


# ── Local variable filtering ──────────────────────────────────────────────────


def _matches_any_pattern(name: str, patterns: List[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def _is_boring_value(obj: Any) -> bool:
    """Return True for values that are almost never interesting in test output."""
    # Module objects, classes, functions — rarely the source of truth in tests
    return inspect.ismodule(obj) or inspect.isclass(obj) or inspect.isbuiltin(obj)


def _is_large(obj: Any, max_items: int = 200) -> bool:
    """Heuristic: is this object too large to render usefully?"""
    try:
        if _is_numpy_array(obj) and obj.size > max_items:
            return True
        if _is_pandas_df(obj) and (obj.shape[0] * obj.shape[1]) > max_items:
            return True
        if isinstance(obj, (list, tuple, set, frozenset, dict)) and len(obj) > max_items:
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


def filter_locals(
    local_vars: Dict[str, Any],
    exclude_patterns: List[str],
    max_vars: int = 10,
) -> List[Tuple[str, Any]]:
    """
    Filter *local_vars* down to the most interesting subset.

    Returns a list of (name, value) pairs, ordered so that:
      1. Variables whose names don't start with '_' come first.
      2. Smaller / simpler objects come before large ones.
    """
    result: List[Tuple[str, Any]] = []
    for name, value in local_vars.items():
        if _matches_any_pattern(name, exclude_patterns):
            continue
        if _is_boring_value(value):
            continue
        result.append((name, value))

    # Sort: non-underscore first, then by type simplicity
    def sort_key(pair: Tuple[str, Any]) -> Tuple[int, int]:
        name, value = pair
        prefix = 0 if not name.startswith("_") else 1
        large = 1 if _is_large(value) else 0
        return (prefix, large)

    result.sort(key=sort_key)
    return result[:max_vars]


# ── Source extraction ─────────────────────────────────────────────────────────


def get_source_lines(
    filename: str,
    lineno: int,
    context: int = 4,
) -> Optional[Tuple[List[str], int, int]]:
    """
    Return (lines, first_lineno, failing_lineno) for the source file,
    or None if the source cannot be read.

    *lines* are the raw source lines (with newlines) for the context window.
    """
    try:
        with open(filename, encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        start = max(0, lineno - 1 - context)
        end = min(len(all_lines), lineno + context)
        return all_lines[start:end], start + 1, lineno
    except (OSError, TypeError):
        return None


# ── Traceback helpers ─────────────────────────────────────────────────────────


def extract_assertion_frame(
    tb: Optional[Any],
) -> Optional[inspect.FrameInfo]:
    """
    Walk the traceback to find the innermost frame that contains an
    assert statement (heuristic: the last user-visible frame).
    """
    if tb is None:
        return None
    frames: List[inspect.FrameInfo] = []
    current = tb
    while current is not None:
        frame = current.tb_frame
        info = inspect.getframeinfo(frame)
        frames.append(  # type: ignore[arg-type]
            inspect.FrameInfo(
                frame=frame,
                filename=info.filename,
                lineno=current.tb_lineno,
                function=info.function,
                code_context=info.code_context,
                index=info.index,
            )
        )
        current = current.tb_next
    # Return the last (innermost) frame
    return frames[-1] if frames else None


def format_exception_only(exc: BaseException) -> str:
    """Return a concise one-line exception summary."""
    lines = traceback.format_exception_only(type(exc), exc)
    return "".join(lines).strip()


# ── Diff helpers ──────────────────────────────────────────────────────────────


def unified_text_diff(expected: str, actual: str, n_context: int = 3) -> List[str]:
    """Return unified diff lines between two multi-line strings."""
    import difflib

    return list(
        difflib.unified_diff(
            expected.splitlines(keepends=True),
            actual.splitlines(keepends=True),
            fromfile="expected",
            tofile="actual",
            n=n_context,
        )
    )


def compute_deep_diff(
    expected: Any,
    actual: Any,
) -> Optional[Dict[str, Any]]:
    """
    Return a DeepDiff result dict, or None if deepdiff is unavailable
    or the objects are not diffable.
    """
    try:
        from deepdiff import DeepDiff  # type: ignore[import-untyped]

        diff = DeepDiff(expected, actual, ignore_order=False, verbose_level=2)
        return dict(diff) if diff else None
    except Exception:  # noqa: BLE001
        return None


# ── String helpers ────────────────────────────────────────────────────────────


def indent(text: str, prefix: str = "  ") -> str:
    return textwrap.indent(text, prefix)


def pluralize(count: int, singular: str, plural: Optional[str] = None) -> str:
    plural = plural or singular + "s"
    return f"{count} {singular if count == 1 else plural}"


def truncate_middle(s: str, max_len: int = 120) -> str:
    """Truncate a long string in the middle, preserving start and end."""
    if len(s) <= max_len:
        return s
    half = (max_len - 3) // 2
    return s[:half] + "…" + s[-half:]


# ── Numeric helpers ───────────────────────────────────────────────────────────


def format_numeric_diff(
    expected: Any,
    actual: Any,
) -> Optional[str]:
    """
    For numeric scalars/arrays, compute and format the absolute and relative
    difference. Returns None if not applicable.
    """
    try:
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            abs_diff = abs(actual - expected)
            if expected != 0:
                rel_diff = abs_diff / abs(expected)
                return f"Δ = {abs_diff:.6g}  ({rel_diff:.2%} relative)"
            return f"Δ = {abs_diff:.6g}"
    except Exception:  # noqa: BLE001
        pass

    # numpy scalar comparison
    if _is_numpy_array(expected) or _is_numpy_array(actual):
        try:
            import numpy as np

            e = np.asarray(expected, dtype=float)
            a = np.asarray(actual, dtype=float)
            abs_diff = np.abs(a - e)
            return (
                f"max|Δ| = {float(np.max(abs_diff)):.6g}, "
                f"mean|Δ| = {float(np.mean(abs_diff)):.6g}"
            )
        except Exception:  # noqa: BLE001
            pass

    return None
