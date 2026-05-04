"""
beacon.rewrite
──────────────
AST-based assertion introspection.

Given a failing ``AssertionError`` and its traceback, this module:

1. Locates the exact ``assert`` statement in the source AST.
2. Extracts the full boolean expression that failed.
3. Evaluates sub-expressions where safe to do so, producing a structured
   breakdown: "left OP right", sub-comparisons, function call results, etc.
4. Falls back gracefully at every step — never raises, never loses the
   original exception.

We use the ``executing`` library (same approach as pytest, devtools, icecream)
for reliable frame → AST node mapping.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .utils import safe_repr, safe_type_name

# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class SubExpr:
    """A single evaluated sub-expression within the failing assertion."""

    source: str
    """The source text of this sub-expression."""

    value: Any
    """The runtime value."""

    type_name: str = field(default="")
    """Human-readable type name."""

    def __post_init__(self) -> None:
        if not self.type_name:
            self.type_name = safe_type_name(self.value)

    @property
    def repr(self) -> str:
        return safe_repr(self.value)


@dataclass
class AssertionBreakdown:
    """
    Structured breakdown of a failing assertion.

    Attributes
    ----------
    raw_source : str
        The full source text of the assert statement.
    expression_source : str
        Just the boolean expression (without ``assert`` and optional message).
    message_source : Optional[str]
        The optional assert message, if present.
    sub_expressions : List[SubExpr]
        Evaluated sub-expressions (comparisons, function calls, …).
    lhs : Optional[SubExpr]
        Left-hand side for binary comparisons.
    rhs : Optional[SubExpr]
        Right-hand side for binary comparisons.
    operator : Optional[str]
        Comparison operator as a string (``"=="``, ``"<"``, ``"in"``, …).
    error : Optional[str]
        If introspection partially failed, a note about what was skipped.
    """

    raw_source: str = ""
    expression_source: str = ""
    message_source: Optional[str] = None
    sub_expressions: List[SubExpr] = field(default_factory=list)
    lhs: Optional[SubExpr] = None
    rhs: Optional[SubExpr] = None
    operator: Optional[str] = None
    error: Optional[str] = None

    @property
    def is_comparison(self) -> bool:
        return self.lhs is not None and self.rhs is not None

    @property
    def has_sub_expressions(self) -> bool:
        return bool(self.sub_expressions)


# ── Operator mapping ──────────────────────────────────────────────────────────

_OP_SYMBOLS: Dict[type, str] = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.Is: "is",
    ast.IsNot: "is not",
    ast.In: "in",
    ast.NotIn: "not in",
    ast.And: "and",
    ast.Or: "or",
}


def _op_symbol(op: ast.cmpop) -> str:
    return _OP_SYMBOLS.get(type(op), type(op).__name__)


# ── Safe AST source extraction ────────────────────────────────────────────────


def _node_source(node: ast.AST, full_source: str) -> str:
    """
    Extract source text for *node* from *full_source*.

    Uses ast node line/col info (Python 3.8+).
    """
    try:
        lines = full_source.splitlines()
        # ast.get_source_segment is the cleanest approach
        seg = ast.get_source_segment(full_source, node)
        if seg:
            return seg.strip()
        # Fallback: grab lines manually
        start = node.lineno - 1  # type: ignore[attr-defined]
        end = node.end_lineno  # type: ignore[attr-defined]
        return " ".join(ln.strip() for ln in lines[start:end])
    except Exception:  # noqa: BLE001
        return "<source unavailable>"


# ── Safe eval in frame ────────────────────────────────────────────────────────


def _safe_eval(source: str, frame: types.FrameType) -> Tuple[Any, bool]:
    """
    Evaluate *source* in *frame*'s scope. Returns (value, success).

    Never raises — on any error returns (None, False).
    """
    try:
        value = eval(source, frame.f_globals, frame.f_locals)  # noqa: S307
        return value, True
    except Exception:  # noqa: BLE001
        return None, False


# ── Core introspection ────────────────────────────────────────────────────────


def _try_get_assert_node(
    frame: types.FrameType,
) -> Optional[ast.Assert]:
    """
    Use ``executing`` to map *frame* → AST node.

    Falls back to None if the frame cannot be mapped (e.g. REPL, .pyc only).
    """
    try:
        import executing  # type: ignore[import-untyped]

        ex = executing.Source.executing(frame)
        node = ex.node
        if node is None:
            return None
        # Walk up to the Assert statement
        src = executing.Source.for_frame(frame)
        for n in ast.walk(src.tree):
            if isinstance(n, ast.Assert) and hasattr(n, "lineno"):
                if n.lineno == frame.f_lineno or (
                    hasattr(n, "end_lineno")
                    and n.lineno <= frame.f_lineno <= n.end_lineno  # type: ignore[attr-defined]
                ):
                    return n
        return None
    except Exception:  # noqa: BLE001
        return None


def _extract_assert_source(frame: types.FrameType) -> Optional[str]:
    """Best-effort: return the raw source of the failing assert statement."""
    try:
        lines, start = inspect.getsourcelines(frame)
        # frame.f_lineno is 1-based; start is 1-based too
        offset = frame.f_lineno - start
        if 0 <= offset < len(lines):
            # Multi-line assert: collect until the expression is balanced
            collected = []
            for line in lines[offset:]:
                collected.append(line)
                joined = "".join(collected).strip()
                if joined.startswith("assert"):
                    try:
                        ast.parse(joined)
                        return textwrap.dedent(joined).strip()
                    except SyntaxError:
                        continue
            return "".join(collected[:1]).strip()
    except Exception:  # noqa: BLE001
        pass
    return None


def _breakdown_from_assert_node(
    node: ast.Assert,
    frame: types.FrameType,
    full_source: str,
) -> AssertionBreakdown:
    """Build an AssertionBreakdown from a concrete ast.Assert node."""
    bd = AssertionBreakdown()
    bd.expression_source = _node_source(node.test, full_source)
    if node.msg is not None:
        bd.message_source = _node_source(node.msg, full_source)

    test = node.test

    # ── Case 1: Simple comparison  a == b, a < b, etc. ────────────────────
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        lhs_src = _node_source(test.left, full_source)
        rhs_src = _node_source(test.comparators[0], full_source)
        lhs_val, lhs_ok = _safe_eval(lhs_src, frame)
        rhs_val, rhs_ok = _safe_eval(rhs_src, frame)
        op_str = _op_symbol(test.ops[0])
        bd.operator = op_str
        if lhs_ok:
            bd.lhs = SubExpr(source=lhs_src, value=lhs_val)
        if rhs_ok:
            bd.rhs = SubExpr(source=rhs_src, value=rhs_val)
        # Also add as sub-expressions
        if lhs_ok:
            bd.sub_expressions.append(SubExpr(source=lhs_src, value=lhs_val))
        if rhs_ok:
            bd.sub_expressions.append(SubExpr(source=rhs_src, value=rhs_val))

    # ── Case 2: Chained comparison  a < b < c ────────────────────────────
    elif isinstance(test, ast.Compare) and len(test.ops) > 1:
        all_nodes = [test.left] + list(test.comparators)
        for n in all_nodes:
            src = _node_source(n, full_source)
            val, ok = _safe_eval(src, frame)
            if ok:
                bd.sub_expressions.append(SubExpr(source=src, value=val))

    # ── Case 3: Boolean expression  a and b, not x, etc. ─────────────────
    elif isinstance(test, (ast.BoolOp, ast.UnaryOp)):
        for n in ast.walk(test):
            if isinstance(n, (ast.Name, ast.Attribute, ast.Call, ast.Subscript)):
                src = _node_source(n, full_source)
                if src and src != bd.expression_source:
                    val, ok = _safe_eval(src, frame)
                    if ok:
                        bd.sub_expressions.append(SubExpr(source=src, value=val))

    # ── Case 4: Generic — evaluate the whole expression ───────────────────
    else:
        val, ok = _safe_eval(bd.expression_source, frame)
        if ok:
            bd.sub_expressions.append(SubExpr(source=bd.expression_source, value=val))

    return bd


# ── Public API ────────────────────────────────────────────────────────────────


def introspect_assertion(
    exc_info: Tuple[type, BaseException, Any],
) -> AssertionBreakdown:
    """
    Given the exc_info triple from a caught AssertionError, return a
    structured AssertionBreakdown.

    This is the single entry-point called by the reporter.
    Never raises — returns a minimal breakdown on any failure.
    """
    _exc_type, _exc_value, tb = exc_info
    bd = AssertionBreakdown()

    if tb is None:
        bd.error = "No traceback available"
        return bd

    # Walk to the innermost frame
    innermost_tb = tb
    while innermost_tb.tb_next is not None:
        innermost_tb = innermost_tb.tb_next

    frame: types.FrameType = innermost_tb.tb_frame

    # Try to get raw source first (always useful as fallback)
    raw = _extract_assert_source(frame)
    bd.raw_source = raw or ""

    # Try to get and parse the AST node via executing
    assert_node = _try_get_assert_node(frame)
    if assert_node is None:
        # Fallback: parse the raw source manually
        if raw:
            try:
                tree = ast.parse(raw)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assert):
                        # Get full module source for accurate source segments
                        try:
                            full_src = inspect.getsource(
                                inspect.getmodule(frame) or frame  # type: ignore[arg-type]
                            )
                        except Exception:  # noqa: BLE001
                            full_src = raw
                        return _breakdown_from_assert_node(node, frame, full_src)
            except Exception:  # noqa: BLE001
                pass
        bd.expression_source = raw or ""
        bd.error = "Could not map frame to AST node; showing raw source"
        return bd

    try:
        full_src = inspect.getsource(inspect.getmodule(frame) or frame)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        full_src = raw or ""

    try:
        return _breakdown_from_assert_node(assert_node, frame, full_src)
    except Exception as e:  # noqa: BLE001
        bd.error = f"Breakdown partially failed: {e}"
        return bd
