"""
tests/test_examples.py
──────────────────────
Showcase / documentation tests.

These tests are designed to FAIL intentionally when run directly,
demonstrating the quality of Beacon's output. They are skipped in CI
unless you pass --run-examples.

To see Beacon's output, run:

    pytest tests/test_examples.py --run-examples -v

Each test has been carefully crafted to show a different Beacon feature.
"""
from __future__ import annotations

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def run_examples(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--run-examples", default=False))


def _skip_unless_examples(request: pytest.FixtureRequest) -> None:
    if not request.config.getoption("--run-examples", default=False):
        pytest.skip("Pass --run-examples to run showcase tests")


# ─────────────────────────────────────────────────────────────────────────────
# Feature 1: Simple scalar comparison
# ─────────────────────────────────────────────────────────────────────────────


def test_scalar_comparison(request: pytest.FixtureRequest) -> None:
    """Beacon shows LHS/RHS values and a Δ for numeric diffs."""
    _skip_unless_examples(request)

    expected_pnl = 1_250_000.0
    actual_pnl = 1_187_432.57

    assert actual_pnl == expected_pnl


# ─────────────────────────────────────────────────────────────────────────────
# Feature 2: Dict comparison
# ─────────────────────────────────────────────────────────────────────────────


def test_dict_comparison(request: pytest.FixtureRequest) -> None:
    """Beacon renders a DeepDiff table for dict mismatches."""
    _skip_unless_examples(request)

    expected = {
        "symbol": "NIFTY",
        "strike": 22000,
        "expiry": "2024-12-26",
        "delta": 0.52,
        "gamma": 0.004,
    }
    actual = {
        "symbol": "NIFTY",
        "strike": 22000,
        "expiry": "2024-12-26",
        "delta": 0.48,  # wrong
        "gamma": 0.004,
        "vega": 12.3,  # extra key
    }

    assert actual == expected


# ─────────────────────────────────────────────────────────────────────────────
# Feature 3: Author notes via decorator
# ─────────────────────────────────────────────────────────────────────────────

import beacon


@beacon.note("Kelly fraction must be bounded to [0, 1] for all valid inputs.")
@beacon.note("This is critical for risk management — unconstrained Kelly can blow up capital.")
def test_kelly_bounded(request: pytest.FixtureRequest) -> None:
    """Notes surface in the failure panel, providing instant context."""
    _skip_unless_examples(request)

    def unconstrained_kelly(mu: float, sigma: float) -> float:
        return mu / (sigma**2)

    # Deliberately using unconstrained Kelly which can exceed 1
    f = unconstrained_kelly(mu=0.5, sigma=0.3)  # returns ~5.6
    assert 0.0 <= f <= 1.0, f"Kelly fraction {f:.4f} out of bounds"


# ─────────────────────────────────────────────────────────────────────────────
# Feature 4: Author notes via context manager
# ─────────────────────────────────────────────────────────────────────────────


def test_inline_annotation(request: pytest.FixtureRequest) -> None:
    """annotate() context manager attaches inline notes to the failure."""
    _skip_unless_examples(request)

    z_scores = [0.1, -0.3, 4.2, 0.8]  # 4.2 is out of bounds

    with beacon.annotate("z-scores must be bounded to [-3, 3] post-winsorisation"):
        for z in z_scores:
            assert -3.0 <= z <= 3.0, f"z-score {z} is out of bounds"


# ─────────────────────────────────────────────────────────────────────────────
# Feature 5: Chained / complex boolean assertion
# ─────────────────────────────────────────────────────────────────────────────


def test_complex_assertion(request: pytest.FixtureRequest) -> None:
    """Beacon breaks down complex boolean expressions."""
    _skip_unless_examples(request)

    bid = 100.50
    ask = 100.45  # crossed market!
    spread = ask - bid

    assert bid < ask and spread > 0, f"Crossed market: bid={bid}, ask={ask}"


# ─────────────────────────────────────────────────────────────────────────────
# Feature 6: Multi-line string diff
# ─────────────────────────────────────────────────────────────────────────────


def test_multiline_string_diff(request: pytest.FixtureRequest) -> None:
    """Beacon renders a coloured unified diff for multi-line string mismatches."""
    _skip_unless_examples(request)

    expected_report = (
        "Portfolio Summary\n"
        "=================\n"
        "Total PnL: $1,250,000\n"
        "Max Drawdown: 4.8%\n"
        "Sharpe Ratio: 1.82\n"
    )
    actual_report = (
        "Portfolio Summary\n"
        "=================\n"
        "Total PnL: $1,187,432\n"
        "Max Drawdown: 6.2%\n"
        "Sharpe Ratio: 1.54\n"
    )

    assert actual_report == expected_report


# ─────────────────────────────────────────────────────────────────────────────
# Feature 7: NumPy array diff
# ─────────────────────────────────────────────────────────────────────────────


def test_numpy_array_diff(request: pytest.FixtureRequest) -> None:
    """Beacon shows shape, dtype, and element-wise diff stats for arrays."""
    pytest.importorskip("numpy")
    _skip_unless_examples(request)

    import numpy as np

    expected_weights = np.array([0.3, 0.3, 0.2, 0.2])
    actual_weights = np.array([0.35, 0.25, 0.25, 0.15])

    beacon.assert_array_equal(actual_weights, expected_weights, rtol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Feature 8: Pandas DataFrame diff
# ─────────────────────────────────────────────────────────────────────────────


def test_dataframe_diff(request: pytest.FixtureRequest) -> None:
    """Beacon shows shape, column, and cell-level diffs for DataFrames."""
    pytest.importorskip("pandas")
    _skip_unless_examples(request)

    import pandas as pd

    expected = pd.DataFrame(
        {
            "symbol": ["NIFTY", "BANKNIFTY", "FINNIFTY"],
            "delta": [0.52, 0.48, 0.51],
            "gamma": [0.004, 0.003, 0.0035],
        }
    )
    actual = pd.DataFrame(
        {
            "symbol": ["NIFTY", "BANKNIFTY", "FINNIFTY"],
            "delta": [0.52, 0.55, 0.51],  # BANKNIFTY delta is wrong
            "gamma": [0.004, 0.003, 0.0040],  # FINNIFTY gamma is wrong
        }
    )

    beacon.assert_frame_equal(actual, expected)


# ─────────────────────────────────────────────────────────────────────────────
# Feature 9: beacon.assert_raises
# ─────────────────────────────────────────────────────────────────────────────


def test_assert_raises_example(request: pytest.FixtureRequest) -> None:
    """beacon.assert_raises catches the expected exception and returns it."""
    _skip_unless_examples(request)

    def validate_strike(strike: float) -> None:
        if strike <= 0:
            raise ValueError(f"Strike must be positive, got {strike}")

    # This will fail because validate_strike raises, but we expect TypeError
    exc = beacon.assert_raises(TypeError, validate_strike, -100.0)


# ─────────────────────────────────────────────────────────────────────────────
# Feature 10: Parameterized tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "price,expected_valid",
    [
        (100.0, True),
        (0.0, False),
        (-50.0, False),
        (1e6, True),
        (float("inf"), False),  # this will fail — inf is not a valid price
    ],
    ids=["normal", "zero", "negative", "large", "inf"],
)
def test_parameterized_price_validation(
    request: pytest.FixtureRequest,
    price: float,
    expected_valid: bool,
) -> None:
    """
    Beacon attaches the parameterized ID to each failure, making it
    immediately clear which parameter set failed.
    """
    _skip_unless_examples(request)

    import math

    def is_valid_price(p: float) -> bool:
        return p > 0 and math.isfinite(p)

    actual_valid = is_valid_price(price)
    assert actual_valid == expected_valid, (
        f"Price {price}: expected valid={expected_valid}, got {actual_valid}"
    )
