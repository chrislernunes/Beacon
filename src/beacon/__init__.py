"""
beacon
──────
Beautiful, self-documenting test failure messages.

A modern successor to Two Sigma's marbles library.

Quick start
───────────
Just install beacon and run pytest — the plugin auto-activates::

    pip install beacon
    pytest                    # your existing tests, now with rich output

Rich assertion helpers::

    import beacon

    beacon.assert_equal(result, expected)
    beacon.assert_almost_equal(price, 100.0, rtol=1e-4)
    beacon.assert_frame_equal(df_actual, df_expected)
    beacon.assert_raises(ValueError, fn, bad_input)

Author annotations::

    @beacon.note("Kelly fraction must be bounded to [0, 1]")
    def test_kelly_bounded():
        f = kelly(mu=0.05, sigma=0.2)
        assert 0.0 <= f <= 1.0

    def test_with_context():
        with beacon.annotate("z-score must be bounded post-winsorisation"):
            z = winsorise(raw_z)
            assert -3.0 <= z <= 3.0
"""

from ._version import __version__

# ── Public API ────────────────────────────────────────────────────────────────

from .annotations import annotate, note

from .core import (
    assert_almost_equal,
    assert_array_equal,
    assert_equal,
    assert_false,
    assert_frame_equal,
    assert_in,
    assert_is_none,
    assert_is_not_none,
    assert_not_equal,
    assert_not_in,
    assert_raises,
    assert_true,
    capture_failure,
)

from .config import BeaconConfig, get_config, reset_config

from .reporters import FailureReport, render_failure

__all__ = [
    # Version
    "__version__",
    # Annotations
    "note",
    "annotate",
    # Assertion helpers
    "assert_equal",
    "assert_not_equal",
    "assert_almost_equal",
    "assert_true",
    "assert_false",
    "assert_in",
    "assert_not_in",
    "assert_is_none",
    "assert_is_not_none",
    "assert_raises",
    "assert_frame_equal",
    "assert_array_equal",
    # Core
    "capture_failure",
    # Config
    "BeaconConfig",
    "get_config",
    "reset_config",
    # Reports
    "FailureReport",
    "render_failure",
]
