"""
tests/conftest.py
─────────────────
Shared fixtures for the Beacon test suite.
"""
from __future__ import annotations

import sys
from typing import Any, Generator, Optional, Tuple

import pytest

import beacon
from beacon.config import BeaconConfig, reset_config
from beacon.reporters import FailureReport


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register --run-examples before test collection."""
    parser.addoption(
        "--run-examples",
        action="store_true",
        default=False,
        help="Run the intentionally-failing example tests to see Beacon output",
    )


@pytest.fixture(autouse=True)
def reset_beacon_config() -> Generator[None, None, None]:
    """Ensure config singleton is reset between tests."""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def cfg() -> BeaconConfig:
    """Return a BeaconConfig with terminal output only."""
    c = BeaconConfig()
    c.show_locals = True
    c.show_source = True
    c.show_diff = True
    c.output_formats = ["terminal"]
    return c


@pytest.fixture
def silent_cfg() -> BeaconConfig:
    """BeaconConfig that suppresses all output (for unit tests of internals)."""
    c = BeaconConfig()
    c.output_formats = []  # no output sinks
    return c


def make_exc_info(
    exc_type: type = AssertionError,
    message: str = "test failure",
) -> Tuple[type, BaseException, Any]:
    """Helper to produce a real exc_info triple."""
    try:
        raise exc_type(message)
    except exc_type:
        return sys.exc_info()  # type: ignore[return-value]
