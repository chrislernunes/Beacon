"""Tests for beacon.core assertion helpers."""
from __future__ import annotations

import pytest

import beacon
from beacon.config import BeaconConfig


# Use a silent config so tests don't pollute terminal output
@pytest.fixture
def cfg() -> BeaconConfig:
    c = BeaconConfig()
    c.output_formats = []  # suppress all output during unit tests
    return c


class TestAssertEqual:
    def test_equal_passes(self, cfg: BeaconConfig) -> None:
        beacon.assert_equal(1, 1, config=cfg)  # no exception

    def test_not_equal_raises(self, cfg: BeaconConfig) -> None:
        with pytest.raises(AssertionError):
            beacon.assert_equal(1, 2, config=cfg)

    def test_equal_lists_passes(self, cfg: BeaconConfig) -> None:
        beacon.assert_equal([1, 2, 3], [1, 2, 3], config=cfg)

    def test_unequal_lists_raises(self, cfg: BeaconConfig) -> None:
        with pytest.raises(AssertionError):
            beacon.assert_equal([1, 2, 3], [1, 2, 4], config=cfg)

    def test_custom_message(self, cfg: BeaconConfig) -> None:
        with pytest.raises(AssertionError, match="custom message"):
            beacon.assert_equal(1, 2, msg="custom message", config=cfg)

    def test_none_equal(self, cfg: BeaconConfig) -> None:
        beacon.assert_equal(None, None, config=cfg)

    def test_string_equal(self, cfg: BeaconConfig) -> None:
        beacon.assert_equal("hello", "hello", config=cfg)

    def test_dicts_equal(self, cfg: BeaconConfig) -> None:
        beacon.assert_equal({"a": 1, "b": 2}, {"a": 1, "b": 2}, config=cfg)


class TestAssertNotEqual:
    def test_not_equal_passes(self, cfg: BeaconConfig) -> None:
        beacon.assert_not_equal(1, 2, config=cfg)

    def test_equal_raises(self, cfg: BeaconConfig) -> None:
        with pytest.raises(AssertionError):
            beacon.assert_not_equal(1, 1, config=cfg)


class TestAssertAlmostEqual:
    def test_close_values_pass(self, cfg: BeaconConfig) -> None:
        beacon.assert_almost_equal(1.0, 1.0 + 1e-10, rtol=1e-7, config=cfg)

    def test_far_values_raise(self, cfg: BeaconConfig) -> None:
        with pytest.raises(AssertionError):
            beacon.assert_almost_equal(1.0, 2.0, rtol=1e-7, atol=0.0, config=cfg)

    def test_atol_respected(self, cfg: BeaconConfig) -> None:
        beacon.assert_almost_equal(0.0, 0.0001, atol=0.001, rtol=0.0, config=cfg)

    def test_exact_equality_passes(self, cfg: BeaconConfig) -> None:
        beacon.assert_almost_equal(42.0, 42.0, config=cfg)

    def test_zero_expected_atol(self, cfg: BeaconConfig) -> None:
        beacon.assert_almost_equal(0.0, 1e-9, atol=1e-8, rtol=0.0, config=cfg)


class TestAssertTrue:
    def test_truthy_passes(self, cfg: BeaconConfig) -> None:
        beacon.assert_true(True, config=cfg)
        beacon.assert_true(1, config=cfg)
        beacon.assert_true("non-empty", config=cfg)
        beacon.assert_true([1], config=cfg)

    def test_falsy_raises(self, cfg: BeaconConfig) -> None:
        with pytest.raises(AssertionError):
            beacon.assert_true(False, config=cfg)

    def test_none_raises(self, cfg: BeaconConfig) -> None:
        with pytest.raises(AssertionError):
            beacon.assert_true(None, config=cfg)

    def test_empty_list_raises(self, cfg: BeaconConfig) -> None:
        with pytest.raises(AssertionError):
            beacon.assert_true([], config=cfg)


class TestAssertFalse:
    def test_falsy_passes(self, cfg: BeaconConfig) -> None:
        beacon.assert_false(False, config=cfg)
        beacon.assert_false(0, config=cfg)
        beacon.assert_false(None, config=cfg)
        beacon.assert_false([], config=cfg)

    def test_truthy_raises(self, cfg: BeaconConfig) -> None:
        with pytest.raises(AssertionError):
            beacon.assert_false(True, config=cfg)


class TestAssertIn:
    def test_member_in_list(self, cfg: BeaconConfig) -> None:
        beacon.assert_in(1, [1, 2, 3], config=cfg)

    def test_member_in_dict_keys(self, cfg: BeaconConfig) -> None:
        beacon.assert_in("key", {"key": "value"}, config=cfg)

    def test_member_in_string(self, cfg: BeaconConfig) -> None:
        beacon.assert_in("hello", "hello world", config=cfg)

    def test_member_not_in_raises(self, cfg: BeaconConfig) -> None:
        with pytest.raises(AssertionError):
            beacon.assert_in(99, [1, 2, 3], config=cfg)

    def test_member_in_set(self, cfg: BeaconConfig) -> None:
        beacon.assert_in(5, {1, 3, 5, 7}, config=cfg)


class TestAssertNotIn:
    def test_not_in_passes(self, cfg: BeaconConfig) -> None:
        beacon.assert_not_in(99, [1, 2, 3], config=cfg)

    def test_in_raises(self, cfg: BeaconConfig) -> None:
        with pytest.raises(AssertionError):
            beacon.assert_not_in(1, [1, 2, 3], config=cfg)


class TestAssertIsNone:
    def test_none_passes(self, cfg: BeaconConfig) -> None:
        beacon.assert_is_none(None, config=cfg)

    def test_non_none_raises(self, cfg: BeaconConfig) -> None:
        with pytest.raises(AssertionError):
            beacon.assert_is_none(0, config=cfg)

    def test_false_raises(self, cfg: BeaconConfig) -> None:
        with pytest.raises(AssertionError):
            beacon.assert_is_none(False, config=cfg)


class TestAssertIsNotNone:
    def test_non_none_passes(self, cfg: BeaconConfig) -> None:
        beacon.assert_is_not_none(0, config=cfg)
        beacon.assert_is_not_none(False, config=cfg)
        beacon.assert_is_not_none("", config=cfg)

    def test_none_raises(self, cfg: BeaconConfig) -> None:
        with pytest.raises(AssertionError):
            beacon.assert_is_not_none(None, config=cfg)


class TestAssertRaises:
    def test_expected_exception_caught(self, cfg: BeaconConfig) -> None:
        exc = beacon.assert_raises(ValueError, int, "bad", config=cfg)
        assert isinstance(exc, ValueError)

    def test_no_exception_raises_assertion(self, cfg: BeaconConfig) -> None:
        with pytest.raises(AssertionError):
            beacon.assert_raises(ValueError, lambda: None, config=cfg)

    def test_wrong_exception_raises_assertion(self, cfg: BeaconConfig) -> None:
        def raises_type_error() -> None:
            raise TypeError("type error")

        with pytest.raises(AssertionError):
            beacon.assert_raises(ValueError, raises_type_error, config=cfg)

    def test_exception_with_args(self, cfg: BeaconConfig) -> None:
        def my_fn(x: int) -> int:
            if x < 0:
                raise ValueError(f"negative: {x}")
            return x

        exc = beacon.assert_raises(ValueError, my_fn, -1, config=cfg)
        assert "negative" in str(exc)

    def test_exception_with_kwargs(self, cfg: BeaconConfig) -> None:
        def my_fn(*, strict: bool = False) -> None:
            if strict:
                raise RuntimeError("strict mode")

        exc = beacon.assert_raises(RuntimeError, my_fn, config=cfg, strict=True)
        assert isinstance(exc, RuntimeError)


class TestAssertArrayEqual:
    def test_numpy_required(self, cfg: BeaconConfig) -> None:
        pytest.importorskip("numpy")

    def test_equal_arrays_pass(self, cfg: BeaconConfig) -> None:
        np = pytest.importorskip("numpy")
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        beacon.assert_array_equal(a, b, config=cfg)

    def test_close_arrays_pass(self, cfg: BeaconConfig) -> None:
        np = pytest.importorskip("numpy")
        a = np.array([1.0, 2.0])
        b = np.array([1.0 + 1e-9, 2.0 - 1e-9])
        beacon.assert_array_equal(a, b, rtol=1e-6, config=cfg)

    def test_different_arrays_raise(self, cfg: BeaconConfig) -> None:
        np = pytest.importorskip("numpy")
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 99.0])
        with pytest.raises(AssertionError):
            beacon.assert_array_equal(a, b, config=cfg)


class TestAssertFrameEqual:
    def test_pandas_required(self, cfg: BeaconConfig) -> None:
        pytest.importorskip("pandas")

    def test_equal_frames_pass(self, cfg: BeaconConfig) -> None:
        pd = pytest.importorskip("pandas")
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        beacon.assert_frame_equal(df1, df2, config=cfg)

    def test_different_frames_raise(self, cfg: BeaconConfig) -> None:
        pd = pytest.importorskip("pandas")
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [1, 99]})
        with pytest.raises(AssertionError):
            beacon.assert_frame_equal(df1, df2, config=cfg)
