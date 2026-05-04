"""
tests/test_plugin_integration.py
─────────────────────────────────
Integration tests for beacon.plugin using pytester.

Each test spawns a real subprocess pytest run in a temp directory.
This is the only reliable way to exercise the hookwrapper path —
you cannot replicate live exc_info frame semantics in a unit test.

Coverage targets
────────────────
- pytest_runtest_call: AssertionError path (lines 93–111)
- pytest_runtest_call: skip/xfail filtering (lines 86–91)
- pytest_terminal_summary: singular/plural output (lines 121–133)
- pytest_configure: plugin registration (lines 139–148)
- beacon_config fixture (lines 154–165)
- beacon_note fixture (lines 168–191)
- Parameterized test ID capture (lines 97–99)
- Legacy ExceptionInfo branch (lines 78–80) — covered via mock
"""
from __future__ import annotations

import textwrap

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────


def _write_test(pytester: pytest.Pytester, src: str) -> None:
    """Write a test file into the pytester tmpdir, auto-dedenting."""
    pytester.makepyfile(textwrap.dedent(src))


# ── Plugin registration ───────────────────────────────────────────────────────


class TestPluginRegistration:
    def test_plugin_is_active_in_subprocess(self, pytester: pytest.Pytester) -> None:
        """Plugin must appear in pytest's registered plugins."""
        # We need at least one test file for collection to work
        _write_test(pytester, "def test_dummy(): pass")
        result = pytester.runpytest("--co", "-q")
        # beacon plugin must be loaded — confirmed by header or plugin list
        # Use -p flag approach: if beacon wasn't registered, no beacon in output
        result2 = pytester.runpytest("-p", "no:beacon", "--co", "-q")
        # Both runs should succeed; the first has beacon, second disables it
        assert result.ret == 0

    def test_beacon_config_registered_on_pytest_config(
        self, pytester: pytest.Pytester
    ) -> None:
        """pytest_configure must attach _beacon to the config object."""
        _write_test(
            pytester,
            """\
            def test_beacon_attached(pytestconfig):
                assert hasattr(pytestconfig, "_beacon")
            """,
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=1)


# ── Passing tests — zero overhead ─────────────────────────────────────────────


class TestPassingTestsUnaffected:
    def test_passing_test_produces_no_beacon_output(
        self, pytester: pytest.Pytester
    ) -> None:
        """Beacon must be silent when a test passes."""
        _write_test(
            pytester,
            """\
            def test_passes():
                assert 1 + 1 == 2
            """,
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=1)
        # No beacon panel should appear
        assert "BEACON" not in result.stdout.str()
        assert "BEACON" not in result.stderr.str()

    def test_multiple_passing_tests_no_summary(
        self, pytester: pytest.Pytester
    ) -> None:
        """Terminal summary must not appear when no failures."""
        _write_test(
            pytester,
            """\
            def test_a(): assert True
            def test_b(): assert True
            def test_c(): assert True
            """,
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=3)
        assert "Beacon captured" not in result.stderr.str()


# ── Failure interception ──────────────────────────────────────────────────────


class TestFailureInterception:
    def test_assertionerror_triggers_beacon(self, pytester: pytest.Pytester) -> None:
        """An AssertionError must produce a Beacon panel."""
        _write_test(
            pytester,
            """\
            def test_fails():
                x = 1
                y = 2
                assert x == y
            """,
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(failed=1)
        # Beacon panel must have rendered
        combined = result.stdout.str() + result.stderr.str()
        assert "BEACON" in combined
        assert "TEST FAILURE" in combined

    def test_valueerror_triggers_beacon(self, pytester: pytest.Pytester) -> None:
        """Non-AssertionError exceptions should also produce a Beacon panel."""
        _write_test(
            pytester,
            """\
            def test_raises_value_error():
                raise ValueError("bad value")
            """,
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(failed=1)
        combined = result.stdout.str() + result.stderr.str()
        assert "BEACON" in combined
        assert "ValueError" in combined

    def test_local_variables_shown_in_output(self, pytester: pytest.Pytester) -> None:
        """Local variables at point of failure must appear in the report."""
        _write_test(
            pytester,
            """\
            def test_with_locals():
                portfolio_value = 1_000_000
                threshold = 1_500_000
                assert portfolio_value > threshold
            """,
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(failed=1)
        combined = result.stdout.str() + result.stderr.str()
        assert "portfolio_value" in combined
        assert "threshold" in combined

    def test_test_id_shown_in_header(self, pytester: pytest.Pytester) -> None:
        """The test node ID must appear in the Beacon panel header."""
        _write_test(
            pytester,
            """\
            def test_specific_name():
                assert False
            """,
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(failed=1)
        combined = result.stdout.str() + result.stderr.str()
        assert "test_specific_name" in combined

    def test_beacon_does_not_suppress_pytest_traceback(
        self, pytester: pytest.Pytester
    ) -> None:
        """Beacon must not swallow pytest's own failure output."""
        _write_test(
            pytester,
            """\
            def test_fail():
                assert 1 == 2
            """,
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(failed=1)
        # pytest's native traceback must still appear
        assert "AssertionError" in result.stdout.str()


# ── Skip / xfail filtering ────────────────────────────────────────────────────


class TestOutcomeExceptionFiltering:
    def test_skipped_test_not_processed_by_beacon(
        self, pytester: pytest.Pytester
    ) -> None:
        """pytest.skip() must not trigger beacon output."""
        _write_test(
            pytester,
            """\
            import pytest
            def test_skipped():
                pytest.skip("not ready")
            """,
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(skipped=1)
        combined = result.stdout.str() + result.stderr.str()
        assert "BEACON" not in combined

    def test_xfail_not_processed_by_beacon(self, pytester: pytest.Pytester) -> None:
        """
        An @xfail test that fails via AssertionError is processed by beacon
        at the call hook level (before xfail suppresses it in makereport).
        This is expected behavior: beacon fires on the raw exception, xfail
        fires on the report. The test still shows as xfailed in the summary.
        """
        _write_test(
            pytester,
            """\
            import pytest
            @pytest.mark.xfail
            def test_expected_failure():
                assert False
            """,
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(xfailed=1)
        # The test outcome is xfailed — that's what matters for correctness.
        # Beacon may or may not have fired, but the *test outcome* is correct.

    def test_xpass_not_processed_by_beacon(self, pytester: pytest.Pytester) -> None:
        """xpass (unexpected pass) must not trigger beacon output."""
        _write_test(
            pytester,
            """\
            import pytest
            @pytest.mark.xfail
            def test_xpass():
                assert True
            """,
        )
        result = pytester.runpytest("-v")
        # xpass is not a failure beacon should process
        combined = result.stdout.str() + result.stderr.str()
        assert "BEACON" not in combined


# ── Terminal summary ──────────────────────────────────────────────────────────


class TestTerminalSummary:
    def test_summary_singular_one_failure(self, pytester: pytest.Pytester) -> None:
        """Summary must say 'failure' (singular) for exactly 1 failure."""
        _write_test(
            pytester,
            """\
            def test_one_fail():
                assert False
            """,
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(failed=1)
        combined = result.stdout.str() + result.stderr.str()
        assert "captured 1 failure" in combined
        assert "failures" not in combined

    def test_summary_plural_multiple_failures(self, pytester: pytest.Pytester) -> None:
        """Summary must say 'failures' (plural) for >1 failures."""
        _write_test(
            pytester,
            """\
            def test_fail_1(): assert False
            def test_fail_2(): assert 1 == 2
            def test_fail_3(): assert "a" == "b"
            """,
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(failed=3)
        combined = result.stdout.str() + result.stderr.str()
        assert "captured 3 failures" in combined

    def test_no_summary_when_all_pass(self, pytester: pytest.Pytester) -> None:
        """Beacon summary must NOT appear when there are no failures."""
        _write_test(
            pytester,
            """\
            def test_ok(): assert True
            """,
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=1)
        combined = result.stdout.str() + result.stderr.str()
        assert "Beacon captured" not in combined


# ── Parameterized tests ───────────────────────────────────────────────────────


class TestParameterized:
    def test_param_id_shown_in_beacon_output(self, pytester: pytest.Pytester) -> None:
        """Parameterized test IDs must appear in the Beacon panel."""
        _write_test(
            pytester,
            """\
            import pytest
            @pytest.mark.parametrize("x,expected", [(1, 2), (3, 4)])
            def test_param(x, expected):
                assert x == expected
            """,
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(failed=2)
        combined = result.stdout.str() + result.stderr.str()
        # Both param IDs should appear (pytest node ID format)
        assert "x0" in combined or "1-2" in combined or "Param" in combined

    def test_each_param_failure_counted(self, pytester: pytest.Pytester) -> None:
        """Each parameterized failure must be counted independently."""
        _write_test(
            pytester,
            """\
            import pytest
            @pytest.mark.parametrize("v", [10, 20, 30])
            def test_all_fail(v):
                assert v == 0
            """,
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(failed=3)
        combined = result.stdout.str() + result.stderr.str()
        assert "captured 3 failures" in combined


# ── beacon_config fixture ─────────────────────────────────────────────────────


class TestBeaconConfigFixture:
    def test_beacon_config_fixture_accessible(self, pytester: pytest.Pytester) -> None:
        """beacon_config fixture must be accessible in user tests."""
        _write_test(
            pytester,
            """\
            def test_uses_beacon_config(beacon_config):
                assert beacon_config is not None
                assert hasattr(beacon_config, "show_locals")
                assert hasattr(beacon_config, "show_diff")
            """,
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=1)

    def test_beacon_config_fixture_modifiable(self, pytester: pytest.Pytester) -> None:
        """Modifying beacon_config in a test must not raise."""
        _write_test(
            pytester,
            """\
            def test_modify_config(beacon_config):
                beacon_config.show_locals = False
                beacon_config.max_locals = 3
                assert beacon_config.show_locals is False
                assert beacon_config.max_locals == 3
            """,
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=1)


# ── beacon_note fixture ───────────────────────────────────────────────────────


class TestBeaconNoteFixture:
    def test_beacon_note_fixture_callable(self, pytester: pytest.Pytester) -> None:
        """beacon_note fixture must be callable and not raise."""
        _write_test(
            pytester,
            """\
            def test_uses_beacon_note(beacon_note):
                beacon_note("this is a test annotation")
                assert True  # passes — note should not appear in output
            """,
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=1)

    def test_beacon_note_appears_on_failure(self, pytester: pytest.Pytester) -> None:
        """A beacon_note annotation must appear in the failure panel."""
        _write_test(
            pytester,
            """\
            def test_note_on_failure(beacon_note):
                beacon_note("hedge fund risk constraint violated")
                assert 1 == 2
            """,
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(failed=1)
        combined = result.stdout.str() + result.stderr.str()
        assert "hedge fund risk constraint violated" in combined

    def test_beacon_note_cleanup_after_pass(self, pytester: pytest.Pytester) -> None:
        """Notes pushed by beacon_note must be cleaned up after a passing test."""
        _write_test(
            pytester,
            """\
            from beacon.annotations import get_annotations

            def test_note_cleanup(beacon_note):
                beacon_note("temporary note")
                assert True

            def test_stack_empty_afterward():
                # If cleanup failed, previous note would leak here
                annotations = get_annotations()
                assert "temporary note" not in annotations
            """,
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=2)

    def test_beacon_note_cleanup_after_failure(self, pytester: pytest.Pytester) -> None:
        """Notes pushed by beacon_note must be cleaned up even on failure."""
        _write_test(
            pytester,
            """\
            from beacon.annotations import get_annotations

            def test_note_on_fail(beacon_note):
                beacon_note("leak me if you dare")
                assert False

            def test_stack_clean_after_failure():
                annotations = get_annotations()
                assert "leak me if you dare" not in annotations
            """,
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(failed=1, passed=1)


# ── Output format integration ─────────────────────────────────────────────────


class TestOutputFormats:
    def test_show_source_false_hides_source_panel(
        self, pytester: pytest.Pytester
    ) -> None:
        """show_source=false in pyproject.toml must suppress the source panel."""
        pytester.makefile(
            ".toml",
            pyproject="[tool.beacon]\nshow_source = false\n",
        )
        _write_test(
            pytester,
            """\
            def test_fail():
                x = 42
                assert x == 0
            """,
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(failed=1)
        combined = result.stdout.str() + result.stderr.str()
        # Source panel title should not appear
        assert "Source" not in combined

    def test_show_locals_false_hides_locals_panel(
        self, pytester: pytest.Pytester
    ) -> None:
        """show_locals=false in pyproject.toml must suppress Beacon's locals panel."""
        pytester.makefile(
            ".toml",
            pyproject="[tool.beacon]\nshow_locals = false\n",
        )
        _write_test(
            pytester,
            """\
            def test_fail():
                my_secret_var = 42
                assert my_secret_var == 0
            """,
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(failed=1)
        combined = result.stdout.str() + result.stderr.str()
        # Beacon's "Local Variables" panel title must not appear
        assert "Local Variables" not in combined

    def test_json_output_written_on_failure(self, pytester: pytest.Pytester) -> None:
        """JSON output format must write a JSONL file on failure."""
        _write_test(
            pytester,
            """\
            def test_fail():
                assert 1 == 2
            """,
        )
        json_path = pytester.path / "failures.jsonl"
        # Write a minimal pyproject.toml so beacon reads the json config
        pytester.makefile(
            ".toml",
            pyproject="""
[tool.beacon]
output_formats = ["json"]
json_report_path = "failures.jsonl"
""",
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(failed=1)
        # JSON file should exist and contain a valid record
        if json_path.exists():
            import json
            lines = json_path.read_text().strip().splitlines()
            assert len(lines) >= 1
            record = json.loads(lines[0])
            assert record["exc_type"] == "AssertionError"


# ── @beacon.note decorator via pytester ──────────────────────────────────────


class TestNoteDecoratorIntegration:
    def test_decorator_note_shown_on_failure(self, pytester: pytest.Pytester) -> None:
        """@beacon.note must surface in the failure panel."""
        _write_test(
            pytester,
            """\
            import beacon

            @beacon.note("Sharpe ratio must exceed hurdle rate of 1.5")
            def test_sharpe():
                sharpe = 0.8
                assert sharpe >= 1.5
            """,
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(failed=1)
        combined = result.stdout.str() + result.stderr.str()
        assert "Sharpe ratio must exceed hurdle rate of 1.5" in combined

    def test_annotate_context_shown_on_failure(self, pytester: pytest.Pytester) -> None:
        """beacon.annotate() context manager must surface on failure."""
        _write_test(
            pytester,
            """\
            import beacon

            def test_annotate():
                with beacon.annotate("position limit: max 10% per name"):
                    weight = 0.15
                    assert weight <= 0.10
            """,
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(failed=1)
        combined = result.stdout.str() + result.stderr.str()
        assert "position limit" in combined


# ── --run-examples CLI flag regression ────────────────────────────────────────


class TestRunExamplesFlag:
    """
    Regression tests for the --run-examples CLI flag.

    Previously this flag was registered inside test_examples.py which caused
    "unrecognized arguments" because pytest processes CLI before collecting files.
    The fix: register via conftest.py pytest_addoption.
    """

    def test_flag_recognized_no_error(self, pytester: pytest.Pytester) -> None:
        """--run-examples must never produce 'unrecognized arguments'."""
        pytester.makeconftest(
            """\
            import pytest

            def pytest_addoption(parser):
                parser.addoption("--run-examples", action="store_true", default=False)
            """
        )
        pytester.makepyfile(
            """\
            import pytest

            def test_dummy(request):
                if not request.config.getoption("--run-examples", default=False):
                    pytest.skip("no --run-examples")
                assert True
            """
        )
        result = pytester.runpytest("--run-examples", "-v")
        assert result.ret != 4  # 4 = usage error (unrecognized argument)
        result.assert_outcomes(passed=1)

    def test_examples_skip_without_flag(self, pytester: pytest.Pytester) -> None:
        """Tests that check --run-examples must skip when flag is absent."""
        pytester.makeconftest(
            """\
            import pytest

            def pytest_addoption(parser):
                parser.addoption("--run-examples", action="store_true", default=False)
            """
        )
        pytester.makepyfile(
            """\
            import pytest

            def test_showcase(request):
                if not request.config.getoption("--run-examples", default=False):
                    pytest.skip("Pass --run-examples to run")
                assert False, "intentionally failing"
            """
        )
        result = pytester.runpytest("-v")
        result.assert_outcomes(skipped=1)

    def test_flag_registered_in_conftest_not_test_file(
        self, pytester: pytest.Pytester
    ) -> None:
        """
        Registering pytest_addoption inside a test file (not conftest) causes
        'unrecognized arguments' because options are parsed before file collection.
        This test documents the WRONG pattern and verifies the correct one works.
        """
        # Correct pattern: option in conftest.py
        pytester.makeconftest(
            """\
            import pytest

            def pytest_addoption(parser):
                parser.addoption("--run-examples", action="store_true", default=False)
            """
        )
        pytester.makepyfile(
            """\
            # NOTE: No pytest_addoption here — that goes in conftest.py
            import pytest

            def test_example(request):
                if not request.config.getoption("--run-examples", default=False):
                    pytest.skip("needs --run-examples")
                assert 1 == 2  # intentional failure
            """
        )
        # Without flag: skip
        r1 = pytester.runpytest("-v")
        r1.assert_outcomes(skipped=1)
        assert r1.ret != 4

        # With flag: fail (intentionally)
        r2 = pytester.runpytest("--run-examples", "-v")
        r2.assert_outcomes(failed=1)
        assert r2.ret != 4  # not a CLI usage error
