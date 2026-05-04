"""Tests for beacon.config."""
from __future__ import annotations

import os

import pytest

from beacon.config import BeaconConfig, get_config, reset_config


class TestBeaconConfigDefaults:
    def test_show_locals_default_true(self) -> None:
        cfg = BeaconConfig()
        assert cfg.show_locals is True

    def test_show_source_default_true(self) -> None:
        cfg = BeaconConfig()
        assert cfg.show_source is True

    def test_show_diff_default_true(self) -> None:
        cfg = BeaconConfig()
        assert cfg.show_diff is True

    def test_max_locals_default(self) -> None:
        cfg = BeaconConfig()
        assert cfg.max_locals == 10

    def test_output_formats_default(self) -> None:
        cfg = BeaconConfig()
        assert "terminal" in cfg.output_formats

    def test_theme_default(self) -> None:
        cfg = BeaconConfig()
        assert cfg.theme == "monokai"

    def test_llm_explain_default_false(self) -> None:
        cfg = BeaconConfig()
        assert cfg.llm_explain is False


class TestBeaconConfigEnvOverride:
    def test_env_show_locals_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BEACON_SHOW_LOCALS", "false")
        cfg = BeaconConfig()
        cfg._apply_env()
        assert cfg.show_locals is False

    def test_env_show_locals_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BEACON_SHOW_LOCALS", "1")
        cfg = BeaconConfig()
        cfg._apply_env()
        assert cfg.show_locals is True

    def test_env_max_locals(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BEACON_MAX_LOCALS", "5")
        cfg = BeaconConfig()
        cfg._apply_env()
        assert cfg.max_locals == 5

    def test_env_theme(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BEACON_THEME", "dracula")
        cfg = BeaconConfig()
        cfg._apply_env()
        assert cfg.theme == "dracula"

    def test_env_invalid_int_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Invalid env values should be silently ignored, not crash."""
        monkeypatch.setenv("BEACON_MAX_LOCALS", "not-a-number")
        cfg = BeaconConfig()
        cfg._apply_env()
        assert cfg.max_locals == 10  # unchanged default


class TestBeaconConfigDictApply:
    def test_apply_valid_keys(self) -> None:
        cfg = BeaconConfig()
        cfg._apply_dict({"max_locals": 3, "show_locals": False})
        assert cfg.max_locals == 3
        assert cfg.show_locals is False

    def test_apply_unknown_keys_ignored(self) -> None:
        cfg = BeaconConfig()
        cfg._apply_dict({"nonexistent_key": "value"})  # should not raise
        assert not hasattr(cfg, "nonexistent_key")

    def test_apply_private_keys_ignored(self) -> None:
        cfg = BeaconConfig()
        original_overrides = cfg._overrides.copy()
        cfg._apply_dict({"_overrides": {"injected": True}})
        # _overrides should be unchanged
        assert cfg._overrides == original_overrides


class TestBeaconConfigOverride:
    def test_override_returns_new_instance(self) -> None:
        cfg = BeaconConfig()
        new_cfg = cfg.override(max_locals=3)
        assert new_cfg is not cfg

    def test_override_changes_field(self) -> None:
        cfg = BeaconConfig()
        new_cfg = cfg.override(max_locals=3)
        assert new_cfg.max_locals == 3
        assert cfg.max_locals == 10  # original unchanged

    def test_override_ignores_unknown(self) -> None:
        cfg = BeaconConfig()
        new_cfg = cfg.override(totally_fake_field=42)
        assert not hasattr(new_cfg, "totally_fake_field")


class TestGetConfigSingleton:
    def test_returns_same_instance(self) -> None:
        reset_config()
        a = get_config()
        b = get_config()
        assert a is b

    def test_reset_produces_new_instance(self) -> None:
        reset_config()
        a = get_config()
        reset_config()
        b = get_config()
        assert a is not b
