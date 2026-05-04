"""
beacon.config
─────────────
Typed configuration for Beacon, read from pyproject.toml [tool.beacon]
or from environment variables / programmatic overrides.

All fields have sensible defaults so zero configuration is needed
for the standard use case.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib  # type: ignore[no-redef]
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            tomllib = None  # type: ignore[assignment]


def _find_pyproject() -> Optional[Path]:
    """Walk up from cwd looking for pyproject.toml."""
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        candidate = parent / "pyproject.toml"
        if candidate.exists():
            return candidate
    return None


def _load_pyproject_section() -> dict[str, Any]:
    """Load [tool.beacon] from the nearest pyproject.toml, if any."""
    if tomllib is None:
        return {}
    path = _find_pyproject()
    if path is None:
        return {}
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return data.get("tool", {}).get("beacon", {})
    except Exception:  # noqa: BLE001
        return {}


@dataclass
class BeaconConfig:
    """
    Central configuration object for Beacon.

    Precedence (highest → lowest):
      1. Programmatic / per-test overrides
      2. Environment variables (BEACON_*)
      3. pyproject.toml [tool.beacon]
      4. Hard-coded defaults below
    """

    # ── Display ───────────────────────────────────────────────────────────────
    show_locals: bool = True
    """Show filtered local variables captured at the point of failure."""

    max_locals: int = 10
    """Maximum number of local variables to display."""

    show_source: bool = True
    """Show a source code snippet around the failing assertion."""

    source_context_lines: int = 4
    """Lines of source context above and below the failing line."""

    show_diff: bool = True
    """Show structured diff for complex objects (dicts, lists, dataframes…)."""

    theme: str = "monokai"
    """Pygments syntax-highlighting theme for source snippets."""

    # ── Filtering ─────────────────────────────────────────────────────────────
    locals_exclude_patterns: List[str] = field(
        default_factory=lambda: ["__*", "_pytest*", "builtins"]
    )
    """Glob-style patterns for variable names to exclude from locals display."""

    max_repr_length: int = 500
    """Maximum character length for a single variable repr before truncation."""

    max_collection_items: int = 20
    """Maximum number of items shown when rendering a collection."""

    # ── Output ────────────────────────────────────────────────────────────────
    output_formats: List[str] = field(default_factory=lambda: ["terminal"])
    """Output sinks: 'terminal', 'json', 'html'."""

    json_report_path: Optional[str] = None
    """Path for JSON report output (only when 'json' in output_formats)."""

    html_report_path: Optional[str] = None
    """Path for HTML report output (only when 'html' in output_formats)."""

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_explain: bool = False
    """Enable LLM-powered failure explanation (requires openai extra)."""

    llm_model: str = "gpt-4o-mini"
    """OpenAI model to use for failure explanation."""

    # ── Internals ─────────────────────────────────────────────────────────────
    _overrides: dict[str, Any] = field(default_factory=dict, repr=False)

    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def load(cls) -> "BeaconConfig":
        """
        Build a BeaconConfig by merging defaults ← pyproject.toml ← env vars.
        """
        cfg = cls()
        toml_section = _load_pyproject_section()
        cfg._apply_dict(toml_section)
        cfg._apply_env()
        return cfg

    def _apply_dict(self, data: dict[str, Any]) -> None:
        """Apply a dict of overrides, ignoring unknown keys."""
        for key, value in data.items():
            if hasattr(self, key) and not key.startswith("_"):
                setattr(self, key, value)

    def _apply_env(self) -> None:
        """Apply BEACON_* environment variables."""
        env_map: dict[str, tuple[str, type]] = {
            "BEACON_SHOW_LOCALS": ("show_locals", bool),
            "BEACON_MAX_LOCALS": ("max_locals", int),
            "BEACON_SHOW_SOURCE": ("show_source", bool),
            "BEACON_SOURCE_CONTEXT_LINES": ("source_context_lines", int),
            "BEACON_SHOW_DIFF": ("show_diff", bool),
            "BEACON_THEME": ("theme", str),
            "BEACON_LLM_EXPLAIN": ("llm_explain", bool),
            "BEACON_LLM_MODEL": ("llm_model", str),
        }
        for env_key, (attr, cast) in env_map.items():
            raw = os.environ.get(env_key)
            if raw is not None:
                if cast is bool:
                    setattr(self, attr, raw.lower() in ("1", "true", "yes"))
                else:
                    try:
                        setattr(self, attr, cast(raw))
                    except ValueError:
                        pass

    def override(self, **kwargs: Any) -> "BeaconConfig":
        """Return a shallow copy with the given fields overridden."""
        import copy

        new = copy.copy(self)
        for k, v in kwargs.items():
            if hasattr(new, k):
                setattr(new, k, v)
        return new


# Module-level singleton — loaded once per process.
_CONFIG: Optional[BeaconConfig] = None


def get_config() -> BeaconConfig:
    """Return the process-level BeaconConfig singleton."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = BeaconConfig.load()
    return _CONFIG


def reset_config() -> None:
    """Reset the singleton (mainly useful in tests)."""
    global _CONFIG
    _CONFIG = None
