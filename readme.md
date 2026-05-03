<h1 align="center">Beacon</h1>

<p align="center"> <img src="beacon.avif" width="420"/> </p>

Beacon is a tool that makes test failures much easier to understand. Instead of just saying “this test failed,” it shows exactly what went wrong, 
what values were compared, how they differ, and even the context around the failure - all in a clean, readable format. 
It automatically plugs into pytest, so when something breaks, the error message itself becomes a clear explanation of the problem, 
almost like the test is documenting its own failure.

<p align="center">
  <img src="https://img.shields.io/badge/version-0.1.1-111111.svg" />
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" />
  <img src="https://img.shields.io/badge/pytest-plugin-orange.svg" />
  <img src="https://img.shields.io/badge/Rich-terminal-blueviolet" />
</p>

---

## Why Beacon?

A failing test should be a **self-documenting source of truth**. The moment a test breaks in CI or on a colleague's machine, the failure output alone should tell you:

- **What** failed — the exact expression, with evaluated sub-expressions
- **Why** it failed — a structured diff of the actual vs. expected values
- **The intent** — author annotations attached at write-time, not buried in a comment

Standard `assert` output and even pytest's rewriting are often not enough. Beacon goes further.

---

## Features

| Feature | Description |
|---|---|
| **Rich terminal output** | Colours, syntax highlighting, clean diffs, tables |
| **AST introspection** | Full expression breakdown with evaluated sub-expressions |
| **Author annotations** | `@beacon.note(...)` decorator and `beacon.annotate(...)` context manager |
| **Smart diffs** | Structured diffs for dicts, DataFrames, numpy arrays, strings |
| **pytest plugin** | Auto-activates on install — zero configuration needed |
| **Zero overhead** | No cost on passing tests |
| **Rich assertion helpers** | `assert_equal`, `assert_frame_equal`, `assert_raises`, and more |
| **Fully typed** | `mypy --strict` clean |

---

## Installation

```bash
pip install beacon
```

That's it. The pytest plugin **auto-activates** — no `conftest.py` changes needed.

### Optional extras

```bash
pip install beacon[numpy]    # numpy array diffs
pip install beacon[pandas]   # DataFrame diffs
pip install beacon[llm]      # LLM-powered failure explanation (experimental)
pip install beacon[dev]      # development tools
```

---

## Quick Start

### It just works with existing `assert` statements

```python
# tests/test_trading.py

def test_portfolio_pnl():
    expected_pnl = 1_250_000.0
    actual_pnl = compute_pnl(positions)   # returns 1_187_432.57

    assert actual_pnl == expected_pnl
```

Run `pytest` — Beacon intercepts the failure and renders:

```
╭──────────────────────────────────────────────────────────╮
│  BEACON  TEST FAILURE                                    │
│  Test: tests/test_trading.py::test_portfolio_pnl         │
│  File: tests/test_trading.py:8                           │
│  Error: AssertionError                                   │
╰──────────────────────────────────────────────────────────╯

╭─ Assertion Breakdown ────────────────────────────────────╮
│  assert actual_pnl == expected_pnl                       │
│                                                          │
│  left  actual_pnl   = 1187432.57   (float)               │
│        ==                                                │
│  right expected_pnl = 1250000.0    (float)               │
│                                                          │
│  Δ  62567.43  (5.01% relative)                           │
╰──────────────────────────────────────────────────────────╯

╭─ Source  tests/test_trading.py ──────────────────────────╮
│  5   def test_portfolio_pnl():                           │
│  6       expected_pnl = 1_250_000.0                      │
│  7       actual_pnl = compute_pnl(positions)             │
│  8 ▶     assert actual_pnl == expected_pnl               │
│  9                                                       │
╰──────────────────────────────────────────────────────────╯

╭─ Local Variables ────────────────────────────────────────╮
│  expected_pnl │ float │ 1250000.0                        │
│  actual_pnl   │ float │ 1187432.57                       │
╰──────────────────────────────────────────────────────────╯
```

### Author annotations

```python
import beacon

@beacon.note("Kelly fraction must be bounded to [0, 1] for all valid inputs.")
@beacon.note("Critical for risk management — unconstrained Kelly blows up capital.")
def test_kelly_bounded():
    f = kelly_fraction(mu=0.5, sigma=0.3)
    assert 0.0 <= f <= 1.0
```

### Inline annotations

```python
def test_signal_processing():
    raw_z = compute_raw_z_scores(prices)

    with beacon.annotate("z-scores must be bounded to [-3, 3] post-winsorisation"):
        z = winsorise(raw_z)
        assert all(-3.0 <= zi <= 3.0 for zi in z)
```

### Rich assertion helpers

```python
import beacon

# Scalar / collection equality
beacon.assert_equal(result, expected)
beacon.assert_not_equal(actual_status, "ERROR")

# Numeric closeness
beacon.assert_almost_equal(computed_delta, 0.52, rtol=1e-4)

# Boolean
beacon.assert_true(order.is_filled())
beacon.assert_is_not_none(session_token)

# Membership
beacon.assert_in(symbol, VALID_SYMBOLS)

# Exception testing — returns the caught exception
exc = beacon.assert_raises(ValueError, validate_strike, -100.0)
assert "positive" in str(exc)

# NumPy
beacon.assert_array_equal(portfolio_weights, expected_weights, rtol=1e-6)

# Pandas
beacon.assert_frame_equal(df_actual, df_expected)
```

---

## Configuration

Add a `[tool.beacon]` section to `pyproject.toml`:

```toml
[tool.beacon]
show_locals = true
max_locals = 10
show_source = true
source_context_lines = 4
show_diff = true
theme = "monokai"            # any pygments theme
output_formats = ["terminal", "json"]
json_report_path = "beacon_failures.jsonl"
llm_explain = false          # set true + OPENAI_API_KEY for AI explanations
```

### Environment variable overrides

| Variable | Type | Description |
|---|---|---|
| `BEACON_SHOW_LOCALS` | bool | Show local variables |
| `BEACON_MAX_LOCALS` | int | Max variables to display |
| `BEACON_SHOW_SOURCE` | bool | Show source snippet |
| `BEACON_SOURCE_CONTEXT_LINES` | int | Lines of context |
| `BEACON_SHOW_DIFF` | bool | Show structured diffs |
| `BEACON_THEME` | str | Pygments theme name |
| `BEACON_OUTPUT_FORMATS` | JSON list | Output sinks, e.g. `["terminal", "json"]` |
| `BEACON_JSON_REPORT_PATH` | str | JSONL report path |
| `BEACON_HTML_REPORT_PATH` | str | HTML report path |
| `BEACON_LLM_EXPLAIN` | bool | Enable LLM explanation |

---

## Project Structure

```
beacon/
├── src/beacon/
│   ├── __init__.py       # Public API
│   ├── plugin.py         # pytest plugin hooks
│   ├── core.py           # Failure capture + assertion helpers
│   ├── rewrite.py        # AST-based assertion introspection
│   ├── reporters.py      # Rich terminal / JSON / HTML output
│   ├── annotations.py    # @note decorator + annotate() context manager
│   ├── utils.py          # Pure helper functions
│   ├── config.py         # Typed configuration
│   └── _version.py
├── tests/
├── pyproject.toml
└── README.md
```

---

## Development Setup

```bash
git clone https://github.com/beacon-testing/beacon
cd beacon

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows

# Install in editable mode with dev extras
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run the test suite
pytest

# Run with coverage
pytest --cov=beacon --cov-report=term-missing

# Type check
mypy src/beacon

# Lint
ruff check src/ tests/
ruff format src/ tests/

# See all Beacon's own output (intentionally failing showcase tests)
pytest tests/test_examples.py --run-examples -v
```

PowerShell uses a different environment-variable syntax from Bash/Zsh:

```powershell
$env:BEACON_OUTPUT_FORMATS='["json"]'
pytest tests/test_examples.py --run-examples -v
```

```bash
BEACON_OUTPUT_FORMATS='["json"]' pytest tests/test_examples.py --run-examples -v
```

---

## Roadmap

### v0.2 — Hardening
- `unittest.TestCase` compatibility layer
- Full HTML report with collapsible sections
- Structured JSON output with schema validation
- `--beacon-disable` flag for opt-out per run
- Better multi-line assertion support

### v0.3 — Intelligence
- LLM-powered failure explanation (opt-in, OpenAI/Anthropic)
- Suggested fix hints for common assertion patterns
- Regex / datetime / UUID / UUID-aware diffs

### v0.4 — Ecosystem
- VS Code extension: inline failure decorations
- GitHub Actions summary integration
- Slack / Discord webhook for CI failures
- Plugin / extension API for custom reporters

---

## License

MIT — see [LICENSE](LICENSE).

