# Beacon — Development Roadmap

## Guiding Philosophy

Every new feature must pass one test: **does it make a failing test more self-documenting?**
We never add complexity for its own sake.

---

## v0.1.0 — Foundation ✅ (current)

- [x] Rich terminal output with panels, tables, syntax highlighting
- [x] AST-based assertion introspection via `executing`
- [x] Sub-expression evaluation (LHS/RHS breakdown for comparisons)
- [x] Local variable capture and smart filtering
- [x] Source snippet with failing line highlighted
- [x] `@beacon.note` decorator for test-level author annotations
- [x] `beacon.annotate()` context manager for inline annotations
- [x] Full set of rich assertion helpers (`assert_equal`, `assert_raises`, etc.)
- [x] NumPy array diff (`assert_array_equal`, shape/dtype/Δ table)
- [x] Pandas DataFrame diff (`assert_frame_equal`)
- [x] DeepDiff integration for dicts and sequences
- [x] Unified text diff for multi-line strings
- [x] pytest plugin with zero-overhead on passing tests
- [x] JSON output sink (JSONL append)
- [x] pyproject.toml configuration
- [x] Environment variable overrides
- [x] Full type annotations (`mypy --strict`)
- [x] Comprehensive test suite

---

## v0.2.0 — Hardening & Compatibility

### unittest Compatibility Layer
Provide a `BeaconTestCase(unittest.TestCase)` base class that:
- Overrides `assertEqual`, `assertRaises`, etc. with Beacon-enhanced versions
- Works seamlessly with Django's test runner, nose2, and plain `python -m unittest`
- Injects beacon output before Django/nose2 takes over the failure message

### Enhanced HTML Report
- Self-contained single-file HTML (no external deps)
- Collapsible sections (click to expand source, locals, diff)
- Dark/light theme toggle
- Copy-to-clipboard for assertion expressions
- Shareable: generates a permalink hash from the failure content

### Structured JSON Schema
- Publish a JSON Schema for the JSONL failure records
- `beacon.load_report(path)` utility for programmatic consumption
- Integration guide for ingesting into Datadog, Grafana, ELK

### CLI Improvements
- `beacon run pytest ...` — wrapper that sets up output formats
- `beacon report failures.jsonl` — pretty-print a JSON report file
- `beacon diff a.jsonl b.jsonl` — compare two report runs

### Robustness
- `--beacon-disable` pytest flag to fully suppress beacon output
- `--beacon-theme=<name>` pytest flag
- Graceful handling of `.pyc`-only environments (frozen apps)
- Support for `assert` inside comprehensions and lambdas

---

## v0.3.0 — Intelligence

### LLM-Powered Failure Explanation (opt-in)
- `llm_explain = true` in `pyproject.toml` + `OPENAI_API_KEY`
- Sends: expression source, LHS/RHS values, local vars, author notes
- Returns: 2–4 sentence plain-English diagnosis + suggested fix
- Supports OpenAI (GPT-4o) and Anthropic (Claude 3.5 Sonnet) via config
- Caches responses locally to avoid re-calling for identical failures
- Respects `--offline` flag — never calls external API in CI without explicit opt-in

### Smart Type-Aware Diffs
- **Datetime/timedelta**: show human-readable difference ("3 hours 42 minutes earlier")
- **UUID**: highlight differing segments (only the variant nibble changed)
- **Enum**: show enum name, not just integer value
- **Decimal/Fraction**: show exact rational difference
- **Regex match**: show what matched vs. what was expected to match
- **Dataclass**: field-by-field diff table

### Suggested Fix Hints
Pattern-match common assertion failures and suggest fixes:
- `1 != 1.0` → "Did you mean `assert_almost_equal`? Integer and float comparison."
- `[] != []` → "Two empty lists — check if the function returned early."
- `None != expected` → "Got None — function may be missing a return statement."

---

## v0.4.0 — Ecosystem Integration

### VS Code Extension
- Inline failure decorations in the editor (ghost text showing LHS/RHS)
- Hover panel with full Beacon output
- "Run test with Beacon" right-click context menu item
- Reads `beacon_failures.jsonl` from workspace root

### GitHub Actions Summary
- Write `$GITHUB_STEP_SUMMARY` with HTML-formatted failure report
- Action: `beacon-testing/beacon-action@v1`
- Example: each failing test gets a collapsible `<details>` block in the PR checks tab

### Webhook / Notification Plugins
- `beacon[slack]`: post rich Slack blocks on failure
- `beacon[discord]`: post embed to a Discord webhook
- `beacon[pagerduty]`: create PagerDuty incident for production test failures
- Plugin API: `beacon.register_reporter(MyReporter)` — implement `render(report)`

### Plugin / Extension API
```python
# Custom reporter example
import beacon
from beacon.reporters import FailureReport

class MyDatadogReporter:
    def render(self, report: FailureReport) -> None:
        # ship to Datadog
        ...

beacon.register_reporter(MyDatadogReporter())
```

---

## v0.5.0 — Performance & Scale

### Parallel Test Suite Support
- Thread-safe annotation stack (already done) + process-safe JSON sink
- `beacon[xdist]`: integration with `pytest-xdist` for parallel runs
- Merge JSONL shards from worker processes into a single report

### Large Object Handling
- Streaming repr for objects > 10MB
- Configurable depth limits for deeply nested diffs
- Lazy evaluation: only compute expensive diffs if the failure panel is actually rendered

### Benchmarks
- Micro-benchmark suite: measure per-test overhead on passing tests
- Target: < 50µs overhead per passing test
- Publish results in README

---

## Long-Term Vision

Beacon should become the **standard way Python developers understand test failures** — not just a prettier output, but a genuine productivity multiplier. The goal is that reading a Beacon failure report is as informative as sitting next to the engineer who wrote the test.

Key principles we will never compromise:
1. **Zero overhead on passing tests** — always.
2. **Never crash the test run** — all rendering is wrapped in try/except.
3. **Graceful degradation** — if rich, executing, or deepdiff are unavailable, Beacon still works, just less richly.
4. **Opt-in for anything external** — LLM calls, webhooks, file I/O — always explicit.
