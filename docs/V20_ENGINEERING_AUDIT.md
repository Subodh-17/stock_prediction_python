# V20 engineering audit

Evidence labels: **CONFIRMED** = reproduced with a command whose output was seen · **SUSPECTED** = a risk from reading code, not reproduced ·
**NOT RUN** = not attempted or impossible here. An untested feature is *not* called broken.

## 1. Environment and method
* Hardware: **1 vCPU (Intel(R) Xeon(R) Processor), 3997 MB RAM**, Linux x86_64, CPython 3.12.3. One vCPU / 4 GB means two TensorFlow processes at once can crash
  (this killed a baseline run in V19) and cold-start costs are high; absolute timings below are specific to this machine.
* Baseline: the delivered `stock_prediction_python_v19.zip` (sha256 `efdddc916e8d3db93fc9271739697185102eb0d0276e8cd2e0d1fb8d3c5799ac`), extracted untouched and tested in a clean virtual environment created
  from its own `requirements-dev-lock.txt`. Working tree: a separate extraction. No Git repository, CI configuration, Makefile, Dockerfile or packaging
  script existed (checked); there were no uncommitted user changes to preserve.
* V19 baseline result: **442 passed, 0 failed in 262 s**. Compilation, `pip check`, a real TensorFlow import and the lock comparison were also
  re-done (section 7). V19's three unresolved placeholders (in its verification matrix and test section) were resolved from these results.

## 2. Architecture and data flow
`market_data.py` (instrument resolution, quote/history providers, TTL caches, request coalescing, freshness classification, failure caching)
-> `forecast_service.py` (single background worker, job registry, result cache keyed by last completed bar, warm-up) -> `pipeline.py`
(feature engineering, Linear Regression, LSTM with provenance-checked checkpoints, ensemble) -> `evaluation.py` (same-window naive/drift/LR/LSTM/
ensemble comparison, MASE, block-bootstrap intervals, rolling past-only conformal intervals) -> `backtest.py` (next-open execution, costs, slippage,
exposure/turnover). `ai_integrations/providers.py` + `research.py` produce validated, fact-grounded commentary that is **never** part of the numeric
forecast. `webapp/` serves the workstation UI (`/api/v2`) and the classic forms/JSON API; paper trading only is reachable from the workstation.
`data_sources/` holds the legacy history adapters (Yahoo, Upstox, Groww); `market_session.py` removes bars for sessions still in progress.

## 3. Feature inventory (all present in V19; V20 changes marked)
Quotes with freshness classes and visible fallback · daily history, indicators, data-quality report (V20: new checks) · LR / LSTM / ensemble forecast as a
background job · baseline-relative evaluation (V20: MASE, intervals) · prediction intervals (rolling, past-only) · backtest (V20: corrected trade P&L,
exposure, turnover) · AI commentary with provider status · paper portfolio, risk page, position sizing · health and measured performance page ·
classic `/predict`, `/api/*` (V20: ticker validation) · scripts: provider verification, multi-ticker evaluation, benchmark, browser audit (V20: release
gate, packager).

## 4. Findings

### 4.1 CONFIRMED defects (fixed in V20; each has a regression test)
| ID | Defect | Evidence (reproduced against V19) | Fix | Tests |
|---|---|---|---|---|
| D1 | **Trade-level P&L omitted the entry fee** (original engine, carried through V19): ledger, win rate, profit factor, average trade return overstated | flat price, 1% cost each way: equity change **-19.90**, ledger P&L **-9.90**, avg trade return -1.00% (true -1.99%); a +0.5% trade with 1% fees counted as a *win* | cost basis includes the entry fee; ledger now reconciles with equity | `test_v20_backtest_accounting` (incl. a property test over 8 cost/slippage/mode settings x 12 random paths) |
| D2 | **Classic `/predict` accepted any text as a ticker** and passed it to the provider layer | `../../etc/passwd`, `AAPL?x=1&y=2`, `<script>...`, a 500-character string all reached the compute function (HTTP 302/202) | one shared rule (1-20 chars, `A-Z 0-9 . - & =`, optional leading `^`) on every route | `test_v20_security` (9 hostile strings x 5 routes; 10 valid forms still accepted) |
| D3 | **Data-quality blind spots**: a frozen feed (20 identical O=H=L=C days) and a 6-day zero-volume stretch reported as clean; `inf` reported only indirectly as an "impossible OHLC relationship"; a timezone-aware index was never normalised (comparison with naive dates raises `TypeError`) | clean probes with otherwise valid frames (the checker correctly caught unsorted, duplicate, NaN, negative/zero price, High<Low, negative volume and gap cases) | new warnings/issue, tz normalised in `standardize_dataframe` and defensively in `evaluation` | `test_v20_data_integrity` |
| D4 | **Model-vs-naive verdicts rested on a point estimate**, and a plain 95% bootstrap interval would have been overconfident | calibration experiment (synthetic, no true skill; iid-normal errors with log-AR(1) volatility; n = 60/120/250; 300 replications): the 95% interval reported "beats naive" in **4-8%** of trials (nominal 2.5% one-sided) and excluded 0 in 8-13% (nominal 5%); the 99% interval reported "beats" in **0.3-3.3%** | block-bootstrap intervals; the verdict uses the **99%** interval; both are shown | `test_v20_evaluation_stats` (incl. a seeded calibration regression) |
| D5 | No scaled error metric; evaluation horizon not declared | code read | MASE (training-period naive MAE as scale, strictly past data); explicit one-day horizon statement | same file |

### 4.2 Hardening and presentation fixes (not defects in the strict sense)
* **Model loading:** Keras `safe_mode=True` is the default, but it is now passed explicitly; a checkpoint with a Python `Lambda` layer and *matching* integrity hashes is
  refused and its code does not run (test writes a marker file from inside the Lambda and asserts it never appears).
* **Backtest view:** a zero-trade or low-exposure result showed "Strategy minus Buy & Hold +3.96%" in green with no context; it now says the strategy stayed in cash
  and that the difference is the market's move, not skill.
* **Test coverage gaps closed:** `news_feed.py` (27.8% -> parser, XML-entity refusal, size cap, dedupe, status logic tested) and the `trade.py` CLI's live-trading guard.

### 4.3 SUSPECTED risks / not fixed (documented, not hidden)
| ID | Risk | Status |
|---|---|---|
| R1 | Every provider integration (Yahoo, Upstox, Groww, Gemini, Groq, OpenRouter, ...) is unverified against a live endpoint | **NOT RUN**: no network or credentials in the build environment |
| R2 | No trading calendar: holidays and half-days are not modelled; close times are an approximation table | SUSPECTED; the session rule errs toward dropping a bar |
| R3 | `submit_forecast` fetches history on the request thread (bounded by timeouts; other requests are not blocked - tested) | SUSPECTED, low impact |
| R4 | Programmatic calls without `async` hold a request thread up to `LEGACY_PREDICT_WAIT_SECONDS` (default 120 s) - required by the original tests | by design, documented |
| R5 | The Flask development server is the only server; it binds to 127.0.0.1 only | SUSPECTED if someone exposes it; use a production WSGI server and authentication before any exposure |
| R6 | Only a one-day horizon exists; multi-horizon evaluation is impossible without new models | limitation |
| R7 | An LLM contribution to numeric forecasts remains unvalidated (needs point-in-time historical LLM outputs that do not exist); forced to 0 by default | limitation |
| R8 | No CI configuration; the release gate is a local script | limitation |

## 5. Test coverage and execution
* **Line coverage (statements, excluding tests): 85.0% of 5585** - measured by running the full suite under `coverage` 7.16 in a
  throw-away install (removed afterwards; the environment was re-checked against the lock). **23 of 46** modules with >= 20 statements are at or above 90%.
  Subprocess-based tests (scripts, server start) are not traced, so `scripts/*` read low. This run (528 tests) was taken *before* the 15 news/trade-CLI tests were added.
  Lowest modules with >= 40 statements in that run:

| Module | Statements | Covered |
|---|---|---|
| `trade.py` | 50 | 0.0% |
| `news_feed.py` | 72 | 27.8% |
| `scripts/release_gate.py` | 111 | 36.9% |
| `data_sources/upstox_source.py` | 109 | 47.7% |
| `scripts/package_release.py` | 70 | 52.9% |
| `visualize.py` | 69 | 65.2% |
| `main.py` | 73 | 67.1% |
| `trading/trade_executor.py` | 114 | 74.6% |

* Suites: original 188 (unmodified) + V17 110 + V18 82 + V19 62 = 442 at baseline. **Final: 545 collected and 545 passed** in the gate run, i.e. V20 adds 103 tests in new
  `test_v20_*` files (data integrity, evaluation statistics, backtest accounting, security, release tools, CLI and news parsing, plus a small file for the
  multi-ticker summary and the backtest view). **All 34 pre-existing test and helper files are byte-identical to the V19 baseline** (checked with `cmp`); nothing was edited, removed or skipped.
* Mutation checks performed in V20 (each made the relevant tests fail; files restored): the XML-entity guard, the live-trading CLI guard, and (earlier in the V19-V20
  work) the cached-quote reclassification, the evaluated-residual leak into calibration, and the next-open fill.

## 6. Dependency and security
* **Vulnerability scan:** `pip-audit` 2.10.1 against PyPI advisory data on the 80 pins of `requirements-lock.txt`: **80 of 80 pins checked, 0 with known
  vulnerabilities**. A first scan of `requirements-dev-lock.txt` checked only 4 packages (the tool does not follow `-r` includes with `--disable-pip`) and was
  discarded rather than reported as a clean result. Advisory databases lag; this is a point-in-time check.
* Secrets: repository scan (packager patterns) finds none outside deliberately fake fixtures under `tests/`; `.env.example` holds placeholders only.
* Checked and found sound: no `pickle`/`yaml.load`/`eval`/`exec`/`subprocess` in application code; templates never disable autoescaping and the UI inserts text via
  `textContent`; CSRF on all state-changing routes; per-IP rate limits; 256 KB request cap; HttpOnly + SameSite cookies; localhost-only bind; no SQL layer exists.
* Not applicable here: authentication/authorisation (single local user by design), uploads (none), database (none).

## 7. Performance (measured; 1 vCPU, 3997 MB; synthetic 520-bar fixture; a fake provider with a fixed 300 ms delay stands in for upstream)
| Measurement | V19 | V20 |
|---|---|---|
| Quote cache hit, median (500 requests) | 0.0184 ms | 0.0188 ms |
| Upstream calls for 501 quote requests | 1 | 1 |
| First forecast job, 5 epochs | 9.51 s (no warm-up; this run) | 3.72 s (after background warm-up of 4236.9 ms) |
| Evaluation stage | 11.9-14.7 ms | 23.1 ms (includes the new bootstrap) |
| Same bar again (result cache) | 0.18 ms | 0.13 ms |
| Checkpoint reuse, end to end | 0.56 s | 0.7 s |
| Backtest median / p95 | 2.0 / 12.7 ms | 2.0 / 4.8 ms |

Sample sizes: one forecast run per column (quote: 500, backtest: 20). Cold start varies widely with the file cache (the same V19 code measured 9.5 s and 23.1 s
without warm-up). The bootstrap costs a median of 15.9 / 37.4 / 66.8 ms at 100 / 250 / 500 out-of-sample days (5 repetitions each). Bottleneck: LSTM training
(3.2 s for 5 epochs here), already off the request thread and checkpoint-reused. Provider latency was not measurable (no network).

## 8. Release-gate run on the working tree (recorded by `scripts/release_gate.py`; commands, exit codes and elapsed seconds are the script's own)
The gate was invoked twice: `--skip-tests` (environment and start-up checks) and `--tests-only` (the full suite). A single ten-minute invocation was started first
and was killed by the sandbox part-way through with no result written, so it is **not counted**; splitting the run keeps each part short enough to complete.
| Step | Command | Result | Exit | Seconds | Detail |
|---|---|---|---|---|---|
| python version | `python --version` | **INFO** | 0 | 0 | 3.12.3 |
| pip check | `python -m pip check` | **PASS** | 0 | 3.96 | No broken requirements found. |
| real TensorFlow import | `python -c "import tensorflow as tf, keras; print('t..."` | **PASS** | 0 | 27.9 | tensorflow 2.21.0 keras 3.15.1 |
| installed set == lock | `pip freeze vs requirements-dev-lock.txt` | **PASS** | 0 | 0.29 | identical (84 packages compared; pip/setuptools/wheel ignored) |
| compile all sources | `python -m compileall -q .` | **PASS** | 0 | 0.16 |  |
| server start-up smoke test | `python webapp/app.py (random localhost port)` | **PASS** | 0 | 5.09 | {"/": 200, "/api/v2/meta": 200, "/api/v2/health": 200, "/api/v2/ai/status": 200, "/static/ |
| full test suite | `-m pytest tests webapp/test_app.py -q -p no:cacheprovider --ignore=tes` | **PASS** | 0 | 284.54 | {"passed": 545, "pytest_reported_seconds": 272.12} |

Full suite: **545 passed, 0 failed, 0 skipped, 0 errors**.

## 9. Prioritised remediation plan
| Priority | Item | Impact | Effort | Regression risk | Status |
|---|---|---|---|---|---|
| 1 | Trade P&L must include entry fee (D1) | high (misleading strategy metrics) | low | low (no original test pinned it) | **done** |
| 2 | One ticker rule on every route (D2) | medium (input validation) | low | low | **done** |
| 3 | CI-based verdict + MASE (D4, D5) | high (prevents over-claiming skill) | medium | low | **done** |
| 4 | Data-quality gaps (D3) | medium | low | low | **done** |
| 5 | Verify providers with real credentials (`scripts/verify_providers.py`) | **highest** (every data/AI claim depends on it) | low, needs credentials | none | not run |
| 6 | Run `scripts/evaluate_universe.py` on a real, pre-declared ticker list | high (the only way to learn whether any model beats naive) | low, needs data access | none | not run |
| 7 | Trading calendar + move history fetch into the job | medium | medium | medium | open |
| 8 | Production WSGI server + authentication before any network exposure | high if exposed | medium | medium | open |
