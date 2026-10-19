# V19 final audit

> *Placeholders in this document were resolved during V20 from results obtained in V20 Phase A; see `docs/V20_ENGINEERING_AUDIT.md`.*

Classification: **PASS** = executed and passed · **FAIL** = executed and failed · **BLOCKED** = could not execute (no network /
credentials / platform) · **NOT RUN** = not attempted. A mocked HTTP response is **not** a provider test. A passing unit test
establishes mechanics, **not** real-world prediction accuracy.

## 1. Verification matrix

| Item | Result | Evidence / reason |
|---|---|---|
| V18 baseline (untouched rollback copy), full suite | **PASS** | 380 passed in 385 s. (A first attempt died part-way because it ran concurrently with another TensorFlow job on a 4 GB machine; it produced no summary and was **not** counted. It was re-run alone.) |
| Python compilation | **PASS** | V18 rollback and extracted V19 tree |
| V19 full suite from the extracted ZIP in a venv created from its dev lock | **PASS (resolved during V20)** -- 442 passed, 0 failed, 0 skipped in 262 s (4:22), run in V20 Phase A from a pristine extraction of `stock_prediction_python_v19.zip` (sha256 begins `efdddc91`) inside a clean virtual environment created from that archive's `requirements-dev-lock.txt` (CPython 3.12.3, Linux x86_64, 1 vCPU, 4 GB). The pytest summary line was recorded; the separate exit-code line was not captured for this run (the wrapper's trailing `echo` never wrote), so the exit status is inferred from the summary only. V19's own attempt to run the suite from its extracted ZIP at release time was interrupted before completion (no summary line) and is **not** counted anywhere | |
| Clean install from the delivered lockfiles (CPython 3.12.3, Linux x86_64) | **PASS (resolved during V20)** -- a virtual environment created from the extracted V19 ZIP's `requirements-dev-lock.txt`: the first attempt failed with `OSError: No space left on device` (disk full from earlier environments; the half-installed environment even passed `pip check`, so that check alone was not sufficient evidence), the second, after freeing disk, ended `PIP_EXIT=0` with `Successfully installed`; a real `import tensorflow` succeeded (tensorflow 2.21.0, keras 3.15.1, numpy 2.5.4, pandas 3.0.6); `pip check`: *No broken requirements found*; the 84 installed packages equal the V18-verified set plus pyflakes. Linux x86_64 / Python 3.12 only | |
| pyflakes: no undefined names anywhere (now a test) | **PASS** | found one real defect, see 2.9 |
| Real-browser audit (Chromium; 1440 / 820 / 390 px; all 10 routes; 30 pages) | **PASS** | 0 console/CSP errors; 0 horizontal overflow; focus outline on every tabbed control; **axe-core 4.14 WCAG 2.0/2.1/2.2 A+AA: 0 violations**; **0 rendering artefacts** (`[object`, `undefined`, `NaN`, bare `null`) in visible text. Run against the QA harness (synthetic inputs + fake AI HTTP) -- it audits the UI, not provider behaviour. axe covers a subset of accessibility issues; no screen-reader testing |
| Real Yahoo / Upstox / Groww requests | **BLOCKED** | hosts unreachable from the build sandbox; no tokens |
| Real Gemini / Groq / OpenRouter / OpenAI / Anthropic / xAI / DeepSeek requests | **BLOCKED** | hosts unreachable; no API keys (re-checked at the start of this session) |
| Windows / macOS install | **NOT RUN** | no such platform available |
| Official documentation re-consulted | done | Google Gemini deprecations page (dated **2026-10-12**): `gemini-3.5-flash-lite` (default) is current and recommended for new projects, no shutdown date; `gemini-2.0-flash` shut down 2026-06-01; 2.5 family limited to accounts that used it. Upstox market-quote docs (see V18 record) |

**No provider is verified.** Run `python scripts/verify_providers.py` with your own credentials to change that (see
`docs/V18_CHANGES.md` section 8).

## 2. Defects found against V18 and fixed

Each was **reproduced against the V18 code first** (a one-off script), then fixed, then covered by a regression test that fails on the
old behaviour.

| # | Defect (V18) | Impact | Fix | Regression test |
|---|---|---|---|---|
| 2.1 | A **stale real-time quote, once cached, came back as `unverified` with `is_stale=False`** (re-classified without remembering the provider's real-time entitlement) | the paper-order gate blocks only `stale`/`unavailable`, so a stale price could be traded | quote remembers `provider_entitled_realtime`; cached quotes re-classified with the same rules | `test_v19_regressions` (service, ageing, and the paper-order route); mutation-checked |
| 2.2 | A failing **history** request was retried on every UI refresh (5 refreshes -> 5 upstream calls) | hammering a failing provider | negative cache (`HISTORY_FAILURE_TTL_SECONDS`, default 30 s; rate limits honour `Retry-After`) | `test_history_failures_are_negatively_cached...` |
| 2.3 | A **17-day data gap was scored as a "next-day" target** | multi-day errors mixed into next-day metrics, intervals and backtests | rows with > 5 calendar days between base and target are excluded and counted (`n_dropped_gap`) | `test_a_multi_week_data_gap...` |
| 2.4 | `LLM_PANEL_WEIGHT=0.3` was honoured with no evidence (weights summed to 1.3) | an LLM could move the numeric price | forced to 0 unless `LLM_PANEL_WEIGHT_EVIDENCE_ACK=I_HAVE_OUT_OF_SAMPLE_EVIDENCE`; reason exposed (`LLM_WEIGHT_NOTE`, Settings) | subprocess tests |
| 2.5 | **Checkpoint dependency compatibility was never checked** (`package_versions` stored, never compared; `keras` not even recorded) | a model saved under another major TensorFlow/Keras/NumPy could be reused | major-version check for tensorflow/keras/numpy; manifest without versions is unprovable | `test_load_checkpoint_refuses_incompatible_dependency_versions_end_to_end` |
| 2.6 | `/api/signal` and `/api/backtest` still ran the whole pipeline inside the request | blocked a web worker | background jobs with the same `async` opt-in; generic `GET /api/jobs/<id>/result`; job ids lengthened to 96 bits | `test_v19_routes` (incl. "slow job never blocks quotes/health/history") |
| 2.7 | The **classic/CLI fetch path kept an unfinished current-day bar**, and V17's single 16:00 cut-off would keep a forming bar on later-closing exchanges (Xetra, Euronext, SGX, LSE) | intraday price treated as a settled close | `market_session.py`: per-exchange close table + buffer; applied in `data_sources.fetch_stock_data`; reported on the results page | `test_v19_session` (8 exchanges either side of the boundary, legacy path with a pinned clock, full pipeline) |
| 2.8 | **Index symbols (`^GSPC`, `^NSEI`...) were rejected** by the resolver; an Indian index would also have used New York's clock | cannot watch indices; wrong session logic | leading `^` accepted; explicit exchange map for common indices | `test_index_symbols_resolve...` |
| 2.9 | `scripts/browser_audit.py` (documented as reproducible) **did not start**: `os` used before import | the V18 audit could not be reproduced | fixed; pyflakes undefined-name check added to the suite | `test_v19_static_analysis` |
| 2.10 | **UI rendered `[object HTMLSpanElement]` instead of badges** whenever a quote came from the server cache (and the same pattern for data-quality flags); Settings printed bare `null` | visibly broken status line; **missed by the V18 browser audit because it always forced a fresh quote** | DOM insertion is now array-safe everywhere; `badge()` returns one element; the audit re-loads from cache and scans every route's text for artefacts | audit scan + mutation test against the old code |
| 2.11 | The next-day **forecast never appeared on the price chart**; a page reload discarded it | forecast and observed data not shown together | model estimate drawn in a fixed-width zone right of a divider (dashed whisker + open diamond, never a candle), legend, tooltip, accessible name; restored after reload for the same symbol only | `test_v19_ui`; screenshot reviewed |
| 2.12 | **~4-19 s cold-start cost paid by the first user's forecast** (TensorFlow import + LSTM tracing) | slow first forecast | background `warm_up()` at server start (opt-out `WARMUP_ON_START=false`) | `test_v19_perf` + measurements below |

Claims in the V18 notes that were re-verified rather than trusted: rolling conformal calibration (tests + mutation check: leaking the
evaluated residual into its own calibration makes two tests fail), checkpoint provenance (19 tests), lock reproducibility (fresh venv),
next-open execution, and redaction. The V18 statement that the browser audit was clean was **true but incomplete** (2.9, 2.10).

## 3. Methodology (unchanged core; V19 additions)
Same-window comparison of naive / drift / LR / LSTM / LR+LSTM; rolling past-only conformal intervals; checkpoint provenance manifest;
fixed (untuned) ensemble weights; next-open backtest with slippage and no-fill on a missing open -- as documented in
`docs/V18_CHANGES.md`. V19 adds: the gap-exclusion rule (2.3), the per-exchange session rule (2.7), the LLM evidence gate (2.4), the
dependency-major rule (2.5), and `scripts/evaluate_universe.py` (see 4).

**Ensemble integrity.** The LLM numeric weight is 0 and cannot be raised without the explicit acknowledgement above. No leakage-free
chronological experiment showing incremental LLM value exists (it cannot be built without point-in-time historical LLM outputs), so
AI remains commentary only.

## 4. Model evaluation results -- what exists and what does not
* **No real-market evaluation was produced.** The sandbox has no market-data access, so there are **no** real MAE/RMSE/directional
  numbers, no evaluation period and no ticker list to report. Nothing was cherry-picked because nothing was run on real data.
* The only numbers that exist are from **synthetic random-walk fixtures** (the QA harness): on 102 held-out synthetic days LR / LSTM /
  ensemble were 1.0 % / 1.3 % / 0.8 % **worse** than the naive last-close forecast on RMSE. That is the expected outcome on a random
  walk, shows the evaluation does not flatter models, and says **nothing** about real markets.
* To produce real evidence without selection, use `python scripts/evaluate_universe.py SYM1 SYM2 ...` with your provider access. It lists
  **every** symbol (failures and insufficient-data cases included, none retried or dropped), and summarises per model: how many
  instruments beat naive, median/best/worst RMSE improvement, and an exact two-sided sign-test p-value. Instruments are not independent
  (markets co-move) and beating naive on RMSE is not a trading edge; the summary says so. Tests cover membership, order, one fetch per
  symbol, failure reporting with redaction, and the sign test.

## 5. Measured performance (synthetic fixture bars, CPU of the build machine; **not** real-provider latency)
`python scripts/benchmark.py --epochs 5 --bars 520` (a fake provider with a fixed 300 ms delay stands in for upstream). Values below are
copied from `outputs/benchmark.json` of the run noted; each is a wall-clock measurement taken by that run.

| Measurement | Value |
|---|---|
| Quote, upstream stand-in miss | 300.9 ms |
| Quote, cache hit (500 requests) | median 0.0184 ms, p95 0.0257 ms; upstream calls for 501 requests: 1 |
| `submit_forecast` returned after (first call) | 1029.8 ms (cold imports); a repeat for the same bar returns in 0.18 ms (cache hit: True) |
| First forecast job end-to-end, no warm-up (this run) | 9.51 s (stage timings: {'history': 5.1, 'models': 5598.0, 'evaluation': 16.9}) |
| Same bar, result cache cleared, checkpoint reused | 0.56 s |
| Backtest | median 2.0 ms, p95 12.7 ms |

**Cold start (separate fresh-process runs, one sample each):** first forecast job **23.1 s without warm-up vs 3.7 s after a
background warm-up** (warm-up itself took 3.7 s, in the background). An earlier cold run measured 9.5 s, so cold-start time varies
widely with the machine's file cache; the improvement is real but its size is machine-dependent.

Per-stage latencies recorded by the app itself (also on the Data & Model Health page):

| Operation | Samples | Median ms | p95 ms | Max ms |
|---|---|---|---|---|
| `backtest.run` | 20 | 0.29 | 0.5 | 0.5 |
| `forecast.job_total` | 2 | 9486.72 | 9486.72 | 9486.72 |
| `history.upstream` | 1 | 0.12 | 0.12 | 0.12 |
| `pipeline.feature_engineering_lstm` | 2 | 1.1 | 1.1 | 1.1 |
| `pipeline.lr_train_eval` | 2 | 4.37 | 4.37 | 4.37 |
| `pipeline.lstm_checkpoint_load` | 1 | 149.07 | 149.07 | 149.07 |
| `pipeline.lstm_inference` | 2 | 676.79 | 676.79 | 676.79 |
| `pipeline.lstm_train` | 1 | 4701.53 | 4701.53 | 4701.53 |
| `quote.upstream.yahoo` | 1 | 300.58 | 300.58 | 300.58 |

Bottlenecks seen: LSTM **training** (about 4.7 s for 5 epochs here) dominates and is already off the request
thread and checkpoint-reused (about 0.56 s on reuse); LSTM first-call inference includes graph tracing.

## 6. Provider and AI status
Unchanged from `docs/V18_CHANGES.md` sections 4-5 (all **BLOCKED**), plus: Gemini default re-checked against Google's page dated 2026-10-12
(section 1); `mock_verified` can never display as `verified`; the Health page lists only records written by `scripts/verify_providers.py`.

## 7. Tests
Collected in the V20 baseline (442): original 188 (`test_final_pass` 18, `test_hardening` 54, `test_phase_improvements` 116), V17 110 (`ai` 30, `backtest` 10, `evaluation` 9, `market_data` 31, `webapp` 30), V18 82 (`checkpoints` 19, `conformal` 12, `forecast_e2e` 4, `jobs` 16, `native_preload` 2, `providers` 13, `security` 9, `ui` 7), V19 62 (`perf` 6, `regressions` 12, `routes` 7, `session` 26, `static_analysis` 2, `ui` 6, `universe` 3). No earlier test file was edited in V19 (byte-compared against the V18 tree). Result of the full run: see the Suite row of the verification matrix

## 8. Known limitations / unresolved
* **Nothing was tested against a live provider.** Windows/macOS installs unverified. No screen-reader testing.
* No real-market model evaluation or backtest results exist (section 4).
* `submit_forecast` still fetches history on the request thread (bounded by timeouts; other requests are not blocked -- tested).
* Programmatic `/predict` and `/api/*` calls **without** `async` still hold a request thread for up to `LEGACY_PREDICT_WAIT_SECONDS`
  (default 120 s) because the original tests expect synchronous results; HTML forms and `async=1` clients return at once.
* Session close times are an approximation table (no half-days/early closes/holidays); the drop rule errs toward removing a bar.
* Cancellation does not interrupt a running Keras fit. The `ImpersonateError` mechanism (V18) remains a hypothesis; the fix was verified only on Linux.
* Warm-up uses roughly the memory of one TensorFlow process at server start; disable with `WARMUP_ON_START=false` on small machines
  (a 4 GB machine could not run two TensorFlow jobs at once during this work).
* Interval coverage guarantees are **not** claimed (serial dependence / non-stationarity); LLM numeric contribution remains unvalidated.
