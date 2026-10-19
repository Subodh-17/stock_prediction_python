# V20 release notes

## Status
**PASS WITH LIMITATIONS.** Everything that could be executed offline was executed and passed (section "Verification"); every live-provider check is **NOT RUN**
(no network or credentials). No model-quality claim is made: no real-market evaluation exists.

## Supported environment
CPython **3.12** on **Linux x86_64** (verified). Windows and macOS: **not verified** (the previous Windows/3.11 lock is kept as
`requirements-lock-windows-py311-legacy.txt`). CPU only (`tensorflow-cpu`). Recommended: >= 4 GB RAM and >= 6 GB free disk for the environment (a TensorFlow
environment is about 2 GB; check `df -h` first). Do not run two TensorFlow processes at once on a 4 GB machine.

## Clean installation (commands marked **ran** were executed in V20)
```bash
python3.12 -m venv .venv && source .venv/bin/activate          # ran
pip install -r requirements-dev-lock.txt                       # ran: exit 0, then `pip check` clean, real TensorFlow import OK (runtime only: requirements-lock.txt)
cp .env.example .env                                           # edit placeholders; the app starts with NO keys
python scripts/release_gate.py                                 # ran (below)
```
```powershell
# Windows PowerShell - NOT RUN in V20 (no Windows host); unpinned install
py -3.12 -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r webapp/requirements.txt ; Copy-Item .env.example .env
```

## Environment variables
All are in `.env.example` with **placeholders only**; none is required. Groups: data (`DATA_SOURCE`, `UPSTOX_*`, `GROWW_*`, freshness/TTL settings), AI providers
(`GOOGLE_API_KEY`, `GOOGLE_MODEL`, `GROQ_*`, `OPENROUTER_*`, `OPENAI_*`, `ANTHROPIC_*`, ...), model (`LOOKBACK_WINDOW`, `LSTM_*`, `LR_WEIGHT`, `LSTM_WEIGHT`,
`LLM_PANEL_WEIGHT` which is **ignored unless** `LLM_PANEL_WEIGHT_EVIDENCE_ACK` carries the documented acknowledgement), trading (paper default;
`ENABLE_LIVE_TRADING` and a 32+ character `WEBAPP_LIVE_TRADING_TOKEN` are both needed for any live path), web (`WEBAPP_SECRET_KEY` - leave unset for a random one,
`WEBAPP_PORT`, `WEBAPP_EXPENSIVE_RPM`), jobs (`LEGACY_PREDICT_WAIT_SECONDS`, `JOB_TIMEOUT_SECONDS`, `WARMUP_ON_START`), `DROP_OPEN_SESSION_BAR`, `HISTORY_FAILURE_TTL_SECONDS`.
Never commit `.env`; the packager refuses to include it.

## Provider setup and limits
Yahoo (via yfinance) needs no key and is **never labelled Live**; Upstox needs a daily token; Gemini needs a key from Google AI Studio (default model
`gemini-3.5-flash-lite`, confirmed current in Google's deprecation page dated 2026-10-12; free-tier eligibility and quotas are per project - check Google's pages).
Run `python scripts/verify_providers.py` with your credentials: it makes real read-only requests and records PASS / FAIL / BLOCKED / NOT_CONFIGURED without secrets.
**Verification status in V20: NOT RUN for every provider** (ran offline: exit 0, all BLOCKED or NOT_CONFIGURED).

## Commands
| Purpose | Command | V20 status |
|---|---|---|
| Web app (localhost only) | `python webapp/app.py` | ran via the gate's start-up smoke test (HTTP 200 on 6 URLs) |
| CLI prediction | `python main.py --ticker AAPL` | `--help` ran (exit 0); a real run needs network: NOT RUN |
| Paper/live CLI | `python trade.py --ticker AAPL` (paper by default) | guards tested; a run needs network: NOT RUN |
| Full tests | `python -m pytest tests webapp/test_app.py -q` | ran (see Verification) |
| Release gate | `python scripts/release_gate.py` | ran |
| Package | `python scripts/package_release.py --out stock_prediction_python_v20.zip` | ran |
| Multi-ticker evaluation | `python scripts/evaluate_universe.py AAPL NSE:TCS --epochs 15` | ran offline: exit 0, every row reported `failed` (no data), nothing invented |
| Benchmark | `python scripts/benchmark.py --epochs 5 --bars 520 --warmup` | ran: exit 0 |
| Browser audit | `AXE_JS=... python scripts/browser_audit.py` (needs Playwright, Chromium, axe-core, and `python tests/qa_harness.py` running) | ran: exit 0, 0 axe violations on 30 pages |

## Model artefacts and checkpoints
LSTM checkpoints live under `CHECKPOINT_DIR`, one directory per name, each with `model.keras`, `scaler.json` (no pickle) and `metadata.json` (manifest v2: training
start/end, data fingerprints, hashes, versions, seed, feature schema). A checkpoint is reused only if its training data ended strictly before the first evaluated date,
the history still matches, the versions' majors match and the hashes verify; otherwise it is retrained and only that name is overwritten. Models load with Keras
`safe_mode=True`. Checkpoints are excluded from the release archive.

## Reading the numbers
* **MAE / RMSE** are in price units on the next-day close; **MASE** divides MAE by the average absolute daily change in the *training* period (below 1 beats that scale).
* **RMSE vs naive** is `1 - RMSE_model/RMSE_naive`. The table shows 95% and 99% moving-block bootstrap intervals. **The verdict says "beats" only if the 99% interval is
  above zero**, because in a calibration experiment the 95% interval overstated skill. Intervals assume weak dependence beyond one block and are indicative.
* **Directional accuracy** excludes flat days and has its own interval. A model that matches naive has no directional call (reported unavailable).
* **Prediction interval:** rolling, past-only conformal on return residuals. "Nominal" is the target; "observed" is measured historical coverage of the *same procedure*. No
  guarantee is claimed for dependent, non-stationary returns.
* **Backtest:** next-open fills, costs and slippage per side, no fill if the open is missing. Trade P&L includes the entry fee. **Exposure** is the share of days holding a
  position; **turnover** is traded value divided by initial capital. A low-exposure strategy beats Buy & Hold in a falling market without skill. Forecast quality and strategy
  profit are separate questions; neither is claimed here.

## Troubleshooting
| Symptom | Cause / fix |
|---|---|
| `OSError: No space left on device` during `pip install`; later `No module named 'tensorflow.python'` | disk full during extraction; the half-installed environment can still pass `pip check`. Free space (an environment is ~2 GB), delete the venv, reinstall, then verify with a real `import tensorflow` |
| Yahoo requests fail with `ImpersonateError ... chrome150` | TensorFlow was imported before yfinance's HTTP client; import `config` first (the project does) |
| Process killed / tests die part-way on a 4 GB machine | two TensorFlow processes at once; run one at a time, set `WARMUP_ON_START=false` |
| Gemini `model_unavailable` / 404 | the model id is retired or not available to your key; use AI Research -> Test connection to list usable models |
| `insufficient_quota` / 429 | free-tier quota exhausted; wait, add credit, or use another provider (numerical features are unaffected) |
| Port in use | set `WEBAPP_PORT` |
| `POST /predict` returns 400 for a ticker | tickers are 1-20 characters of letters, digits and `. - & =` (optionally starting `^`); use `NSE:` / `BSE:` prefixes in the workstation |

## Known limitations and unverified integrations
See `docs/V20_ENGINEERING_AUDIT.md` section 4.3. In short: no live provider verified; no real-market model evaluation; no trading calendar; Windows/macOS untested;
one-day horizon only; LLM numeric contribution unvalidated; Flask development server only.

## Upgrade notes from V19 (behaviour changes)
1. **Backtest trade P&L now includes the entry fee** -> win rate, profit factor and average trade return will be lower than V19 for the same data; equity curve and total return are unchanged.
2. **Verdict text and rule changed:** "beats naive" needs the 99% interval above zero; evaluation rows gain `mase`, `rmse_improvement_ci95/99`, `directional_accuracy_ci95`.
3. **Classic `/predict` validates tickers** (400 on invalid); `&`, `=` and a leading `^` are now accepted everywhere.
4. New data-quality warnings (frozen feed, zero-volume stretch, tz-aware index); a tz-aware provider index is converted to naive exchange-local dates.
5. New backtest fields `exposure_pct`, `turnover_x`. New scripts `release_gate.py`, `package_release.py`; `pyflakes` is in the dev lock.
6. Test files: no earlier test file was modified; new files are `test_v20_*`.

## Verification (work tree; the extracted-archive run is reported in the final summary, not in this file)
Gate steps and results are in `docs/V20_ENGINEERING_AUDIT.md` section 8: pip check, real TensorFlow import, installed set vs lock, compile, start-up smoke test, full suite.
