"""
Functional tests for webapp/app.py using Flask's test client -- unlike the
Scilab GUI, a Flask app can be genuinely tested end-to-end without a real
browser or display: the test client sends real HTTP requests in-process and
returns real HTML responses to assert against.

Run from the project root:
    python webapp/test_app.py
"""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

import config
import app as webapp_module

app = webapp_module.app
app.config["TESTING"] = True

n_pass = 0


def check(condition, message):
    global n_pass
    if condition:
        print(f"PASS: {message}")
        n_pass += 1
    else:
        print(f"FAIL: {message}")
        raise AssertionError(f"Test failed: {message}")


# ---------------------------------------------------------------------------
# Build a synthetic dataset + mock pipeline result, same pattern used in the
# CLI's own integration test, so these tests run fast and need no network,
# no API keys, and no real LSTM training wait.
# ---------------------------------------------------------------------------
np.random.seed(11)
n = 300
dates = pd.bdate_range("2024-01-01", periods=n)
returns = np.random.normal(loc=0.0006, scale=0.014, size=n)
close = 180 * np.exp(np.cumsum(returns))
synthetic_df = pd.DataFrame({
    "Open": close * 1.001, "High": close * 1.01, "Low": close * 0.99,
    "Close": close, "Volume": np.random.randint(1_000_000, 5_000_000, n),
}, index=dates)


def _fake_pipeline_result(ticker="TESTX", skip_llm=True):
    """Runs the REAL pipeline logic (LR + LSTM fit/eval) against synthetic
    data via a mocked data fetch, so results are genuinely computed, not
    hand-faked -- just fast and offline.

    IMPORTANT: this must be called and its result captured BEFORE
    app.pipeline.run_pipeline is patched with it, since app.py's `import
    pipeline` and this file's `import pipeline as pipeline_module` are the
    same shared module object -- patching app.pipeline.run_pipeline also
    changes what pipeline_module.run_pipeline points to. Calling the real
    one from inside a side_effect that has already been substituted for
    itself would recurse forever.
    """
    import pipeline as pipeline_module
    with patch("data_loader.fetch_stock_data", return_value=synthetic_df):
        return pipeline_module.run_pipeline(
            ticker=ticker, period="2y", epochs=3, source="yfinance",
            skip_llm=skip_llm, verbose=False,
        )


# Compute the real result ONCE, up front, before any patching happens.
_PRECOMPUTED_RESULT = _fake_pipeline_result("TESTX", skip_llm=True)


client = app.test_client()


def _csrf(c):
    """Fetches a valid CSRF token bound to `c`'s session -- the same way a
    real browser gets one, by first GETting a page that renders a form."""
    c.get("/")
    with c.session_transaction() as sess:
        return sess["csrf_token"]


# ---------------------------------------------------------------------------
# Test 0: CSRF protection on state-changing routes
# ---------------------------------------------------------------------------
csrf_client = app.test_client()
resp = csrf_client.post("/predict", data={"ticker": "TESTX", "period": "2y", "epochs": "3"})
check(resp.status_code == 403, "POST /predict with NO csrf_token -> 403, not processed")

resp = csrf_client.post("/predict", data={
    "ticker": "TESTX", "period": "2y", "epochs": "3", "csrf_token": "totally-forged-value",
})
check(resp.status_code == 403, "POST /predict with an INVALID csrf_token -> 403, not processed")

resp = csrf_client.get("/")
check(resp.status_code == 200, "GET / is completely unaffected by CSRF protection")

# ---------------------------------------------------------------------------
# Test 1: index page loads and contains the form
# ---------------------------------------------------------------------------
resp = client.get("/")
check(resp.status_code == 200, "GET / returns 200")
check(b"ticker" in resp.data.lower(), "GET / contains the ticker input field")
check(b"Run Prediction" in resp.data, "GET / contains the submit button")

# ---------------------------------------------------------------------------
# Test 2: predicting with an empty ticker is rejected cleanly (400, not a crash)
# ---------------------------------------------------------------------------
resp = client.post("/predict", data={"ticker": "", "period": "2y", "epochs": "3",
                                      "csrf_token": _csrf(client)})
check(resp.status_code == 400, "POST /predict with empty ticker -> 400, not a crash")
check(b"ticker symbol" in resp.data.lower(), "POST /predict empty ticker shows a clear message")

# ---------------------------------------------------------------------------
# Test 3: a full real prediction, through the actual route, with the data
# fetch mocked (so it's fast/offline) but the LR+LSTM math genuinely runs
# ---------------------------------------------------------------------------
with patch("app.pipeline.run_pipeline", return_value=_PRECOMPUTED_RESULT):
    resp = client.post("/predict", data={
        "ticker": "TESTX", "period": "2y", "epochs": "3", "source": "yfinance", "skip_llm": "on",
        "csrf_token": _csrf(client),
    }, follow_redirects=True)
    check(resp.status_code == 200, "POST /predict (full real LR+LSTM run) -> 200 after redirect")
    check(b"TESTX" in resp.data, "Results page shows the ticker")
    check(b"Ensemble Vote" in resp.data, "Results page shows the vote strip section")
    check(b"FINAL" in resp.data, "Results page shows a final signal badge")
    check(b"base64" in resp.data, "Results page embeds charts as base64 images")
    check(b"Export CSV" in resp.data, "Results page shows the Export CSV button")
    check(b"Export PDF" in resp.data, "Results page shows the Export PDF button")

    # --- CSV export: real content, not just a 200 ---
    resp_csv = client.get("/export/csv")
    check(resp_csv.status_code == 200, "GET /export/csv -> 200")
    check(resp_csv.mimetype == "text/csv", "CSV export has the correct mimetype")
    check(b"attachment" in resp_csv.headers.get("Content-Disposition", "").encode(),
          "CSV export sets a download Content-Disposition header")
    csv_text = resp_csv.data.decode()
    check("TESTX" in csv_text, "CSV export contains the ticker")
    check("RMSE" in csv_text, "CSV export contains the metrics section")
    check("Directional Accuracy" in csv_text, "CSV export contains the accuracy metric")
    csv_lines = csv_text.strip().split("\n")
    check(len(csv_lines) > 10, "CSV export contains actual date rows, not just a header")

    # --- PDF export: real content, not just a 200 ---
    resp_pdf = client.get("/export/pdf")
    check(resp_pdf.status_code == 200, "GET /export/pdf -> 200")
    check(resp_pdf.mimetype == "application/pdf", "PDF export has the correct mimetype")
    check(resp_pdf.data[:4] == b"%PDF", "PDF export produces an actual valid PDF file (starts with %PDF)")
    check(len(resp_pdf.data) > 5000, "PDF export is a substantial file, not an empty/broken one")

# --- exports with no prior prediction redirect cleanly, not a crash ---
fresh_client2 = app.test_client()
resp = fresh_client2.get("/export/csv", follow_redirects=False)
check(resp.status_code == 302, "GET /export/csv with no session data redirects, not a crash")
resp = fresh_client2.get("/export/pdf", follow_redirects=False)
check(resp.status_code == 302, "GET /export/pdf with no session data redirects, not a crash")

# ---------------------------------------------------------------------------
# Test 4: results page with no prior prediction redirects to index, not a crash
# ---------------------------------------------------------------------------
fresh_client = app.test_client()
resp = fresh_client.get("/results", follow_redirects=False)
check(resp.status_code == 302, "GET /results with no session data redirects (302)")
check("/" in resp.headers.get("Location", ""), "GET /results redirects toward index")

# ---------------------------------------------------------------------------
# Test 5: paper trade executes and shows a result
# ---------------------------------------------------------------------------
config.PAPER_STARTING_CASH = 100000
config.PAPER_PORTFOLIO_PATH = "/tmp/webapp_test_portfolio.json"
config.TRADE_LOG_PATH = "/tmp/webapp_test_trade_log.csv"
if os.path.exists(config.PAPER_PORTFOLIO_PATH):
    os.remove(config.PAPER_PORTFOLIO_PATH)

resp = client.post("/trade", data={"mode": "paper", "csrf_token": _csrf(client)}, follow_redirects=True)
check(resp.status_code == 200, "POST /trade (paper) -> 200")
check(b"Trade Result" in resp.data, "Paper trade shows the trade result page")

# ---------------------------------------------------------------------------
# Test 6: live trade with ENABLE_LIVE_TRADING=false must be blocked at
# /trade/confirm, never reaching an actual broker call
# ---------------------------------------------------------------------------
original_flag = config.ENABLE_LIVE_TRADING
config.ENABLE_LIVE_TRADING = False
resp = client.post("/trade", data={"mode": "live", "broker": "upstox",
                                    "csrf_token": _csrf(client)}, follow_redirects=True)
check(resp.status_code == 403, "GET /trade/confirm with ENABLE_LIVE_TRADING=false -> 403, not executed")
check(b"disabled" in resp.data.lower(), "Blocked live-trade page explains why")

# ---------------------------------------------------------------------------
# Test 7: even with ENABLE_LIVE_TRADING=true, /trade/execute with a WRONG
# confirmation phrase must not execute (this is the critical safety path --
# tested here without ever touching a real broker, by mocking place_order
# to fail loudly if it's ever actually called)
# ---------------------------------------------------------------------------
config.ENABLE_LIVE_TRADING = True

# --- 7a: enabled in .env but NO operator secret configured -> fail closed ---
# ENABLE_LIVE_TRADING alone must NOT be enough to reach a live order over
# HTTP. With no WEBAPP_LIVE_TRADING_TOKEN set, the live routes refuse.
original_token = config.WEBAPP_LIVE_TRADING_TOKEN
config.WEBAPP_LIVE_TRADING_TOKEN = ""
with patch("app.execute_trade_decision") as mock_execute:
    resp = client.post("/trade", data={"mode": "live", "broker": "upstox",
                                        "csrf_token": _csrf(client)}, follow_redirects=True)
    check(resp.status_code == 403,
          "Live trade with ENABLE_LIVE_TRADING=true but no operator secret configured -> 403")
    mock_execute.assert_not_called()
    print("PASS: CRITICAL -- execute_trade_decision was NEVER called with no operator secret configured")
    n_pass += 1

# --- 7b: a too-short secret is treated as no protection at all ---
config.WEBAPP_LIVE_TRADING_TOKEN = "short"
with patch("app.execute_trade_decision") as mock_execute:
    resp = client.post("/trade", data={"mode": "live", "broker": "upstox",
                                        "csrf_token": _csrf(client)}, follow_redirects=True)
    check(resp.status_code == 403, "Live trade with a too-short operator secret -> 403")
    mock_execute.assert_not_called()
    print("PASS: CRITICAL -- execute_trade_decision was NEVER called with a too-short operator secret")
    n_pass += 1

# --- 7c: correct length secret configured, but the WRONG one submitted ---
TEST_OPERATOR_SECRET = "x" * 40
config.WEBAPP_LIVE_TRADING_TOKEN = TEST_OPERATOR_SECRET
with patch("app.execute_trade_decision") as mock_execute:
    resp = client.post("/trade", data={"mode": "live", "broker": "upstox",
                                        "operator_secret": "wrong-secret-entirely",
                                        "csrf_token": _csrf(client)},
                        follow_redirects=True)
    check(resp.status_code == 401, "Live trade with a WRONG operator secret -> 401")
    mock_execute.assert_not_called()
    print("PASS: CRITICAL -- execute_trade_decision was NEVER called with a wrong operator secret")
    n_pass += 1

# --- 7d: /trade/confirm cannot be reached directly without authenticating ---
with patch("app.execute_trade_decision") as mock_execute:
    # follow_redirects=False so the redirect itself is observable -- with
    # redirects followed, a bounce to the index page also reports 200.
    resp = client.get("/trade/confirm", follow_redirects=False)
    check(resp.status_code in (302, 401),
          "GET /trade/confirm without authenticating -> not the confirm page")
    check(b"Confirm Live Trade" not in resp.data,
          "Unauthenticated /trade/confirm never renders the live order details")
    mock_execute.assert_not_called()

# --- 7e: with the CORRECT operator secret, the normal flow proceeds ---
resp = client.post("/trade", data={"mode": "live", "broker": "upstox",
                                    "operator_secret": TEST_OPERATOR_SECRET,
                                    "csrf_token": _csrf(client)},
                    follow_redirects=True)
check(resp.status_code == 200,
      "GET /trade/confirm with live enabled + correct operator secret -> 200 (shows confirm page)")
check(b"Confirm Live Trade" in resp.data, "Confirm page renders")

with client.session_transaction() as sess:
    real_token = sess.get("live_trade_token")
check(real_token is not None, "A confirmation token was minted in the session")

with patch("app.execute_trade_decision") as mock_execute:
    resp = client.post("/trade/execute", data={
        "token": real_token, "confirmation_phrase": "definitely not the phrase",
        "csrf_token": _csrf(client),
    }, follow_redirects=True)
    check(resp.status_code == 200, "POST /trade/execute with wrong phrase -> 200 (cancelled page, not an error)")
    check(b"did not match" in resp.data.lower() or b"cancelled" in resp.data.lower(),
          "Wrong phrase shows a clear cancellation message")
    mock_execute.assert_not_called()
    print("PASS: CRITICAL -- execute_trade_decision was NEVER called with a wrong confirmation phrase")
    n_pass += 1

# ---------------------------------------------------------------------------
# Test 8: a stale/reused token must also be rejected (replay protection)
# ---------------------------------------------------------------------------
with patch("app.execute_trade_decision") as mock_execute:
    resp = client.post("/trade/execute", data={
        "token": "some-fake-token-that-was-never-issued", "confirmation_phrase": webapp_module.CONFIRMATION_PHRASE,
        "csrf_token": _csrf(client),
    }, follow_redirects=True)
    check(resp.status_code == 403, "POST /trade/execute with an invalid token -> 403")
    mock_execute.assert_not_called()
    print("PASS: CRITICAL -- execute_trade_decision was NEVER called with an invalid/forged token")
    n_pass += 1

config.ENABLE_LIVE_TRADING = original_flag
config.WEBAPP_LIVE_TRADING_TOKEN = original_token

# ---------------------------------------------------------------------------
# Test 9: 404 handling is clean, not a raw traceback
# ---------------------------------------------------------------------------
resp = client.get("/this-route-does-not-exist")
check(resp.status_code == 404, "Unknown route -> 404")
check(b"Traceback" not in resp.data, "404 page does not leak a Python traceback")

print(f"\n========================================")
print(f"ALL {n_pass} WEBAPP TESTS PASSED")
print(f"========================================")
