"""
QA HARNESS -- NOT PART OF THE PRODUCT. Boots the real web app with the real pipeline, but swaps the *network
inputs* for deterministic synthetic fixtures (the dev sandbox cannot reach Yahoo/Gemini/etc.), so the UI can be
exercised end to end in a browser. Screenshots taken from this harness are test artifacts, not market data.
Run:  python tests/qa_harness.py  (listens on 127.0.0.1:5057)
"""
import json, os, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "webapp")]
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
tmp = tempfile.mkdtemp(prefix="qa_")
os.environ.update(PAPER_PORTFOLIO_PATH=f"{tmp}/paper.json", TRADE_LOG_PATH=f"{tmp}/log.csv", CHECKPOINT_DIR=f"{tmp}/ck",
                  OUTPUT_DIR=f"{tmp}/out", GOOGLE_API_KEY="QA_FAKE_KEY_not_a_real_key_000000", GOOGLE_MODEL="gemini-3.5-flash-lite",
                  WEBAPP_EXPENSIVE_RPM="1000", QUOTE_TTL_SECONDS="2")
from datetime import datetime, timezone, timedelta
import config, market_data as md
from tests.v17_helpers import fixture_bars, FakeResp
from tests.test_v17_market_data import Fake, raw
import data_sources.yfinance_source as ys
from ai_integrations import providers as P

BARS = fixture_bars(520, seed=11, start="2023-10-02")
ys.fetch = lambda sym, period, interval="1d", **k: BARS.copy()
last = float(BARS["Close"].iloc[-1])
yahoo = Fake("yahoo", raw=dict(raw(price=last * 1.004), prev_close=last))
md._service = md.MarketDataService([yahoo])

def gem(url_models=False):
    def http(method, url, headers=None, json=None, timeout=None):
        if "pageSize" in url:
            return FakeResp(200, {"models": [{"name": "models/gemini-3.5-flash-lite", "supportedGenerationMethods": ["generateContent"]}]})
        txt = {"summary": f"The last completed close was {last:.2f}. The numerical ensemble and baselines are shown in the fact pack.",
               "model_disagreement": "Components differ; see F_PRED_* facts.",
               "observations": [{"text": "RSI is computed from supplied closes.", "fact_ids": ["F_IND_RSI14"]}],
               "risk_factors": [{"text": "Interval coverage may degrade in volatile regimes.", "fact_ids": ["F_INTERVAL"]}],
               "hypotheses": [{"text": "Mean reversion could matter (hypothesis, not a fact); a 250 level is unsupported.", "fact_ids": []}],
               "news": [], "limitations": ["No news feed configured."], "missing_information": ["Earnings calendar"]}
        return FakeResp(200, {"candidates": [{"content": {"parts": [{"text": __import__("json").dumps(txt)}]}}]})
    return http
P._registry = P.Registry(http=gem())
import app as webapp
if __name__ == "__main__":
    webapp.app.run("127.0.0.1", 5057, threaded=True)
