import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = """
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, %r)
import config                      # the project's own import order: config first ...
import tensorflow                  # ... then TensorFlow
import yfinance as yf
try:
    yf.Ticker("AAPL").info
    print("RESULT ok")
except Exception as e:
    print("RESULT", type(e).__name__, str(e)[:120])
"""


def test_yahoo_client_survives_tensorflow_being_imported_after_config():
    """Regression for the verified V18 finding: with TensorFlow loaded first, yfinance fails locally with
    ImpersonateError. With the project's import order it must never do so (it either succeeds or fails on the NETWORK)."""
    pytest.importorskip("tensorflow"); pytest.importorskip("yfinance")
    out = subprocess.run([sys.executable, "-c", CODE % ROOT], capture_output=True, text=True, timeout=180,
                         cwd=ROOT).stdout
    result = [l for l in out.splitlines() if l.startswith("RESULT")]
    assert result, out
    assert "ImpersonateError" not in result[-1] and "Impersonating" not in result[-1], result[-1]


def test_client_init_failures_are_reported_with_an_actionable_message(monkeypatch):
    import market_data as md
    class T:
        def __init__(self, *a): pass
        @property
        def info(self):
            raise RuntimeError("Impersonating chrome150 is not supported")
    import yfinance
    monkeypatch.setattr(yfinance, "Ticker", T)
    with pytest.raises(md.ProviderError) as e:
        md.YahooProvider().fetch_raw(md.resolve_instrument("AAPL"))
    assert e.value.kind == "client_init" and "imported before yfinance" in str(e.value) and not e.value.retryable
