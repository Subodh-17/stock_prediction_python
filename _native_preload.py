"""
_native_preload.py -- import-order guard for yfinance's HTTP client.

VERIFIED FINDING (V18, clean environment built from requirements-lock.txt: yfinance 1.7.0, curl_cffi 0.16.3,
tensorflow-cpu 2.21.0, Linux, Python 3.12): if TensorFlow is imported BEFORE yfinance's curl_cffi session is created, every
Yahoo request fails *locally* (no network involved) with
    ImpersonateError: Impersonating chrome150 is not supported
Creating the session first works. The mechanism (native-library load order) is a hypothesis, not proven; the behaviour
was reproduced deterministically in both orders. Windows/macOS: UNVERIFIED.

`config` calls preload() first thing, and every entry point imports `config` before the models, so TensorFlow can
never win the race inside this project. Code that imports TensorFlow itself before importing this project should call
`_native_preload.preload()` first.
"""


def preload() -> bool:
    try:
        import yfinance as yf
        yf.Ticker("SPY")        # constructing a Ticker initialises yfinance's shared curl_cffi session (no network call)
        return True
    except Exception:           # noqa: BLE001 - yfinance missing/broken must not stop non-Yahoo features
        return False
