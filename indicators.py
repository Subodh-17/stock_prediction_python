"""indicators.py -- standard technical indicators on real OHLCV bars (None during warm-up).
SMA: simple mean. Bollinger: SMA(20) +/- 2 population-std. RSI: Wilder smoothing (14).
MACD: EMA12 - EMA26, signal EMA9 of MACD (EMAs seeded with the first value, alpha=2/(n+1))."""
import numpy as np


def _nan_to_none(a):
    return [None if not np.isfinite(x) else float(x) for x in a]


def sma(c, n):
    c = np.asarray(c, float)
    out = np.full(len(c), np.nan)
    if len(c) >= n:
        cs = np.cumsum(np.insert(c, 0, 0.0))
        out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


def ema(c, n):
    c = np.asarray(c, float)
    out = np.empty(len(c))
    if len(c) == 0:
        return out
    a = 2.0 / (n + 1)
    out[0] = c[0]
    for i in range(1, len(c)):
        out[i] = a * c[i] + (1 - a) * out[i - 1]
    return out


def bollinger(c, n=20, k=2.0):
    c = np.asarray(c, float)
    mid = sma(c, n)
    sd = np.full(len(c), np.nan)
    for i in range(n - 1, len(c)):
        sd[i] = np.std(c[i - n + 1:i + 1])
    return mid - k * sd, mid, mid + k * sd


def rsi(c, n=14):
    c = np.asarray(c, float)
    out = np.full(len(c), np.nan)
    if len(c) <= n:
        return out
    d = np.diff(c)
    gain, loss = np.where(d > 0, d, 0.0), np.where(d < 0, -d, 0.0)
    ag, al = gain[:n].mean(), loss[:n].mean()
    out[n] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    for i in range(n, len(d)):
        ag = (ag * (n - 1) + gain[i]) / n
        al = (al * (n - 1) + loss[i]) / n
        out[i + 1] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    return out


def macd(c, fast=12, slow=26, sig=9):
    line = ema(c, fast) - ema(c, slow)
    sg = ema(line, sig)
    out_line, out_sig = line.copy(), sg.copy()
    out_line[:slow - 1] = np.nan
    out_sig[:slow - 1 + sig - 1] = np.nan
    return out_line, out_sig, out_line - out_sig


def compute(bars):
    c = [b["c"] for b in bars]
    lo, mid, up = bollinger(c)
    m, s, h = macd(c)
    series = {"sma20": sma(c, 20), "sma50": sma(c, 50), "bb_lower": lo, "bb_mid": mid, "bb_upper": up,
              "rsi14": rsi(c), "macd": m, "macd_signal": s, "macd_hist": h}
    latest = {k: (None if not len(v) or not np.isfinite(v[-1]) else float(v[-1])) for k, v in series.items()}
    return {"series": {k: _nan_to_none(v) for k, v in series.items()}, "latest": latest}
