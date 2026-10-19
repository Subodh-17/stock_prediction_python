"""Test-only fixtures. NEVER imported by application code. Synthetic data lives here and nowhere else."""
import numpy as np
import pandas as pd


def fixture_bars(n=300, seed=0, start="2024-01-02", p0=100.0):
    rng = np.random.default_rng(seed)
    close = p0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    open_ = np.r_[p0, close[:-1]] * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    idx = pd.bdate_range(start, periods=n, name="Date")
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                         "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float)}, index=idx)


class FakeResp:
    def __init__(self, status=200, body=None, headers=None):
        import json
        self.status_code = status
        self._body = body if body is not None else {}
        self.text = body if isinstance(body, str) else json.dumps(self._body)
        self.headers = headers or {}

    def json(self):
        if isinstance(self._body, str):
            raise ValueError("not json")
        return self._body
