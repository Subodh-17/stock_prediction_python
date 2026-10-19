"""V20 security regressions: model-artifact loading, checkpoint names, output escaping. Real code paths, no mocks of the checks."""
import hashlib
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "webapp"), ROOT]

import checkpoints as ck
import config


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHECKPOINT_DIR", str(tmp_path / "ck"))
    monkeypatch.setattr(config, "TRAIN_TEST_SPLIT", 0.8)
    return tmp_path


def test_a_checkpoint_containing_a_python_lambda_is_refused_and_its_code_never_runs(store):
    """A tampered .keras file with a Lambda layer would run attacker code on load if safe_mode were off."""
    import tensorflow as tf
    marker = store / "PWNED"
    def evil(x):
        open(str(marker), "w").write("code executed")
        return x
    m = tf.keras.Sequential([tf.keras.layers.Input((10, 1)), tf.keras.layers.Lambda(evil), tf.keras.layers.Flatten(), tf.keras.layers.Dense(1)])
    d = store / "ck" / "evil"; d.mkdir(parents=True)
    m.save(str(d / "model.keras"))
    marker.unlink(missing_ok=True)                                       # saving may have traced the function; only LOADING matters
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    sc = MinMaxScaler().fit(np.linspace(0, 1, 20).reshape(-1, 1))
    (d / "scaler.json").write_text(json.dumps(ck._scaler_to_dict(sc)))
    meta = {"ticker": "T", "period": "2y", "source": "t", "lookback": 10, "units": 4, "lstm_prediction_target": "return",
            "train_test_split": 0.8, "saved_at": "2026-01-01T00:00:00+00:00", "n_train_rows": 10, "random_seed": 1, "package_versions": {},
            "manifest_version": 1,
            "model_sha256": hashlib.sha256((d / "model.keras").read_bytes()).hexdigest(),     # hashes MATCH: integrity alone is not the defence
            "scaler_sha256": hashlib.sha256((d / "scaler.json").read_bytes()).hexdigest()}
    (d / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ck.CheckpointMismatchError):
        ck.load_checkpoint("evil", ticker="T", lookback=10, target_mode="return", train_test_split=0.8)
    assert not marker.exists(), "the Lambda payload executed during checkpoint load"


def test_safe_mode_is_requested_explicitly():
    assert "safe_mode=True" in open(os.path.join(ROOT, "checkpoints.py")).read()


@pytest.mark.parametrize("bad", ["../x", "a/b", "..\\x", "a.b", "", "/etc/passwd", "x/../../y", "name with/slash"])
def test_checkpoint_names_cannot_escape_the_checkpoint_directory(store, bad):
    with pytest.raises((ValueError, ck.CheckpointMismatchError, ck.CheckpointNotFoundError)):
        ck.checkpoint_exists(bad) if False else ck._checkpoint_dir(bad)


def test_a_valid_checkpoint_name_resolves_inside_the_checkpoint_directory(store):
    p = os.path.realpath(ck._checkpoint_dir("auto_NSE_RELIANCE_2y"))
    assert p.startswith(os.path.realpath(config.CHECKPOINT_DIR) + os.sep)


HOSTILE = ["<script>alert(1)</script><img src=x onerror=alert(2)>", "../../etc/passwd", "AAPL?x=1&y=2", "A B;rm -rf /", "a" * 500,
           "AAPL\nSet-Cookie: x=1", "ＡＡＰＬ", "AAPL%00", "'; DROP TABLE t;--"]
VALID = ["AAPL", "RELIANCE.NS", "M&M.NS", "BAJAJ-AUTO.NS", "BRK-B", "^NSEI", "^GSPC", "EURUSD=X", "7203.T", "A" * 20]


@pytest.fixture
def client(monkeypatch):
    """A client whose background compute is REPLACED, so no test here can reach a real provider."""
    import app as webapp
    from webapp import api_v2
    api_v2._hits.clear(); monkeypatch.setattr(config, "WEBAPP_EXPENSIVE_RPM", 100000)
    called = []
    monkeypatch.setattr(webapp, "_compute_legacy_predict", lambda p: (called.append(p["ticker"]), webapp._store_result({"created_at": 0}))[1])
    monkeypatch.setattr(webapp, "_compute_api_signal", lambda *a: (called.append(a[0]), a[-1].__setitem__("_payload", {}), "x")[2])
    monkeypatch.setattr(webapp, "_compute_api_predict", lambda *a: (called.append(a[0]), a[-1].__setitem__("_payload", {}), "x")[2])
    monkeypatch.setattr(webapp, "_compute_api_backtest", lambda *a: (called.append(a[0]), a[-1].__setitem__("_payload", {}), "x")[2])
    c = webapp.app.test_client()
    return c, c.get("/api/v2/meta").get_json()["csrf_token"], called


@pytest.mark.parametrize("bad", HOSTILE)
def test_hostile_ticker_text_is_rejected_on_every_route_and_never_reaches_the_providers(client, bad):
    c, tok, called = client
    r = c.post("/predict", data={"ticker": bad, "csrf_token": tok})
    body = r.get_data(as_text=True)
    assert r.status_code == 400 and "<script>alert(1)" not in body and "onerror=alert(2)>" not in body
    for path in ("/api/predict", "/api/signal", "/api/backtest"):
        assert c.post(path, json={"ticker": bad}).status_code == 400, path
    assert c.get("/api/v2/quote", query_string={"symbol": bad}).status_code == 400
    assert called == []                                                    # nothing was ever handed to a compute/provider path


@pytest.mark.parametrize("good", VALID)
def test_legitimate_tickers_are_still_accepted_everywhere(client, good):
    c, tok, called = client
    assert c.post("/predict", data={"ticker": good, "csrf_token": tok, "async": "1"}).status_code == 202
    assert c.post("/api/signal", json={"ticker": good, "async": True}).status_code == 202
    import time
    deadline = time.time() + 5
    while good not in called and time.time() < deadline:                  # the compute runs on the background worker
        time.sleep(0.02)
    assert good in called
