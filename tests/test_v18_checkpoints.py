import json
import os

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler

import checkpoints as ck
import config
from models.lstm_model import LSTMPredictor
from tests.v17_helpers import fixture_bars

DF = fixture_bars(220, seed=5)
TRAIN_END = str(DF.index[150])[:10]
FIRST_EVAL = str(DF.index[151])[:10]


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CHECKPOINT_DIR", str(tmp_path / "ck"))
    monkeypatch.setattr(config, "TRAIN_TEST_SPLIT", 0.8)
    return tmp_path / "ck"


def save(name="a", df=DF, train_end=TRAIN_END, with_prov=True, target="return", ticker="TST"):
    sc = MinMaxScaler().fit(np.linspace(0, 1, 40).reshape(-1, 1))
    m = LSTMPredictor(lookback=10)
    return ck.save_checkpoint(name, m, sc, ticker=ticker, period="2y", source="t", n_train_rows=100, target_mode=target,
                              exchange="US/UNSPECIFIED", provenance=ck.training_provenance(df, train_end) if with_prov else None,
                              training_config={"epochs": 1})


def load(name="a", **kw):
    base = dict(ticker="TST", lookback=10, target_mode="return", train_test_split=0.8, require_provenance=True,
                train_frame=DF, first_eval_date=FIRST_EVAL, exchange="US/UNSPECIFIED", feature_names=["Close_return"], horizon_days=1)
    base.update(kw)
    return ck.load_checkpoint(name, **base)


def test_manifest_records_the_required_provenance(store):
    save()
    m = json.load(open(store / "a" / "metadata.json"))
    for k in ("model_type", "model_impl_version", "exchange", "forecast_horizon_days", "feature_names", "feature_version",
              "train_start", "train_end", "train_fingerprint", "anchor_fingerprint", "scaler_config", "training_config",
              "model_sha256", "scaler_sha256", "saved_at", "random_seed", "package_versions", "lookback", "lstm_prediction_target"):
        assert m.get(k) not in (None, ""), k
    assert m["manifest_version"] == 2 and m["train_end"] == TRAIN_END and m["train_start"] == str(DF.index[0])[:10]


def test_matching_checkpoint_is_reused(store):
    save()
    model, scaler, meta = load()
    assert model.is_fitted and meta["train_end"] == TRAIN_END


@pytest.mark.parametrize("kw,needle", [
    (dict(target_mode="price"), "lstm_prediction_target"),
    (dict(feature_names=["Close"]), "feature schema"),
    (dict(horizon_days=5), "horizon"),
    (dict(lookback=20), "lookback"),
    (dict(ticker="OTHER"), "ticker"),
    (dict(exchange="NSE"), "exchange"),
])
def test_incompatible_checkpoints_are_rejected_with_a_reason(store, kw, needle):
    save()
    with pytest.raises(ck.CheckpointMismatchError, match=needle):
        load(**kw)


def test_model_implementation_or_feature_version_change_rejects(store, monkeypatch):
    save()
    monkeypatch.setattr(ck, "MODEL_IMPL_VERSION", "lstm-v99")
    with pytest.raises(ck.CheckpointProvenanceError, match="implementation version"):
        load()
    monkeypatch.undo()
    monkeypatch.setattr(config, "CHECKPOINT_DIR", str(store))
    monkeypatch.setattr(ck, "FEATURE_VERSION", "feat-v2")
    with pytest.raises(ck.CheckpointProvenanceError, match="feature-engineering"):
        load()


def test_missing_manifest_is_rejected_when_provenance_is_required_but_legacy_load_still_works(store):
    save(with_prov=False)
    with pytest.raises(ck.CheckpointProvenanceError, match="no provenance manifest"):
        load()
    model, _, meta = ck.load_checkpoint("a", ticker="TST", lookback=10, target_mode="return", train_test_split=0.8)
    assert model.is_fitted and meta["manifest_version"] == 1           # documented legacy behaviour is unchanged


def test_checkpoint_trained_on_the_evaluation_period_is_rejected(store):
    save()
    with pytest.raises(ck.CheckpointProvenanceError, match="NOT before the first evaluated date"):
        load(first_eval_date=TRAIN_END)                                  # same day: overlap
    with pytest.raises(ck.CheckpointProvenanceError, match="future-contaminated"):
        load(first_eval_date=str(DF.index[100])[:10])                    # historical evaluation cutoff before training end


def test_restated_history_is_detected_by_the_fingerprint(store):
    save()
    adj = DF.copy()
    adj.loc[adj.index[:151], "Close"] *= 0.97                            # e.g. a later dividend adjustment
    with pytest.raises(ck.CheckpointProvenanceError, match="no longer match"):
        load(train_frame=adj)


def test_training_end_missing_from_current_data_is_rejected(store):
    save()
    with pytest.raises(ck.CheckpointProvenanceError, match="not present in the current data"):
        load(train_frame=DF.iloc[160:])


@pytest.mark.parametrize("victim", ["model.keras", "scaler.json", "metadata.json"])
def test_corrupted_files_are_rejected_safely_without_touching_other_checkpoints(store, victim):
    save("a"); save("b")
    p = store / "a" / victim
    p.write_bytes(p.read_bytes()[: max(1, p.stat().st_size // 3)])
    with pytest.raises(ck.CheckpointMismatchError):
        load("a")
    assert (store / "a").is_dir() and (store / "b" / "metadata.json").exists()    # nothing was deleted
    assert load("b")[0].is_fitted


def test_saving_replaces_only_its_own_directory_and_leaves_no_temp_dirs(store):
    save("a"); save("b")
    b_before = (store / "b" / "metadata.json").read_text()
    save("a", train_end=str(DF.index[140])[:10])
    assert (store / "b" / "metadata.json").read_text() == b_before
    assert sorted(os.listdir(store)) == ["a", "b"]
    assert json.load(open(store / "a" / "metadata.json"))["train_end"] == str(DF.index[140])[:10]


# ---- end to end through the real pipeline (tiny LSTM, 1 epoch; synthetic data lives only in this test)
@pytest.fixture
def trained(monkeypatch):
    """Counts LSTM trainings; patched ONCE per test (re-patching would nest wrappers and over-count)."""
    c, orig = {"trained": 0}, LSTMPredictor.train
    def counting(self, *a, **k):
        c["trained"] += 1
        return orig(self, *a, **k)
    monkeypatch.setattr(LSTMPredictor, "train", counting)
    return c


def _run(df, name, monkeypatch, counter):
    import pipeline
    return pipeline.run_pipeline(ticker="TST", period="2y", epochs=1, skip_llm=True, verbose=False, checkpoint_name=name, df=df)


def test_pipeline_reuses_a_valid_checkpoint_and_retrains_when_it_would_be_future_contaminated(store, monkeypatch, trained):
    monkeypatch.setattr(config, "LSTM_PREDICTION_TARGET", "return")
    c = trained
    full = fixture_bars(260, seed=9)
    _run(full, "p", monkeypatch, c);                  assert c["trained"] == 1            # first run trains + saves manifest
    m1 = json.load(open(store / "p" / "metadata.json"));  assert m1["manifest_version"] == 2
    _run(full, "p", monkeypatch, c);                  assert c["trained"] == 1            # same data: reused
    grown = fixture_bars(275, seed=9)                                                      # more bars appended (same prefix)
    _run(grown, "p", monkeypatch, c);                 assert c["trained"] == 1            # test window moved later: still reusable
    shrunk = full.iloc[:200]                                                                # a historical cut-off BEFORE the training end
    _run(shrunk, "p", monkeypatch, c);                assert c["trained"] == 2            # rejected -> retrained
    m2 = json.load(open(store / "p" / "metadata.json"))
    assert m2["train_end"] < m1["train_end"]


def test_pipeline_retrains_a_legacy_checkpoint_and_overwrites_only_that_name(store, monkeypatch, trained):
    monkeypatch.setattr(config, "LSTM_PREDICTION_TARGET", "return")
    c = trained
    df = fixture_bars(260, seed=9)
    save("legacy", df=df, with_prov=False)
    save("other", df=df)
    other_before = (store / "other" / "metadata.json").read_text()
    _run(df, "legacy", monkeypatch, c)
    assert c["trained"] == 1 and json.load(open(store / "legacy" / "metadata.json"))["manifest_version"] == 2
    assert (store / "other" / "metadata.json").read_text() == other_before
