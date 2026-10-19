"""V20 data-integrity regressions (found by probing the V19 checker with corrupt frames). Synthetic frames, tests only."""
import numpy as np
import pandas as pd
import pytest

import data_loader
from data_sources.base import standardize_dataframe
from tests.test_v17_evaluation import make_result
from tests.v17_helpers import fixture_bars
import evaluation as ev


def report(df):
    return data_loader.check_data_quality(df, ticker="T")


# ---- timezone-aware indexes
def test_standardize_converts_a_tz_aware_index_to_the_same_local_dates():
    df = fixture_bars(30, seed=1)
    aware = df.copy(); aware.index = aware.index.tz_localize("America/New_York")
    out = standardize_dataframe(aware)
    assert out.index.tz is None and [str(d)[:10] for d in out.index] == [str(d)[:10] for d in df.index]
    assert out["Close"].tolist() == df["Close"].tolist()
    assert standardize_dataframe(df).index.equals(df.index)                          # naive input is untouched


def test_standardize_keeps_the_trading_date_across_a_dst_change():
    idx = pd.date_range("2026-03-05", periods=8, freq="B", tz="America/New_York")     # spans the US spring-forward on 2026-03-08
    df = fixture_bars(8, seed=2); df.index = idx
    out = standardize_dataframe(df)
    assert [str(d)[:10] for d in out.index] == [str(d)[:10] for d in idx] and (out.index.hour == 0).all()


def test_a_raw_tz_aware_frame_is_flagged_and_the_evaluation_tolerates_one():
    df = fixture_bars(120, seed=3); df.index = df.index.tz_localize("UTC")
    assert any("timezone-aware" in w for w in report(df).warnings)
    r = make_result(n=340, n_test=120)
    naive = ev.aligned_frame(r)
    r_aware = dict(r); d = r["df"].copy(); d.index = d.index.tz_localize("America/New_York"); r_aware["df"] = d
    aware = ev.aligned_frame(r_aware)                                               # used to raise TypeError (naive vs aware compare)
    assert aware.index.tolist() == naive.index.tolist() and np.allclose(aware["base"], naive["base"])


# ---- new diagnostics
def test_infinite_values_are_named_explicitly():
    df = fixture_bars(120, seed=4); df.iloc[30, df.columns.get_loc("Close")] = np.inf
    r = report(df)
    assert not r.is_valid and any("non-finite (infinite)" in i for i in r.issues)


def test_a_frozen_feed_is_flagged_but_a_short_run_is_not():
    df = fixture_bars(120, seed=4); v = df.iloc[80]["Close"]; df.iloc[80:100, :4] = v
    r = report(df)
    assert r.is_valid and any("20 consecutive day(s) with identical Open/High/Low/Close" in w for w in r.warnings)     # warning, not an error
    short = fixture_bars(120, seed=4); short.iloc[80:84, :4] = short.iloc[80]["Close"]
    assert not any("identical Open/High/Low/Close" in w for w in report(short).warnings)


def test_zero_volume_stretch_is_flagged_only_for_instruments_that_report_volume():
    df = fixture_bars(120, seed=4); df.iloc[30:36, df.columns.get_loc("Volume")] = 0
    assert any("6 consecutive day(s) with zero Volume" in w for w in report(df).warnings)
    index_like = fixture_bars(120, seed=4); index_like["Volume"] = 0.0                # an index: volume is always 0
    assert not any("zero Volume" in w for w in report(index_like).warnings)


def test_a_clean_frame_still_produces_no_messages_and_old_checks_still_fire():
    r = report(fixture_bars(120, seed=4))
    assert r.is_valid and r.issues == [] and r.warnings == []
    d = fixture_bars(120, seed=4); d.iloc[40, d.columns.get_loc("High")] = d.iloc[40]["Low"] * 0.9
    assert any("impossible OHLC" in i for i in report(d).issues)
    assert any("not in chronological" in i for i in report(fixture_bars(120, seed=4).iloc[::-1]).issues)
