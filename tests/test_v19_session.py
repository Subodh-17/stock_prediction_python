from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import config
import market_data as md
import market_session as ms
from tests.v17_helpers import fixture_bars


def local(tz, y, m, d, hh, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=ZoneInfo(tz)).astimezone(timezone.utc)


def df_ending(date_str, n=60):
    return fixture_bars(n, seed=2).set_axis(pd.bdate_range(end=date_str, periods=n, name="Date"))


# (symbol, last bar date (a Wednesday), tz, last minute BEFORE the table's session end, first minute AFTER)
CASES = [("AAPL", "America/New_York", 16, 10, 16, 20), ("NSE:TCS", "Asia/Kolkata", 15, 40, 15, 50), ("BSE:TCS", "Asia/Kolkata", 15, 40, 15, 50),
         ("VOD.L", "Europe/London", 16, 40, 16, 50), ("SAP.DE", "Europe/Berlin", 17, 40, 17, 50), ("7203.T", "Asia/Tokyo", 15, 40, 15, 50),
         ("D05.SI", "Asia/Singapore", 17, 10, 17, 20), ("0700.HK", "Asia/Hong_Kong", 16, 10, 16, 20)]


@pytest.mark.parametrize("sym,tz,h1,m1,h2,m2", CASES)
def test_bar_is_dropped_until_the_exchange_session_has_ended(sym, tz, h1, m1, h2, m2):
    inst = md.resolve_instrument(sym)
    assert inst.timezone == tz
    df = df_ending("2026-09-30")                                     # a Wednesday
    out, dropped = ms.drop_partial_bar(df, inst, None, now=local(tz, 2026, 9, 30, h1, m1))
    assert dropped == "2026-09-30" and len(out) == len(df) - 1 and str(out.index[-1])[:10] < "2026-09-30"
    out2, dropped2 = ms.drop_partial_bar(df, inst, None, now=local(tz, 2026, 9, 30, h2, m2))
    assert dropped2 is None and len(out2) == len(df)


def test_the_old_16_00_rule_would_have_kept_a_forming_bar_on_late_closing_exchanges():
    """Regression for V17's single 16:00 cutoff: at 16:30 in Frankfurt the session is still open."""
    inst = md.resolve_instrument("SAP.DE")
    out, dropped = ms.drop_partial_bar(df_ending("2026-09-30"), inst, None, now=local("Europe/Berlin", 2026, 9, 30, 16, 30))
    assert dropped == "2026-09-30"


def test_settled_bars_weekends_and_known_closed_sessions_are_not_dropped():
    inst = md.resolve_instrument("AAPL")
    df = df_ending("2026-09-30")
    assert ms.drop_partial_bar(df, inst, None, now=local("America/New_York", 2026, 10, 1, 11))[1] is None      # newest bar is yesterday's
    assert ms.drop_partial_bar(df_ending("2026-10-02"), inst, None, now=local("America/New_York", 2026, 10, 3, 11))[1] is None   # Saturday
    assert ms.drop_partial_bar(df, inst, "closed", now=local("America/New_York", 2026, 9, 30, 11))[1] is None  # caller knows it is closed
    assert ms.drop_partial_bar(df, inst, "regular", now=local("America/New_York", 2026, 9, 30, 20))[1] == "2026-09-30"   # known open wins over the clock
    assert ms.drop_partial_bar(df.iloc[:0], inst, None)[1] is None


# ---- the legacy fetch path
@pytest.fixture
def legacy(monkeypatch):
    import data_sources as ds
    import data_sources.yfinance_source as ys
    monkeypatch.setattr(ys, "fetch", lambda *a, **k: df_ending("2026-09-30", 300))
    def pin(tz, hh, mm=0):
        monkeypatch.setattr(ds, "_now", lambda: local(tz, 2026, 9, 30, hh, mm))
    return ds, pin


def test_legacy_fetch_drops_the_forming_bar_and_says_so(legacy):
    import data_loader
    ds, pin = legacy
    pin("America/New_York", 11)
    out = data_loader.fetch_stock_data("AAPL", period="2y", source="yfinance")
    assert str(out.index[-1])[:10] == "2026-09-29" and out.attrs["dropped_open_session_bar"] == "2026-09-30" and len(out) == 299
    pin("America/New_York", 17)
    out = data_loader.fetch_stock_data("AAPL", period="2y", source="yfinance")
    assert str(out.index[-1])[:10] == "2026-09-30" and "dropped_open_session_bar" not in out.attrs
    pin("Asia/Kolkata", 12)
    assert str(data_loader.fetch_stock_data("RELIANCE.NS", period="2y", source="yfinance").index[-1])[:10] == "2026-09-29"
    assert str(data_loader.fetch_stock_data("RELIANCE.BO", period="2y", source="yfinance").index[-1])[:10] == "2026-09-29"


def test_the_drop_can_be_switched_off_and_never_breaks_on_odd_symbols(legacy, monkeypatch):
    import data_loader
    ds, pin = legacy
    pin("America/New_York", 11)
    monkeypatch.setattr(config, "DROP_OPEN_SESSION_BAR", False)
    assert len(data_loader.fetch_stock_data("AAPL", period="2y", source="yfinance")) == 300
    monkeypatch.setattr(config, "DROP_OPEN_SESSION_BAR", True)
    assert len(data_loader.fetch_stock_data("^GSPC", period="2y", source="yfinance")) == 299
    assert len(data_loader.fetch_stock_data("BTC-USD", period="2y", source="yfinance")) in (299, 300)    # never raises


def test_pipeline_result_excludes_the_forming_bar_and_reports_it(legacy, monkeypatch, tmp_path):
    import pipeline
    monkeypatch.setattr(config, "CHECKPOINT_DIR", str(tmp_path / "ck"))
    ds, pin = legacy
    pin("America/New_York", 11)
    r = pipeline.run_pipeline(ticker="AAPL", period="2y", epochs=1, skip_llm=True, verbose=False)
    assert r["dropped_open_session_bar"] == "2026-09-30" and str(r["df"].index[-1])[:10] == "2026-09-29"
    assert all(str(d)[:10] <= "2026-09-29" for d in r["lr_dates_test"])


def test_results_page_tells_the_user_when_a_bar_was_excluded():
    import os
    html = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "webapp/templates/results.html")).read()
    assert "Unfinished session excluded" in html and "dropped_open_session_bar" in html


# ---- index symbols (V17/V18 rejected anything starting with ^)
@pytest.mark.parametrize("sym,exch,tz", [("^GSPC", "US/UNSPECIFIED", "America/New_York"), ("^NSEI", "NSE", "Asia/Kolkata"),
                                          ("^BSESN", "BSE", "Asia/Kolkata"), ("^FTSE", "LSE", "Europe/London"), ("^N225", "TSE", "Asia/Tokyo"),
                                          ("^GDAXI", "XETRA", "Europe/Berlin"), ("^HSI", "HKEX", "Asia/Hong_Kong")])
def test_index_symbols_resolve_with_the_right_exchange_and_session_clock(sym, exch, tz):
    inst = md.resolve_instrument(sym)
    assert inst.exchange == exch and inst.timezone == tz and inst.yahoo_symbol == sym and inst.key == f"{exch}:{sym}"


def test_an_indian_index_uses_the_indian_session_clock_not_new_yorks():
    inst = md.resolve_instrument("^NSEI")
    df = df_ending("2026-09-30")
    # 20:00 in India is long after the NSE close but still 10:30 in New York: must NOT be dropped
    assert ms.drop_partial_bar(df, inst, None, now=local("Asia/Kolkata", 2026, 9, 30, 20))[1] is None
    assert ms.drop_partial_bar(df, inst, None, now=local("Asia/Kolkata", 2026, 9, 30, 12))[1] == "2026-09-30"


@pytest.mark.parametrize("bad", ["^", "^^GSPC", "^ GSPC;", "^" + "A" * 30])
def test_malformed_index_like_symbols_are_still_rejected(bad):
    with pytest.raises(md.InvalidSymbol):
        md.resolve_instrument(bad)
