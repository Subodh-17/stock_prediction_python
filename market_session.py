"""
market_session.py -- is the newest daily bar still forming?

A daily bar dated "today" that the provider returns while the exchange is open is NOT a settled close: using it as a training
target, an evaluation actual or "the last close" mixes an intraday price into a daily series (look-ahead in spirit, and wrong
in fact). The newest bar is therefore dropped when it is dated today in the exchange's time zone and the session has not
ended. Rules:
  * local exchange date/time via the instrument's IANA zone;
  * a bar dated today is kept only if the exchange has closed (close time + a buffer for closing auctions / data lag), the
    day is a weekend, or the caller KNOWS the session is closed/post;
  * unknown exchange -> 16:00 local (US-style).  Exchange holidays are not modelled: on a holiday no "today" bar exists, so
    nothing is dropped, which is correct.
The table values are regular-session closes in local time PLUS a ~15 minute buffer; they are approximations (early closes,
half-days and special sessions are not modelled).
"""
from datetime import datetime, timezone
from typing import Optional

CLOSE_HOUR_LOCAL = {      # hour as decimal, regular close + buffer
    "NSE": 15.75, "BSE": 15.75,                  # 15:30
    "US/UNSPECIFIED": 16.25,                     # 16:00 (NYSE/Nasdaq)
    "LSE": 16.75,                                # 16:30
    "XETRA": 17.75, "EURONEXT-PA": 17.75,        # 17:30
    "TSE": 15.75,                                # 15:30 (since Nov 2024)
    "HKEX": 16.25,                               # 16:00 + closing auction
    "TSX": 16.25, "ASX": 16.25, "SGX": 17.25,    # 16:00 / 16:00 + auction / 17:00
}
DEFAULT_CLOSE_HOUR = 16.25


def session_is_over(local_dt: datetime, exchange: str) -> bool:
    return local_dt.hour + local_dt.minute / 60 >= CLOSE_HOUR_LOCAL.get(exchange, DEFAULT_CLOSE_HOUR)


def drop_partial_bar(df, inst, session: Optional[str], now: Optional[datetime] = None):
    """Returns (df_without_forming_bar, dropped_date_or_None). `inst` needs .timezone and .exchange."""
    from zoneinfo import ZoneInfo
    if df is None or len(df) == 0:
        return df, None
    now = now or datetime.now(timezone.utc)
    local = now.astimezone(ZoneInfo(inst.timezone))
    last = df.index[-1]
    if str(last)[:10] != local.date().isoformat():
        return df, None                       # newest bar is from an earlier day: settled
    if session in ("closed", "post"):
        return df, None
    if local.weekday() >= 5:
        return df, None
    if session is None and session_is_over(local, inst.exchange):
        return df, None
    return df.iloc[:-1], str(last)[:10]
