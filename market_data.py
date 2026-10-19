"""
market_data.py
---------------
Quote + history layer with an EXPLICIT freshness policy.

Why this exists: the original data layer returned only a DataFrame of daily
bars. Nothing recorded which provider answered, when the last price was
actually produced, or whether it was real-time, delayed, or end-of-day, and
the pipeline treated the last historical close as "the current price" (which
the trade executor then used to price orders).

Rules enforced here:
  * "live" is only ever assigned when the provider is documented/entitled to
    supply real-time data AND the source timestamp is within
    config.LIVE_MAX_AGE_SECONDS. A recent *retrieval* time never implies freshness.
  * yfinance (an unofficial wrapper over Yahoo's endpoints) can NEVER be
    "live": Yahoo does not guarantee real-time data for personal-use access.
    It yields "delayed" (delay known) or "unverified" (delay unknown).
  * Falling back to a lower-freshness provider is recorded in the quote's
    `selection_trail` and `downgraded_from`; the UI must show it.
  * Nothing is fabricated: if no provider answers, the result is an
    "unavailable" Quote with the reasons.
"""
from __future__ import annotations

import random
import re
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Callable, Dict, List, Optional

import config
import perf
import security

# ---------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------
_SUFFIX_EXCHANGE = {
    ".NS": ("NSE", "Asia/Kolkata", "INR"), ".BO": ("BSE", "Asia/Kolkata", "INR"),
    ".L": ("LSE", "Europe/London", "GBP"), ".T": ("TSE", "Asia/Tokyo", "JPY"),
    ".HK": ("HKEX", "Asia/Hong_Kong", "HKD"), ".TO": ("TSX", "America/Toronto", "CAD"),
    ".AX": ("ASX", "Australia/Sydney", "AUD"), ".DE": ("XETRA", "Europe/Berlin", "EUR"),
    ".PA": ("EURONEXT-PA", "Europe/Paris", "EUR"), ".SI": ("SGX", "Asia/Singapore", "SGD"),
}
_SYMBOL_RE = re.compile(r"^\^?[A-Z0-9][A-Z0-9.\-&^=]{0,19}$")      # a leading ^ is Yahoo's index prefix (^GSPC, ^NSEI)
# Yahoo index symbols carry no exchange suffix, so the exchange (and therefore the session clock) is stated here. An index not
# listed stays US/UNSPECIFIED: correct for ^GSPC/^DJI/^IXIC/^VIX, and visibly "unspecified" for anything else.
_INDEX_EXCHANGE = {"^NSEI": "NSE", "^NSEBANK": "NSE", "^CNXIT": "NSE", "^BSESN": "BSE", "^FTSE": "LSE", "^N225": "TSE", "^HSI": "HKEX",
                   "^GDAXI": "XETRA", "^FCHI": "EURONEXT-PA", "^STI": "SGX", "^GSPTSE": "TSX", "^AXJO": "ASX"}
_EXCHANGE_META = {v[0]: (v[1], v[2]) for v in _SUFFIX_EXCHANGE.values()}


class InvalidSymbol(ValueError):
    pass


@dataclass(frozen=True)
class Instrument:
    """A listing, not just a ticker: the same ticker on two exchanges is two Instruments."""
    symbol: str          # bare trading symbol, e.g. "RELIANCE"
    exchange: str        # "NSE", "BSE", "LSE", ... or "US/UNSPECIFIED" for a bare symbol
    yahoo_symbol: str    # what Yahoo needs, e.g. "RELIANCE.NS"
    timezone: str        # exchange timezone (IANA)
    currency_hint: Optional[str]

    @property
    def key(self) -> str:
        return f"{self.exchange}:{self.symbol}"

    def to_dict(self) -> dict:
        return {**asdict(self), "key": self.key}


def resolve_instrument(raw: str) -> Instrument:
    """
    Accepts "NSE:RELIANCE", "BSE:RELIANCE", "RELIANCE.NS", "VOD.L", "AAPL".
    A bare symbol is NOT guessed to be Indian; it stays "US/UNSPECIFIED" (what Yahoo
    resolves a bare symbol to) so NSE:TCS and a US "TCS" can never be conflated.
    """
    if not raw or not isinstance(raw, str):
        raise InvalidSymbol("A symbol is required.")
    text = raw.strip().upper().replace(" ", "")
    if ":" in text:
        exch, sym = text.split(":", 1)
        suffix = {"NSE": ".NS", "BSE": ".BO"}.get(exch)
        if suffix is None:
            raise InvalidSymbol(f"Unsupported exchange prefix '{exch}'. Use NSE:, BSE:, or a Yahoo suffix like VOD.L.")
        text = sym + suffix
    if not _SYMBOL_RE.match(text):
        raise InvalidSymbol("Symbol contains characters that are not valid in a ticker.")
    if text in _INDEX_EXCHANGE:
        exch = _INDEX_EXCHANGE[text]
        tz, cur = _EXCHANGE_META.get(exch, ("America/New_York", None))
        return Instrument(text, exch, text, tz, cur)
    for suffix, (exch, tz, cur) in _SUFFIX_EXCHANGE.items():
        if text.endswith(suffix):
            return Instrument(text[: -len(suffix)], exch, text, tz, cur)
    return Instrument(text, "US/UNSPECIFIED", text, "America/New_York", None)


# ---------------------------------------------------------------------------
# Quote model + freshness policy
# ---------------------------------------------------------------------------
FRESHNESS_LIVE = "live"
FRESHNESS_DELAYED = "delayed"
FRESHNESS_EOD = "end_of_day"
FRESHNESS_HISTORICAL = "historical"
FRESHNESS_STALE = "stale"
FRESHNESS_UNVERIFIED = "unverified"
FRESHNESS_UNAVAILABLE = "unavailable"


@dataclass
class Quote:
    provider: str
    symbol: str
    exchange: str
    currency: Optional[str] = None
    price: Optional[float] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    prev_close: Optional[float] = None
    volume: Optional[float] = None
    source_ts: Optional[str] = None        # when the PROVIDER says the price was produced (UTC ISO)
    retrieved_ts: Optional[str] = None     # when THIS app received it (UTC ISO)
    timezone: Optional[str] = None
    session: Optional[str] = None          # regular / pre / post / closed / None (unknown)
    data_type: str = "unknown"             # realtime | delayed | eod | unknown  (provider's own claim)
    delay_seconds: Optional[float] = None  # known exchange delay, if the provider reports it
    freshness: str = FRESHNESS_UNAVAILABLE
    age_seconds: Optional[float] = None
    is_stale: bool = False
    realtime_eligible: bool = False
    from_cache: bool = False
    cache_age_seconds: Optional[float] = None
    quality_flags: List[str] = field(default_factory=list)
    selection_trail: List[dict] = field(default_factory=list)
    provider_entitled_realtime: bool = False   # remembered so a CACHED quote is re-classified with the same rules
    downgraded_from: Optional[str] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_ts(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and re.fullmatch(r"\s*\d{9,14}(\.\d+)?\s*", value):
        value = float(value)         # Upstox documents last_trade_time as epoch MILLISECONDS in a string
    if isinstance(value, (int, float)):
        if value <= 0:
            return None
        v = float(value)
        if v > 1e11:                 # > year 5138 in seconds => the value is in milliseconds
            v /= 1000.0
        try:
            return datetime.fromtimestamp(v, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def classify_freshness(*, price, source_ts, data_type: str, delay_seconds: Optional[float],
                       supplies_realtime: bool, now: Optional[datetime] = None) -> dict:
    """
    The one place freshness is decided. Pure function -> unit-testable.
    Returns {freshness, age_seconds, is_stale, realtime_eligible, flags}.
    """
    now = now or datetime.now(timezone.utc)
    ts = _parse_ts(source_ts)
    flags: List[str] = []
    if price is None or not (price == price) or price <= 0:
        return {"freshness": FRESHNESS_UNAVAILABLE, "age_seconds": None, "is_stale": False,
                "realtime_eligible": False, "flags": ["no_valid_price"]}
    if ts is None:
        return {"freshness": FRESHNESS_UNVERIFIED, "age_seconds": None, "is_stale": False,
                "realtime_eligible": False, "flags": ["no_source_timestamp"]}
    age = (now - ts).total_seconds()
    if age < -60:
        flags.append("source_timestamp_in_future")
        return {"freshness": FRESHNESS_UNVERIFIED, "age_seconds": age, "is_stale": False,
                "realtime_eligible": False, "flags": flags}
    age = max(age, 0.0)

    if data_type == "realtime" and supplies_realtime:
        if age <= config.LIVE_MAX_AGE_SECONDS:
            return {"freshness": FRESHNESS_LIVE, "age_seconds": age, "is_stale": False,
                    "realtime_eligible": True, "flags": flags}
        flags.append("realtime_provider_but_old_timestamp")
        return {"freshness": FRESHNESS_STALE, "age_seconds": age, "is_stale": True,
                "realtime_eligible": False, "flags": flags}
    if data_type == "delayed" or (delay_seconds is not None and delay_seconds > 0):
        limit = (delay_seconds or 0) + config.DELAYED_STALE_AFTER_SECONDS
        stale = age > limit
        if stale:
            flags.append("older_than_expected_delay")
        return {"freshness": FRESHNESS_STALE if stale else FRESHNESS_DELAYED, "age_seconds": age,
                "is_stale": stale, "realtime_eligible": False, "flags": flags}
    if data_type == "eod":
        stale = age > 4 * 86400  # covers a long weekend; beyond that something is wrong
        return {"freshness": FRESHNESS_STALE if stale else FRESHNESS_EOD, "age_seconds": age,
                "is_stale": stale, "realtime_eligible": False, "flags": flags}
    flags.append("provider_does_not_state_delay")
    return {"freshness": FRESHNESS_UNVERIFIED, "age_seconds": age, "is_stale": False,
            "realtime_eligible": False, "flags": flags}


# ---------------------------------------------------------------------------
# Caching, request coalescing, retries
# ---------------------------------------------------------------------------
class RateLimited(Exception):
    def __init__(self, message="rate limited", retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class ProviderError(Exception):
    def __init__(self, message, retryable=False, kind="error"):
        super().__init__(message)
        self.retryable = retryable
        self.kind = kind


class TTLCache:
    """Bounded LRU cache with per-entry TTL. Keys must include everything that changes the value."""
    def __init__(self, maxsize=256):
        self._d: "OrderedDict[str, tuple]" = OrderedDict()
        self._lock = threading.Lock()
        self.maxsize = maxsize
        self.hits = 0
        self.misses = 0

    def get(self, key, allow_stale_for: float = 0.0):
        with self._lock:
            item = self._d.get(key)
            if item is None:
                self.misses += 1
                return None
            value, stored, ttl = item
            age = time.monotonic() - stored
            if age <= ttl or age <= ttl + allow_stale_for:
                self._d.move_to_end(key)
                self.hits += 1
                return value, age, age > ttl
            self.misses += 1
            return None

    def set(self, key, value, ttl: float):
        with self._lock:
            self._d[key] = (value, time.monotonic(), ttl)
            self._d.move_to_end(key)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)

    def stats(self) -> dict:
        with self._lock:
            return {"entries": len(self._d), "maxsize": self.maxsize, "hits": self.hits, "misses": self.misses}


class SingleFlight:
    """Concurrent identical calls share ONE upstream request."""
    def __init__(self):
        self._lock = threading.Lock()
        self._inflight: Dict[str, dict] = {}
        self.coalesced = 0

    def do(self, key: str, fn: Callable):
        with self._lock:
            slot = self._inflight.get(key)
            leader = slot is None
            if leader:
                slot = {"event": threading.Event(), "result": None, "error": None}
                self._inflight[key] = slot
            else:
                self.coalesced += 1
        if leader:
            try:
                slot["result"] = fn()
            except BaseException as e:  # noqa: BLE001 - re-raised to every waiter
                slot["error"] = e
            finally:
                with self._lock:
                    self._inflight.pop(key, None)
                slot["event"].set()
        else:
            slot["event"].wait(timeout=120)
        if slot["error"] is not None:
            raise slot["error"]
        return slot["result"]


def call_with_backoff(fn: Callable, *, retries: int = 2, base: float = 0.6, cap: float = 8.0,
                      sleep: Callable[[float], None] = time.sleep, rng: random.Random = random):
    """Bounded retries; exponential backoff with full jitter; honours Retry-After."""
    attempt = 0
    while True:
        try:
            return fn()
        except RateLimited as e:
            if attempt >= retries:
                raise
            delay = e.retry_after if e.retry_after is not None else base * (2 ** attempt)
            sleep(min(cap, delay) + rng.uniform(0, 0.25))
        except ProviderError as e:
            if not e.retryable or attempt >= retries:
                raise
            sleep(rng.uniform(0, min(cap, base * (2 ** attempt))))
        attempt += 1


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
class ProviderHealth:
    def __init__(self, name: str):
        self.name = name
        self.successes = 0
        self.failures = 0
        self.consecutive_failures = 0
        self.no_data_queries = 0
        self.last_success_ts: Optional[str] = None
        self.last_error: Optional[str] = None
        self.last_error_ts: Optional[str] = None
        self.last_no_data: Optional[str] = None
        self.last_no_data_ts: Optional[str] = None
        self.latencies_ms: deque = deque(maxlen=50)
        self.open_until: float = 0.0   # simple circuit breaker (monotonic)
        self._lock = threading.Lock()

    def ok(self, latency_ms: float):
        with self._lock:
            self.successes += 1
            self.consecutive_failures = 0
            self.last_success_ts = datetime.now(timezone.utc).isoformat()
            self.latencies_ms.append(latency_ms)

    def record_no_data(self, message: str):
        """Record an instrument-specific query returning no data (e.g. unknown symbol).
        Does NOT increment provider infrastructure failure counters or trip circuit breaker."""
        with self._lock:
            self.no_data_queries += 1
            self.last_no_data = security.redact(message)[:300]
            self.last_no_data_ts = datetime.now(timezone.utc).isoformat()

    def fail(self, message: str, cooldown: float = 0.0):
        with self._lock:
            self.failures += 1
            self.consecutive_failures += 1
            self.last_error = security.redact(message)[:300]
            self.last_error_ts = datetime.now(timezone.utc).isoformat()
            if self.consecutive_failures >= 3 or cooldown:
                self.open_until = time.monotonic() + (cooldown or min(60.0, 5.0 * self.consecutive_failures))

    @property
    def available(self) -> bool:
        return time.monotonic() >= self.open_until

    def snapshot(self) -> dict:
        with self._lock:
            lat = sorted(self.latencies_ms)
            return {
                "provider": self.name, "successes": self.successes, "failures": self.failures,
                "no_data_queries": self.no_data_queries,
                "consecutive_failures": self.consecutive_failures, "circuit_open": not self.available,
                "last_success_ts": self.last_success_ts, "last_error": self.last_error,
                "last_error_ts": self.last_error_ts,
                "last_no_data": self.last_no_data, "last_no_data_ts": self.last_no_data_ts,
                "latency_ms_median": lat[len(lat) // 2] if lat else None,
                "latency_ms_p95": lat[min(len(lat) - 1, int(len(lat) * 0.95))] if lat else None,
                "latency_samples": len(lat),
            }


class QuoteProvider:
    name = "base"
    supplies_realtime = False       # documented real-time entitlement for this provider/credential
    covers_exchanges: tuple = ()    # () = any

    def __init__(self):
        self.health = ProviderHealth(self.name)
        self.sem = threading.BoundedSemaphore(config.PROVIDER_MAX_CONCURRENCY)

    def configured(self) -> bool:
        return True

    def supports(self, inst: Instrument) -> bool:
        return not self.covers_exchanges or inst.exchange in self.covers_exchanges

    def fetch_raw(self, inst: Instrument) -> dict:   # pragma: no cover - interface
        raise NotImplementedError

    def describe(self) -> dict:
        return {"provider": self.name, "configured": self.configured(),
                "supplies_realtime": self.supplies_realtime,
                "covers_exchanges": list(self.covers_exchanges) or "any Yahoo-supported exchange"}


class YahooProvider(QuoteProvider):
    """
    yfinance. Free, no key, unofficial. Never classified "live" (see module docstring).
    Prefers Yahoo's own `regularMarketTime` + `exchangeDataDelayedBy`; if unavailable,
    falls back to the last 1-minute bar's timestamp with delay UNKNOWN -> "unverified".
    """
    name = "yahoo"
    supplies_realtime = False

    def fetch_raw(self, inst: Instrument) -> dict:
        import yfinance as yf
        t = yf.Ticker(inst.yahoo_symbol)
        info = {}
        try:
            info = t.info or {}
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            if "429" in msg or "too many" in msg:
                raise RateLimited("Yahoo rate limit", retry_after=None)
            if "impersonat" in msg:
                raise ProviderError(_CLIENT_INIT_MSG, retryable=False, kind="client_init")
        price = info.get("regularMarketPrice") or info.get("currentPrice")
        ts = info.get("regularMarketTime")
        delay = info.get("exchangeDataDelayedBy")
        if price and ts:
            state = str(info.get("marketState", "")).upper()
            session = {"REGULAR": "regular", "PRE": "pre", "POST": "post", "POSTPOST": "post",
                       "PREPRE": "pre", "CLOSED": "closed"}.get(state)
            return {
                "price": float(price), "open": info.get("regularMarketOpen") or info.get("open"),
                "high": info.get("regularMarketDayHigh") or info.get("dayHigh"),
                "low": info.get("regularMarketDayLow") or info.get("dayLow"),
                "prev_close": info.get("regularMarketPreviousClose") or info.get("previousClose"),
                "volume": info.get("regularMarketVolume") or info.get("volume"),
                "currency": info.get("currency") or inst.currency_hint, "source_ts": ts, "session": session,
                "data_type": "delayed" if (delay or 0) > 0 else "unknown",
                "delay_seconds": float(delay) * 60 if delay else (0.0 if delay == 0 else None),
                "timezone": info.get("exchangeTimezoneName") or inst.timezone,
            }
        # Fallback 1: fast_info if available (modern yfinance)
        fast = getattr(t, "fast_info", None)
        if fast is not None:
            try:
                f_price = getattr(fast, "last_price", None) or (fast.get("last_price") if isinstance(fast, dict) else None)
                if f_price is not None and f_price > 0:
                    f_prev = getattr(fast, "previous_close", None) or (fast.get("previous_close") if isinstance(fast, dict) else None)
                    f_open = getattr(fast, "open", None) or (fast.get("open") if isinstance(fast, dict) else None)
                    f_high = getattr(fast, "day_high", None) or (fast.get("day_high") if isinstance(fast, dict) else None)
                    f_low = getattr(fast, "day_low", None) or (fast.get("day_low") if isinstance(fast, dict) else None)
                    f_vol = getattr(fast, "last_volume", None) or (fast.get("last_volume") if isinstance(fast, dict) else None)
                    f_cur = getattr(fast, "currency", None) or (fast.get("currency") if isinstance(fast, dict) else None) or inst.currency_hint
                    f_tz = getattr(fast, "timezone", None) or (fast.get("timezone") if isinstance(fast, dict) else None) or inst.timezone
                    return {
                        "price": float(f_price), "open": float(f_open) if f_open else None,
                        "high": float(f_high) if f_high else None, "low": float(f_low) if f_low else None,
                        "prev_close": float(f_prev) if f_prev else None, "volume": float(f_vol) if f_vol else None,
                        "currency": f_cur, "source_ts": ts or datetime.now(timezone.utc).isoformat(),
                        "session": "closed" if not ts else None,
                        "data_type": "delayed" if (delay or 0) > 0 else "unknown",
                        "delay_seconds": float(delay) * 60 if delay else None,
                        "timezone": f_tz,
                    }
            except Exception:
                pass
        # Fallback 2: last 1m bar (during or immediately after market hours)
        try:
            h = t.history(period="1d", interval="1m")
            if h is not None and not h.empty:
                last = h.iloc[-1]
                return {
                    "price": float(last["Close"]), "open": float(h.iloc[0]["Open"]),
                    "high": float(h["High"].max()), "low": float(h["Low"].min()), "prev_close": None,
                    "volume": float(h["Volume"].sum()), "currency": inst.currency_hint,
                    "source_ts": h.index[-1].to_pydatetime(), "session": None,
                    "data_type": "unknown", "delay_seconds": None, "timezone": str(h.index.tz) if h.index.tz else inst.timezone,
                }
        except Exception:
            pass
        # Fallback 3: last 5d daily bars (outside market hours / weekends)
        try:
            h_daily = t.history(period="5d", interval="1d")
            if h_daily is not None and not h_daily.empty:
                last = h_daily.iloc[-1]
                prev = h_daily.iloc[-2]["Close"] if len(h_daily) >= 2 else None
                return {
                    "price": float(last["Close"]), "open": float(last["Open"]),
                    "high": float(last["High"]), "low": float(last["Low"]),
                    "prev_close": float(prev) if prev is not None else None,
                    "volume": float(last["Volume"]), "currency": inst.currency_hint,
                    "source_ts": h_daily.index[-1].to_pydatetime(), "session": "closed",
                    "data_type": "eod", "delay_seconds": None,
                    "timezone": str(h_daily.index.tz) if h_daily.index.tz else inst.timezone,
                }
        except Exception:
            pass
        hint = ""
        if inst.exchange == "US/UNSPECIFIED" and not inst.symbol.startswith("^"):
            hint = f" (If this is an Indian stock, try 'NSE:{inst.symbol}' or '{inst.symbol}.NS')"
        raise ProviderError(f"Yahoo returned no data for {inst.yahoo_symbol}.{hint}", retryable=False, kind="no_data")


class UpstoxQuoteProvider(QuoteProvider):
    """
    NSE/BSE quote via the authenticated Upstox market-quote API, using the same cached
    daily token the history adapter uses (never an interactive login inside a web request).

    Endpoint: GET https://api.upstox.com/v2/market-quote/quotes?instrument_key=... with `Authorization: Bearer <token>`
    (official docs: up to 500 keys; "snapshots obtained directly from the exchanges at the time of request";
    `last_trade_time` = epoch MILLISECONDS as a string -- V17 could not parse that, V18 does).

    STATUS: UNVERIFIED against the live API (no network/credentials where this was built). Run
    `python scripts/verify_providers.py` with your token to verify it. Real-time entitlement depends on your Upstox
    account. Caveat: for an illiquid instrument the LAST TRADE can be older than LIVE_MAX_AGE_SECONDS even though the
    feed is live -- the quote is then (safely) classified stale/unverified, never "live".
    """
    name = "upstox"
    covers_exchanges = ("NSE", "BSE")
    QUOTE_URL = "https://api.upstox.com/v2/market-quote/quotes"

    def configured(self) -> bool:
        return bool(config.UPSTOX_CLIENT_ID and self._token())

    def _token(self):
        try:
            from data_sources import upstox_source
            return upstox_source._load_cached_token()
        except Exception:  # noqa: BLE001
            return None

    @property
    def supplies_realtime(self):  # type: ignore[override]
        return self.configured()

    def fetch_raw(self, inst: Instrument) -> dict:
        import requests
        from data_sources import upstox_source
        token = self._token()
        if not token:
            raise ProviderError("No Upstox access token cached for today. Log in once via the CLI "
                                "(python main.py --source upstox).", kind="auth")
        key = upstox_source._resolve_instrument_key(inst.symbol, inst.exchange)
        r = requests.get(self.QUOTE_URL, params={"instrument_key": key}, timeout=config.DATA_FETCH_TIMEOUT_SECONDS,
                         headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
        if r.status_code == 429:
            raise RateLimited("Upstox rate limit", retry_after=_retry_after(r))
        if r.status_code in (401, 403):
            raise ProviderError("Upstox rejected the token (expired or not entitled).", kind="auth")
        if r.status_code >= 500:
            raise ProviderError(f"Upstox server error {r.status_code}.", retryable=True, kind="outage")
        if r.status_code != 200:
            raise ProviderError(security.safe_upstream_detail(r.text, r.status_code), kind="error")
        data = (r.json() or {}).get("data") or {}
        if not data:
            raise ProviderError("Upstox returned no quote data.", kind="no_data")
        q = next(iter(data.values()))
        ohlc = q.get("ohlc") or {}
        ts = q.get("last_trade_time") or q.get("timestamp")
        return {
            "price": q.get("last_price"), "open": ohlc.get("open"), "high": ohlc.get("high"),
            "low": ohlc.get("low"), "prev_close": ohlc.get("close"), "volume": q.get("volume"),
            "currency": "INR", "source_ts": ts, "session": None,
            "data_type": "realtime" if ts else "unknown", "delay_seconds": None, "timezone": "Asia/Kolkata",
        }


_CLIENT_INIT_MSG = ("Yahoo's HTTP client could not initialise (curl_cffi impersonation error). Known cause: TensorFlow was "
                    "imported before yfinance. Import `config` (or call _native_preload.preload()) before TensorFlow; "
                    "see docs/V18_CHANGES.md.")


def _retry_after(resp) -> Optional[float]:
    try:
        return float(resp.headers.get("Retry-After"))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Service: deterministic selection + visible fallback
# ---------------------------------------------------------------------------
_FRESHNESS_RANK = {FRESHNESS_LIVE: 5, FRESHNESS_DELAYED: 4, FRESHNESS_EOD: 3, FRESHNESS_UNVERIFIED: 2,
                   FRESHNESS_STALE: 1, FRESHNESS_UNAVAILABLE: 0, FRESHNESS_HISTORICAL: 0}


class MarketDataService:
    def __init__(self, providers: Optional[List[QuoteProvider]] = None):
        self.providers: List[QuoteProvider] = providers if providers is not None else [UpstoxQuoteProvider(), YahooProvider()]
        self.quote_cache = TTLCache(512)
        self.history_cache = TTLCache(64)
        self.failure_cache = TTLCache(64)       # recent history FAILURES: a failing provider is not re-hit on every refresh
        self.failure_cache_hits = 0
        self.flight = SingleFlight()
        self.timings: Dict[str, deque] = {"quote": deque(maxlen=200), "history": deque(maxlen=200)}

    # -- provider order: deterministic -----------------------------------
    def candidates(self, inst: Instrument, preferred: Optional[str] = None) -> List[QuoteProvider]:
        eligible = [p for p in self.providers if p.supports(inst) and p.configured()]
        # Priority = list order (real-time-capable providers are listed first), preferred pinned to front.
        eligible.sort(key=lambda p: (p.name != preferred, self.providers.index(p)))
        return eligible

    def _query_one(self, p: QuoteProvider, inst: Instrument) -> Quote:
        t0 = time.perf_counter()
        with p.sem:
            raw = call_with_backoff(lambda: p.fetch_raw(inst), retries=2)
        latency = (time.perf_counter() - t0) * 1000
        perf.record(f"quote.upstream.{p.name}", latency)
        cls = classify_freshness(price=_num(raw.get("price")), source_ts=raw.get("source_ts"),
                                 data_type=raw.get("data_type", "unknown"), delay_seconds=raw.get("delay_seconds"),
                                 supplies_realtime=p.supplies_realtime)
        flags = list(cls["flags"])
        src = _parse_ts(raw.get("source_ts"))
        o, h, l, px = (_num(raw.get(k)) for k in ("open", "high", "low", "price"))
        if None not in (o, h, l, px) and (h < max(o, px, l) - 1e-9 or l > min(o, px, h) + 1e-9):
            flags.append("invalid_ohlc_relationship")
        return Quote(
            provider=p.name, symbol=inst.symbol, exchange=inst.exchange, currency=raw.get("currency") or inst.currency_hint,
            price=_num(raw.get("price")), open=o, high=h, low=l, prev_close=_num(raw.get("prev_close")),
            volume=_num(raw.get("volume")), source_ts=src.isoformat() if src else None,
            retrieved_ts=datetime.now(timezone.utc).isoformat(), timezone=raw.get("timezone"),
            session=raw.get("session"), data_type=raw.get("data_type", "unknown"),
            provider_entitled_realtime=bool(p.supplies_realtime),
            delay_seconds=raw.get("delay_seconds"), freshness=cls["freshness"], age_seconds=cls["age_seconds"],
            is_stale=cls["is_stale"], realtime_eligible=cls["realtime_eligible"], quality_flags=flags,
            latency_ms=round(latency, 1),
        ), latency

    def get_quote(self, raw_symbol: str, *, preferred: Optional[str] = None, force: bool = False) -> Quote:
        inst = resolve_instrument(raw_symbol)
        key = f"q|{inst.key}|{preferred or ''}"
        if not force:
            hit = self.quote_cache.get(key)
            if hit:
                q, age, _ = hit
                return _as_cached(q, age)

        def work() -> Quote:
            trail, best = [], None
            cands = self.candidates(inst, preferred)
            if not cands:
                return Quote(provider="none", symbol=inst.symbol, exchange=inst.exchange, freshness=FRESHNESS_UNAVAILABLE,
                             retrieved_ts=datetime.now(timezone.utc).isoformat(),
                             error="No quote provider is configured for this instrument.",
                             selection_trail=[{"provider": "-", "outcome": "none_eligible"}])
            for p in cands:
                if not p.health.available:
                    trail.append({"provider": p.name, "outcome": "skipped", "reason": "circuit open after repeated failures"})
                    continue
                try:
                    q, latency = self._query_one(p, inst)
                    p.health.ok(latency)
                    self.timings["quote"].append(latency)
                    trail.append({"provider": p.name, "outcome": "ok", "freshness": q.freshness})
                    if best is None or _FRESHNESS_RANK[q.freshness] > _FRESHNESS_RANK[best.freshness]:
                        best = q
                    if q.freshness in (FRESHNESS_LIVE,):
                        break                      # cannot do better than live
                    if q.freshness in (FRESHNESS_DELAYED, FRESHNESS_EOD, FRESHNESS_UNVERIFIED):
                        continue                   # ask the next provider only to see if it is fresher
                except RateLimited as e:
                    p.health.fail(str(e), cooldown=e.retry_after or 30)
                    trail.append({"provider": p.name, "outcome": "rate_limited", "reason": str(e)})
                except ProviderError as e:
                    if e.kind == "no_data":
                        p.health.record_no_data(str(e))
                    else:
                        p.health.fail(str(e))
                    trail.append({"provider": p.name, "outcome": e.kind, "reason": security.redact(str(e))[:200]})
                except Exception as e:  # noqa: BLE001 - never let an adapter bug 500 the UI
                    p.health.fail(f"{type(e).__name__}: {e}")
                    trail.append({"provider": p.name, "outcome": "error", "reason": security.safe_exception_text(e)[:200]})
            if best is None:
                reasons = "; ".join(f"{t['provider']}: {t.get('reason') or t['outcome']}" for t in trail)
                return Quote(provider=cands[0].name, symbol=inst.symbol, exchange=inst.exchange,
                             freshness=FRESHNESS_UNAVAILABLE, retrieved_ts=datetime.now(timezone.utc).isoformat(),
                             error=reasons or "No provider returned a quote.", selection_trail=trail)
            best.selection_trail = trail
            top = cands[0]
            if best.provider != top.name:
                best.downgraded_from = top.name
                best.quality_flags.append(f"fallback_from_{top.name}")
            return best

        q = self.flight.do(key + "|flight", work)
        if q.freshness != FRESHNESS_UNAVAILABLE:
            self.quote_cache.set(key, q, config.QUOTE_TTL_SECONDS)
        else:  # serve a recent good quote during an outage, clearly marked stale/cached
            stale = self.quote_cache.get(key, allow_stale_for=300)
            if stale:
                sq, age, _ = stale
                out = _as_cached(sq, age)
                out.quality_flags.append("served_cached_during_outage")
                out.error = q.error
                return out
        return q

    # -- history -----------------------------------------------------------
    def get_history(self, raw_symbol: str, period: str = "1y", interval: str = "1d") -> dict:
        inst = resolve_instrument(raw_symbol)
        if interval != "1d":
            raise ValueError("Only daily bars are supported by this build (interval='1d').")
        if not re.match(r"^\d+(d|mo|y)$|^(ytd|max)$", period):
            raise ValueError("period must look like 30d, 6mo, 2y, ytd or max.")
        key = f"h|{inst.key}|{period}|{interval}"
        hit = self.history_cache.get(key)
        if hit:
            payload, age, _ = hit
            return {**payload, "from_cache": True, "cache_age_seconds": round(age, 1)}
        failed = self.failure_cache.get(key)
        if failed:
            (kind, msg, retry_after), age, _ = failed
            self.failure_cache_hits += 1
            note = f" (recent failure, not retried for {max(1, int(config.HISTORY_FAILURE_TTL_SECONDS - age))}s)"
            if kind == "rate_limited":
                raise RateLimited(msg + note, retry_after=retry_after)
            raise ProviderError(msg + note, retryable=False, kind="recent_failure")

        def work():
            import data_loader
            from data_sources import yfinance_source
            t0 = time.perf_counter()
            df = call_with_backoff(lambda: _history_or_raise(yfinance_source, inst.yahoo_symbol, period, interval), retries=2)
            ms = (time.perf_counter() - t0) * 1000
            self.timings["history"].append(ms)
            perf.record("history.upstream", ms)
            report = data_loader.check_data_quality(df, ticker=inst.key)
            last = df.index[-1]
            payload = {
                "instrument": inst.to_dict(), "provider": "yahoo", "interval": interval, "period": period,
                "adjusted": True, "adjustment_note": "Prices are split/dividend-adjusted (yfinance auto_adjust=True).",
                "freshness": FRESHNESS_HISTORICAL, "rows": int(len(df)),
                "first_bar": str(df.index[0])[:10], "last_bar": str(last)[:10],
                "retrieved_ts": datetime.now(timezone.utc).isoformat(),
                "data_quality": report.to_dict(), "latency_ms": round(ms, 1),
                "bars": _bars(df),
            }
            return payload

        try:
            payload = self.flight.do(key + "|flight", work)
        except (ProviderError, RateLimited) as e:
            if isinstance(e, RateLimited):
                self.failure_cache.set(key, ("rate_limited", str(e), e.retry_after), max(config.HISTORY_FAILURE_TTL_SECONDS, e.retry_after or 0))
            elif getattr(e, "kind", "") != "client_init":
                self.failure_cache.set(key, ("error", str(e), None), config.HISTORY_FAILURE_TTL_SECONDS)
            raise
        self.history_cache.set(key, payload, config.HISTORY_TTL_SECONDS)
        return {**payload, "from_cache": False, "cache_age_seconds": 0.0}

    def health(self) -> dict:
        def pct(dq, p):
            v = sorted(dq)
            return round(v[min(len(v) - 1, int(len(v) * p))], 1) if v else None
        return {
            "providers": [{**p.describe(), **p.health.snapshot()} for p in self.providers],
            "caches": {"quote": self.quote_cache.stats(), "history": self.history_cache.stats()},
            "coalesced_requests": self.flight.coalesced, "history_failure_cache_hits": self.failure_cache_hits,
            "measured_latency_ms": {k: {"median": pct(v, 0.5), "p95": pct(v, 0.95), "samples": len(v)}
                                    for k, v in self.timings.items()},
        }


def _history_or_raise(src, ysym, period, interval):
    from data_sources.base import DataSourceError
    try:
        return src.fetch(ysym, period, interval, max_retries=0)
    except DataSourceError as e:
        msg = str(e).lower()
        if "impersonat" in msg:
            raise ProviderError(_CLIENT_INIT_MSG, retryable=False, kind="client_init")
        retry = any(h in msg for h in ("429", "too many", "timed out", "timeout", "connection", "503", "502"))
        if "429" in msg or "too many" in msg:
            raise RateLimited(str(e))
        raise ProviderError(security.redact(str(e)), retryable=retry, kind="no_data" if "no data" in msg else "error")


def _bars(df) -> List[dict]:
    return [{"t": str(i)[:10], "o": float(r.Open), "h": float(r.High), "l": float(r.Low),
             "c": float(r.Close), "v": float(r.Volume)} for i, r in df.iterrows()]


def _num(v) -> Optional[float]:
    try:
        f = float(v)
        return f if f == f and f not in (float("inf"), float("-inf")) else None
    except (TypeError, ValueError):
        return None


def _as_cached(q: Quote, age: float) -> Quote:
    import copy
    c = copy.deepcopy(q)
    c.from_cache = True
    c.cache_age_seconds = round(age, 1)
    if c.source_ts:   # age since the SOURCE produced it keeps growing while cached
        classified = classify_freshness(price=c.price, source_ts=c.source_ts, data_type=c.data_type,
                                        delay_seconds=c.delay_seconds, supplies_realtime=c.provider_entitled_realtime)
        c.freshness, c.age_seconds = classified["freshness"], classified["age_seconds"]
        c.is_stale, c.realtime_eligible = classified["is_stale"], classified["realtime_eligible"]
    return c


_service: Optional[MarketDataService] = None
_service_lock = threading.Lock()


def get_service() -> MarketDataService:
    global _service
    with _service_lock:
        if _service is None:
            _service = MarketDataService()
        return _service
