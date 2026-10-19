"""
provider_verification.py -- GENUINE provider checks, run by YOU with YOUR credentials (scripts/verify_providers.py).

Outcomes (never inferred, never mocked):
  PASS            a real request was made and its response passed validation
  FAIL            a real request was made and failed validation / was rejected (error class recorded)
  BLOCKED         the provider host could not be reached from this machine (no real request completed)
  NOT_CONFIGURED  no credentials/token (or not applicable) so nothing was attempted

Records contain provider, endpoint class (history vs quote vs LLM), instrument, UTC time and evidence -- never a secret.
The web app only DISPLAYS this file; it never writes "PASS" on its own.
"""
import json
import os
import socket
from datetime import datetime, timezone
from urllib.parse import urlparse

import config
import security

PATH = lambda: os.path.join(config.OUTPUT_DIR, "provider_verification.json")
HOSTS = {"yahoo": "https://query1.finance.yahoo.com", "upstox": "https://api.upstox.com",
         "gemini": "https://generativelanguage.googleapis.com", "groq": "https://api.groq.com",
         "openrouter": "https://openrouter.ai", "openai": "https://api.openai.com", "anthropic": "https://api.anthropic.com",
         "xai": "https://api.x.ai", "deepseek": "https://api.deepseek.com"}


def reachable(url: str, timeout: float = 6.0) -> bool:
    """TCP-level reachability of the provider host. False means we cannot even connect (=> BLOCKED)."""
    u = urlparse(url)
    try:
        with socket.create_connection((u.hostname, u.port or 443), timeout=timeout):
            return True
    except OSError:
        return False


def _rec(provider, endpoint_class, outcome, instrument=None, evidence=None, detail=None, transport="real"):
    return {"provider": provider, "endpoint_class": endpoint_class, "instrument": instrument, "outcome": outcome,
            "utc": datetime.now(timezone.utc).isoformat(), "transport": transport,
            "evidence": evidence or {}, "detail": security.redact(detail or "")[:300] or None}


def check_yahoo_history(svc, symbol="AAPL"):
    if not reachable(HOSTS["yahoo"]):
        return _rec("yahoo", "history (daily bars)", "BLOCKED", symbol, detail="host unreachable from this machine")
    try:
        h = svc.get_history(symbol, "1mo")
        b = h["bars"]
        ok = len(b) >= 5 and all(x["h"] >= max(x["o"], x["c"]) - 1e-9 and x["l"] <= min(x["o"], x["c"]) + 1e-9 for x in b) \
            and h["last_bar"] <= datetime.now(timezone.utc).date().isoformat()
        return _rec("yahoo", "history (daily bars)", "PASS" if ok else "FAIL", symbol,
                    {"rows": h["rows"], "first_bar": h["first_bar"], "last_bar": h["last_bar"], "adjusted": h["adjusted"],
                     "data_quality_issues": h["data_quality"].get("issues")}, None if ok else "bars failed validation")
    except Exception as e:  # noqa: BLE001
        return _rec("yahoo", "history (daily bars)", "FAIL", symbol, detail=f"{type(e).__name__}: {e}")


def _check_quote(name, provider, symbol, require_config=True):
    import market_data as md
    if require_config and not provider.configured():
        return _rec(name, "quote", "NOT_CONFIGURED", symbol, detail="credentials/token not available")
    if not reachable(HOSTS[name]):
        return _rec(name, "quote", "BLOCKED", symbol, detail="host unreachable from this machine")
    try:
        inst = md.resolve_instrument(symbol)
        q = md.MarketDataService([provider])._query_one(provider, inst)[0]
        valid = q.price is not None and q.price > 0 and q.source_ts is not None and q.symbol == inst.symbol \
            and q.exchange == inst.exchange and "invalid_ohlc_relationship" not in q.quality_flags
        return _rec(name, "quote", "PASS" if valid else "FAIL", symbol,
                    {"freshness_category": q.freshness, "age_seconds": q.age_seconds, "source_ts": q.source_ts,
                     "retrieved_ts": q.retrieved_ts, "currency": q.currency, "exchange": q.exchange,
                     "realtime_eligible": q.realtime_eligible, "flags": q.quality_flags},
                    None if valid else "quote failed validation (price/timestamp/symbol/exchange/OHLC)")
    except Exception as e:  # noqa: BLE001
        return _rec(name, "quote", "FAIL", symbol, detail=f"{type(e).__name__}: {e}")


def check_ai(provider):
    if not provider.api_key:
        return _rec(provider.id, "LLM generateContent/chat", "NOT_CONFIGURED", detail=f"{provider.key_attr} not set")
    if not reachable(HOSTS.get(provider.id, "https://example.invalid")):
        return _rec(provider.id, "LLM generateContent/chat", "BLOCKED", detail="host unreachable from this machine")
    r = provider.test_connection()
    return _rec(provider.id, "LLM generateContent/chat", "PASS" if r["ok"] else "FAIL",
                evidence={"model": provider.model, "latency_ms": r.get("latency_ms"), "kind": r["kind"],
                          "models_available_to_key": (r.get("available_models") or [])[:25]},
                detail=None if r["ok"] else r.get("message"), transport="real" if provider.real_transport else "TEST DOUBLE")


def run_all(svc=None, registry=None, yahoo_symbol="AAPL", nse_symbol="NSE:RELIANCE") -> list:
    import market_data as md
    from ai_integrations import providers
    svc = svc or md.get_service()
    registry = registry or providers.Registry()
    out = [check_yahoo_history(svc, yahoo_symbol),
           _check_quote("yahoo", md.YahooProvider(), yahoo_symbol, require_config=False),
           _check_quote("upstox", md.UpstoxQuoteProvider(), nse_symbol)]
    out += [check_ai(p) for p in registry.providers.values()]
    return out


def save(records: list) -> str:
    path = PATH()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"written_utc": datetime.now(timezone.utc).isoformat(), "records": records}, f, indent=2)
    return path


def load() -> dict:
    try:
        with open(PATH()) as f:
            d = json.load(f)
        return d if isinstance(d, dict) and isinstance(d.get("records"), list) else {"records": []}
    except (OSError, ValueError):
        return {"records": []}
