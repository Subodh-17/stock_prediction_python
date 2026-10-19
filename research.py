"""
research.py -- evidence-grounded AI commentary.

The model receives a FACT PACK (computed numbers + retrieved headlines, each with an id) and must
answer in a fixed JSON schema that cites fact ids. Server-side validation then:
  * rejects malformed / non-schema output,
  * drops any citation to an id that was not supplied,
  * lists every number in the prose that does not appear in the fact pack ("unverified_numbers"),
  * ignores any price target / recommendation field (the schema has none).
The AI never trades, never changes settings, and is never part of the numerical forecast.
"""
import hashlib
import json
import re
import threading
import time
from datetime import datetime, timezone

import config
from ai_integrations import providers

SYSTEM = (
    "You are a cautious equity-research assistant. You are given a FACT PACK as JSON. "
    "Use ONLY facts in it. Never state a price, return, date, earnings figure, headline or "
    "statistic that is not in the pack. Do not predict prices, do not give buy/sell advice, and do not "
    "claim certainty. Separate observations (cite fact ids) from hypotheses (label them). "
    "List anything you would need but was not supplied under missing_information. "
    "Respond with JSON ONLY, no markdown, exactly this shape: "
    '{"summary": str, "model_disagreement": str, '
    '"observations": [{"text": str, "fact_ids": [str]}], '
    '"risk_factors": [{"text": str, "fact_ids": [str]}], '
    '"hypotheses": [{"text": str, "fact_ids": [str]}], '
    '"news": [{"news_id": str, "summary": str}], '
    '"limitations": [str], "missing_information": [str]}'
)

_cache, _lock = {}, threading.Lock()


def build_fact_pack(*, instrument: dict, quote: dict | None, forecast: dict | None,
                    evaluation: dict | None, news: list | None, indicators: dict | None) -> dict:
    facts = {}
    def add(fid, label, value, unit=None, asof=None):
        if value is not None:
            facts[fid] = {"label": label, "value": value, "unit": unit, "as_of": asof}
    if quote and quote.get("price") is not None:
        add("F_QUOTE", "last price", quote["price"], quote.get("currency"), quote.get("source_ts"))
        add("F_FRESH", "quote freshness", quote.get("freshness"))
        add("F_PROV", "quote provider", quote.get("provider"))
    for k, v in (indicators or {}).items():
        add(f"F_IND_{k.upper()}", f"indicator {k}", v)
    if forecast:
        for name, v in (forecast.get("components") or {}).items():
            add(f"F_PRED_{name.upper()}", f"{name} next-day predicted close", v, None, forecast.get("as_of_bar"))
        add("F_PRED_FINAL", "numerical ensemble next-day predicted close", forecast.get("final_price"))
        add("F_PRED_PCT", "ensemble predicted % change vs last close", forecast.get("pct_change"), "%")
        add("F_TARGET", "forecast target definition", forecast.get("target"))
        add("F_INTERVAL", "empirical interval status", (forecast.get("interval") or {}).get("status"))
    if evaluation:
        for r in evaluation.get("models", []):
            add(f"F_EVAL_{r['model'].upper()}_RMSE", f"{r['model']} out-of-sample RMSE (price)", r.get("rmse"), None,
                evaluation.get("period"))
            add(f"F_EVAL_{r['model'].upper()}_SKILL", f"{r['model']} RMSE improvement vs naive", r.get("rmse_improvement_vs_naive_pct"), "%")
            add(f"F_EVAL_{r['model'].upper()}_DIR", f"{r['model']} directional accuracy", r.get("directional_accuracy_pct"), "%")
        add("F_EVAL_N", "out-of-sample observations", evaluation.get("n_obs"))
    news_items = {f"N{i+1}": {"title": n["title"], "publisher": n["publisher"], "published_ts": n.get("published_ts")}
                  for i, n in enumerate((news or [])[:8])}
    return {"instrument": instrument, "facts": facts, "news": news_items}


def _numbers(text: str) -> set:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def validate_response(raw_text: str, pack: dict) -> dict:
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw_text.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", cleaned, re.S)
    if not m:
        raise ValueError("no JSON object in the response")
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"malformed JSON: {e}")
    if not isinstance(data, dict) or not isinstance(data.get("summary"), str) or not data["summary"].strip():
        raise ValueError("missing 'summary' string")
    fact_ids, news_ids = set(pack["facts"]), set(pack["news"])
    dropped = []

    def items(key):
        out = []
        for it in data.get(key) or []:
            if not isinstance(it, dict) or not isinstance(it.get("text"), str):
                continue
            ids = [i for i in (it.get("fact_ids") or []) if isinstance(i, str)]
            good = [i for i in ids if i in fact_ids]
            dropped.extend(i for i in ids if i not in fact_ids)
            out.append({"text": it["text"].strip(), "fact_ids": good, "grounded": bool(good)})
        return out

    clean = {
        "summary": data["summary"].strip(),
        "model_disagreement": str(data.get("model_disagreement") or "").strip(),
        "observations": items("observations"), "risk_factors": items("risk_factors"),
        "hypotheses": items("hypotheses"),
        "news": [{"news_id": n["news_id"], "summary": str(n.get("summary", "")).strip()}
                 for n in (data.get("news") or []) if isinstance(n, dict) and n.get("news_id") in news_ids],
        "limitations": [str(x) for x in (data.get("limitations") or [])][:8],
        "missing_information": [str(x) for x in (data.get("missing_information") or [])][:8],
        "dropped_citations": sorted(set(dropped)),
    }
    prose = " ".join([clean["summary"], clean["model_disagreement"]] +
                     [i["text"] for k in ("observations", "risk_factors", "hypotheses") for i in clean[k]] +
                     [n["summary"] for n in clean["news"]])
    allowed = _numbers(json.dumps(pack))
    allowed_f = {float(x) for x in allowed}
    def supported(tok):
        v = float(tok)
        return tok in allowed or any(abs(v - a) <= max(0.006 * abs(a), 0.006) for a in allowed_f)
    clean["unverified_numbers"] = sorted({t for t in _numbers(prose) if len(t.replace(".", "")) >= 2 and not supported(t)})
    return clean


def generate(pack: dict, registry=None, force=False) -> dict:
    registry = registry or providers.get_registry()
    key = hashlib.sha256(json.dumps(pack, sort_keys=True, default=str).encode()).hexdigest()
    with _lock:
        hit = _cache.get(key)
    if hit and not force and time.monotonic() - hit[0] < config.AI_RESEARCH_CACHE_SECONDS:
        return {**hit[1], "from_cache": True}
    user = "FACT PACK:\n" + json.dumps(pack, default=str)
    out = registry.generate_first(SYSTEM, user, lambda t: validate_response(t, pack))
    if not out["ok"]:
        return {"status": "unavailable", "reason": out["reason"], "attempts": out["attempts"], "from_cache": False,
                "note": "Numerical forecasts are unaffected. No commentary was generated."}
    r = out["result"]
    res = {"status": "ok", "commentary": r["parsed"], "provider": r["provider"], "model": r["model"],
           "latency_ms": r["latency_ms"], "generated_ts": r["generated_ts"], "attempts": out["attempts"],
           "fact_pack": pack, "from_cache": False,
           "disclaimer": "AI-generated interpretation of the supplied facts. It is not a forecast, not advice, and may be wrong."}
    with _lock:
        _cache[key] = (time.monotonic(), res)
        if len(_cache) > 64:
            _cache.pop(next(iter(_cache)))
    return res
