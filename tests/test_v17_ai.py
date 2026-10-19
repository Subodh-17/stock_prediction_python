import json
import time

import pytest
import requests

import config
import research
from ai_integrations import providers as P
from ai_integrations.providers import AIError, classify_http, GeminiProvider, GroqProvider, Registry
from tests.v17_helpers import FakeResp

KEY = "AIzaSyFAKE_KEY_FOR_TESTS_0123456789abcdef"


@pytest.fixture(autouse=True)
def _cfg(monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_API_KEY", KEY)
    monkeypatch.setattr(config, "GOOGLE_MODEL", "gemini-3.5-flash-lite")
    for k in ("GROQ_API_KEY", "GROQ_MODEL", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "XAI_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.setattr(config, k, "", raising=False)
    monkeypatch.setattr(config, "AI_FAILURE_COOLDOWN_SECONDS", 60)


def gem_ok(text='{"summary":"ok"}'):
    return FakeResp(200, {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}]})


# ---- classification
@pytest.mark.parametrize("status,body,kind", [
    (404, {"error": {"code": 404, "message": "models/gemini-2.0-flash is not found for API version v1beta", "status": "NOT_FOUND"}}, "model_unavailable"),
    (404, "<html>Not Found</html>", "unsupported_endpoint"),
    (400, {"error": {"message": "API key not valid. Please pass a valid API key.", "status": "INVALID_ARGUMENT"}}, "invalid_key"),
    (401, {"error": {"message": "Incorrect API key provided"}}, "invalid_key"),
    (429, {"error": {"message": "You exceeded your current quota, please check your plan and billing details.", "code": "insufficient_quota"}}, "insufficient_quota"),
    (429, {"error": {"message": "Quota exceeded for metric generate_content_free_tier_requests, limit: 20 per day", "status": "RESOURCE_EXHAUSTED"}}, "insufficient_quota"),
    (429, {"error": {"message": "Too many requests, slow down"}}, "rate_limit"),
    (503, {"error": {"message": "The model is overloaded", "status": "UNAVAILABLE"}}, "provider_outage"),
    (400, {"error": {"message": "Your credit balance is too low to access the Anthropic API"}}, "insufficient_quota"),
    (400, {"error": {"message": "Invalid JSON payload"}}, "bad_request"),
])
def test_http_errors_are_classified_from_status_and_body(status, body, kind):
    assert classify_http(status, json.dumps(body) if isinstance(body, dict) else body, "x").kind == kind


def test_error_messages_never_contain_the_key():
    e = classify_http(400, json.dumps({"error": {"message": f"bad key {KEY}"}}), "gemini")
    assert KEY not in str(e)


# ---- Gemini adapter
def test_gemini_uses_header_auth_current_model_and_never_puts_key_in_url():
    seen = {}
    def http(method, url, headers=None, json=None, timeout=None):
        seen.update(method=method, url=url, headers=headers, payload=json, timeout=timeout)
        return gem_ok()
    out = GeminiProvider(http).generate("Return JSON", "hi")
    assert seen["url"].endswith("/v1beta/models/gemini-3.5-flash-lite:generateContent")
    assert KEY not in seen["url"] and seen["headers"]["x-goog-api-key"] == KEY
    assert seen["payload"]["generationConfig"]["responseMimeType"] == "application/json"
    assert out["model"] == "gemini-3.5-flash-lite" and out["provider"] == "gemini"


def test_default_gemini_model_is_not_the_retired_one():
    assert GeminiProvider.default_model != "gemini-2.0-flash"
    import importlib; src = open(config.__file__).read()
    assert '"GOOGLE_MODEL", "gemini-2.0-flash"' not in src


def test_http_200_with_thought_only_or_empty_candidate_is_invalid_response():
    thought = FakeResp(200, {"candidates": [{"content": {"parts": [{"text": "thinking...", "thought": True}]}}]})
    with pytest.raises(AIError) as e:
        GeminiProvider(lambda *a, **k: thought).generate("s", "u")
    assert e.value.kind == "invalid_response"
    blocked = FakeResp(200, {"promptFeedback": {"blockReason": "SAFETY"}})
    with pytest.raises(AIError) as e:
        GeminiProvider(lambda *a, **k: blocked).generate("s", "u")
    assert e.value.kind == "invalid_response" and "SAFETY" in str(e.value)


def test_timeout_and_network_errors_are_classified():
    def boom(exc):
        def f(*a, **k): raise exc
        return f
    with pytest.raises(AIError) as e:
        GeminiProvider(boom(requests.exceptions.Timeout())).generate("s", "u")
    assert e.value.kind == "timeout" and e.value.retryable
    with pytest.raises(AIError) as e:
        GeminiProvider(boom(requests.exceptions.ConnectionError(f"failed {KEY}"))).generate("s", "u")
    assert e.value.kind == "network" and KEY not in str(e.value)


def test_missing_key_makes_no_http_call_and_is_not_verified(monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "")
    called = []
    p = GeminiProvider(lambda *a, **k: called.append(1))
    r = p.test_connection()
    assert not called and r["ok"] is False and r["kind"] == "missing_key"
    assert p.status()["state"] == "missing_key"


def test_key_alone_is_not_verified_until_a_real_request_succeeds():
    p = GeminiProvider(lambda *a, **k: gem_ok("ok"))
    assert p.status()["state"] == "untested"
    p.generate("s", "u")                      # a plain generate does not verify
    assert p.status()["state"] == "untested"
    models = FakeResp(200, {"models": [{"name": "models/gemini-3.5-flash-lite", "supportedGenerationMethods": ["generateContent"]}]})
    p2 = GeminiProvider(lambda m, url, **k: models if url.endswith("pageSize=200") else gem_ok("ok"))
    r = p2.test_connection()
    assert r["ok"] and p2.status()["state"] == "mock_verified" and r["transport"] == "TEST DOUBLE"   # V18: a double is never "verified"
    assert "gemini-3.5-flash-lite" in r["available_models"]


def test_test_connection_for_retired_model_lists_what_the_key_can_use():
    models = FakeResp(200, {"models": [{"name": "models/gemini-3.8-flash", "supportedGenerationMethods": ["generateContent"]},
                                       {"name": "models/embedding-001", "supportedGenerationMethods": ["embedContent"]}]})
    nf = FakeResp(404, {"error": {"message": "models/gemini-2.0-flash is not found", "status": "NOT_FOUND"}})
    config.GOOGLE_MODEL = "gemini-2.0-flash"
    p = GeminiProvider(lambda m, url, **k: models if "pageSize" in url else nf)
    r = p.test_connection()
    assert not r["ok"] and r["kind"] == "model_unavailable" and r["available_models"] == ["gemini-3.8-flash"]
    assert p.status()["state"] == "model_unavailable"


# ---- registry: concurrency, cooldown, budget
def _registry(**http_by_provider):
    reg = Registry()
    for pid, http in http_by_provider.items():
        reg.providers[pid].http = http
    return reg


def _validator(t):
    return json.loads(t)


def test_no_providers_configured_is_reported_not_faked(monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "")
    out = Registry().generate_first("s", "u", _validator)
    assert out["ok"] is False and "no AI provider is configured" in out["reason"]


def test_one_failing_provider_does_not_block_another(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "gsk_testkey_123456789"); monkeypatch.setattr(config, "GROQ_MODEL", "some-model")
    groq_ok = FakeResp(200, {"choices": [{"message": {"content": '{"summary":"from groq"}'}, "finish_reason": "stop"}]})
    reg = _registry(gemini=lambda *a, **k: FakeResp(429, {"error": {"message": "Quota exceeded per day free_tier"}}),
                    groq=lambda *a, **k: groq_ok)
    out = reg.generate_first("s", "u", _validator)
    assert out["ok"] and out["result"]["provider"] == "groq" and out["result"]["model"] == "some-model"
    assert any(a["provider"] == "gemini" and a["outcome"] == "insufficient_quota" for a in out["attempts"])


def test_failed_provider_enters_cooldown_and_is_not_called_again():
    n = {"c": 0}
    def http(*a, **k):
        n["c"] += 1
        return FakeResp(503, {"error": {"message": "overloaded"}})
    reg = _registry(gemini=http)
    reg.generate_first("s", "u", _validator)
    first = n["c"]
    out = reg.generate_first("s", "u", _validator)
    assert n["c"] == first and out["ok"] is False and "cooling down" in out["reason"]


def test_slow_provider_is_abandoned_within_global_budget():
    reg = _registry(gemini=lambda *a, **k: (time.sleep(3), gem_ok())[1])
    t0 = time.time()
    out = reg.generate_first("s", "u", _validator, budget=0.4)
    assert time.time() - t0 < 1.5 and not out["ok"] and out["attempts"][0]["outcome"] == "timeout"


def test_malformed_json_is_rejected_by_validator_not_trusted():
    reg = _registry(gemini=lambda *a, **k: gem_ok("this is not json"))
    out = reg.generate_first("s", "u", _validator)
    assert not out["ok"] and out["attempts"][0]["outcome"] == "invalid_response"


def test_status_never_exposes_secrets():
    reg = _registry(gemini=lambda *a, **k: FakeResp(401, {"error": {"message": f"bad {KEY}"}}))
    reg.providers["gemini"].test_connection()
    assert KEY not in json.dumps(reg.statuses())


# ---- research grounding
PACK = {"instrument": {"symbol": "X"}, "news": {"N1": {"title": "t", "publisher": "p"}},
        "facts": {"F_QUOTE": {"label": "last price", "value": 101.5}, "F_PRED_FINAL": {"label": "pred", "value": 103.2}}}


def test_validation_drops_unknown_citations_and_flags_unverified_numbers():
    raw = json.dumps({"summary": "The close was 101.5 and the model expects 103.2; analysts target 250.",
                      "observations": [{"text": "x", "fact_ids": ["F_QUOTE", "F_MADE_UP"]}],
                      "price_target": 999, "recommendation": "BUY", "news": [{"news_id": "N9", "summary": "invented"}]})
    out = research.validate_response(raw, PACK)
    assert out["dropped_citations"] == ["F_MADE_UP"] and out["observations"][0]["fact_ids"] == ["F_QUOTE"]
    assert "250" in out["unverified_numbers"] and "101.5" not in out["unverified_numbers"]
    assert "price_target" not in out and "recommendation" not in out and out["news"] == []


def test_validation_rejects_non_json_and_missing_summary():
    for bad in ("hello", "{}", '{"summary": ""}', "{broken"):
        with pytest.raises(ValueError):
            research.validate_response(bad, PACK)


def test_research_unavailable_state_when_all_providers_down(monkeypatch):
    monkeypatch.setattr(config, "GOOGLE_API_KEY", "")
    r = research.generate(PACK, registry=Registry(), force=True)
    assert r["status"] == "unavailable" and "commentary" not in r


def test_research_caches_identical_requests():
    n = {"c": 0}
    def http(*a, **k):
        n["c"] += 1
        return gem_ok(json.dumps({"summary": "fine 101.5"}))
    reg = _registry(gemini=http)
    research._cache.clear()
    a = research.generate(PACK, registry=reg); b = research.generate(PACK, registry=reg)
    assert a["status"] == "ok" and b["from_cache"] and n["c"] == 1


def test_llm_is_not_in_the_numeric_ensemble_by_default():
    assert config.ENSEMBLE_WEIGHTS["llm_panel"] == 0.0
    assert abs(sum(config.ENSEMBLE_WEIGHTS.values()) - 1.0) < 1e-9


def test_legacy_panel_ceiling_is_real():
    from ai_integrations import panel
    from ai_integrations.base_client import BaseLLMClient
    class Slow(BaseLLMClient):
        provider_name = "Slow"
        def _call_api(self, prompt): time.sleep(4); return "{}"
    old = config.LLM_PANEL_TIMEOUT_SECONDS
    config.LLM_PANEL_TIMEOUT_SECONDS = 0.5
    try:
        orig = panel.get_all_clients; panel.get_all_clients = lambda: [Slow("k" * 12, "m")]
        t0 = time.time(); res = panel.query_llm_panel("t", verbose=False)
        assert time.time() - t0 < 1.5 and "Timed out" in res[0]["error"]
    finally:
        panel.get_all_clients = orig; config.LLM_PANEL_TIMEOUT_SECONDS = old
