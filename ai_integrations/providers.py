"""
ai_integrations/providers.py
-----------------------------
One interface over Gemini, Groq, OpenRouter, OpenAI, Anthropic, xAI and DeepSeek.

Design rules (each fixes a real defect in the old clients):
  * A provider is "connected" only after a real call succeeded. A key existing proves nothing.
  * Failures are CLASSIFIED (missing_key, invalid_key, model_unavailable, unsupported_endpoint,
    rate_limit, insufficient_quota, network, timeout, invalid_response, provider_outage,
    bad_request) from the HTTP status AND the provider's error body -- the old Gemini client
    called raise_for_status() and discarded the body, so a 404 for a retired model looked
    like a generic failure.
  * HTTP 200 is not success: the body must contain non-empty text (blocked / empty / thought-only
    candidates are `invalid_response`).
  * A failing provider enters a cooldown, so page refreshes never hammer it.
  * The model actually used is always returned. Nothing is silently substituted.
  * Concurrency is bounded by a GLOBAL budget; stragglers are abandoned, never awaited.
"""
from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import config
import perf
import security

KINDS = ("missing_key", "invalid_key", "model_unavailable", "unsupported_endpoint", "rate_limit",
         "insufficient_quota", "network", "timeout", "invalid_response", "provider_outage", "bad_request", "unknown")
RETRYABLE = {"rate_limit", "network", "timeout", "provider_outage"}
ACTIONS = {
    "missing_key": "Set the provider's API key in .env and restart.",
    "invalid_key": "The provider rejected the key. Re-create it and update .env.",
    "model_unavailable": "That model id is not available to this key (it may be retired). Use 'Test connection' to list models your key can use, then set the model in .env.",
    "unsupported_endpoint": "The endpoint was not found for this provider. Check the base URL / API version.",
    "rate_limit": "Rate limit reached. The app will back off automatically; try again shortly.",
    "insufficient_quota": "The account has no remaining quota/credit. Wait for the quota window to reset, add credit, or use another provider. Numerical forecasts are unaffected.",
    "network": "Could not reach the provider. Check connectivity / firewall.",
    "timeout": "The provider did not answer in time.",
    "invalid_response": "The provider answered, but the response contained no usable text.",
    "provider_outage": "The provider reports a server-side problem. Try again later.",
    "bad_request": "The provider rejected the request format.",
    "unknown": "Unexpected error. See server logs.",
}


class AIError(Exception):
    def __init__(self, kind: str, message: str, http_status: Optional[int] = None):
        super().__init__(message)
        self.kind = kind if kind in KINDS else "unknown"
        self.http_status = http_status
        self.retryable = self.kind in RETRYABLE


def _body_message(body_text: str) -> str:
    try:
        data = json.loads(body_text)
        err = data.get("error", data) if isinstance(data, dict) else {}
        if isinstance(err, dict):
            return str(err.get("message") or err.get("status") or err.get("code") or "")[:240]
        return str(err)[:240]
    except (ValueError, AttributeError):
        return (body_text or "")[:160]


def classify_http(status: int, body_text: str, provider: str) -> AIError:
    msg = _body_message(body_text)
    low = (body_text or "").lower()
    safe = security.redact(f"HTTP {status}: {msg}".strip())
    if status in (401,) or (status in (400, 403) and ("api key not valid" in low or "api_key_invalid" in low
                                                      or "invalid api key" in low or "permission_denied" in low
                                                      or "invalid x-api-key" in low)):
        return AIError("invalid_key", safe, status)
    if status == 402 or "insufficient_quota" in low or "credit balance is too low" in low or "billing" in low and status in (400, 403, 429):
        return AIError("insufficient_quota", safe, status)
    if status == 404:
        if "model" in low:          # a bare "Not Found" page is an endpoint problem, not a model problem
            return AIError("model_unavailable", safe, status)
        return AIError("unsupported_endpoint", safe, status)
    if status == 429:
        if "quota" in low and ("exceeded your current quota" in low or "per day" in low or "free_tier" in low or "daily" in low):
            return AIError("insufficient_quota", safe, status)
        return AIError("rate_limit", safe, status)
    if status in (500, 502, 503, 504, 529):
        return AIError("provider_outage", safe, status)
    if status in (400, 422):
        return AIError("bad_request", safe, status)
    if status == 403:
        return AIError("invalid_key", safe, status)
    return AIError("unknown", safe, status)


@dataclass
class ProviderState:
    last_success_ts: Optional[str] = None
    last_error_kind: Optional[str] = None
    last_error_message: Optional[str] = None
    last_error_ts: Optional[str] = None
    last_latency_ms: Optional[float] = None
    cooldown_until: float = 0.0
    connection_verified: bool = False
    verified_model: Optional[str] = None
    calls: int = 0
    failures: int = 0
    available_models: Optional[List[str]] = None
    verified_ts: Optional[str] = None


class Provider:
    id = "base"
    label = "Base"
    key_attr = ""
    model_attr = ""
    default_model = ""
    free_tier_note = ""
    data_use_note = ""

    def __init__(self, http: Optional[Callable] = None):
        import requests
        self.http = http or requests.request
        # A provider can only ever be "verified" through the real HTTP transport. Anything injected (tests, harnesses)
        # is a test double, and can at most reach "mock_verified".
        self.real_transport = http is None
        self.state = ProviderState()
        self._lock = threading.Lock()

    # -- config (read live so tests/.env edits apply) ---------------------
    @property
    def api_key(self) -> str:
        return getattr(config, self.key_attr, "") or ""

    @property
    def model(self) -> str:
        return (getattr(config, self.model_attr, "") or self.default_model or "").strip()

    def configured(self) -> bool:
        return bool(self.api_key) and bool(self.model)

    def in_cooldown(self) -> bool:
        return time.monotonic() < self.state.cooldown_until

    # -- HTTP --------------------------------------------------------------
    def _request(self, method, url, headers, payload=None):
        import requests
        try:
            r = self.http(method, url, headers=headers, json=payload, timeout=config.AI_REQUEST_TIMEOUT_SECONDS)
        except requests.exceptions.Timeout:
            raise AIError("timeout", "Request timed out.")
        except requests.exceptions.RequestException as e:
            raise AIError("network", security.safe_exception_text(e)[:200])
        if r.status_code != 200:
            err = classify_http(r.status_code, r.text, self.id)
            ra = r.headers.get("Retry-After") if hasattr(r, "headers") else None
            if err.kind == "rate_limit" and ra:
                try:
                    self.state.cooldown_until = time.monotonic() + min(300.0, float(ra))
                except ValueError:
                    pass
            raise err
        try:
            return r.json()
        except ValueError:
            raise AIError("invalid_response", "Response was not JSON.")

    # -- to be implemented -------------------------------------------------
    def _generate(self, system: str, user: str, max_tokens: int) -> dict:  # -> {"text":..., "usage":...}
        raise NotImplementedError

    def list_models(self) -> List[str]:
        raise NotImplementedError

    # -- public ------------------------------------------------------------
    def generate(self, system: str, user: str, max_tokens: int = 900) -> dict:
        if not self.api_key:
            raise AIError("missing_key", f"{self.label}: no API key configured.")
        if not self.model:
            raise AIError("model_unavailable", f"{self.label}: no model configured (set it in .env).")
        t0 = time.perf_counter()
        with self._lock:
            self.state.calls += 1
        try:
            out = self._generate(system, user, max_tokens)
            text = (out.get("text") or "").strip()
            if not text:
                raise AIError("invalid_response", out.get("why") or "Empty response text.")
        except AIError as e:
            self._record_failure(e)
            perf.record(f"ai.{self.id}.failed_request", (time.perf_counter() - t0) * 1000)
            raise
        ms = (time.perf_counter() - t0) * 1000
        perf.record(f"ai.{self.id}.request", ms)
        with self._lock:
            self.state.last_success_ts = datetime.now(timezone.utc).isoformat()
            self.state.last_latency_ms = round(ms, 1)
            self.state.last_error_kind = None
            self.state.cooldown_until = 0.0
        return {"provider": self.id, "label": self.label, "model": self.model, "text": text,
                "latency_ms": round(ms, 1), "usage": out.get("usage"),
                "generated_ts": datetime.now(timezone.utc).isoformat()}

    def _record_failure(self, e: AIError):
        with self._lock:
            s = self.state
            s.failures += 1
            s.last_error_kind, s.last_error_message = e.kind, security.redact(str(e))[:240]
            s.last_error_ts = datetime.now(timezone.utc).isoformat()
            if e.kind in ("invalid_key", "insufficient_quota", "model_unavailable", "unsupported_endpoint"):
                s.connection_verified = False
            if e.kind != "bad_request":
                s.cooldown_until = max(s.cooldown_until, time.monotonic() + config.AI_FAILURE_COOLDOWN_SECONDS)

    def test_connection(self) -> dict:
        """A REAL call. Lists models (when supported), then makes a tiny generation with the configured model."""
        self.state.cooldown_until = 0.0
        result = {"provider": self.id, "label": self.label, "model": self.model, "ok": False,
                  "tested_ts": datetime.now(timezone.utc).isoformat()}
        if not self.api_key:
            return {**result, "kind": "missing_key", "message": ACTIONS["missing_key"]}
        models = None
        try:
            models = self.list_models()
            self.state.available_models = models
        except AIError as e:
            if e.kind in ("invalid_key", "network", "timeout"):
                self._record_failure(e)
                return {**result, "kind": e.kind, "message": str(e), "action": ACTIONS[e.kind]}
        if not self.model:
            return {**result, "kind": "model_unavailable", "message": "No model configured.",
                    "action": ACTIONS["model_unavailable"], "available_models": (models or [])[:40]}
        try:
            out = self.generate("Reply with the single word: ok", "ping", max_tokens=16)
        except AIError as e:
            return {**result, "kind": e.kind, "message": str(e), "action": ACTIONS[e.kind],
                    "retryable": e.retryable, "available_models": (models or [])[:40]}
        self.state.connection_verified, self.state.verified_model = True, self.model
        self.state.verified_ts = datetime.now(timezone.utc).isoformat()
        return {**result, "ok": True, "kind": "ok", "transport": "real" if self.real_transport else "TEST DOUBLE",
                "latency_ms": out["latency_ms"],
                "available_models": (models or [])[:40], "message": "Connection verified with a real request."}

    def status(self) -> dict:
        s = self.state
        if not self.api_key:
            state = "missing_key"
        elif s.connection_verified and s.last_error_kind is None:
            state = "verified" if self.real_transport else "mock_verified"
        elif s.last_error_kind:
            state = s.last_error_kind
        else:
            state = "untested"
        return {"id": self.id, "label": self.label, "configured": self.configured(), "model": self.model or None,
                "state": state, "verified_model": s.verified_model, "verified_ts": s.verified_ts,
                "transport": "real" if self.real_transport else "test double", "last_success_ts": s.last_success_ts,
                "last_error_kind": s.last_error_kind, "last_error": s.last_error_message,
                "last_error_ts": s.last_error_ts, "last_latency_ms": s.last_latency_ms,
                "cooling_down": self.in_cooldown(), "calls": s.calls, "failures": s.failures,
                "action": ACTIONS.get(s.last_error_kind) if s.last_error_kind else
                          (ACTIONS["missing_key"] if not self.api_key else None),
                "key_env": self.key_attr, "model_env": self.model_attr,
                "free_tier_note": self.free_tier_note, "data_use_note": self.data_use_note}


class GeminiProvider(Provider):
    id, label = "gemini", "Google Gemini"
    key_attr, model_attr, default_model = "GOOGLE_API_KEY", "GOOGLE_MODEL", "gemini-3.5-flash-lite"
    BASE = "https://generativelanguage.googleapis.com/v1beta"
    free_tier_note = "Google lists a free tier for several Flash/Flash-Lite models; limits are per project -- see ai.google.dev rate-limits."
    data_use_note = "On the free tier Google may use prompts/responses to improve its products. Only market data and computed results are sent."

    def _h(self):
        return {"Content-Type": "application/json", "x-goog-api-key": self.api_key}

    def list_models(self):
        data = self._request("GET", f"{self.BASE}/models?pageSize=200", self._h())
        return sorted(m["name"].split("/", 1)[-1] for m in data.get("models", [])
                      if "generateContent" in (m.get("supportedGenerationMethods") or []))

    def _generate(self, system, user, max_tokens):
        payload = {"systemInstruction": {"parts": [{"text": system}]},
                   "contents": [{"role": "user", "parts": [{"text": user}]}],
                   "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens,
                                        "responseMimeType": "application/json" if "JSON" in system else "text/plain"}}
        data = self._request("POST", f"{self.BASE}/models/{self.model}:generateContent", self._h(), payload)
        block = (data.get("promptFeedback") or {}).get("blockReason")
        if block:
            raise AIError("invalid_response", f"Prompt blocked by provider: {block}")
        cands = data.get("candidates") or []
        if not cands:
            raise AIError("invalid_response", "No candidates returned.")
        parts = (cands[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))   # skip thinking parts
        if not text.strip():
            raise AIError("invalid_response", f"Empty candidate (finishReason={cands[0].get('finishReason')}).")
        return {"text": text, "usage": data.get("usageMetadata")}


class OpenAICompatProvider(Provider):
    base_url = ""
    def _h(self):
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
    def list_models(self):
        data = self._request("GET", f"{self.base_url}/models", self._h())
        return sorted(m.get("id", "") for m in data.get("data", []) if m.get("id"))
    def _generate(self, system, user, max_tokens):
        payload = {"model": self.model, "temperature": 0.2, "max_tokens": max_tokens,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        data = self._request("POST", f"{self.base_url}/chat/completions", self._h(), payload)
        ch = (data.get("choices") or [{}])[0]
        return {"text": (ch.get("message") or {}).get("content") or "", "usage": data.get("usage"),
                "why": f"finish_reason={ch.get('finish_reason')}"}


class GroqProvider(OpenAICompatProvider):
    id, label = "groq", "Groq"
    key_attr, model_attr, default_model = "GROQ_API_KEY", "GROQ_MODEL", ""
    base_url = "https://api.groq.com/openai/v1"
    free_tier_note = "Groq offers a free developer tier with rate limits; the models available to you are listed by Test connection."
    data_use_note = "Check Groq's current terms for data use on the free tier."

class OpenRouterProvider(OpenAICompatProvider):
    id, label = "openrouter", "OpenRouter"
    key_attr, model_attr, default_model = "OPENROUTER_API_KEY", "OPENROUTER_MODEL", ""
    base_url = "https://openrouter.ai/api/v1"
    free_tier_note = "OpenRouter lists some models as free (limits apply and change); choose an id from Test connection."
    data_use_note = "Free models may be served by third parties with their own data policies."

class OpenAIProvider(OpenAICompatProvider):
    id, label = "openai", "OpenAI"
    key_attr, model_attr, default_model = "OPENAI_API_KEY", "OPENAI_MODEL", "gpt-4o"
    base_url = "https://api.openai.com/v1"
    free_tier_note = "Paid API. A ChatGPT subscription does not include API credit."

class XAIProvider(OpenAICompatProvider):
    id, label = "xai", "xAI Grok"
    key_attr, model_attr, default_model = "XAI_API_KEY", "XAI_MODEL", "grok-4"
    base_url = "https://api.x.ai/v1"
    free_tier_note = "Paid API."

class DeepSeekProvider(OpenAICompatProvider):
    id, label = "deepseek", "DeepSeek"
    key_attr, model_attr, default_model = "DEEPSEEK_API_KEY", "DEEPSEEK_MODEL", "deepseek-chat"
    base_url = "https://api.deepseek.com"
    free_tier_note = "Paid API."


class AnthropicProvider(Provider):
    id, label = "anthropic", "Anthropic Claude"
    key_attr, model_attr, default_model = "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "claude-sonnet-4-6"
    BASE = "https://api.anthropic.com/v1"
    free_tier_note = "Paid API. A Claude.ai subscription does not include API credit."
    def _h(self):
        return {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    def list_models(self):
        data = self._request("GET", f"{self.BASE}/models?limit=100", self._h())
        return sorted(m.get("id", "") for m in data.get("data", []) if m.get("id"))
    def _generate(self, system, user, max_tokens):
        payload = {"model": self.model, "max_tokens": max_tokens, "system": system,
                   "messages": [{"role": "user", "content": user}]}
        data = self._request("POST", f"{self.BASE}/messages", self._h(), payload)
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        return {"text": text, "usage": data.get("usage"), "why": f"stop_reason={data.get('stop_reason')}"}


PROVIDER_CLASSES = [GeminiProvider, GroqProvider, OpenRouterProvider, AnthropicProvider, OpenAIProvider, XAIProvider, DeepSeekProvider]


class Registry:
    def __init__(self, http=None):
        self.providers: Dict[str, Provider] = {c.id: c(http) for c in PROVIDER_CLASSES}

    def statuses(self) -> List[dict]:
        return [p.status() for p in self.providers.values()]

    def usable(self) -> List[Provider]:
        return [p for p in self.providers.values() if p.configured() and not p.in_cooldown()]

    def generate_first(self, system: str, user: str, validator: Callable[[str], dict],
                       max_tokens: int = 900, budget: Optional[float] = None) -> dict:
        """
        Ask every usable provider concurrently; return the FIRST answer that passes `validator`.
        Slow/failed providers are abandoned (threads are not awaited). Returns
        {"ok": bool, "result": ..., "attempts": [...]} -- never raises.
        """
        budget = config.AI_GLOBAL_BUDGET_SECONDS if budget is None else budget
        provs = self.usable()
        attempts: List[dict] = []
        if not provs:
            why = "no AI provider is configured" if not any(p.configured() for p in self.providers.values()) \
                else "all configured providers are cooling down after recent failures"
            return {"ok": False, "result": None, "attempts": [], "reason": why}
        ex = ThreadPoolExecutor(max_workers=len(provs), thread_name_prefix="ai")
        futs = {ex.submit(p.generate, system, user, max_tokens): p for p in provs}
        deadline = time.monotonic() + budget
        pending = set(futs)
        winner = None
        try:
            while pending and winner is None:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                done, pending = wait(pending, timeout=left, return_when=FIRST_COMPLETED)
                for f in done:
                    p = futs[f]
                    try:
                        raw = f.result()
                        parsed = validator(raw["text"])
                        winner = {**raw, "parsed": parsed}
                        attempts.append({"provider": p.id, "model": p.model, "outcome": "ok"})
                        break
                    except AIError as e:
                        attempts.append({"provider": p.id, "model": p.model, "outcome": e.kind, "message": str(e)[:200]})
                    except ValueError as e:   # validator rejected the content
                        p._record_failure(AIError("invalid_response", str(e)))
                        attempts.append({"provider": p.id, "model": p.model, "outcome": "invalid_response",
                                         "message": security.redact(str(e))[:200]})
            for f in pending:
                attempts.append({"provider": futs[f].id, "model": futs[f].model, "outcome": "timeout",
                                 "message": f"no answer within the {budget:.0f}s global budget; abandoned"})
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
        return {"ok": winner is not None, "result": winner, "attempts": attempts,
                "reason": None if winner else "no provider returned a valid answer"}


_registry: Optional[Registry] = None
_reg_lock = threading.Lock()

def get_registry() -> Registry:
    global _registry
    with _reg_lock:
        if _registry is None:
            _registry = Registry()
        return _registry
