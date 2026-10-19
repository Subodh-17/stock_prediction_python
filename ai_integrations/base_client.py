"""
ai_integrations/base_client.py
--------------------------------
Shared interface + prompt template for every LLM provider client. Each
concrete client (Claude, ChatGPT, Gemini, Grok, DeepSeek) only needs to
implement `_call_api`; parsing and error handling are shared here so all
five behave consistently in the ensemble.
"""

import json
import re
from abc import ABC, abstractmethod

import numpy as np

import config
import security

VALID_DIRECTIONS = {"up", "down", "flat"}
# A next-day percent change beyond this magnitude is treated as almost
# certainly a hallucinated/garbled value rather than a genuine forecast --
# real single-day moves this large are extremely rare for the kind of
# liquid equities this project targets. This excludes the response from
# the ensemble (see BaseLLMClient.get_prediction) rather than silently
# clamping it, since a value that far out is more likely nonsense than a
# genuine extreme prediction worth down-weighting.
MAX_SANE_PCT_CHANGE = 25.0

PROMPT_TEMPLATE = """You are a financial analysis assistant. Based ONLY on the trend data below,
give a short, disciplined forecast for the next trading day's closing price direction.

{trend_summary}

Respond with STRICT JSON only, no markdown fences, no commentary outside the JSON,
in exactly this shape:
{{
  "direction": "up" | "down" | "flat",
  "confidence": <float between 0 and 1>,
  "estimated_pct_change": <float, your estimated next-day percent change, e.g. 0.5 or -1.2>,
  "reasoning": "<one or two sentences>"
}}
"""


class BaseLLMClient(ABC):
    provider_name = "base"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def build_prompt(self, trend_summary: str) -> str:
        return PROMPT_TEMPLATE.format(trend_summary=trend_summary)

    @abstractmethod
    def _call_api(self, prompt: str) -> str:
        """Returns the raw text response from the provider."""
        raise NotImplementedError

    def get_prediction(self, trend_summary: str) -> dict:
        """
        Returns a normalized dict:
        {provider, direction, confidence, estimated_pct_change, reasoning, error}
        Never raises -- errors (including a malformed/invalid response) are
        captured in the "error" field so one failing or hallucinating
        provider never crashes the whole ensemble run, and ensemble.py
        already excludes any result with error is not None from the blend.
        """
        if not self.is_configured():
            return self._error_result("No API key configured")

        prompt = self.build_prompt(trend_summary)
        try:
            raw_text = self._call_api(prompt)
            if not raw_text or not raw_text.strip():
                raise ValueError("Empty response from provider.")
            parsed = self._parse_json_response(raw_text)
            parsed["provider"] = self.provider_name
            parsed["error"] = None
            return parsed
        except Exception as e:
            return self._error_result(_redact_secrets(str(e)))

    def _parse_json_response(self, raw_text: str) -> dict:
        """
        Parses and VALIDATES the provider's JSON response. Anything that
        doesn't look like a trustworthy, well-formed prediction raises
        ValueError/TypeError, which get_prediction() turns into an "error"
        result excluded from the ensemble -- never a default/guessed value
        silently passed through as if it were real model output.
        """
        # Strip markdown code fences if the model added them despite instructions
        cleaned = re.sub(r"^```(json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
        # Grab the first {...} block in case there's stray text around it
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError(f"Could not find JSON in response: {raw_text[:200]}")

        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON in response: {e}")

        if not isinstance(data, dict):
            raise ValueError(f"Expected a JSON object, got {type(data).__name__}.")

        if "direction" not in data:
            raise ValueError("Response is missing the required 'direction' field.")
        direction = str(data["direction"]).strip().lower()
        if direction not in VALID_DIRECTIONS:
            raise ValueError(
                f"Invalid 'direction' value {data['direction']!r} -- must be one of "
                f"{sorted(VALID_DIRECTIONS)}."
            )

        try:
            confidence = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            raise ValueError(f"'confidence' is not numeric: {data.get('confidence')!r}")
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"'confidence' out of the valid [0, 1] range: {confidence}")

        try:
            estimated_pct_change = float(data.get("estimated_pct_change", 0.0))
        except (TypeError, ValueError):
            raise ValueError(
                f"'estimated_pct_change' is not numeric: {data.get('estimated_pct_change')!r}"
            )
        if not np.isfinite(estimated_pct_change):
            raise ValueError("'estimated_pct_change' is not a finite number (NaN/inf).")
        if abs(estimated_pct_change) > MAX_SANE_PCT_CHANGE:
            raise ValueError(
                f"'estimated_pct_change' ({estimated_pct_change}) exceeds the sane bound of "
                f"+/-{MAX_SANE_PCT_CHANGE}% -- treating as a likely hallucinated value."
            )

        return {
            "direction": direction,
            "confidence": confidence,
            "estimated_pct_change": estimated_pct_change,
            "reasoning": data.get("reasoning", ""),
        }

    def _error_result(self, error_msg: str) -> dict:
        return {
            "provider": self.provider_name,
            "direction": None,
            "confidence": 0.0,
            "estimated_pct_change": 0.0,
            "reasoning": None,
            # Redacted again here (belt and suspenders on top of the
            # get_prediction() call site) since this is also reachable
            # directly (e.g. the "No API key configured" case above) and is
            # the one chokepoint every provider's error message passes
            # through before being stored in llm_results -- which the
            # webapp displays and report.json/CSV/PDF exports include.
            "error": _redact_secrets(error_msg) if error_msg else error_msg,
        }


def _redact_secrets(text: str) -> str:
    """
    Strips every configured LLM API key out of a string before it's ever
    stored in an "error" field -- which get_prediction()/_error_result()
    funnel every provider's failure message through, and which the webapp
    displays and report.json/CSV/PDF exports all include verbatim.

    Concrete motivation (a real finding from a security review, not a
    hypothetical): Google's Gemini REST API takes its key as a URL query
    parameter. When that request fails, requests.exceptions.HTTPError's
    default string representation includes the full URL -- key included --
    verbatim (confirmed directly: "400 Client Error: ... for url:
    https://...?key=<the actual key>"). Redacting here, at the one
    chokepoint every provider's exception already passes through, closes
    that leak (and protects against the same class of bug in any other
    provider or future code path) regardless of which URL scheme any given
    provider's HTTP client happens to use.
    """
    # Delegates to security.redact(), which covers every secret this project
    # holds -- not just the five LLM keys this function originally knew about
    # -- plus pattern-based scrubbing for credentials that belong to an
    # upstream service and were never in our config at all. Kept as a thin
    # wrapper under its original name so existing call sites and the tests
    # that target it directly keep working.
    return security.redact(text)
