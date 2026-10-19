"""
Gemini (Google) client for the AI prediction panel.

Implemented as a plain REST call (via `requests`) rather than the
`google-generativeai` SDK. The official SDK pulls in a protobuf/grpc version
that conflicts with TensorFlow's requirements in the same environment; the
REST API avoids that dependency clash entirely and needs nothing extra
beyond the `requests` library already used elsewhere in this project.
"""

from .base_client import BaseLLMClient
import config


class GeminiClient(BaseLLMClient):
    provider_name = "Gemini"
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self):
        super().__init__(api_key=config.GOOGLE_API_KEY, model=config.GOOGLE_MODEL)

    def _call_api(self, prompt: str) -> str:
        return self._post(prompt, auth_mode=config.GEMINI_AUTH_MODE)

    def _post(self, prompt: str, auth_mode: str) -> str:
        import requests
        # SECURITY: the key is sent as an `x-goog-api-key` REQUEST HEADER by
        # default, not as a `?key=` URL query parameter.
        #
        # The query-parameter form (Google's other documented auth scheme for
        # this API, and what this client used to do) puts the secret in the
        # URL, which means requests.exceptions.HTTPError's default string
        # representation contains the key verbatim -- e.g. "400 Client Error:
        # ... for url: https://...?key=<the actual key>". That string is what
        # get_prediction() stores as this provider's "error" field, which is
        # displayed in the webapp and written into report.json/CSV/PDF
        # exports. base_client._redact_secrets() strips it there, but a URL
        # still carries the key through anything else that records URLs
        # (proxy logs, request tracing, a stack trace printed server-side).
        # Header auth removes the exposure at its source; redaction stays in
        # place as defense in depth.
        #
        # GEMINI_AUTH_MODE=query restores the old behavior as an escape hatch,
        # since header auth could NOT be verified against the live Google API
        # from this environment (no network access to it, and no key) -- see
        # README's "LLM integrations" section.
        import requests
        headers = {"Content-Type": "application/json"}
        url = f"{self.BASE_URL}/{self.model}:generateContent"
        if auth_mode == "query":
            url = f"{url}?key={self.api_key}"
        else:
            headers["x-goog-api-key"] = self.api_key
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        if resp.status_code != 200:
            # Keep the provider's error BODY (raise_for_status() discarded it, so a 404 for a retired
            # model looked like a generic failure). classify_http redacts and bounds it.
            from .providers import classify_http
            err = classify_http(resp.status_code, resp.text, "gemini")
            raise RuntimeError(f"Gemini {err.kind}: {err}")
        data = resp.json()
        cands = data.get("candidates") or []
        parts = ((cands[0].get("content") or {}).get("parts") or []) if cands else []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if not text.strip():
            raise RuntimeError("Gemini invalid_response: no text in the candidate "
                               f"(blockReason={(data.get('promptFeedback') or {}).get('blockReason')}).")
        return text
