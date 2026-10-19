"""
security.py
------------
One place that knows every secret this project holds, and one place that
decides what an upstream error is allowed to say.

WHY THIS MODULE EXISTS (a real, verified leak, not a hypothetical):
`ai_integrations/base_client.py` already redacted the five LLM API keys out
of provider error messages. Nothing did the same for BROKER and DATA-SOURCE
credentials, and those paths embed raw upstream response bodies verbatim:

  - `data_sources/groww_source.get_access_token()` raised
    "Groww token response missing 'token': {resp.text}". That body is the
    response from Groww's TOKEN endpoint -- the single most likely place in
    the whole project for a live token to appear. Confirmed directly: a
    token returned under an unexpected key (e.g. "accessToken" rather than
    "token") was echoed in full into the exception message.
  - `data_sources/upstox_source._interactive_login()` did the same with
    Upstox's OAuth token-exchange response.
  - Every non-200 path in `trading/brokers/*.py` put `resp.text` straight
    into the returned message. That message is written to
    `outputs/trade_log.csv` and rendered in the web UI. Confirmed directly:
    an upstream body echoing an Authorization header reached both.

So: redaction now covers ALL configured secrets, not just LLM keys, and
upstream response bodies are truncated as well as redacted before they can
reach a log, an audit file, or a browser.

DESIGN NOTE -- redaction is the LAST line of defence, not the first. The
first is not putting secrets somewhere they can leak (see the Gemini client
switching from a `?key=` URL to a header). This module exists to contain the
blast radius of code paths that haven't done that, including future ones.
"""

import re

import config

# Anything shorter than this is not redacted. Redacting a 3-character
# placeholder like "x" or "dev" would corrupt unrelated text far more often
# than it would protect anything real.
MIN_REDACTABLE_LENGTH = 8

REDACTED = "[REDACTED]"

# How much of an upstream response body may ever be quoted back. Enough to
# diagnose ("invalid instrument key", "rate limit exceeded"), short enough
# that a large JSON blob full of account data can't be dumped into an audit
# log or a browser.
MAX_UPSTREAM_DETAIL_CHARS = 300

# Patterns that look like credentials regardless of whether they match a
# value this process happens to have configured -- covers secrets belonging
# to the UPSTREAM service, other tenants, or a key the user set in an env
# var this project doesn't read.
_CREDENTIAL_PATTERNS = (
    re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)\b(authorization\"?\s*[:=]\s*\"?)(?:bearer\s+)?[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)\b((?:access[_-]?|api[_-]?|secret[_-]?|private[_-]?|client[_-]?)"
                r"(?:token|key|secret)\"?\s*[:=]\s*\"?)[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)([?&](?:key|token|access_token|api_key|client_secret)=)[^&\s\"']+"),
    re.compile(r"\b(sk-(?:ant-)?)[A-Za-z0-9_-]{12,}"),
    re.compile(r"\b(AIza)[A-Za-z0-9_-]{20,}"),
    re.compile(r"\b(xai-)[A-Za-z0-9]{12,}"),
)


def configured_secrets() -> list:
    """
    Every secret value this process currently holds. Read from `config` on
    each call rather than captured at import time, because the test suite
    (and anyone using this interactively) reassigns `config` attributes at
    runtime -- a snapshot taken at import would silently stop redacting the
    value that is actually in use.
    """
    candidates = (
        # LLM providers
        config.ANTHROPIC_API_KEY, config.OPENAI_API_KEY, config.GOOGLE_API_KEY,
        config.XAI_API_KEY, config.DEEPSEEK_API_KEY,
        getattr(config, "GROQ_API_KEY", ""), getattr(config, "OPENROUTER_API_KEY", ""),
        # Brokers / market data
        config.UPSTOX_CLIENT_ID, config.UPSTOX_CLIENT_SECRET,
        config.GROWW_ACCESS_TOKEN, config.GROWW_API_KEY, config.GROWW_API_SECRET,
        # Web app
        config.WEBAPP_LIVE_TRADING_TOKEN,
        getattr(config, "WEBAPP_SECRET_KEY", "") or "",
    )
    return [s for s in candidates if s and len(str(s)) >= MIN_REDACTABLE_LENGTH]


def redact(text) -> str:
    """
    Removes every configured secret from `text`, then scrubs anything that
    merely LOOKS like a credential (bearer tokens, `?key=` query params,
    provider-specific key prefixes). Safe to call on any string, including
    None or a non-string, which is returned unchanged/stringified.

    Longest-first ordering matters: if one secret is a substring of another
    (a client id embedded in a longer composite value, say), redacting the
    short one first would leave a mangled fragment of the long one behind.
    """
    if not text:
        return text
    text = str(text)

    for secret in sorted(configured_secrets(), key=len, reverse=True):
        text = text.replace(str(secret), REDACTED)

    for pattern in _CREDENTIAL_PATTERNS:
        text = pattern.sub(lambda m: f"{m.group(1)}{REDACTED}", text)

    return text


def safe_upstream_detail(body, status_code=None) -> str:
    """
    Turns an upstream HTTP response body into something safe to put in an
    exception message, an audit log, or a web page: redacted, collapsed to a
    single line, and truncated.

    Never returns the empty string -- an error with no detail at all is
    harder to act on than one that says "(no response body)".
    """
    detail = redact(body)
    if detail:
        detail = " ".join(str(detail).split())
    if not detail:
        detail = "(no response body)"
    if len(detail) > MAX_UPSTREAM_DETAIL_CHARS:
        detail = detail[:MAX_UPSTREAM_DETAIL_CHARS] + "... (truncated)"
    return f"HTTP {status_code}: {detail}" if status_code is not None else detail


def safe_exception_text(exc) -> str:
    """
    Redacted string form of an exception. Use this anywhere an exception
    from a network client could carry a URL, a header, or a request body --
    `requests`' exception reprs routinely include the full request URL.
    """
    return redact(f"{type(exc).__name__}: {exc}")
