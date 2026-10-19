"""
logging_utils.py
-----------------
Structured, redaction-safe logging for the server-side paths.

The project previously logged with bare `print()`. That is fine for the CLI
(where the "log" IS the user-facing output, deliberately) but poor for the
Flask app and the network layers: no levels, no timestamps, no module
attribution, no way to raise or lower verbosity, and -- importantly -- no
chokepoint where a secret could be stripped before it hits a log file.

WHAT THIS DOES AND DOES NOT CHANGE:
  - The CLI's own `print()` reporting in main.py / trade.py / pipeline.py is
    left alone. It is the program's OUTPUT, not diagnostics, and rewriting it
    as logging would change what users see for no correctness benefit.
  - Diagnostics in the webapp and the network layers go through here.

Every record is passed through security.redact() by a filter, so a secret
cannot be logged even by a future call site that forgets to redact.
LOG_LEVEL in .env controls verbosity (default INFO).
"""

import logging
import os
import sys

import security

_CONFIGURED = False


class _RedactionFilter(logging.Filter):
    """
    Redacts the formatted message and every positional arg. Implemented as a
    filter rather than a formatter so it applies to EVERY handler attached to
    these loggers, including any a deployment adds later (file, syslog, an
    aggregator) -- a formatter would only protect the handler it is set on.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = security.redact(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: security.redact(v) if isinstance(v, str) else v
                               for k, v in record.args.items()}
            else:
                record.args = tuple(
                    security.redact(a) if isinstance(a, str) else a for a in record.args
                )
        if record.exc_info:
            # Keep the traceback for developers, but redact the exception's
            # own message, which is where a URL or header would appear.
            exc_type, exc_value, exc_tb = record.exc_info
            if exc_value is not None:
                try:
                    exc_value.args = tuple(
                        security.redact(a) if isinstance(a, str) else a
                        for a in exc_value.args
                    )
                except (AttributeError, TypeError):
                    pass
        return True


def configure(level: str = None) -> None:
    """
    Idempotent root configuration. Logs to stderr so stdout stays clean for
    the CLI's own report output and for anything piping it.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    ))
    handler.addFilter(_RedactionFilter())
    root = logging.getLogger("stockpred")
    root.setLevel(getattr(logging, level_name, logging.INFO))
    root.handlers = [handler]
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """A namespaced logger with redaction guaranteed by the root handler."""
    configure()
    logger = logging.getLogger(f"stockpred.{name}")
    logger.addFilter(_RedactionFilter())
    return logger
