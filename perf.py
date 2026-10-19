"""perf.py -- tiny in-process latency recorder. Only MEASURED values are ever stored or shown (nothing is pre-filled);
a metric with no samples reports `samples: 0`. Bounded memory (last 500 samples per metric). Thread-safe."""
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager

_lock = threading.Lock()
_data = defaultdict(lambda: deque(maxlen=500))
_counts = defaultdict(int)


def record(name: str, ms: float) -> None:
    with _lock:
        _data[name].append(float(ms))
        _counts[name] += 1


@contextmanager
def timed(name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        record(name, (time.perf_counter() - t0) * 1000.0)


def snapshot() -> dict:
    with _lock:
        out = {}
        for k, v in sorted(_data.items()):
            s = sorted(v)
            out[k] = {"samples": _counts[k], "window": len(s), "median_ms": round(s[len(s) // 2], 2),
                      "p95_ms": round(s[min(len(s) - 1, int(len(s) * 0.95))], 2), "max_ms": round(s[-1], 2),
                      "last_ms": round(v[-1], 2)}
        return out


def reset() -> None:
    with _lock:
        _data.clear(); _counts.clear()
