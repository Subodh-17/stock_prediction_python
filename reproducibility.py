"""
reproducibility.py
--------------------
Best-effort determinism for the ML pipeline: seeds Python's `random`, NumPy,
and TensorFlow/Keras so repeated runs on the same data and config produce
the same (or near-identical) model weights and predictions.

HONEST LIMITS -- this does NOT guarantee bit-for-bit determinism in every
environment:
  - On GPU, several cuDNN/CUDA ops (e.g. some LSTM/RNN kernels, reductions)
    are non-deterministic by default for performance reasons. Setting
    TF_DETERMINISTIC_OPS=1 makes TensorFlow prefer deterministic kernel
    implementations where available, but can be slower and some ops may
    still not have a deterministic GPU implementation depending on the
    installed TensorFlow/CUDA/cuDNN version.
  - On CPU, floating-point reduction order can still vary slightly with the
    number of threads used (e.g. multi-threaded BLAS), which can produce
    tiny (usually immaterial) floating-point differences.
  - Any code path that also draws from an unseeded external source (a
    different random number generator, network timing affecting async LLM
    results, etc.) will still vary between runs.

In short: seeding here makes runs *reproducible in practice* for the LR and
LSTM models on CPU (the default environment for this project, given
tensorflow-cpu in requirements.txt), not a formal determinism guarantee.
"""

import os
import random
import platform
from datetime import datetime, timezone

import numpy as np

import config


def set_global_seed(seed: int = 42, deterministic_ops: bool = True) -> None:
    """
    Seeds Python's random module, NumPy, and TensorFlow (if importable).

    Call this once, as early as possible in a run (before building/training
    any model), so every source of randomness (weight init, dropout masks,
    train/validation shuffling inside model.fit, etc.) starts from the same
    state.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if deterministic_ops:
        # Ask TensorFlow to prefer deterministic kernel implementations
        # where available. Must be set before TensorFlow is imported/used
        # for full effect, so we set it unconditionally here even though
        # importing tensorflow below will "lock in" the process either way.
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
        os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass  # TensorFlow not installed in this environment -- nothing to seed


def get_package_versions() -> dict:
    """
    Returns the installed version of every library whose behavior could
    meaningfully affect reproducing a run's results (model training
    libraries, not e.g. Flask). Never includes anything secret -- this is
    purely package name -> version string.
    """
    versions = {"python": platform.python_version()}
    for pkg in ("numpy", "pandas", "scikit-learn", "tensorflow", "keras", "yfinance"):
        import_name = {"scikit-learn": "sklearn"}.get(pkg, pkg)
        try:
            mod = __import__(import_name)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not installed"
    return versions


def capture_run_metadata(*, ticker: str, source: str, period: str, seed: int,
                          lstm_target_mode: str, lstm_epochs: int, lookback: int,
                          train_test_split: float, ensemble_weighting_mode: str,
                          n_rows_total: int, n_train_rows: int = None,
                          n_test_rows: int = None) -> dict:
    """
    Captures everything needed to describe (not perfectly replay, given the
    documented nondeterminism limits above, and given market data itself
    can be revised/adjusted after the fact by the data provider) how a
    given run's results were produced: what ticker/source/period was
    fetched, when, with what random seed and model configuration, and what
    software versions were involved.

    Deliberately excludes API keys, broker credentials, and any other
    secret -- this is metadata about the RUN, not about how to authenticate
    as the person who ran it.
    """
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "data_source": source,
        "period": period,
        "random_seed": seed,
        "model_config": {
            "lstm_prediction_target": lstm_target_mode,
            "lstm_epochs": lstm_epochs,
            "lstm_units": config.LSTM_UNITS,
            "lstm_batch_size": config.LSTM_BATCH_SIZE,
            "lookback_window": lookback,
            "train_test_split": train_test_split,
            "ensemble_weighting_mode": ensemble_weighting_mode,
        },
        "data_shape": {
            "n_rows_total": n_rows_total,
            "n_train_rows": n_train_rows,
            "n_test_rows": n_test_rows,
        },
        "package_versions": get_package_versions(),
    }
