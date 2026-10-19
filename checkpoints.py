"""
checkpoints.py
----------------
Optional LSTM checkpointing: train once, save, and reuse across runs
instead of retraining every time. OFF by default -- the CLI/pipeline never
loads a checkpoint unless explicitly told to (a `--checkpoint NAME` flag),
since silently reusing a stale model would be its own correctness trap
(market data changes every trading day; a week-old checkpoint's "next day
prediction" is stale the moment a new day of data exists).

A checkpoint is a directory (checkpoints/<name>/) containing:
  - model.keras     the trained Keras model, in Keras's native format
  - scaler.json      the fitted MinMaxScaler's parameters (not part of the
                     Keras model itself, but required to interpret its
                     output), as plain JSON numbers -- see SECURITY note below
  - metadata.json    everything needed to verify compatibility before
                     reusing this checkpoint for a new run

SECURITY -- no pickle, anywhere in this file. A checkpoint's scaler is a
fitted sklearn MinMaxScaler, whose entire fitted state is a handful of
numbers (min/scale/data_min/data_max/data_range, plus a couple of scalar
settings) -- nothing about it requires arbitrary object deserialization, so
it's written and read as plain JSON (`_scaler_to_dict` / `_scaler_from_dict`
below) instead of `pickle.dump`/`pickle.load`. Unpickling executes arbitrary
code embedded in the file; a JSON number cannot. This closes what would
otherwise be an arbitrary-code-execution path if a checkpoint directory (or
the one file in it) were ever swapped out from under the app -- there is no
"trusted directory" argument needed for the scaler at all, because there is
no deserialization gadget available in JSON to exploit in the first place.
`model.keras` is still loaded via `tf.keras.models.load_model`, which Keras
3 runs in `safe_mode=True` by default (blocking arbitrary Lambda-layer code
execution) -- not overridden anywhere in this file.

COMPATIBILITY CHECKING -- the entire point of this file, safety-wise:
Loading a checkpoint trained for the WRONG ticker, a different lookback
window, or a different prediction target (return vs. price) would either
crash outright (a lookback mismatch changes the model's input shape) or,
worse, silently produce predictions that look plausible but are actually
meaningless (e.g. a model trained on one ticker's price scale, or on
returns, reused as if it were trained on another ticker's prices).
load_checkpoint() always compares the checkpoint's saved metadata against
the requested run's parameters first and refuses outright
(CheckpointMismatchError) on any mismatch that would make reusing it
invalid -- it never silently proceeds with an incompatible checkpoint.

The checkpoint directory itself is still an application-controlled trust
boundary: _checkpoint_dir() rejects any name containing a path separator or
"..", so a checkpoint name can never escape config.CHECKPOINT_DIR. NEVER
point CHECKPOINT_DIR at a location untrusted input can write to.
"""

import os
import json
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import numpy as np
from sklearn.preprocessing import MinMaxScaler

import config
import reproducibility


class CheckpointMismatchError(Exception):
    """Raised when a checkpoint's saved configuration doesn't match the requested run."""


class CheckpointNotFoundError(Exception):
    """Raised when a named checkpoint doesn't exist."""


# Every attribute a fitted MinMaxScaler needs for transform()/inverse_transform()
# to behave identically to the original -- and nothing else. All are plain
# numbers, short arrays of numbers, or a 2-tuple of numbers: fully JSON-safe.
_SCALER_ARRAY_ATTRS = ("scale_", "min_", "data_min_", "data_max_", "data_range_")
_SCALER_SCALAR_ATTRS = ("n_features_in_", "n_samples_seen_", "copy", "clip")


def _scaler_to_dict(scaler: MinMaxScaler) -> dict:
    """Serializes a FITTED MinMaxScaler's parameters to a plain JSON-safe dict."""
    missing = [a for a in _SCALER_ARRAY_ATTRS if not hasattr(scaler, a)]
    if missing:
        raise ValueError(
            f"Cannot save an unfitted scaler (missing: {', '.join(missing)}). "
            f"Fit the scaler before checkpointing."
        )
    d = {"feature_range": list(scaler.feature_range)}
    for attr in _SCALER_ARRAY_ATTRS:
        d[attr] = np.asarray(getattr(scaler, attr)).tolist()
    for attr in _SCALER_SCALAR_ATTRS:
        d[attr] = getattr(scaler, attr, None)
    return d


def _scaler_from_dict(d: dict) -> MinMaxScaler:
    """
    Reconstructs a MinMaxScaler from `_scaler_to_dict`'s output. This sets
    exactly the attributes `.fit()` itself would have set -- no different
    from what pickle would restore, just without pickle's ability to run
    arbitrary code to get there. A dict missing a required field raises
    cleanly (KeyError/ValueError) rather than producing a half-restored,
    silently-wrong scaler.
    """
    required = (*_SCALER_ARRAY_ATTRS, "feature_range")
    missing = [k for k in required if k not in d]
    if missing:
        raise ValueError(f"Scaler data is missing required field(s): {', '.join(missing)}.")
    scaler = MinMaxScaler(feature_range=tuple(d["feature_range"]))
    for attr in _SCALER_ARRAY_ATTRS:
        scaler.__dict__[attr] = np.asarray(d[attr], dtype=float)
    for attr in _SCALER_SCALAR_ATTRS:
        if attr in d and d[attr] is not None:
            scaler.__dict__[attr] = d[attr]
    return scaler


@dataclass
class CheckpointMetadata:
    ticker: str
    lookback: int
    units: int
    lstm_prediction_target: str
    train_test_split: float
    saved_at: str
    n_train_rows: int
    data_period: str
    data_source: str
    random_seed: int
    package_versions: dict
    # ---- V18 provenance manifest (manifest_version 2). Absent/None on checkpoints saved by older versions.
    manifest_version: int = 1
    model_type: str = "lstm"
    model_impl_version: str = None
    exchange: str = None
    forecast_horizon_days: int = 1
    feature_names: list = None
    feature_version: str = None
    train_start: str = None            # first date whose close entered the training data
    train_end: str = None              # LAST date whose close entered the training data
    train_fingerprint: str = None      # sha256 over (date, close) of every training row
    anchor_fingerprint: str = None     # sha256 over the last `anchor_rows` training rows
    anchor_rows: int = None
    scaler_config: dict = None
    training_config: dict = None
    model_sha256: str = None           # integrity: detects a swapped / truncated / corrupted file
    scaler_sha256: str = None

    def to_dict(self) -> dict:
        return asdict(self)


MODEL_IMPL_VERSION = "lstm-v18.1"      # bump when the architecture/training code changes incompatibly
FEATURE_VERSION = "lstm-univariate-v1"  # bump when feature construction changes
ANCHOR_ROWS = 60


class CheckpointProvenanceError(CheckpointMismatchError):
    """The checkpoint cannot PROVE it is safe to reuse for this evaluation (missing/contradicting provenance).
    Subclass of CheckpointMismatchError so callers that already fall back to retraining keep working."""


def _sha256_file(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_rows(df, start=None, end=None) -> str:
    """Stable hash of (date, close) for rows start..end inclusive (dates as 'YYYY-MM-DD'). Closes are rounded so
    harmless float noise does not change it, but a restated/dividend-adjusted history does."""
    import hashlib
    sub = df.loc[start:end] if (start is not None or end is not None) else df
    h = hashlib.sha256()
    for d, c in zip(sub.index, sub["Close"].to_numpy(dtype=float)):
        h.update(f"{str(d)[:10]}:{c:.6g}\n".encode())
    return h.hexdigest()


def training_provenance(df, train_end_date) -> dict:
    """Provenance for a model trained on rows of `df` up to and including `train_end_date`."""
    end = str(train_end_date)[:10]
    idx = [str(d)[:10] for d in df.index]
    if end not in idx:
        raise ValueError(f"train_end {end} is not a date in the supplied data")
    j = idx.index(end)
    lo = max(0, j - ANCHOR_ROWS + 1)
    return {"train_start": idx[0], "train_end": end, "train_fingerprint": fingerprint_rows(df, idx[0], end),
            "anchor_fingerprint": fingerprint_rows(df, idx[lo], end), "anchor_rows": j - lo + 1}


def _checkpoint_dir(name: str) -> str:
    if not name or any(c in name for c in "/\\.."):
        raise ValueError(f"Invalid checkpoint name: {name!r} (no path separators or '..').")
    return os.path.join(config.CHECKPOINT_DIR, name)


def save_checkpoint(name: str, lstm_model, scaler, *, ticker: str, period: str,
                     source: str, n_train_rows: int, seed: int = None,
                     target_mode: str = None, provenance: dict = None, exchange: str = None,
                     training_config: dict = None) -> str:
    """
    Saves `lstm_model` (an LSTMPredictor) + its scaler + metadata to
    checkpoints/<name>/, overwriting any existing checkpoint of that name.
    Returns the checkpoint directory path.

    `target_mode`: records what the model was ACTUALLY trained to predict
    ("return" or "price"). Defaults to config.LSTM_PREDICTION_TARGET for
    convenience, but callers that train with an explicit, different mode
    than the current global config MUST pass it explicitly here --
    otherwise the saved metadata would describe a different model than the
    one actually being checkpointed, and load_checkpoint()'s compatibility
    check (the entire safety point of this file) would be comparing against
    the wrong value.
    """
    seed = config.RANDOM_SEED if seed is None else seed
    target_mode = target_mode or config.LSTM_PREDICTION_TARGET
    final_dir = _checkpoint_dir(name)
    # Write to a sibling temp dir, then swap in: a crash mid-save can never leave a half-written checkpoint under
    # `name`, and only THIS name is ever touched (other checkpoints are never deleted or modified).
    ckpt_dir = f"{final_dir}.tmp-{os.getpid()}"
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    lstm_model.model.save(os.path.join(ckpt_dir, "model.keras"))
    with open(os.path.join(ckpt_dir, "scaler.json"), "w") as f:
        json.dump(_scaler_to_dict(scaler), f)

    metadata = CheckpointMetadata(
        ticker=ticker,
        lookback=lstm_model.lookback,
        units=lstm_model.units,
        lstm_prediction_target=target_mode,
        train_test_split=config.TRAIN_TEST_SPLIT,
        saved_at=datetime.now(timezone.utc).isoformat(),
        n_train_rows=n_train_rows,
        data_period=period,
        data_source=source,
        random_seed=seed,
        package_versions=reproducibility.get_package_versions(),
        manifest_version=2 if provenance else 1,
        model_type="lstm", model_impl_version=MODEL_IMPL_VERSION, exchange=exchange, forecast_horizon_days=1,
        feature_names=["Close_return" if target_mode == "return" else "Close"], feature_version=FEATURE_VERSION,
        scaler_config={"type": "MinMaxScaler", "n_features": int(getattr(scaler, "n_features_in_", 0) or 0),
                       "feature_range": list(getattr(scaler, "feature_range", (0, 1)))},
        training_config=training_config,
        model_sha256=_sha256_file(os.path.join(ckpt_dir, "model.keras")),
        scaler_sha256=_sha256_file(os.path.join(ckpt_dir, "scaler.json")),
        **(provenance or {}),
    )
    with open(os.path.join(ckpt_dir, "metadata.json"), "w") as f:
        json.dump(metadata.to_dict(), f, indent=2)

    shutil.rmtree(final_dir, ignore_errors=True)       # replaces ONLY this checkpoint's directory
    os.replace(ckpt_dir, final_dir)
    return final_dir


def checkpoint_exists(name: str) -> bool:
    try:
        ckpt_dir = _checkpoint_dir(name)
    except ValueError:
        return False
    return os.path.exists(os.path.join(ckpt_dir, "metadata.json"))


def load_checkpoint(name: str, *, ticker: str, lookback: int = None, target_mode: str = None,
                     train_test_split: float = None, require_provenance: bool = False,
                     train_frame=None, first_eval_date=None, exchange: str = None,
                     feature_names: list = None, horizon_days: int = None):
    """
    Loads a checkpoint by name, first verifying it's compatible with the
    requested `ticker` / `lookback` / `target_mode` / `train_test_split`.
    Any mismatch on these raises CheckpointMismatchError immediately -- the
    checkpoint is refused outright and never silently reused, since a
    mismatch here means either the model's input shape won't match
    (lookback), its output would mean something different than what's
    expected (ticker, target_mode), or the evaluation done against it would
    not be out-of-sample (train_test_split).

    On train_test_split specifically -- this check was missing and is a real
    data-leakage hole, not a cosmetic one. pipeline.py rebuilds the test
    sequences at the CURRENT config.TRAIN_TEST_SPLIT when it reuses a
    checkpoint, but the checkpoint's weights and scaler were fit against
    whatever split was in force when it was SAVED. If the current split is
    smaller than the saved one (e.g. saved at 0.8, re-run at 0.6), the
    rebuilt "test" window starts inside rows the checkpoint was actually
    trained on, so lstm_eval reports in-sample performance while presenting
    it as held-out. The metadata to catch this was already being saved --
    it just wasn't being compared. Now it is.

    Returns (LSTMPredictor_with_loaded_weights, scaler, metadata_dict).
    """
    ckpt_dir = _checkpoint_dir(name)
    metadata_path = os.path.join(ckpt_dir, "metadata.json")
    model_path = os.path.join(ckpt_dir, "model.keras")
    scaler_path = os.path.join(ckpt_dir, "scaler.json")

    if not os.path.exists(metadata_path):
        raise CheckpointNotFoundError(f"No checkpoint named '{name}' found at {ckpt_dir}.")

    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
        if not isinstance(metadata, dict):
            raise ValueError("metadata is not an object")
        for k in ("ticker", "lookback", "lstm_prediction_target", "units"):
            if k not in metadata:
                raise ValueError(f"metadata is missing '{k}'")
    except (ValueError, OSError) as e:
        raise CheckpointMismatchError(f"Checkpoint '{name}' has unreadable/corrupt metadata ({type(e).__name__}); "
                                       f"it will NOT be used.") from e

    target_mode = target_mode or config.LSTM_PREDICTION_TARGET
    lookback = lookback or config.LOOKBACK_WINDOW
    train_test_split = (config.TRAIN_TEST_SPLIT if train_test_split is None
                         else train_test_split)

    mismatches = []
    saved_split = metadata.get("train_test_split")
    if saved_split is None:
        mismatches.append(
            "train_test_split: this checkpoint predates the train/test-split "
            "compatibility check and doesn't record which split it was trained "
            "under, so it cannot be verified as out-of-sample for this run"
        )
    elif abs(float(saved_split) - float(train_test_split)) > 1e-9:
        mismatches.append(
            f"train_test_split: checkpoint was trained with {saved_split}, this run "
            f"uses {train_test_split} (the test window would no longer be fully "
            f"out-of-sample for the saved weights -- reusing it would report "
            f"in-sample performance as held-out)"
        )
    if metadata["ticker"] != ticker:
        mismatches.append(f"ticker: checkpoint is for '{metadata['ticker']}', requested '{ticker}'")
    if metadata["lookback"] != lookback:
        mismatches.append(f"lookback: checkpoint used {metadata['lookback']}, requested {lookback} "
                           f"(this changes the model's input shape -- cannot be reused)")
    if metadata["lstm_prediction_target"] != target_mode:
        mismatches.append(f"lstm_prediction_target: checkpoint was trained for "
                           f"'{metadata['lstm_prediction_target']}', requested '{target_mode}' "
                           f"(this changes what the model's output represents)")

    if mismatches:
        raise CheckpointMismatchError(
            f"Checkpoint '{name}' is not compatible with this run and will NOT be used:\n  "
            + "\n  ".join(mismatches)
        )

    # ---- V18 provenance / integrity -------------------------------------------------------------
    prov = validate_provenance(metadata, require=require_provenance, train_frame=train_frame,
                               first_eval_date=first_eval_date, exchange=exchange, feature_names=feature_names,
                               horizon_days=horizon_days)
    if prov:
        raise CheckpointProvenanceError(
            f"Checkpoint '{name}' cannot be proven safe to reuse and will NOT be used:\n  " + "\n  ".join(prov))
    # Integrity hashes are VERIFIED BEFORE THE MODEL IS RETURNED, but evaluated after the semantic checks below so
    # that a swapped file is reported with the most specific diagnosis (e.g. the wrong input shape) first.
    integrity_problems = []
    for fname, key in (("model.keras", "model_sha256"), ("scaler.json", "scaler_sha256")):
        want = metadata.get(key)
        path = os.path.join(ckpt_dir, fname)
        if want and (not os.path.exists(path) or _sha256_file(path) != want):
            integrity_problems.append(f"{fname} does not match its recorded hash (corrupted, truncated or swapped)")

    import tensorflow as tf
    from models.lstm_model import LSTMPredictor

    try:
        # safe_mode=True is Keras's default; it is stated explicitly so a library change can never silently re-enable
        # deserialising arbitrary Python (e.g. a Lambda layer) from a model file. A checkpoint containing one is refused.
        keras_model = tf.keras.models.load_model(model_path, safe_mode=True)
    except Exception as e:  # noqa: BLE001 - any failure to deserialize means the artifact is unusable
        raise CheckpointMismatchError(f"Checkpoint '{name}': model file could not be loaded "
                                       f"({type(e).__name__}); it will NOT be used.") from e
    if not os.path.exists(scaler_path):
        raise CheckpointNotFoundError(
            f"Checkpoint '{name}' is missing its scaler.json file at {scaler_path}. "
            f"(Checkpoints saved by older versions of this project used a "
            f"pickle-based scaler.pkl instead; those are no longer accepted -- "
            f"retrain and re-save this checkpoint.)"
        )
    try:
        with open(scaler_path) as f:
            scaler = _scaler_from_dict(json.load(f))
    except (ValueError, KeyError, TypeError, OSError) as e:
        raise CheckpointMismatchError(f"Checkpoint '{name}': scaler.json is corrupt ({type(e).__name__}); "
                                       f"it will NOT be used.") from e

    # The checks above compare METADATA against the run. Metadata is a plain
    # JSON file sitting next to the weights -- it can be hand-edited, or go
    # out of sync if model.keras is swapped for a different one. So verify
    # the loaded model ITSELF, not just what the sidecar file claims:
    #   - input shape must be (None, lookback, 1)  -- wrong sequence length
    #     would otherwise fail deep inside Keras with a shape error, or
    #     silently accept a differently-shaped window
    #   - output shape must be (None, 1)            -- a model with a
    #     different head is predicting something other than one next value
    arch_problems = []
    try:
        input_shape = tuple(keras_model.input_shape)
        output_shape = tuple(keras_model.output_shape)
    except (AttributeError, TypeError) as e:
        raise CheckpointMismatchError(
            f"Checkpoint '{name}' contains a model whose input/output shape could not "
            f"be inspected ({type(e).__name__}), so it cannot be verified as compatible."
        ) from e

    expected_input = (None, metadata["lookback"], 1)
    if input_shape != expected_input:
        arch_problems.append(
            f"model input shape is {input_shape}, expected {expected_input} "
            f"(the saved weights do not match the lookback this checkpoint claims)"
        )
    if output_shape != (None, 1):
        arch_problems.append(
            f"model output shape is {output_shape}, expected (None, 1) -- this model "
            f"does not predict a single next value"
        )
    if arch_problems:
        raise CheckpointMismatchError(
            f"Checkpoint '{name}' has weights that disagree with its own metadata and "
            f"will NOT be used:\n  " + "\n  ".join(arch_problems)
        )

    # The scaler is equally load-bearing: a MinMaxScaler that was never fitted,
    # or one fitted on multi-feature data, would silently produce meaningless
    # inverse-transforms rather than failing.
    n_scaler_features = getattr(scaler, "n_features_in_", None)
    if n_scaler_features is None:
        raise CheckpointMismatchError(
            f"Checkpoint '{name}' contains a scaler that was never fitted "
            f"(no n_features_in_), so its outputs could not be interpreted."
        )
    if n_scaler_features != 1:
        raise CheckpointMismatchError(
            f"Checkpoint '{name}' contains a scaler fitted on {n_scaler_features} "
            f"features, but this pipeline scales a single univariate series."
        )

    if integrity_problems:
        raise CheckpointMismatchError(f"Checkpoint '{name}' failed its integrity check and will NOT be used: "
                                       + "; ".join(integrity_problems))

    predictor = LSTMPredictor(lookback=metadata["lookback"], units=metadata["units"])
    predictor.model = keras_model
    predictor.is_fitted = True

    return predictor, scaler, metadata


# Libraries whose MAJOR version must match between saving and loading: a different major can change the serialisation
# format or numerics. (Minor/patch differences are accepted; a model file that cannot be deserialised is rejected anyway.)
COMPAT_MAJOR_PACKAGES = ("tensorflow", "keras", "numpy")


def _major(v):
    try:
        return int(str(v).split(".")[0])
    except (ValueError, TypeError):
        return None


def validate_provenance(metadata: dict, *, require: bool, train_frame=None, first_eval_date=None,
                        exchange=None, feature_names=None, horizon_days=None, current_versions: dict = None) -> list:
    """Returns a list of problems (empty = acceptable). Pure function: no file access, easy to test.

    A checkpoint is reusable for an evaluation whose FIRST evaluated target date is `first_eval_date` only if
    every observation it was trained on is strictly EARLIER than that date -- otherwise the 'out-of-sample'
    evaluation would partly be in-sample (or, in a historical backtest, the model would have seen the simulated
    future)."""
    problems = []
    has = metadata.get("manifest_version", 1) >= 2 and metadata.get("train_end")
    if not has:
        if require:
            problems.append("no provenance manifest (saved by a version that did not record training dates). "
                            "Documented migration path: retrain; the new checkpoint overwrites this one only.")
        return problems
    if metadata.get("model_type", "lstm") != "lstm":
        problems.append(f"model_type is '{metadata.get('model_type')}', expected 'lstm'")
    if metadata.get("model_impl_version") != MODEL_IMPL_VERSION:
        problems.append(f"model implementation version '{metadata.get('model_impl_version')}' != current "
                        f"'{MODEL_IMPL_VERSION}' (architecture/training code changed)")
    if metadata.get("feature_version") != FEATURE_VERSION:
        problems.append(f"feature-engineering version '{metadata.get('feature_version')}' != current '{FEATURE_VERSION}'")
    if feature_names is not None and list(metadata.get("feature_names") or []) != list(feature_names):
        problems.append(f"feature schema {metadata.get('feature_names')} != requested {list(feature_names)}")
    if horizon_days is not None and metadata.get("forecast_horizon_days") != horizon_days:
        problems.append(f"forecast horizon {metadata.get('forecast_horizon_days')}d != requested {horizon_days}d")
    if exchange is not None and metadata.get("exchange") not in (None, exchange):
        problems.append(f"exchange '{metadata.get('exchange')}' != requested '{exchange}'")
    saved = metadata.get("package_versions") or {}
    if current_versions is None:
        import reproducibility
        current_versions = reproducibility.get_package_versions()
    if not saved:
        problems.append("manifest has no dependency versions, so library compatibility cannot be verified")
    for pkg in COMPAT_MAJOR_PACKAGES:
        a, b = _major(saved.get(pkg)), _major(current_versions.get(pkg))
        if a is not None and b is not None and a != b:
            problems.append(f"saved with {pkg} {saved.get(pkg)} but the current {pkg} is {current_versions.get(pkg)} "
                            f"(different major version)")
    sc = metadata.get("scaler_config") or {}
    if sc.get("n_features") not in (None, 0, 1):
        problems.append(f"scaler was fitted on {sc.get('n_features')} features; a single feature is expected")
    te = str(metadata["train_end"])[:10]
    if first_eval_date is not None and not te < str(first_eval_date)[:10]:
        problems.append(f"training data extends to {te}, which is NOT before the first evaluated date "
                        f"{str(first_eval_date)[:10]}: the model has seen the evaluation period "
                        f"(future-contaminated for this evaluation)")
    if train_frame is not None:
        idx = [str(d)[:10] for d in train_frame.index]
        if te not in idx:
            problems.append(f"training end date {te} is not present in the current data, so the training "
                            f"history cannot be verified")
        else:
            j = idx.index(te)
            n = int(metadata.get("anchor_rows") or 0)
            if n and metadata.get("anchor_fingerprint"):
                lo = j - n + 1
                if lo < 0 or fingerprint_rows(train_frame, idx[lo], te) != metadata["anchor_fingerprint"]:
                    problems.append("the last training rows no longer match the data the model was trained on "
                                    "(history was restated, e.g. a dividend/split adjustment, or the symbol differs)")
            else:
                problems.append("manifest has no anchor fingerprint, so the training history cannot be verified")
            ts = str(metadata.get("train_start", ""))[:10]
            if metadata.get("train_fingerprint") and ts == idx[0] \
                    and fingerprint_rows(train_frame, ts, te) != metadata["train_fingerprint"]:
                problems.append("full training-range fingerprint differs from the current data")
    return problems


def list_checkpoints() -> list:
    """Returns every saved checkpoint's metadata (plus its name), newest first."""
    base = config.CHECKPOINT_DIR
    if not os.path.isdir(base):
        return []
    results = []
    for name in os.listdir(base):
        meta_path = os.path.join(base, name, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            meta["name"] = name
            results.append(meta)
    results.sort(key=lambda m: m.get("saved_at", ""), reverse=True)
    return results


def delete_checkpoint(name: str) -> bool:
    ckpt_dir = _checkpoint_dir(name)
    if os.path.isdir(ckpt_dir):
        shutil.rmtree(ckpt_dir)
        return True
    return False
