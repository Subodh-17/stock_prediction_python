"""
pipeline.py
------------
The core "fetch data -> train LR -> train LSTM -> query LLM panel -> blend
ensemble" pipeline, extracted into one reusable function so both `main.py`
(reporting/charts) and `trade.py` (buy/sell decisions) call the exact same
logic rather than maintaining two copies of it.
"""

from datetime import datetime, timezone

import config
import data_loader
import reproducibility
import perf
from models.linear_regression_model import LinearRegressionPredictor
from models.lstm_model import LSTMPredictor
from ai_integrations.panel import query_llm_panel
import ensemble
import validation as validation_mod
import confidence as confidence_mod
import checkpoints as checkpoints_mod


def run_pipeline(ticker: str, period: str = None, epochs: int = None,
                  source: str = None, skip_llm: bool = False, verbose: bool = True,
                  walk_forward: bool = False, ensemble_weighting_mode: str = None,
                  seed: int = None, checkpoint_name: str = None, retrain: bool = False,
                  for_backtest: bool = False, df=None) -> dict:
    """
    Runs the full prediction pipeline for `ticker` and returns everything
    downstream consumers need:

        df, current_price,
        lr_model, lr_eval, lr_next_price, lr_dates_test, y_test,
        lstm_model, lstm_eval, lstm_next_price, lstm_dates_test, scaler,
        llm_results, llm_panel_summary,
        final_ensemble
        (and, when requested: walk_forward_lr, walk_forward_lstm)

    `walk_forward`: also run expanding-window walk-forward validation for
    both models (Phase 2 requirement). This is additional to, and does not
    replace, the single train/test split evaluation above -- it costs extra
    runtime (retrains the LSTM a few more times), so it's opt-in.

    `ensemble_weighting_mode`: "fixed" (default, from config.ENSEMBLE_WEIGHTS)
    or "dynamic" (weights derived from walk-forward validation performance,
    see ensemble.compute_dynamic_weights). Defaults to
    config.ENSEMBLE_WEIGHTING_MODE. "dynamic" mode runs walk-forward
    validation internally even if `walk_forward=False`, since it needs that
    data to compute weights -- but uses cheaper defaults (fewer LSTM folds)
    than an explicit `walk_forward=True` request.

    `checkpoint_name`: if given, the LSTM is loaded from (and/or saved to)
    checkpoints/<checkpoint_name>/ instead of always training fresh --
    "train once, reuse many times". Behavior:
      - No checkpoint of this name exists yet, or `retrain=True`: train the
        LSTM fresh as usual, then SAVE it under this name for next time.
      - A checkpoint of this name exists, is compatible with `ticker` and
        the current LSTM config (lookback, prediction target), and
        `retrain=False` (default): LOAD its weights instead of training --
        saves the training time entirely. The LSTM is still EVALUATED
        against the current run's own fresh test split (never against
        stale cached numbers), so lstm_eval always reflects genuine
        current-data performance regardless of whether training happened
        this run or not.
      - A checkpoint of this name exists but is NOT compatible (different
        ticker, lookback, or prediction target): this is never silently
        used. The mismatch is logged clearly and the LSTM trains fresh
        (then overwrites the checkpoint with the new, compatible one) --
        see checkpoints.py's module docstring for why this matters.

    `for_backtest`: hints that the caller intends to run an ensemble
    backtest against this result afterward (backtest.backtest_from_pipeline_result).
    Only affects behavior when combined with dynamic weighting: a
    genuinely historically-valid, day-by-day dynamic-weight backtest (see
    backtest.py's _compute_historical_dynamic_weights) needs walk-forward
    folds that conclude BEFORE the backtest's test period even starts --
    the "cheap" dynamic-weights-only path normally requests just enough
    LSTM folds to clear the live-prediction safety floor (no spare
    "buffer"), which leaves no fold old enough to precede the backtest
    window at all. Setting this requests a few extra folds so a real
    historical backtest actually has a chance to use genuine per-day
    weights instead of falling back to fixed for every single day.
    """
    period = period or config.PERIOD
    epochs = epochs or config.LSTM_EPOCHS
    source = source or config.DATA_SOURCE
    ensemble_weighting_mode = ensemble_weighting_mode or config.ENSEMBLE_WEIGHTING_MODE
    seed = config.RANDOM_SEED if seed is None else seed

    reproducibility.set_global_seed(seed)

    def log(msg):
        if verbose:
            print(msg)

    if df is None:
        log(f"[1/5] Fetching historical data (source={source})...")
        df = data_loader.fetch_stock_data(ticker=ticker, period=period, source=source)
        resolved_ticker = df.attrs.get("resolved_ticker", ticker)
        dropped_open_bar = df.attrs.get("dropped_open_session_bar")
    else:   # V17: caller supplied already-validated bars (explicit exchange, partial bar removed)
        resolved_ticker = ticker
        dropped_open_bar = None
        log("[1/5] Using caller-supplied historical data.")
    data_loader.validate_sufficient_data(df)
    current_price = float(df["Close"].iloc[-1])
    log(f"      Loaded {len(df)} rows. Current price: {current_price:.2f}")

    quality_report = data_loader.check_data_quality(df, ticker=ticker)
    if quality_report.issues:
        log(f"      DATA QUALITY ISSUES found (proceeding, but treat results with extra "
            f"caution): {'; '.join(quality_report.issues)}")
    if quality_report.warnings:
        log(f"      Data quality notes: {'; '.join(quality_report.warnings)}")

    # --- Linear Regression ---
    log("\n[2/5] Training Linear Regression model...")
    X_train, X_test, y_train, y_test, feature_names, lr_dates_test = \
        data_loader.prepare_linear_regression_data(df)
    with perf.timed("pipeline.lr_train_eval"):
        lr_model = LinearRegressionPredictor().train(X_train, y_train, feature_names)
        lr_eval = lr_model.evaluate(X_test, y_test)
    log(f"      RMSE: {lr_eval['rmse']:.3f}  MAE: {lr_eval['mae']:.3f}  "
        f"R2: {lr_eval['r2']:.3f}  MAPE: {lr_eval['mape']:.2f}%  "
        f"Directional accuracy: {lr_eval['accuracy_pct']:.1f}%")
    # NOTE: predict_next uses a freshly-built feature row for the ACTUAL
    # most recent day (today), not X_test[-1] -- see
    # data_loader.build_latest_feature_row's docstring for why X_test[-1]
    # would silently predict a day that's already known and is blind to
    # today's real data (a real bug, confirmed and fixed).
    lr_next_price = lr_model.predict_next(data_loader.build_latest_feature_row(df))

    # --- LSTM ---
    log(f"\n[3/5] Preparing LSTM model ({epochs} epochs if training)...")
    lstm_target_mode = config.LSTM_PREDICTION_TARGET
    lstm_X_train, lstm_X_test, lstm_y_train, lstm_y_test, scaler, lstm_dates_test, lstm_base_prices_test = \
        data_loader.prepare_lstm_training_data(df, target_mode=lstm_target_mode)  # timed below as feature engineering

    # V18: provenance of the model this run would TRAIN, and the first date it will be EVALUATED on. A reused
    # checkpoint must have been trained strictly before that first evaluated date.
    split_rows = int((len(df) - (1 if lstm_target_mode == "return" else 0)) * config.TRAIN_TEST_SPLIT)
    last_train_pos = split_rows if lstm_target_mode == "return" else split_rows - 1
    last_train_pos = max(0, min(last_train_pos, len(df) - 1))
    train_end_date = str(df.index[last_train_pos])[:10]
    first_eval_date = str(lstm_dates_test[0])[:10] if len(lstm_dates_test) else None
    try:
        from market_data import resolve_instrument
        _exchange = resolve_instrument(ticker).exchange
    except Exception:  # noqa: BLE001
        _exchange = None
    _features = ["Close_return" if lstm_target_mode == "return" else "Close"]

    lstm_model = None
    lstm_loaded_from_checkpoint, ckpt_meta = False, {}
    if checkpoint_name and not retrain and checkpoints_mod.checkpoint_exists(checkpoint_name):
        try:
            _t_load = __import__("time").perf_counter()
            lstm_model, scaler, ckpt_meta = checkpoints_mod.load_checkpoint(
                checkpoint_name, ticker=ticker, lookback=config.LOOKBACK_WINDOW,
                target_mode=lstm_target_mode,
                train_test_split=config.TRAIN_TEST_SPLIT,
                require_provenance=True, train_frame=df, first_eval_date=first_eval_date,
                exchange=_exchange, feature_names=_features, horizon_days=1,
            )
            lstm_loaded_from_checkpoint = True
            perf.record("pipeline.lstm_checkpoint_load", (__import__("time").perf_counter() - _t_load) * 1000)
            log(f"      Loaded LSTM checkpoint '{checkpoint_name}' "
                f"(saved {ckpt_meta['saved_at']}, trained on {ckpt_meta['n_train_rows']} rows) "
                f"-- skipping training.")
            try:
                saved_at = datetime.fromisoformat(ckpt_meta["saved_at"])
                age_days = (datetime.now(timezone.utc) - saved_at).total_seconds() / 86400
                if age_days >= config.CHECKPOINT_STALE_WARNING_DAYS:
                    log(f"      NOTE: this checkpoint is {age_days:.0f} day(s) old. Market "
                        f"conditions/data since then aren't reflected in the loaded weights -- "
                        f"use --retrain-checkpoint if you want a fresh fit on current data.")
            except (ValueError, KeyError):
                pass  # saved_at missing/malformed -- not worth failing the run over an age warning
            # The checkpoint's scaler was fit on ITS OWN training data, not
            # necessarily this run's -- rebuild this run's test sequences
            # using that scaler (transform only, never re-fit), following
            # the exact same train/test boundary and windowing logic as a
            # freshly-trained run, so lstm_eval stays a genuine, leakage-
            # safe, apples-to-apples evaluation against CURRENT data rather
            # than stale numbers from whenever the checkpoint was saved.
            close = df["Close"].values
            if lstm_target_mode == "return":
                raw_series = ((close[1:] - close[:-1]) / close[:-1]).reshape(-1, 1)
            else:
                raw_series = close.reshape(-1, 1)
            split_idx = int(len(raw_series) * config.TRAIN_TEST_SPLIT)
            lb = ckpt_meta["lookback"]
            train_scaled = scaler.transform(raw_series[:split_idx])
            test_scaled = scaler.transform(raw_series[split_idx:])
            _, _, lstm_X_test, lstm_y_test = data_loader.window_scaled_series(
                train_scaled, test_scaled, lb
            )
            lstm_X_test = lstm_X_test.reshape((lstm_X_test.shape[0], lb, 1))
        except checkpoints_mod.CheckpointMismatchError as e:
            log(f"      Checkpoint '{checkpoint_name}' exists but is NOT compatible with this "
                f"run -- training fresh instead (checkpoint will be overwritten):\n      {e}")
            lstm_model = None

    if lstm_model is None:
        log(f"      Training LSTM model ({epochs} epochs)...")
        lstm_model = LSTMPredictor()
        with perf.timed("pipeline.lstm_train"):
            lstm_model.train(lstm_X_train, lstm_y_train, epochs=epochs, verbose=0)
        if checkpoint_name:
            checkpoints_mod.save_checkpoint(
                checkpoint_name, lstm_model, scaler, ticker=ticker, period=period,
                source=source, n_train_rows=len(lstm_X_train), seed=seed,
                target_mode=lstm_target_mode, exchange=_exchange,
                provenance=checkpoints_mod.training_provenance(df, train_end_date),
                training_config={"epochs": epochs, "units": lstm_model.units, "lookback": lstm_model.lookback},
            )
            log(f"      Saved LSTM checkpoint as '{checkpoint_name}'.")

    lstm_checkpoint_info = {"name": checkpoint_name, "reused": lstm_loaded_from_checkpoint,
                            "train_end": train_end_date if not lstm_loaded_from_checkpoint else ckpt_meta.get("train_end"),
                            "first_eval_date": first_eval_date,
                            "manifest_version": 2 if not lstm_loaded_from_checkpoint else ckpt_meta.get("manifest_version", 1)}
    _t_inf = __import__("time").perf_counter()
    lstm_eval = lstm_model.evaluate(lstm_X_test, lstm_y_test, scaler=scaler,
                                     base_prices=lstm_base_prices_test)
    log(f"      [target: {lstm_target_mode}] RMSE: {lstm_eval['rmse']:.3f}  MAE: {lstm_eval['mae']:.3f}  "
        f"R2: {lstm_eval['r2']:.3f}  MAPE: {lstm_eval['mape']:.2f}%  "
        f"Directional accuracy: {lstm_eval['accuracy_pct']:.1f}%")
    # Same fix as LR above: build a fresh window for the actual most recent
    # day (today) rather than reusing lstm_X_test[-1], which would predict
    # a day that's already known and ignores today's real closing price.
    lstm_last_window, lstm_base_price = data_loader.build_latest_lstm_window(
        df, scaler, lstm_model.lookback, lstm_target_mode
    )
    lstm_next_price = lstm_model.predict_next(lstm_last_window, scaler, base_price=lstm_base_price)
    perf.record("pipeline.lstm_inference", (__import__("time").perf_counter() - _t_inf) * 1000)

    # --- Walk-forward validation (optional, additional to the split above) ---
    wf_lr = wf_lstm = None
    if walk_forward or ensemble_weighting_mode == "dynamic":
        log("\n[opt] Running walk-forward (expanding-window) validation...")
        try:
            wf_lr = validation_mod.walk_forward_validate_lr(df, n_splits=5)
            # IMPORTANT: the LSTM fold count requested here must never be
            # lower than config.DYNAMIC_WEIGHTS_MIN_FOLDS -- the minimum
            # compute_dynamic_weights() requires before it will trust the
            # result. Requesting fewer folds than that threshold would make
            # dynamic weighting structurally unreachable (always fall back)
            # regardless of how much data is actually available, which is
            # exactly what used to happen here: this "cheap" path requested
            # only 2 LSTM folds while the safety check required 3.
            if walk_forward or for_backtest:
                # A full --walk-forward run, or a caller intending to
                # backtest afterward, is already paying for a thorough
                # validation -- ask for a few MORE folds than the bare
                # minimum (not just exactly the minimum), so there's some
                # "buffer" of folds old enough to have concluded before a
                # later backtest's test period even starts. Without this
                # buffer, the last fold's own test period necessarily
                # extends to near the end of the dataset (expanding-window
                # folds by construction), which usually overlaps with or
                # comes after the single train/test split's test period --
                # leaving NO point in the backtest window with enough PRIOR
                # fold history, so a genuinely historical day-by-day
                # dynamic-weight backtest would silently never engage.
                # Confirmed empirically: DYNAMIC_WEIGHTS_MIN_FOLDS folds
                # exactly (no buffer) produced 0 dynamic-weighted backtest
                # days out of ~100; +2 folds of buffer let dynamic weights
                # actually apply across virtually the whole backtest period
                # for a typical 2-year daily dataset.
                lstm_splits = max(5, config.DYNAMIC_WEIGHTS_MIN_FOLDS + 2)
                lstm_epochs = max(5, epochs // 3) if walk_forward else 5
            else:
                # Dynamic weighting alone, with no walk-forward report and
                # no backtest intended afterward: ask for exactly enough
                # folds to be ABLE to clear the safety bar if the data
                # supports it -- no more, to keep this path cheap -- and no
                # fewer, or it can never succeed.
                lstm_splits = config.DYNAMIC_WEIGHTS_MIN_FOLDS
                lstm_epochs = 5
            wf_lstm = validation_mod.walk_forward_validate_lstm(
                df, n_splits=lstm_splits, epochs=lstm_epochs, target_mode=lstm_target_mode
            )
            log(f"      LR walk-forward: mean RMSE {wf_lr.mean_rmse:.3f} "
                f"over {wf_lr.n_folds} folds, mean directional accuracy "
                f"{wf_lr.mean_directional_accuracy_pct:.1f}%")
            log(f"      LSTM walk-forward: mean RMSE {wf_lstm.mean_rmse:.3f} "
                f"over {wf_lstm.n_folds} folds, mean directional accuracy "
                f"{wf_lstm.mean_directional_accuracy_pct:.1f}%")
        except data_loader.InsufficientDataError as e:
            log(f"      Walk-forward validation skipped: {e}")

    # --- LLM panel ---
    llm_results = []
    llm_panel_summary = None
    if not skip_llm:
        log("\n[4/5] Querying LLM panel (Claude, ChatGPT, Gemini, Grok, DeepSeek)...")
        trend_summary = data_loader.recent_trend_summary(df, ticker=ticker)
        try:
            llm_results = query_llm_panel(trend_summary, verbose=verbose)
            llm_panel_summary = ensemble.blend_llm_panel(llm_results, current_price)
        except Exception as e:
            # query_llm_panel already isolates PER-PROVIDER failures (one
            # bad/timed-out/malformed response never affects the others --
            # see ai_integrations/base_client.py and panel.py). This catches
            # the narrower, rarer case of the panel-level orchestration
            # itself failing (e.g. a client failing to even construct) --
            # without this, that would crash the whole run instead of
            # falling back to "no LLM panel available," exactly the outcome
            # skip_llm=True already produces on purpose.
            log(f"      LLM panel failed entirely (not just one provider) -- continuing with "
                f"LR + LSTM only. Error: {e}")
            llm_results = []
            llm_panel_summary = None
    else:
        log("\n[4/5] Skipping LLM panel (skip_llm=True)")

    # --- Ensemble ---
    log("\n[5/5] Computing final ensemble prediction...")
    ensemble_weights = None
    if ensemble_weighting_mode == "dynamic":
        ensemble_weights = ensemble.compute_dynamic_weights(
            wf_lr, wf_lstm, min_folds_required=config.DYNAMIC_WEIGHTS_MIN_FOLDS
        )
        if ensemble_weights["mode"] == "dynamic":
            log(f"      Weighting mode: dynamic")
            log(f"      LR validation RMSE:  {ensemble_weights['basis']['lr_mean_rmse']:.4f} "
                f"({ensemble_weights['basis']['lr_folds']} folds)")
            log(f"      LSTM validation RMSE: {ensemble_weights['basis']['lstm_mean_rmse']:.4f} "
                f"({ensemble_weights['basis']['lstm_folds']} folds)")
            log(f"      LLM contribution (fixed, not derived from validation): "
                f"{ensemble_weights['llm_panel']:.0%}")
            log(f"      Final weights -> LR: {ensemble_weights['linear_regression']:.0%}  "
                f"LSTM: {ensemble_weights['lstm']:.0%}  "
                f"LLM: {ensemble_weights['llm_panel']:.0%}")
        else:
            log(f"      Weighting mode: fixed_fallback")
            log(f"      Reason: {ensemble_weights['basis']['reason']}")
            log(f"      Final weights -> LR: {ensemble_weights['linear_regression']:.0%}  "
                f"LSTM: {ensemble_weights['lstm']:.0%}  "
                f"LLM: {ensemble_weights['llm_panel']:.0%}")

    final = ensemble.compute_ensemble_prediction(
        lr_price=lr_next_price,
        lstm_price=lstm_next_price,
        llm_panel_summary=llm_panel_summary,
        current_price=current_price,
        weights=ensemble_weights,
    )
    log(f"      Linear Regression -> {lr_next_price:.2f}")
    log(f"      LSTM                -> {lstm_next_price:.2f}")
    if llm_panel_summary:
        log(f"      LLM Panel ({llm_panel_summary['num_providers']} providers) "
            f"-> {llm_panel_summary['panel_price_estimate']:.2f} "
            f"({llm_panel_summary['majority_direction']})")
    log(f"      >>> FINAL ENSEMBLE -> {final['final_predicted_price']:.2f} "
        f"({final['predicted_pct_change']:+.2f}%)")

    wf_lr_dict = wf_lr.to_dict() if wf_lr else None
    wf_lstm_dict = wf_lstm.to_dict() if wf_lstm else None

    confidence_summary = confidence_mod.compute_prediction_confidence(
        ensemble_result=final, lr_eval=lr_eval, lstm_eval=lstm_eval,
        walk_forward_lr=wf_lr_dict, walk_forward_lstm=wf_lstm_dict,
    )
    log(f"      Confidence: {confidence_summary['confidence_label']}  "
        f"(model agreement: {confidence_summary['model_agreement']['agreement_label']})")
    if confidence_summary["approximate_price_range"]:
        r = confidence_summary["approximate_price_range"]
        log(f"      Approximate price range (not a statistical CI): "
            f"{r['low']:.2f} - {r['high']:.2f}")

    return {
        "ticker": ticker,
        "df": df,
        "current_price": current_price,
        "data_quality": quality_report.to_dict(),
        "run_metadata": reproducibility.capture_run_metadata(
            ticker=ticker, source=source, period=period, seed=seed,
            lstm_target_mode=lstm_target_mode, lstm_epochs=epochs,
            lookback=lstm_model.lookback, train_test_split=config.TRAIN_TEST_SPLIT,
            ensemble_weighting_mode=ensemble_weighting_mode,
            n_rows_total=len(df), n_train_rows=len(X_train), n_test_rows=len(X_test),
        ),
        "lr_model": lr_model, "lr_eval": lr_eval, "lr_next_price": lr_next_price,
        "resolved_ticker": resolved_ticker, "dropped_open_session_bar": dropped_open_bar, "lstm_checkpoint": lstm_checkpoint_info,
        "lr_dates_test": lr_dates_test, "y_test": y_test,
        "lstm_model": lstm_model, "lstm_eval": lstm_eval, "lstm_next_price": lstm_next_price,
        "lstm_dates_test": lstm_dates_test, "scaler": scaler,
        "llm_results": llm_results, "llm_panel_summary": llm_panel_summary,
        "final_ensemble": final,
        "walk_forward_lr": wf_lr_dict,
        "walk_forward_lstm": wf_lstm_dict,
        "confidence": confidence_summary,
    }
