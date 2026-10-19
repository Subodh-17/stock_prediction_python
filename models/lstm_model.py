"""
models/lstm_model.py
----------------------
A stacked LSTM network for next-day closing price prediction, trained on
sliding windows of scaled historical closes. Captures sequential / temporal
patterns that the linear regression model can't (nonlinear momentum, longer
lookback dependencies, etc.).
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from metrics_utils import directional_accuracy, safe_mape

import config


class LSTMPredictor:
    def __init__(self, lookback: int = None, units: int = None):
        self.lookback = lookback or config.LOOKBACK_WINDOW
        self.units = units or config.LSTM_UNITS
        self.model = self._build_model()
        self.scaler = None  # set externally after data prep (shared MinMaxScaler)
        self.is_fitted = False

    def _build_model(self) -> Sequential:
        model = Sequential([
            Input(shape=(self.lookback, 1)),
            LSTM(self.units, return_sequences=True),
            Dropout(0.2),
            LSTM(self.units // 2, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mean_squared_error")
        return model

    def train(self, X_train, y_train, epochs=None, batch_size=None, validation_split=0.1, verbose=0):
        epochs = epochs or config.LSTM_EPOCHS
        batch_size = batch_size or config.LSTM_BATCH_SIZE
        early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=verbose,
        )
        self.is_fitted = True
        return history

    def predict(self, X) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before predicting.")
        # batch_size=len(X): predicts the whole array in a single batch
        # instead of Keras's default batch_size=32. This doesn't change any
        # computed value (batching never affects a feedforward/LSTM
        # inference result), but it does avoid a common, well-documented
        # source of unnecessary tf.function retracing: with the default
        # batch_size, a test set whose length isn't a multiple of 32 gets a
        # differently-shaped final partial batch, which triggers an extra
        # trace every time. This matters here specifically because
        # walk-forward validation creates several distinct LSTM models in a
        # loop (one per fold) and calls .predict() on each -- multiplying
        # any per-call retracing overhead by the number of folds.
        return self.model.predict(X, batch_size=len(X), verbose=0).flatten()

    def evaluate(self, X_test, y_test, scaler=None, base_prices=None) -> dict:
        """
        Evaluates in scaled space by default. If `scaler` is provided, also
        returns metrics in real price units (more interpretable).

        `base_prices`: pass this when the model was trained on RETURNS
        rather than raw prices (see data_loader.prepare_lstm_return_data).
        When given, the scaled y/predictions are inverse-transformed into
        real RETURNS first, then reconstructed into a price via
        `base_prices[i] * (1 + return[i])` -- so the resulting rmse/mae/r2/
        mape/accuracy_pct/predictions/actual are in the same price units and
        directly comparable to price-mode's output, regardless of which
        target the model was actually trained on.
        """
        preds_scaled = self.predict(X_test)
        result = {
            "rmse_scaled": float(np.sqrt(mean_squared_error(y_test, preds_scaled))),
            "predictions_scaled": preds_scaled,
        }
        if scaler is not None:
            y_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            preds_real = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

            if base_prices is not None:
                base_prices_arr = np.asarray(base_prices, dtype=float)
                result["actual_return"] = y_real
                result["predicted_return"] = preds_real
                y_price = base_prices_arr * (1 + y_real)
                preds_price = base_prices_arr * (1 + preds_real)
            else:
                y_price = y_real
                preds_price = preds_real

            result["rmse"] = float(np.sqrt(mean_squared_error(y_price, preds_price)))
            result["mae"] = float(mean_absolute_error(y_price, preds_price))
            result["r2"] = float(r2_score(y_price, preds_price))
            result["mape"] = safe_mape(y_price, preds_price)
            result["accuracy_pct"] = directional_accuracy(y_price, preds_price)
            result["predictions"] = preds_price
            result["actual"] = y_price  # real-price-unit ground truth, handy for backtesting/plots
        return result

    def predict_next(self, last_window: np.ndarray, scaler, base_price: float = None) -> float:
        """
        Predicts the next closing price given the most recent `lookback`
        scaled values. `last_window` shape: (lookback,)

        `base_price`: pass this (the most recent actual close) when the
        model was trained on RETURNS rather than raw prices -- the
        prediction is then reconstructed as `base_price * (1 + return)`.
        """
        x = last_window.reshape(1, self.lookback, 1)
        pred_scaled = self.predict(x)[0]
        pred_real = scaler.inverse_transform([[pred_scaled]])[0][0]
        if base_price is not None:
            return float(base_price * (1 + pred_real))
        return float(pred_real)
