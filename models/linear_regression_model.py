"""
models/linear_regression_model.py
----------------------------------
A straightforward multivariate Linear Regression baseline for next-day
closing price prediction. Serves as the interpretable, fast-to-train
counterweight to the LSTM in the ensemble.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from metrics_utils import directional_accuracy, safe_mape


class LinearRegressionPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray, feature_names=None):
        self.feature_names = feature_names
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before predicting.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        preds = self.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        mape = safe_mape(y_test, preds)
        accuracy_pct = directional_accuracy(y_test, preds)
        return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape,
                "accuracy_pct": accuracy_pct, "predictions": preds}

    def coefficients(self) -> dict:
        """Returns feature -> coefficient mapping, useful for interpretability."""
        if not self.is_fitted:
            raise RuntimeError("Model must be trained first.")
        names = self.feature_names or [f"feature_{i}" for i in range(len(self.model.coef_))]
        return dict(zip(names, self.model.coef_.tolist()))

    def predict_next(self, latest_features: np.ndarray) -> float:
        """Predicts the very next closing price from the most recent feature row."""
        return float(self.predict(latest_features.reshape(1, -1))[0])
