"""
Tests for metrics_utils.py's directional_accuracy() -- hand-verified exact
expected outcomes, same rigor as the rest of this project's test suite.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from metrics_utils import directional_accuracy

# Every prediction correctly calls the direction of the next move
y_true = np.array([100, 105, 110, 108])
y_pred = np.array([100, 103, 112, 107])
assert directional_accuracy(y_true, y_pred) == 100.0
print("Test 1 (perfect directional calls -> 100%) passed")

# Every prediction calls the wrong direction
y_true2 = np.array([100, 105, 110])
y_pred2 = np.array([100, 95, 90])
assert directional_accuracy(y_true2, y_pred2) == 0.0
print("Test 2 (every call wrong -> 0%) passed")

# Exactly half right
y_true3 = np.array([100, 105, 100])
y_pred3 = np.array([100, 110, 110])
assert directional_accuracy(y_true3, y_pred3) == 50.0
print("Test 3 (mixed -> exactly 50%) passed")

# Too few points -> NaN, not a crash or a divide-by-zero
result = directional_accuracy(np.array([100]), np.array([100]))
assert result != result  # NaN check
print("Test 4 (single point -> NaN, not a crash) passed")

# Mismatched lengths -> NaN, not an exception
result2 = directional_accuracy(np.array([100, 101]), np.array([100, 101, 102]))
assert result2 != result2
print("Test 5 (mismatched lengths -> NaN, not an exception) passed")

# Sanity check against a real fitted model on real bundled-style data
import pandas as pd
from models.linear_regression_model import LinearRegressionPredictor
import data_loader

np.random.seed(5)
n = 300
dates = pd.bdate_range("2024-01-01", periods=n)
returns = np.random.normal(loc=0.0005, scale=0.015, size=n)
close = 150 * np.exp(np.cumsum(returns))
df = pd.DataFrame({
    "Open": close * 1.001, "High": close * 1.01, "Low": close * 0.99,
    "Close": close, "Volume": np.random.randint(1_000_000, 5_000_000, n),
}, index=dates)

X_train, X_test, y_train, y_test, feature_names, _ = data_loader.prepare_linear_regression_data(df)
model = LinearRegressionPredictor().train(X_train, y_train, feature_names)
ev = model.evaluate(X_test, y_test)
assert "accuracy_pct" in ev
assert 0 <= ev["accuracy_pct"] <= 100
print(f"Test 6 (real LR model produces a sane accuracy_pct: {ev['accuracy_pct']:.1f}%) passed")

print("\nALL METRICS_UTILS TESTS PASSED")
