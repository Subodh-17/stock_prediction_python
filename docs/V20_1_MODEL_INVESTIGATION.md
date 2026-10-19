# V20.1 Real-Market Forecasting Investigation

Investigation Date: **2026-10-19**  
Subject: **Root-Cause Analysis of Model vs. Naive Baseline Performance on Real Equities**  
Target Universe: **AAPL, MSFT, GOOGL, AMZN** (Daily adjusted bars, 2-year history)  
Investigation Verdict: **CONFIRMED NEGATIVE RESULT — RETAIN NAIVE RANDOM-WALK BASELINE AS PREFERRED BENCHMARK**

---

## 1. Executive Summary

In the V20 release verification, Linear Regression (LR), Long Short-Term Memory (LSTM), and the LR+LSTM ensemble were evaluated against real out-of-sample market data across four major US equities (`AAPL`, `MSFT`, `GOOGL`, `AMZN`). None of the models achieved statistically significant improvement over the zero-effort naive persistence baseline ($\hat{P}_{t+1} = P_t$). In fact, the LSTM underperformed naive across all four assets and was statistically worse at 99% bootstrap confidence on two assets (`GOOGL`, `AMZN`), while Linear Regression underperformed naive on three of four assets.

This V20.1 investigation performed an exhaustive empirical and mathematical audit to determine the root cause of this failure, evaluated a pre-declared series of regularized and stationary candidate models, and conducted multi-fold expanding-window walk-forward validation across the entire universe.

### Key Investigation Findings
1. **The Equities Market as a Near-Martingale**: Daily closing prices for mega-cap equities closely follow a random walk with small drift ($E[P_{t+1}|\mathcal{F}_t] \approx P_t(1 + \mu)$). The predictable component of daily returns from historical technical features is near zero ($R^2 < 0.005$) and completely submerged within ~2% daily idiosyncratic volatility.
2. **Variance-Covariance Penalty of Active Forecasts**:
   $$\Delta MSE = MSE(model) - MSE(naive) = E[\hat{\Delta}_{t+1}^2] - 2 E[\Delta_{t+1} \cdot \hat{\Delta}_{t+1}]$$
   Because the signal-to-noise ratio is so low, any parameterized model predicting a non-zero move ($\hat{\Delta}_{t+1} = \hat{P}_{t+1} - P_t \neq 0$) introduces estimation variance ($E[\hat{\Delta}_{t+1}^2] > 0$) that routinely exceeds twice its covariance with the actual next-day move ($2 E[\Delta_{t+1} \cdot \hat{\Delta}_{t+1}]$). The zero-variance naive persistence forecast ($Var(\hat{\Delta}) = 0$) mathematically minimizes out-of-sample MSE whenever true conditional autocorrelation is weak.
3. **Multicollinearity in Linear Regression Features**: Raw price levels (`Close`, `Lag_1`, `Open`, `MA_5`, `MA_10`, `Low`, `High`) exhibit pairwise correlations exceeding 0.99. In unregularized OLS, this causes severe coefficient variance (oscillating weights such as +60 on `Close` and -38 on `Lag_1`), rendering out-of-sample predictions sensitive to small noise fluctuations.
4. **Regime-Shift Bias in LSTM**: The stacked LSTM network learned the strong upward drift of the 2024 to early-2026 tech bull market, predicting an average daily return of +0.3% to +0.5% (annualized +100% to +280%). When tested on the May-October 2026 consolidation and pullback period, this systematic positive drift added substantial bias error ($(E[\hat{y}] - E[y])^2$) to the MSE.
5. **Candidate Model Ablation**: Predeclared candidate improvements—including Ridge regression on price levels, autoregressive models in stationary return space, stationary multi-feature Ridge regression, and naive shrinkage—were systematically evaluated. While return-space models eliminated the collinearity blowup of price-level OLS, **none of the candidates achieved statistically significant superiority over the naive baseline** (all 99% bootstrap confidence intervals bracket zero).
6. **Walk-Forward Validation Across 20 Folds**: Across 5-fold expanding-window walk-forward validation on all four equities (20 total folds), the naive random-walk baseline beat the Linear Regression model on **17 out of 20 folds (85% of folds)**.

In accordance with release gate principles, **the naive persistence baseline is retained as the preferred benchmark**, and this negative result is preserved and documented rather than obscured.

---

## 2. Baseline Reproduction

### Provenance and Configuration
- **Host Environment**: CPython 3.11.0, Windows AMD64, TensorFlow-CPU 2.21.0, Keras 3.15.1, NumPy 2.4.6, Pandas 3.0.5, Scikit-Learn 1.9.1, yfinance 1.7.0.
- **Data Source**: Yahoo Finance daily adjusted bars via yfinance 1.7.0. Partial open session bars dropped.
- **Evaluation Period**: 2026-05-28 to 2026-10-16 (99 out-of-sample trading days).
- **History Period**: 2 years (~500 historical trading days total; 80% train / 20% test split).
- **Random Seed**: 42 (fixed across NumPy, Python random, TensorFlow).
- **Execution Command**:
  ```bash
  python scripts/evaluate_universe.py AAPL MSFT GOOGL AMZN --epochs 5 --period 2y
  ```

### Reproduced Out-of-Sample Performance Table
*(99 out-of-sample observations per ticker, identical evaluation dates)*

| Ticker | Model | RMSE | MAE | MASE | Dir. Acc. | vs Naive RMSE | 99% Bootstrap CI | Diebold-Mariano |
|---|---|---|---|---|---|---|---|---|
| **AAPL** | Naive | **5.534** | **3.988** | 1.446 | - | 0.00% | - | - |
| AAPL | Drift | 5.527 | 3.963 | 1.437 | 54.5% | +0.14% | [-0.91%, +1.73%] | z = +0.269, p = 0.788 |
| AAPL | Linear Regression | 5.767 | 4.215 | 1.528 | 55.6% | -4.21% | [-12.29%, +3.58%] | z = -0.934, p = 0.350 |
| AAPL | LSTM (return) | 5.809 | 4.166 | 1.510 | 54.5% | -4.95% | [-10.71%, +3.22%] | z = -1.453, p = 0.146 |
| AAPL | LR+LSTM Ensemble | 5.560 | 3.994 | 1.448 | 53.5% | -0.46% | [-6.08%, +4.63%] | z = -0.223, p = 0.824 |
| **MSFT** | Naive | **10.395** | **7.404** | 1.556 | - | 0.00% | - | - |
| MSFT | Drift | 10.388 | 7.397 | 1.554 | 55.6% | +0.06% | [-0.17%, +0.21%] | z = +1.086, p = 0.278 |
| MSFT | Linear Regression | 10.487 | 7.456 | 1.566 | 51.5% | -0.88% | [-6.37%, +3.97%] | z = -0.535, p = 0.593 |
| MSFT | LSTM (return) | 10.486 | 7.488 | 1.573 | 53.5% | -0.88% | [-11.58%, +2.86%] | z = -0.496, p = 0.620 |
| MSFT | LR+LSTM Ensemble | 10.343 | 7.317 | 1.537 | 60.6% | +0.50% | [-2.64%, +2.01%] | z = +0.652, p = 0.514 |
| **GOOGL** | Naive | **7.233** | **5.351** | 1.602 | - | 0.00% | - | - |
| GOOGL | Drift | 7.347 | 5.392 | 1.614 | 49.5% | -1.58% | [-3.72%, +1.16%] | z = -1.353, p = 0.176 |
| GOOGL | Linear Regression | 7.338 | 5.342 | 1.599 | 52.5% | -1.45% | [-4.99%, +1.65%] | z = -0.963, p = 0.336 |
| GOOGL | LSTM (return) | 7.517 | 5.516 | 1.651 | 49.5% | -3.93% | [-6.91%, -0.88%] | z = -2.760, p = 0.006 |
| GOOGL | LR+LSTM Ensemble | 7.417 | 5.414 | 1.620 | 48.5% | -2.54% | [-5.53%, +0.43%] | z = -1.840, p = 0.066 |
| **AMZN** | Naive | **5.911** | **4.072** | 1.246 | - | 0.00% | - | - |
| AMZN | Drift | 5.933 | 4.138 | 1.266 | 41.4% | -0.36% | [-2.01%, +0.61%] | z = -0.757, p = 0.449 |
| AMZN | Linear Regression | 5.848 | 4.019 | 1.230 | 50.5% | +1.07% | [-9.15%, +4.28%] | z = +0.494, p = 0.621 |
| AMZN | LSTM (return) | 6.163 | 4.453 | 1.362 | 41.4% | -4.26% | [-14.60%, -0.13%] | z = -2.609, p = 0.009 |
| AMZN | LR+LSTM Ensemble | 5.930 | 4.174 | 1.277 | 46.5% | -0.32% | [-5.79%, +2.03%] | z = -0.230, p = 0.818 |

---

## 3. Audit of Preprocessing, Feature Engineering, and Alignment

A line-by-line inspection was conducted across `data_loader.py`, `pipeline.py`, `models/linear_regression_model.py`, `models/lstm_model.py`, and `evaluation.py`.

### Leakage and Split Isolation Verification
- **Chronological Split**: Verified. In `prepare_linear_regression_data()` and `prepare_lstm_training_data()`, splits are strictly positional: `X[:split_idx]` and `X[split_idx:]`. No shuffling occurs.
- **Scaler Isolation**: Verified. `StandardScaler` (for LR) and `MinMaxScaler` (for LSTM) call `.fit()` exclusively on the training slice (`X_train` / `train_raw`). Test sets only receive `.transform()`.
- **Target Realization Dates**: Verified. In `data_loader.py` (`lr_target_dates`), the test dates correspond strictly to $t+1$ (the day tomorrow's close is realized), properly aligning LR and LSTM evaluation frames.
- **Prediction Target Alignment**: In `evaluation.aligned_frame()`, defensive assertions confirm that the actual price series in the test set matches `df["Close"]` on that exact date to within $10^{-6}$.

### Structural Weaknesses Identified in Feature Engineering
While leakage was cleanly prevented, the mathematical specification of features in the original Linear Regression model revealed significant structural issues:
1. **Extreme Multicollinearity**:
   In `LR_FEATURE_COLUMNS`:
   - `Close` and `Open`: correlation = 0.998
   - `Close` and `High`: correlation = 0.999
   - `Close` and `Low`: correlation = 0.999
   - `Close` and `Lag_1`: correlation = 0.994
   - `Close` and `MA_5`: correlation = 0.992
   In standard unregularized OLS, fitting on near-collinear variables causes severe variance inflation. On AAPL, the fitted coefficients reached +60.30 on `Close` and -37.84 on `Lag_1`.
2. **Non-Stationarity**:
   Raw price levels are $I(1)$ unit-root non-stationary. Fitting OLS directly on price levels creates a spurious regression problem where the model fits sample-specific trend levels rather than stationary relationship dynamics.

---

## 4. Error Analysis and Mathematical Decompositions

To understand why every model had higher MSE than naive, we executed exact mathematical decompositions across all 99 out-of-sample points.

### Variance-Covariance Decomposition of Forecast MSE
For a model predicting next-day close $\hat{P}_{t+1}$ relative to current close $P_t$, let $\Delta_{act} = P_{t+1} - P_t$ and $\Delta_{pred} = \hat{P}_{t+1} - P_t$.
The out-of-sample MSE difference relative to the naive forecast ($\hat{P}_{naive} = P_t$) is:
$$\Delta MSE = MSE(model) - MSE(naive) = \underbrace{E[\Delta_{pred}^2]}_{\text{Variance Penalty}} - \underbrace{2 E[\Delta_{act} \cdot \Delta_{pred}]}_{\text{Directional Covariance Reward}}$$

| Ticker | Model | Naive MSE | Model MSE | $E[\Delta_{pred}^2]$ (Variance Added) | $2 E[\Delta_{act} \cdot \Delta_{pred}]$ (Covariance Gained) | Net $\Delta MSE$ |
|---|---|---|---|---|---|---|
| **AAPL** | Linear Regression | 30.6305 | 33.2634 | 7.2076 | 4.5746 | **+2.6329** (Worse) |
| AAPL | LSTM | 30.6305 | 33.7410 | 3.2602 | 0.1496 | **+3.1106** (Worse) |
| AAPL | LR+LSTM Ensemble | 30.6305 | 30.9135 | 3.3290 | 3.0459 | **+0.2831** (Worse) |
| **MSFT** | Linear Regression | 108.0555 | 109.9687 | 3.2359 | 1.3226 | **+1.9132** (Worse) |
| MSFT | LSTM | 108.0555 | 109.9620 | 4.6455 | 2.7390 | **+1.9065** (Worse) |
| MSFT | LR+LSTM Ensemble | 108.0555 | 106.9820 | 1.3854 | 2.4589 | **-1.0735** (Better) |
| **GOOGL** | Linear Regression | 52.3108 | 53.8419 | 0.9946 | -0.5365 | **+1.5311** (Worse) |
| GOOGL | LSTM | 52.3108 | 56.5005 | 1.1808 | -3.0088 | **+4.1897** (Worse) |
| GOOGL | LR+LSTM Ensemble | 52.3108 | 55.0062 | 0.8175 | -1.8779 | **+2.6954** (Worse) |
| **AMZN** | Linear Regression | 34.9423 | 34.1996 | 1.3603 | 2.1030 | **-0.7427** (Better) |
| AMZN | LSTM | 34.9423 | 37.9857 | 1.3928 | -1.6507 | **+3.0434** (Worse) |
| AMZN | LR+LSTM Ensemble | 34.9423 | 35.1690 | 0.7788 | 0.5520 | **+0.2268** (Worse) |

### Key Insight from Variance Decomposition
On 10 out of the 12 model evaluations, the variance added by the model's predictions ($E[\Delta_{pred}^2]$) exceeded twice the covariance ($2 E[\Delta_{act} \cdot \Delta_{pred}]$). In several cases (LSTM on GOOGL and AMZN), the covariance was actually **negative**, meaning the model systematically bet in the opposite direction of the actual next-day price movement!

### Bias-Variance Decomposition
$$MSE = (\text{Mean Error})^2 + \text{Var}(\text{Error})$$

| Ticker | Model | $\text{Bias}^2$ | $\text{Var}(\text{Error})$ | Total MSE |
|---|---|---|---|---|
| **AAPL** | Naive | 0.1103 | 30.5202 | 30.6305 |
| AAPL | Linear Regression | 0.0283 | 33.2350 | 33.2634 |
| AAPL | LSTM | **1.9231** | 31.8179 | 33.7410 |
| **MSFT** | Naive | 1.4004 | 106.6551 | 108.0555 |
| MSFT | Linear Regression | 2.6588 | 107.3100 | 109.9687 |
| MSFT | LSTM | 0.2415 | 109.7206 | 109.9620 |
| **GOOGL** | Naive | 0.2327 | 52.0781 | 52.3108 |
| GOOGL | Linear Regression | 1.4891 | 52.3529 | 53.8419 |
| GOOGL | LSTM | **2.2086** | 54.2918 | 56.5004 |
| **AMZN** | Naive | 0.0487 | 34.8935 | 34.9423 |
| AMZN | Linear Regression | 0.0469 | 34.1526 | 34.1996 |
| AMZN | LSTM | **1.6319** | 36.3538 | 37.9857 |

The LSTM suffered from severe positive bias on AAPL ($\text{Bias}^2 = 1.92$), GOOGL ($\text{Bias}^2 = 2.21$), and AMZN ($\text{Bias}^2 = 1.63$). The model learned an artificial upward drift during the bull market training phase that failed when market regimes normalized.

---

## 5. Candidate Model Improvements & Ablation Results

We established a small, pre-declared set of five justified candidate improvements:
- **Candidate A1, A2, A3**: Ridge regression on price-level features with $L_2$ regularization penalties ($\alpha \in \{10, 100, 1000\}$) to penalize collinearity.
- **Candidate B1**: Autoregressive OLS in stationary return space ($\hat{R}_{t+1} = \alpha + \sum_{i=0}^2 \beta_i R_{t-i}$; $\hat{P}_{t+1} = P_t(1 + \hat{R}_{t+1})$).
- **Candidate B2**: Autoregressive Ridge in stationary return space ($\alpha = 10$).
- **Candidate C**: Multi-feature Ridge regression in stationary space (lagged returns $R_t, R_{t-1}, R_{t-2}$, 5-day and 10-day realized volatility, normalized moving averages $(P - MA)/MA$, and volume log-change; $\alpha = 50$).
- **Candidate D**: Zero-shrinkage ensemble (shrinking model return predictions toward zero, i.e., toward the naive persistence forecast).

### Empirical Out-of-Sample Performance Comparison

| Model Architecture | AAPL RMSE (Imp %) | MSFT RMSE (Imp %) | GOOGL RMSE (Imp %) | AMZN RMSE (Imp %) | Median Imp % vs Naive |
|---|---|---|---|---|---|
| **Naive Baseline (persistence)** | **5.534** (0.0%) | **10.395** (0.0%) | **7.233** (0.0%) | **5.911** (0.0%) | **0.00%** |
| Baseline LR (OLS on levels) | 5.767 (-4.21%) | 10.487 (-0.88%) | 7.338 (-1.45%) | 5.848 (+1.07%) | -1.17% |
| Candidate A1 (Ridge $\alpha=10$ levels) | 5.695 (-2.90%) | 11.202 (-7.77%) | 7.834 (-8.31%) | 5.833 (+1.33%) | -5.34% |
| Candidate A2 (Ridge $\alpha=100$ levels) | 6.335 (-14.46%) | 12.964 (-24.72%) | 7.911 (-9.39%) | 6.095 (-3.12%) | -11.93% |
| Candidate A3 (Ridge $\alpha=1000$ levels) | 18.760 (-238.97%) | 19.963 (-92.05%) | 23.263 (-221.64%) | 9.642 (-63.11%) | -156.85% |
| **Candidate B1 (OLS on returns)** | 5.458 (+0.98%) | 10.368 (-0.19%) | 7.307 (-1.54%) | 5.953 (-0.59%) | **-0.39%** |
| **Candidate B2 (Ridge on returns $\alpha=10$)** | 5.503 (+0.18%) | 10.344 (+0.04%) | 7.310 (-1.57%) | 5.935 (-0.28%) | **-0.12%** |
| **Candidate C (Stationary Multi-Feature)** | 5.493 (+1.21%) | 10.416 (-0.67%) | 7.355 (-1.19%) | 6.066 (-2.16%) | **-0.93%** |

### Bootstrap Uncertainty Analysis for Candidates
Block-bootstrap confidence intervals (1,000 replications, block length $L = \text{max}(5, 2 n^{1/3})$) were calculated for the best-performing return-space models:

| Ticker | Candidate Model | RMSE Imp % | 95% Bootstrap CI | 99% Bootstrap CI | Statistically Significant at 99% CI? |
|---|---|---|---|---|---|
| **AAPL** | Cand B1 (OLS returns) | +0.98% | [-1.20%, +2.99%] | [-1.89%, +3.75%] | **NO** |
| AAPL | Cand C (Stationary Ridge) | +1.21% | [-1.75%, +2.92%] | [-3.03%, +3.40%] | **NO** |
| **MSFT** | Cand B1 (OLS returns) | -0.19% | [-1.46%, +0.25%] | [-1.81%, +0.53%] | **NO** |
| MSFT | Cand C (Stationary Ridge) | -0.67% | [-3.94%, +0.62%] | [-4.84%, +2.01%] | **NO** |
| **GOOGL** | Cand B1 (OLS returns) | -1.54% | [-3.65%, +0.68%] | [-4.25%, +1.24%] | **NO** |
| GOOGL | Cand C (Stationary Ridge) | -1.19% | [-3.47%, +1.40%] | [-4.59%, +2.07%] | **NO** |
| **AMZN** | Cand B1 (OLS returns) | -0.59% | [-1.64%, +0.01%] | [-1.96%, +0.32%] | **NO** |
| AMZN | Cand C (Stationary Ridge) | -2.16% | [-3.47%, -0.48%] | [-4.15%, -0.00%] | **NO** |

Every single candidate's 99% confidence interval brackets zero (or is strictly negative). **Not a single candidate model achieves statistically significant superiority over the naive baseline.**

---

## 6. Walk-Forward Expanding-Window Validation (20 Folds)

To eliminate the risk that a single fixed test split was unrepresentative, we executed expanding-window walk-forward validation across five sequential folds for all four equities (20 total out-of-sample folds spanning March 2025 through October 2026).

### Detailed Fold-by-Fold Results

| Fold | Ticker | Train Period | Test Period | Naive RMSE | LR-OLS RMSE | LR-Ridge RMSE | Ret-AR RMSE | LR vs Naive Imp % |
|---|---|---|---|---|---|---|---|---|
| F1 | AAPL | 2024-10-31..2025-02-28 | 2025-03-03..2025-06-27 | **5.656** | 7.172 | 6.367 | 5.696 | -26.80% |
| F2 | AAPL | 2024-10-31..2025-06-27 | 2025-06-30..2025-10-23 | **3.614** | 3.811 | 3.958 | 3.659 | -5.45% |
| F3 | AAPL | 2024-10-31..2025-10-23 | 2025-10-24..2026-02-23 | **3.449** | 3.672 | 3.755 | 3.439 | -6.45% |
| F4 | AAPL | 2024-10-31..2026-02-23 | 2026-02-24..2026-06-22 | **3.968** | 4.394 | 4.280 | 3.983 | -10.74% |
| F5 | AAPL | 2024-10-31..2026-06-22 | 2026-06-23..2026-10-16 | **5.667** | 5.690 | 5.837 | 5.638 | -0.41% |
| *Mean* | *AAPL* | *Across 5 Folds* | *Average RMSE* | **4.471** | **4.948** | **4.839** | **4.483** | **-10.67%** |
| F1 | MSFT | 2024-10-31..2025-02-28 | 2025-03-03..2025-06-27 | **7.545** | 10.589 | 9.113 | 7.645 | -40.34% |
| F2 | MSFT | 2024-10-31..2025-06-27 | 2025-06-30..2025-10-23 | **5.044** | 5.490 | 5.322 | 5.007 | -8.85% |
| F3 | MSFT | 2024-10-31..2025-10-23 | 2025-10-24..2026-02-23 | **8.705** | 9.219 | 9.313 | 8.880 | -5.91% |
| F4 | MSFT | 2024-10-31..2026-02-23 | 2026-02-24..2026-06-22 | 7.820 | **7.812** | 8.289 | 7.816 | +0.11% |
| F5 | MSFT | 2024-10-31..2026-06-22 | 2026-06-23..2026-10-16 | **10.179** | 10.625 | 11.334 | 10.189 | -4.39% |
| *Mean* | *MSFT* | *Across 5 Folds* | *Average RMSE* | **7.859** | **8.747** | **8.674** | **7.907** | **-11.30%** |
| F1 | GOOGL | 2024-10-31..2025-02-28 | 2025-03-03..2025-06-27 | **3.776** | 5.360 | 3.842 | 3.781 | -41.94% |
| F2 | GOOGL | 2024-10-31..2025-06-27 | 2025-06-30..2025-10-23 | **3.626** | 4.458 | 4.865 | 3.606 | -22.92% |
| F3 | GOOGL | 2024-10-31..2025-10-23 | 2025-10-24..2026-02-23 | **5.362** | 5.511 | 5.794 | 5.344 | -2.77% |
| F4 | GOOGL | 2024-10-31..2026-02-23 | 2026-02-24..2026-06-22 | 7.384 | **7.372** | 7.838 | 7.383 | +0.16% |
| F5 | GOOGL | 2024-10-31..2026-06-22 | 2026-06-23..2026-10-16 | **6.997** | 7.302 | 7.460 | 7.040 | -4.35% |
| *Mean* | *GOOGL* | *Across 5 Folds* | *Average RMSE* | **5.429** | **6.000** | **5.960** | **5.431** | **-10.52%** |
| F1 | AMZN | 2024-10-31..2025-02-28 | 2025-03-03..2025-06-27 | **5.113** | 6.393 | 5.338 | 5.088 | -25.04% |
| F2 | AMZN | 2024-10-31..2025-06-27 | 2025-06-30..2025-10-23 | 4.004 | **3.974** | 3.791 | 3.991 | +0.74% |
| F3 | AMZN | 2024-10-31..2025-10-23 | 2025-10-24..2026-02-23 | **4.828** | 4.972 | 5.164 | 4.968 | -2.98% |
| F4 | AMZN | 2024-10-31..2026-02-23 | 2026-02-24..2026-06-22 | **4.598** | 5.007 | 4.920 | 4.594 | -8.90% |
| F5 | AMZN | 2024-10-31..2026-06-22 | 2026-06-23..2026-10-16 | **5.890** | 5.953 | 5.889 | 5.921 | -1.06% |
| *Mean* | *AMZN* | *Across 5 Folds* | *Average RMSE* | **4.887** | **5.260** | **5.020** | **4.912** | **-7.64%** |

### Walk-Forward Summary
- Across all 20 individual folds, the naive random-walk baseline beat the Linear Regression model on **17 out of 20 folds (85% of folds)**.
- Across all 4 assets, the mean walk-forward RMSE deficit of Linear Regression relative to naive was:
  - AAPL: -10.67%
  - MSFT: -11.30%
  - GOOGL: -10.52%
  - AMZN: -7.64%
  - **Overall Average Deficit: -10.03%**
- Operating in stationary return space (`Ret-AR`) avoids the catastrophic collinearity errors of early folds (e.g. MSFT Fold 1 where LR-OLS had RMSE 10.589 vs Naive 7.545), achieving an average deficit of just -0.36% vs Naive. However, it still fails to beat the zero-variance naive persistence baseline on average.

---

## 7. Multiple-Testing and Overfitting Risks

In empirical finance and machine learning, searching over multiple tickers, time periods, and model hyperparameters inevitably produces isolated "wins" purely by chance.

### Demonstrating False Positive Risk
Notice that in the single test split:
- Linear Regression had a nominal +1.07% RMSE improvement on `AMZN`.
- The LR+LSTM Ensemble had a nominal +0.50% RMSE improvement on `MSFT`.
- Candidate C had a nominal +1.21% RMSE improvement on `AAPL`.

If an investigator reported only the AMZN split for Linear Regression or only the AAPL split for Candidate C, they would present a false claim of predictive capability. However:
1. When tested under block-bootstrap uncertainty, all 99% confidence intervals include zero (AMZN LR 99% CI: `[-9.15%, +4.28%]`).
2. When tested across 5 expanding walk-forward folds, Linear Regression loses to Naive on 4 of 5 AMZN folds (Mean: -7.64%).
3. When evaluated across the 4-ticker universe, the median improvement of every candidate is negative.

Under the Bonferroni or False Discovery Rate (FDR) corrections for multiple testing across 4 models $\times$ 4 tickers $\times$ 5 folds = 80 hypothesis tests, the required critical $p$-value threshold drops to $\alpha / 80 \approx 0.0006$. Not a single test in this entire investigation meets this threshold.

---

## 8. Preserved Pipeline and Test Integrity

The existing production architecture, interfaces, and test suite remain fully preserved:
- **Test Suite Results**: 545 collected, **545 passed**, 0 failed, 0 skipped, 0 collection errors in 626.39s.
- **Existing Test Files**: All 34 pre-existing V19 test files remain byte-for-byte identical.
- **Integrity**: No assertions were deleted, weakened, or rewritten.
- **Packaging Verification**: `scripts/package_release.py` runs and verifies the complete distribution.

---

## 9. Conclusion and Release Recommendation

The empirical failure of Linear Regression, LSTM, and the LR+LSTM ensemble to outperform the naive baseline is **not a software defect or coding bug**. It is an expected consequence of:
1. Market efficiency and near-martingale asset dynamics at the daily frequency for liquid large-cap equities.
2. The mathematical variance penalty incurred by any parameterized model in low signal-to-noise regimes.
3. Collinearity and non-stationarity in raw price-level feature representations.
4. Systematic drift bias in unregularized deep neural networks across market regime shifts.

### Release Governance Decision
- **Retain the Naive Persistence Baseline as the Benchmark**: The release gate's strict requirement that models must exclude zero at 99% bootstrap confidence before claiming to "beat naive" is completely sound and must remain enforced.
- **Transparent Reporting**: The application must continue to prominently report whether models beat naive, display MASE and bootstrap confidence intervals, and clearly explain to users that simulated low forecast errors in price space do not imply market-beating forecasting skill.
- **Final Release Gate Status**: **PASS WITH LIMITATIONS** (confirmed negative predictive result documented with full scientific transparency).
