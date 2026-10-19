# Quant Workstation V22 — Quantitative Workstation UI Redesign

## 1. Executive Summary & Design Vision

The V21 release laid essential groundwork by introducing CSS design tokens and theme toggles. However, its layout remained an undifferentiated, single-column vertical stack of generic cards.

**Quant Workstation V22** delivers a complete architectural visual redesign tailored to quantitative finance professionals:
1. **Responsive 12-Column Bento-Grid System**: Replaces monolithic card stacks with a proportional information hierarchy that balances high-density data tables, large candlestick charts, telemetry gauges, and workflow controls.
2. **Institutional Market Overview**: Re-architected as a trading command center featuring an active instrument hero card with large tabular pricing, daily change metrics, live telemetry, an embedded OHLC candlestick chart with forecast diamonds, an interactive watchlist, and data health cards.
3. **Top Command Bar with Instrument Quick Picks**: Features a persistent terminal badge, live operational indicator, symbol search input, and interactive quick-picker ticker pills (`AAPL`, `MSFT`, `NSE:TCS`, `NVDA`) that provide one-click active instrument switching.
4. **Enhanced Navigation**: Left navigation rail augmented with intuitive domain glyphs (`◈`, `⌕`, `⚡`, `☷`, `⇄`, `✦`, `⊞`, `⚠`, `♥`, `⚙`) grouped into clear functional categories: *Market*, *Quant Models*, *Execution & Risk*, and *Infrastructure*.
5. **Rigorous Financial Typography & Contrast**: Strict tabular numerals (`font-variant-numeric: tabular-nums`) across all prices, percentages, dates, and metrics, coupled with verified WCAG AA contrast (minimum 4.5:1) in both dark (graphite) and light (paper) themes.

---

## 2. Bento-Grid Layout System

The styling overhaul in `webapp/static/ws.css` implements a flexible 12-column grid system:

```css
.bento {
  display: grid;
  grid-template-columns: repeat(12, minmax(0, 1fr));
  gap: 14px;
  width: 100%;
}
.c12 { grid-column: span 12; }
.c8  { grid-column: span 8; }
.c7  { grid-column: span 7; }
.c6  { grid-column: span 6; }
.c5  { grid-column: span 5; }
.c4  { grid-column: span 4; }
.c3  { grid-column: span 3; }
```

### Responsive Auto-Reflow Behavior
- **Desktop (>= 1160px)**: Displays full 12-column spans (`c8`+`c4`, `c7`+`c5`, `c6`+`c6`, `c4`+`c4`+`c4`).
- **Tablet (860px - 1160px)**: Asymmetric spans (`c8`, `c7`, `c5`, `c4`) automatically collapse into balanced 6-column half-width tiles or 12-column full-width tiles.
- **Mobile (< 860px)**: The navigation rail reflows into a horizontal scroll bar at the top, and all `.bento` children collapse to `grid-column: span 1` in a single-column touch-friendly view with no horizontal overflow.

---

## 3. Page-by-Page Transformation

### 3.1 Market Overview (`VIEWS.overview`)
- **Before (V21)**: Empty hero area, plain text inputs, and a basic watchlist table stacked above an empty market breadth box.
- **After (V22)**:
  - **Hero Active Instrument Card (`c12`)**: Displays the active ticker, exchange, quote freshness badge, market session, large tabular price typography (`28px`), and daily change percentage pill. Includes telemetry stats (Previous Close, Provider, Freshness, Age) and direct action buttons (`Forecast Lab`, `Full Chart`, `Paper Order`).
  - **Price Action & Forecast Canvas (`c8`)**: Directly renders daily OHLC candlestick bars with SMA 20 and SMA 50 overlays, and shows the model estimate diamond if a forecast exists. Underneath, displays range high/low and average volume metrics.
  - **Watchlist Panel (`c4` or `c8`)**: High-contrast instrument watch table with add-by-symbol input, instant refresh button, and freshness badges.
  - **Forecast Snapshot (`c4`)**: Displays the current next-day prediction, implied move, and 80% interval.
  - **Feed Telemetry & Health (`c4`)**: Shows upstream provider, exchange time, cache age, and latency.
  - **Market Breadth (`c4`)**: Honest indicator stating advance/decline source availability without artificial values.

### 3.2 Instrument Research (`VIEWS.research`)
- **Before (V21)**: Chart stacked above a generic two-column layout.
- **After (V22)**:
  - High-resolution candlestick OHLC canvas occupying full width (`c12`) with date, timezone, adjustment, and retrieval time provenance.
  - Symmetrical analytical row: Range Statistics & Data Quality audit panel (`c6`) and live News Feed headlines (`c6`).

### 3.3 Forecast Lab (`VIEWS.forecast`)
- **Before (V21)**: Form controls followed by raw text output.
- **After (V22)**:
  - Command panel (`c12`) with history period selector, LSTM epochs, and walk-forward checkbox.
  - Dedicated Forecast Hero Card (`c12`) with numerical ensemble predicted price, implied percentage change, last completed bar close, and evaluation summary.
  - Symmetrical analytical tiles: Component Model Weights & Calibrated Prediction Intervals (`c6`), and Run Details & Execution Timings (`c6`).
  - Clear evaluation verdict note (`c12`).

### 3.4 Model Evaluation (`VIEWS.evaluation`)
- **Before (V21)**: Unstyled tables and disconnected charts.
- **After (V22)**:
  - Chronological held-out comparison table (`c12`) with model names, sample counts, MAE, RMSE, directional accuracy, MASE, and Diebold–Mariano statistics.
  - Actual vs. Predicted Close line chart (`c12`) aligned by target date.
  - Dual residual inspection row: Ensemble Residuals Histogram (`c6`) and Residuals Over Time (`c6`).
  - Prediction-Interval Backtest panel (`c12`) scoring nominal vs. observed coverage.
  - Expanding-window Walk-Forward Folds table (`c12`).

### 3.5 Strategy Backtesting (`VIEWS.backtest`)
- **Before (V21)**: Basic form fields and raw numbers.
- **After (V22)**:
  - Configuration panel (`c12`) with capital, costs, slippage, and signal source selection.
  - Comprehensive metrics row (`c12`) contrasting Strategy Return, Buy & Hold Benchmark, Drawdown, Sharpe Ratio, Win Rate, and Turnover.
  - Asymmetric visual analysis: Equity Curve with buy/sell trade fill glyphs (`c8`) paired alongside Drawdown Curve (`c4`).
  - Complete trade ledger table (`c12`) and clear strategy limitation disclosures.

### 3.6 AI Research (`VIEWS.ai`)
- **Before (V21)**: Stacked tables with plain text outputs.
- **After (V22)**:
  - Configured AI Providers status table (`c12`) displaying real-world verification states, transport mechanisms, and measured latency.
  - Commentary generation panel (`c12`) rendering grounded observations, risk factors, hypotheses, news summaries, and citation fact-checking badges.

### 3.7 Portfolio & Paper Trading (`VIEWS.portfolio`)
- **Before (V21)**: Simple form controls for paper orders.
- **After (V22)**:
  - High-level paper metrics row (`c12`) for Cash, Marked Equity, and Today's Trades.
  - Open Paper Positions table (`c12`) with real-time unrealized P&L pills.
  - Asymmetric action split: Simulated Order entry form with safety confirmation checkbox (`c7`) and Starting Balance Reset panel (`c5`).
  - Prominent notice on live trading gating and requirements (`c12`).

### 3.8 Risk & Exposure (`VIEWS.risk`)
- **Before (V21)**: Generic numbers without visual indicators.
- **After (V22)**:
  - Exposure metrics row (`c12`) for Equity, Cash Weight, and Largest Position.
  - Exposure by Position panel (`c7`) featuring graphical progress bar overlays indicating portfolio weights, annualized volatility, and maximum 1-year drawdown.
  - Active server-enforced risk limits (`c5`).
  - Interactive Position-Size Calculator (`c12`).

### 3.9 Data & Model Health (`VIEWS.health`)
- **Before (V21)**: Long list of status boxes.
- **After (V22)**:
  - Market Data Providers table with circuit breaker states (`c12`).
  - Dual infrastructure row: Caches & Coalescing metrics (`c6`) and AI Provider statuses (`c6`).
  - Measured Performance Latencies table (`c12`) recording wall-clock timings.
  - Provider Verification Records table (`c12`).
  - Dual runtime diagnostics: Background Training Jobs (`c6`) and Saved LSTM Checkpoints (`c6`).

### 3.10 Settings (`VIEWS.settings`)
- **Before (V21)**: Simple form.
- **After (V22)**:
  - Browser preferences panel (`c12`) for Color Theme (Auto, Dark, Light), Quote Auto-Refresh, and Default Chart Range.
  - Symmetrical 4-panel bento grid: Data Sources (`c6`), Models (`c6`), Risk & Trading parameters (`c6`), and AI Provider Readiness diagnostics (`c6`).

---

## 4. Screenshot Artifacts

Real before-and-after screenshots across both desktop (`1440x900`) and mobile (`390x844`) viewports have been captured and archived in the repository:

- **Baseline Screenshots (V21)**: `docs/screenshots/before/` (20 files total)
- **Redesigned Screenshots (V22)**: `docs/screenshots/after/` (20 files total)
