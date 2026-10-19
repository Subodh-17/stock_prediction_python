# Quant Workstation V21 — UI Architecture & Design Guide

## 1. Design Vision & Philosophy

Quant Workstation V21 is engineered for quantitative analysts, portfolio researchers, and algorithmic traders who require dense, high-contrast, distraction-free market interfaces.

### Core Visual Principles
1. **Zero External Latency (CSP Strict)**:
   - System fonts only (`Segoe UI`, `SF Pro Text`, `system-ui` for UI; `Cascadia Mono`, `SF Mono`, `ui-monospace` for quantitative figures).
   - Zero CDNs, Google Fonts, or external scripts.
   - Strict `script-src 'self'` policy enforced by default.
2. **Honest Data Presentation**:
   - Direction is never conveyed by color alone; every up/down indicator is paired with an arrow (`▲` / `▼`) or text badge.
   - Forecasts are explicitly demarcated as dashed whiskers / diamonds in their own segregated chart zone; historical observation series are never mutated.
   - Stale or unavailable data is visibly flagged with dashed borders or explicit status badges.
3. **Restrained Glassmorphism**:
   - Translucent top navigation (`backdrop-filter: blur(12px)`) provides sticky orientation without occluding deep data tables.
   - Subtle surface elevation borders replace heavy skeuomorphic drop shadows.

---

## 2. Design Tokens & Theme System

The design system operates via CSS Custom Properties defined on `:root` and overridden via `[data-theme="..."]`.

### 2.1 Dark Theme (Graphite — Default)
- Surface Background (`--bg`): `#101215` (Deep charcoal)
- Panel Surface (`--panel`): `#161a1f` (Graphite card)
- Raised Surface (`--raise`): `#1c2127` (Interactive / metric card)
- Borders (`--line`, `--line2`): `#2a3039`, `#39414d`
- Control Border (`--ctl`): `#646c75`
- Primary Text (`--tx`): `#dde1e7` (High-contrast light grey)
- Muted Text (`--mut`): `#8f98a6`
- Dim / Labels (`--dim`): `#808891`
- Accent (`--acc`): `#c8a35a` (Warm brass — focus & primary indicators)
- Market Gain (`--gain`): `#5cbf8f` (Soft institutional emerald)
- Market Loss (`--loss`): `#e86f63` (Soft coral red)
- Market Info (`--info`): `#7ea8d0` (Soft sapphire)
- Warning / Caveat (`--warn`): `#e0b04f` (Muted amber)

### 2.2 Light Theme (Paper)
- Surface Background (`--bg`): `#f6f8fa` (Clean off-white canvas)
- Panel Surface (`--panel`): `#ffffff` (Pure white card)
- Raised Surface (`--raise`): `#edf1f5` (Subtle grey elevation)
- Borders (`--line`, `--line2`): `#d0d7de`, `#afb8c1`
- Control Border (`--ctl`): `#656d76`
- Primary Text (`--tx`): `#1f2328` (High-contrast dark slate)
- Muted Text (`--mut`): `#454e56` (Legible secondary text)
- Dim / Labels (`--dim`): `#57606a`
- Accent (`--acc`): `#875d00` (Burnished brass / amber)
- Market Gain (`--gain`): `#096e35` (Deep forest green)
- Market Loss (`--loss`): `#cf222e` (Deep crimson)
- Market Info (`--info`): `#0957b8` (Cobalt blue)
- Warning / Caveat (`--warn`): `#8c5800` (Warm amber brown)

### 2.3 Mathematical WCAG AA Compliance
In accordance with WCAG 2.1 AA specifications:
- Every foreground text token (`tx`, `mut`, `dim`, `acc`, `gain`, `loss`, `info`, `warn`) on every surface (`bg`, `panel`, `raise`) exceeds the required **4.5:1** contrast ratio.
- All interactive form control borders (`ctl`) meet or exceed **3.0:1** non-text contrast against adjacent surfaces.

---

## 3. Anti-FOUC Implementation

To prevent unpleasant visual flashes when loading stored user preferences, the application executes an inline script in `<head>` before any stylesheet or body DOM is parsed:

```html
<script>
  (function() {
    var t = 'auto';
    try { t = localStorage.getItem('ws.theme') || 'auto'; } catch (e) {}
    document.documentElement.setAttribute('data-theme', t);
  })();
</script>
```

When `data-theme="auto"`, CSS media query `@media (prefers-color-scheme: light)` automatically applies the light tokens if the user's OS is in light mode, seamlessly adapting without JavaScript runtime overhead.

---

## 4. UI Components & Quantitative Views

### 4.1 Top Navigation Bar
- **Ticker Query**: Fast instrument lookup supporting bare symbols (e.g., `AAPL`), prefixed tickers (e.g., `NSE:TCS`), and exchange suffixes (e.g., `TCS.NS`).
- **Quote Strip**: Real-time price display with freshness badge (`● Live`, `◐ Delayed`, `◼ End of day`, `▤ Historical`, `▲ Stale`, `✕ Unavailable`).
- **Theme Quick-Switch**: Dropdown menu allowing immediate toggle between Auto, Dark, and Light themes.

### 4.2 High-DPI Canvas Charts
- **Isolation of Estimates**: Historical candlestick bars are rendered strictly from observed market data. Model estimates for the next trading day are drawn in a dedicated 56px slot to the right of a dashed divider line, using a dashed whisker and open diamond.
- **Accessible Alternatives**: Every chart component mounts a keyboard-focusable, screen-reader-accessible HTML `<table>` inside a collapsable `<details>` drawer.
- **Dynamic Color Sync**: Charts query CSS variables dynamically (`getComputedStyle(document.documentElement).getPropertyValue(...)`), guaranteeing instant theme adaptation on redraws.

### 4.3 Strategy Backtesting View
- **Summary Metrics**: Displays strategy return, benchmark buy-and-hold return, max drawdown, Sharpe ratio, turnover, and market exposure percentage.
- **Zero-Trade Signal Diagnostics**: If a backtest produces 0 trades because forecasts never exceeded thresholds, the view renders an honest quantitative diagnostic banner detailing the exact forecast range, buy threshold, and threshold gap.
- **Interactive Trade Ledger**: Chronological table of all simulated fills with fill price, shares, transaction costs, and realized P&L.
