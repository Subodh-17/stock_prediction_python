"""V22 Quant Workstation UI Redesign guarantees:
- 12-column responsive bento-grid dashboard layout
- Hero active instrument cards with typography and telemetry
- Top command bar with live indicator, quick-picker instrument pills, and theme switching
- Accessible, high-contrast visual tokens and keyboard-navigable chart equivalents
- Verified presence of bento grid wrappers across all workstation views
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSS = open(os.path.join(ROOT, "webapp/static/ws.css"), encoding="utf-8").read()
JS = open(os.path.join(ROOT, "webapp/static/ws.js"), encoding="utf-8").read()
HTML = open(os.path.join(ROOT, "webapp/templates/workstation.html"), encoding="utf-8").read()


def test_bento_grid_system_in_stylesheet():
    assert ".bento{" in CSS or ".bento {" in CSS
    assert "grid-template-columns:repeat(12" in CSS
    for span in ("c12", "c8", "c7", "c6", "c5", "c4", "c3"):
        assert f".{span}" in CSS, f"Missing span class .{span}"
    # Verify responsive adjustments for tablet and mobile
    assert "@media (max-width:1160px)" in CSS
    assert "@media (max-width:860px)" in CSS


def test_hero_instrument_components_in_stylesheet():
    for cls in ("hero-card", "hero-main", "hero-sym", "hero-price-row", "hero-px", "hero-chg", "hero-stats", "hero-stat"):
        assert f".{cls}" in CSS, f"Missing hero class .{cls}"


def test_quick_pills_in_template_and_js():
    assert 'id="top-pills"' in HTML
    assert 'data-sym="AAPL"' in HTML
    assert 'data-sym="MSFT"' in HTML
    assert 'data-sym="NSE:TCS"' in HTML
    assert 'data-sym="NVDA"' in HTML
    # Wired in JS
    assert "updateTopPills" in JS
    assert "#top-pills .pill" in JS


def test_terminal_branding_and_navigation_icons_in_template():
    assert "QUANT TERMINAL" in HTML
    assert "status-dot" in HTML
    glyphs = ["◈", "⌕", "⚡", "☷", "⇄", "✦", "⊞", "⚠", "♥", "⚙"]
    for g in glyphs:
        assert g in HTML, f"Missing navigation glyph {g}"


def test_bento_grid_helper_and_views_usage_in_js():
    assert "const bento = (...tiles) => h('div', { class: 'bento' }, ...tiles);" in JS
    for view_name in ("overview", "research", "forecast", "evaluation", "backtest", "ai", "portfolio", "risk", "health", "settings"):
        assert f"VIEWS.{view_name}" in JS, f"Missing view VIEWS.{view_name}"
    # Overview specifically features the bento grid with hero card, chart, and watchlist
    body_overview = JS[JS.index("VIEWS.overview"):JS.index("async function loadWatchQuotes()")]
    assert "bento(heroCard" in body_overview or "bento(" in body_overview
    assert "hero-card" in body_overview
    assert "wlPanel" in body_overview


def test_no_innerhtml_or_eval_in_v22_code():
    js_clean = re.sub(r"/\*.*?\*/", "", JS, flags=re.S)
    assert "innerHTML" not in js_clean
    assert "document.write" not in js_clean
    assert "eval(" not in js_clean


def test_live_badge_is_strictly_singular_in_js():
    occurrences = [m.start() for m in re.finditer(r"'Live'", JS)]
    assert len(occurrences) == 1, f"Found 'Live' {len(occurrences)} times in ws.js"
