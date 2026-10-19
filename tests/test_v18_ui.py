"""Static UI guarantees (WCAG contrast computed from the real stylesheet; accessibility hooks present in the real JS).
The real-browser audit (axe-core, Chromium) is scripts/browser_audit.py -- it is NOT run by pytest."""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSS = open(os.path.join(ROOT, "webapp/static/ws.css"), encoding="utf-8").read()
JS = open(os.path.join(ROOT, "webapp/static/ws.js"), encoding="utf-8").read()
VARS = dict(re.findall(r"--([\w-]+):(#[0-9a-fA-F]{6})", CSS.split("}")[0]))


def lum(h):
    c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def ratio(a, b):
    la, lb = sorted((lum(a), lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def test_every_text_colour_meets_wcag_aa_on_every_surface():
    for fg in ("tx", "mut", "dim", "acc", "gain", "loss", "info", "warn"):
        for bg in ("bg", "panel", "raise"):
            assert ratio(VARS[fg], VARS[bg]) >= 4.5, f"--{fg} on --{bg} = {ratio(VARS[fg], VARS[bg]):.2f}"
    assert ratio("#15120a", VARS["acc"]) >= 4.5 and ratio("#1a1405", VARS["warn"]) >= 4.5     # primary button, SIMULATED badge


def test_form_control_borders_meet_the_3_to_1_non_text_contrast_requirement():
    for bg in ("bg", "panel", "raise"):
        assert ratio(VARS["ctl"], VARS[bg]) >= 3.0, bg
    assert "border:1px solid var(--ctl)" in CSS


def test_focus_indicator_is_visible_and_high_contrast():
    assert re.search(r":focus-visible\{outline:2px solid var\(--acc\)", CSS)
    assert all(ratio(VARS["acc"], VARS[bg]) >= 3.0 for bg in ("bg", "panel", "raise"))


def test_every_chart_has_a_keyboard_reachable_table_equivalent():
    assert JS.count("dataTable(") >= 4                                          # definition + line + histogram + candle
    for chart in ("function lineChart", "function histChart", "function candleChart"):
        body = JS[JS.index(chart):]
        body = body[:body.index("\n}\n") + 3]
        assert "dataTable(" in body, chart
    assert "tabindex: '0'" in JS and "role: 'region'" in JS and "scope: 'col'" in JS


def test_scrollable_table_wrappers_are_focusable_and_named():
    m = re.search(r"return h\('div', \{ class: 'tw'[^}]*\}", JS)
    assert m and "tabindex: '0'" in m.group(0) and "aria-label" in m.group(0)


def test_direction_is_never_conveyed_by_colour_alone():
    assert "'▲'" in JS and "'▼'" in JS and "'■'" in JS
    assert "const dir = v => v > 0 ? ['up', '▲']" in JS
    assert "mock_verified" in JS and "TEST DOUBLE" in JS                         # a test double can never look verified


def test_all_form_controls_in_the_template_are_labelled():
    html = open(os.path.join(ROOT, "webapp/templates/workstation.html")).read()
    for ident in re.findall(r'<(?:input|select)[^>]*\sid="([\w-]+)"', html):
        assert f'for="{ident}"' in html, ident
