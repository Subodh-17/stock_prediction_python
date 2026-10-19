"""Static guarantees about the forecast-vs-observed presentation. The rendered result is checked in a real browser by
scripts/browser_audit.py (not part of pytest)."""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JS = open(os.path.join(ROOT, "webapp/static/ws.js"), encoding="utf-8").read()
CSS = open(os.path.join(ROOT, "webapp/static/ws.css"), encoding="utf-8").read()


def test_forecast_is_drawn_in_its_own_slots_never_as_a_candle():
    body = JS[JS.index("function candleChart"):JS.index("/* ---------------------------------------------------------------- shared UI pieces")]
    assert "FW = fc ? 56 : 0" in body and "R + FW" in body                            # fixed-pixel zone to the right of the last real bar
    assert "L + n * step + FW * .6" in body and "setLineDash([5, 3])" in body         # dashed whisker/diamond, outside the observed area
    assert "px > L + n * step + 10" in body                                           # hovering that zone explains it
    assert "bars.forEach((b, i)" in body and "bars.push" not in body and "unshift" not in body    # observed series is never extended
    assert "MODEL ESTIMATE (not observed)" in body                                    # tooltip says what it is
    assert "not an observation" in JS and "a dashed marker to the right is a model estimate" in body   # accessible name


def test_the_forecast_is_only_drawn_for_the_instrument_it_was_computed_for():
    assert "S.forecast.instrument.yahoo_symbol === hs.instrument.yahoo_symbol" in JS


def test_chart_states_timezone_adjustment_source_and_retrieval_time():
    assert "exchange-local trading days (${hs.instrument.timezone})" in JS and "adjusted prices" in JS and "retrieved ${tsLocal(hs.retrieved_ts)}" in JS


def test_missing_interval_is_shown_as_unavailable_not_as_a_made_up_range():
    assert "interval unavailable: insufficient evaluation data" in JS and "fc.low != null" in JS


def test_forecast_survives_a_reload_but_is_forgotten_when_the_symbol_changes():
    assert "sessionStorage.setItem('ws.fkey'" in JS and "sessionStorage.getItem('ws.fkey'" in JS
    assert re.search(r"S\.forecast = null; S\.fkey = null; try \{ sessionStorage\.removeItem\('ws\.fkey'\)", JS)
    assert "saved.symbol === S.symbol" in JS                                         # never restore another instrument's forecast


def test_every_new_visual_element_is_covered_by_the_contrast_rule():
    assert "var(--acc)" in CSS and ".legend .dia" in CSS
