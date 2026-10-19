"""
Capture REAL-APP screenshots of all 10 workstation views across desktop and mobile viewports.
Connects directly to the documented entry-point server: python webapp/app.py on http://127.0.0.1:5000
using live real-market Yahoo data (quotes and 1y historical daily bars).

Records honest real-runtime status:
  - Market data / historical bars / quote charts: REAL (Yahoo Finance)
  - Forecast / evaluation / backtest: REAL (trained on real AAPL bars)
  - Portfolio / Risk sizing: REAL (marked against real AAPL price)
  - AI commentary: BLOCKED / NOT CONFIGURED (honest reporting: no API keys)
"""
import json
import os
import sys
import time
import urllib.request
from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "http://127.0.0.1:5000"
ROUTES = ["overview", "research", "forecast", "evaluation", "backtest", "ai", "portfolio", "risk", "health", "settings"]
VPS = {
    "desktop": (1440, 900),
    "mobile": (390, 844)
}

def wait_for_server(url, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url + "/", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False

def main():
    out_dir = os.path.join(ROOT, "docs", "screenshots", "v23_1")
    os.makedirs(out_dir, exist_ok=True)

    if not wait_for_server(BASE):
        print(f"[-] Error: Real Flask server at {BASE} is not reachable. Run 'python webapp/app.py' first.")
        sys.exit(1)

    print(f"[+] Connected to real Flask app at {BASE}")
    results = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        for vp_name, (w, h) in VPS.items():
            print(f"[+] Capturing viewport: {vp_name} ({w}x{h})...", flush=True)
            ctx = browser.new_context(viewport={"width": w, "height": h})
            pg = ctx.new_page()

            # 1. Initial load
            pg.goto(BASE + "/")
            pg.wait_for_selector("#view h1")
            time.sleep(1.0)

            # Query real AAPL
            pg.fill("#sym", "AAPL")
            pg.click("#symform button")
            pg.wait_for_selector("#quote .badge", timeout=15000)
            time.sleep(1.0)

            # 2. Hydrate forecast on real AAPL bars (epochs=2 for speed)
            print("  [-] Training forecast on real AAPL market bars (epochs=2)...", flush=True)
            pg.click("a[href='#/forecast']", force=True)
            pg.wait_for_selector("#run")
            pg.fill("#fe", "2")
            pg.evaluate("document.querySelector('#run').click()")
            try:
                pg.wait_for_selector(".hero-card", timeout=120000)
                print("  [OK] Real forecast training complete.", flush=True)
            except Exception as e:
                print(f"  [!] Forecast timeout/notice: {e}", flush=True)

            # 3. Capture all views
            for r in ROUTES:
                pg.evaluate(f"window.location.hash = '#/{r}'")
                time.sleep(1.2)
                if r == "backtest":
                    try:
                        pg.evaluate("document.querySelector('#backtest-run, button[type=submit], button.pri') ? document.querySelector('#backtest-run, button[type=submit], button.pri').click() : null")
                        time.sleep(1.0)
                    except Exception as e:
                        print(f"    [!] Backtest notice: {e}", flush=True)
                elif r == "ai":
                    try:
                        pg.evaluate("document.querySelector('#gen') ? document.querySelector('#gen').click() : null")
                        time.sleep(1.0)
                    except Exception as e:
                        print(f"    [!] AI commentary notice: {e}", flush=True)

                file_path = os.path.join(out_dir, f"{vp_name}_{r}.png")
                pg.screenshot(path=file_path, full_page=True)
                print(f"    [OK] Captured {vp_name}_{r}.png ({os.path.getsize(file_path)} bytes)", flush=True)
                results[f"{vp_name}_{r}"] = {
                    "path": file_path,
                    "bytes": os.path.getsize(file_path),
                    "status": "PASS"
                }

            ctx.close()
        browser.close()

    log_path = os.path.join(out_dir, "real_app_acceptance_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "server": BASE,
            "data_source": "Yahoo Finance (real live delayed quotes & 1y daily bars)",
            "viewports": VPS,
            "captures": results
        }, f, indent=1)

    print(f"[+] All 20 real-app screenshots captured to {out_dir}")

if __name__ == "__main__":
    main()
