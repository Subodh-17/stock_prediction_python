"""
Capture real screenshots of all 10 workstation views across desktop and mobile viewports.
Uses tests/qa_harness.py with deterministic synthetic fixtures so no live broker or network calls occur.

Usage:
    python scripts/capture_screenshots.py --mode before
    python scripts/capture_screenshots.py --mode after
"""
import argparse
import os
import subprocess
import sys
import time
import urllib.request

os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "http://127.0.0.1:5057"
ROUTES = ["overview", "research", "forecast", "evaluation", "backtest", "ai", "portfolio", "risk", "health", "settings"]
VPS = {
    "desktop": (1440, 900),
    "mobile": (390, 844)
}

def wait_for_server(url, timeout=45):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url + "/", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.8)
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["before", "after", "v23"], default="v23")
    args = parser.parse_args()

    out_dir = os.path.join(ROOT, "docs", "screenshots", args.mode)
    os.makedirs(out_dir, exist_ok=True)

    server_proc = None
    # Check if server is already running
    try:
        with urllib.request.urlopen(BASE + "/", timeout=1) as r:
            already_running = (r.status == 200)
    except Exception:
        already_running = False

    if not already_running:
        print("[+] Starting qa_harness on 127.0.0.1:5057...")
        server_proc = subprocess.Popen(
            [sys.executable, os.path.join(ROOT, "tests", "qa_harness.py")],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        if not wait_for_server(BASE):
            if server_proc:
                server_proc.kill()
            print("[-] Error: qa_harness did not respond within timeout")
            sys.exit(1)
        print("[+] Server ready.")
    else:
        print("[+] Reusing running server on 127.0.0.1:5057")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            for vp_name, (w, h) in VPS.items():
                print(f"[+] Capturing viewport: {vp_name} ({w}x{h})...", flush=True)
                ctx = browser.new_context(viewport={"width": w, "height": h})
                pg = ctx.new_page()

                # Initial load & ticker input
                pg.goto(BASE + "/")
                pg.wait_for_selector("#view h1")
                pg.fill("#sym", "AAPL")
                pg.click("#symform button")
                pg.wait_for_selector("#quote .badge")

                # Run quick forecast so forecast/evaluation/backtest/ai views are hydrated
                print(f"  [-] Hydrating forecast lab (epochs=2)...", flush=True)
                pg.click("a[href='#/forecast']")
                pg.wait_for_selector("#run")
                pg.fill("#fe", "2")
                pg.click("#run")
                pg.wait_for_selector(".hero-card", timeout=120000)
                print(f"  [-] Forecast ready.", flush=True)

                # Iterate through all routes
                for r in ROUTES:
                    pg.click(f"a[href='#/{r}']")
                    time.sleep(1.0)
                    if r == "backtest":
                        try:
                            pg.click("button:has-text('Run backtest')")
                            pg.wait_for_selector("text=Trade ledger", timeout=15000)
                            time.sleep(0.5)
                        except Exception as e:
                            print(f"    [!] Backtest run notice: {e}", flush=True)
                    if r == "ai":
                        try:
                            pg.click("#gen")
                            pg.wait_for_selector("text=AI-generated", timeout=20000)
                            time.sleep(0.5)
                        except Exception as e:
                            print(f"    [!] AI commentary notice: {e}", flush=True)

                    # Capture screenshot
                    file_path = os.path.join(out_dir, f"{vp_name}_{r}.png")
                    pg.screenshot(path=file_path, full_page=True)
                    print(f"    [OK] Saved {vp_name}_{r}.png", flush=True)

                ctx.close()
            browser.close()
    finally:
        if server_proc:
            print("[+] Shutting down qa_harness server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()

    print(f"[+] All screenshots successfully saved to: {out_dir}")

if __name__ == "__main__":
    main()
