"""
Release gate: runs the verification steps and RECORDS what actually happened (command, exit code, elapsed seconds, parsed counts).
Nothing is inferred: a step that is not run is recorded as NOT RUN with the reason.

    python scripts/release_gate.py                 # everything incl. the full test suite
    python scripts/release_gate.py --skip-tests    # environment/startup checks only
Writes outputs/release_gate.json and prints a table. Exit code 0 only if no step FAILED.
"""
import argparse, json, os, re, socket, subprocess, sys, time, urllib.request
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("PYTHONUTF8", "1")
IGNORED = {"pip", "setuptools", "wheel"}


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_pins(lines):
    out = {}
    for l in lines:
        l = l.strip()
        if not l or l.startswith(("#", "-r", "-c")) or "==" not in l:
            continue
        n, v = l.split("==", 1)
        out[norm(n)] = v.split(";")[0].strip()
    return out


def read_lock(path, seen=None):
    """Pins from a requirements file, following `-r other.txt` includes."""
    seen = seen or set()
    path = os.path.abspath(path)
    if path in seen:
        return {}
    seen.add(path)
    pins, lines = {}, open(path).read().splitlines()
    for l in lines:
        if l.strip().startswith("-r "):
            pins.update(read_lock(os.path.join(os.path.dirname(path), l.strip()[3:].strip()), seen))
    pins.update(parse_pins(lines))
    return pins


def compare_sets(installed: dict, lock: dict):
    inst = {k: v for k, v in installed.items() if k not in IGNORED}
    return {"missing_from_env": sorted(k for k in lock if k not in inst),
            "extra_in_env": sorted(k for k in inst if k not in lock),
            "version_mismatch": sorted(f"{k}: env {inst[k]} != lock {lock[k]}" for k in lock if k in inst and inst[k] != lock[k])}


def parse_pytest_summary(text: str) -> dict:
    out = {}
    m = re.findall(r"(\d+) (passed|failed|skipped|errors?|xfailed|xpassed|deselected)", text.splitlines()[-1] if text.strip() else "")
    for n, k in m:
        out[k.rstrip("s") if k.startswith("error") else k] = int(n)
    t = re.search(r"in ([\d.]+)s", text)
    out["pytest_reported_seconds"] = float(t.group(1)) if t else None
    return out


def run(cmd, timeout, env=None):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout, env=env)
        return p.returncode, (p.stdout + p.stderr), time.time() - t0
    except subprocess.TimeoutExpired as e:
        return -9, f"TIMEOUT after {timeout}s", time.time() - t0


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def smoke_test_server(py):
    port, tmp = free_port(), os.path.join(ROOT, "outputs", "_smoke")
    os.makedirs(tmp, exist_ok=True)
    env = {**os.environ, "WEBAPP_PORT": str(port), "WARMUP_ON_START": "false", "TF_CPP_MIN_LOG_LEVEL": "3",
           "PYTHONUTF8": "1",
           "PAPER_PORTFOLIO_PATH": os.path.join(tmp, "paper.json"), "TRADE_LOG_PATH": os.path.join(tmp, "log.csv")}
    env.pop("ENABLE_LIVE_TRADING", None)
    proc = subprocess.Popen([py, os.path.join(ROOT, "webapp", "app.py")], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    results, ok = {}, False
    try:
        deadline = time.time() + 90
        while time.time() < deadline and proc.poll() is None:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2); ok = True; break
            except Exception:
                time.sleep(1)
        if ok:
            for path in ("/", "/api/v2/meta", "/api/v2/health", "/api/v2/ai/status", "/static/ws.js", "/static/ws.css"):
                try:
                    results[path] = urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10).status
                except Exception as e:
                    results[path] = f"ERROR {type(e).__name__}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    good = ok and all(v == 200 for v in results.values())
    return good, {"port": port, "bound_to": "127.0.0.1", "responses": results, "server_exit_code_after_terminate": proc.returncode}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tests", action="store_true", help="environment/start-up checks only -> outputs/release_gate_env.json")
    ap.add_argument("--tests-only", action="store_true", help="only the full test suite -> outputs/release_gate_tests.json")
    ap.add_argument("--verify-archive", help="verify a release archive against its manifest and sha256 -> outputs/release_gate_archive.json")
    ap.add_argument("--lock", help="override lockfile path (e.g. platform-specific lock)")
    a = ap.parse_args()
    if a.skip_tests and a.tests_only:
        ap.error("--skip-tests and --tests-only are mutually exclusive")
    py, steps = sys.executable, []

    def record(name, command, status, code=None, secs=None, detail=None):
        steps.append({"step": name, "command": command, "status": status, "exit_code": code,
                      "elapsed_s": None if secs is None else round(secs, 2), "detail": detail})
        print(f"{status:8s} {name:34s} exit={code} {'' if secs is None else f'{secs:7.1f}s'}  {str(detail or '')[:110]}")

    record("python version", f"{py} --version", "INFO", 0, 0, sys.version.split()[0])
    if a.verify_archive:
        import package_release
        t0 = time.time()
        probs = package_release.verify_archive(a.verify_archive)
        record("verify release archive", f"verify {a.verify_archive}", "PASS" if not probs else "FAIL", 0 if not probs else 1, time.time() - t0, "clean: manifest, sha256, and members match" if not probs else "; ".join(probs))
        name = "release_gate_archive.json"
        os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
        with open(os.path.join(ROOT, "outputs", name), "w") as f:
            json.dump({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "python": sys.version.split()[0], "steps": steps}, f, indent=1)
        sys.exit(1 if probs else 0)
    if a.tests_only:
        return finish(steps, py, a, [run_suite(py, record, steps)])
    c, out, t = run([py, "-m", "pip", "check"], 120)
    record("pip check", f"{py} -m pip check", "PASS" if c == 0 else "FAIL", c, t, out.strip().splitlines()[-1] if out.strip() else "")
    code = "import tensorflow as tf, keras; print('tensorflow', tf.__version__, 'keras', keras.__version__)"
    c, out, t = run([py, "-c", code], 300, {**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"})
    ver_line = next((l for l in out.splitlines() if l.startswith("tensorflow ")), "")
    record("real TensorFlow import", f"{py} -c \"{code[:40]}...\"", "PASS" if c == 0 and ver_line else "FAIL", c, t, ver_line[:80] or out.strip()[-80:])
    c, out, t = run([py, "-m", "pip", "freeze"], 120)
    installed = parse_pins(out.splitlines())
    lockfile = a.lock or ("requirements-dev-lock.txt" if os.path.exists(os.path.join(ROOT, "requirements-dev-lock.txt")) else "requirements-lock.txt")
    diff = compare_sets(installed, read_lock(os.path.join(ROOT, lockfile)))
    record("installed set == lock", f"pip freeze vs {lockfile}", "PASS" if not any(diff.values()) else "FAIL", 0, t,
           "identical (%d packages compared; pip/setuptools/wheel ignored)" % len([k for k in installed if k not in IGNORED]) if not any(diff.values()) else json.dumps(diff)[:300])
    c, out, t = run([py, "-m", "compileall", "-q", "."], 300)
    record("compile all sources", f"{py} -m compileall -q .", "PASS" if c == 0 else "FAIL", c, t, out.strip()[:100])
    t0 = time.time(); good, info = smoke_test_server(py)
    record("server start-up smoke test", "python webapp/app.py (random localhost port)", "PASS" if good else "FAIL", 0 if good else 1, time.time() - t0, json.dumps(info["responses"]))
    if a.skip_tests:
        record("full test suite", "pytest tests webapp/test_app.py", "NOT RUN", None, None, "--skip-tests given")
    else:
        run_suite(py, record, steps)
    finish(steps, py, a, [])


def run_suite(py, record, steps):
    cmd = [py, "-m", "pytest", "tests", "webapp/test_app.py", "-q", "-p", "no:cacheprovider", "--ignore=tests/qa_harness.py"]
    c, out, t = run(cmd, 3000, {**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3", "PYTHONUTF8": "1"})
    counts = parse_pytest_summary(out)
    bad = counts.get("failed", 0) + counts.get("error", 0)
    record("full test suite", " ".join(cmd[1:]), "PASS" if c == 0 and bad == 0 and counts.get("passed", 0) > 0 else "FAIL", c, t, json.dumps(counts))
    steps[-1]["counts"] = counts
    steps[-1]["tail"] = out.strip().splitlines()[-3:]


def finish(steps, py, a, _):
    name = "release_gate_tests.json" if a.tests_only else "release_gate_env.json" if a.skip_tests else "release_gate.json"
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    with open(os.path.join(ROOT, "outputs", name), "w") as f:
        json.dump({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "python": sys.version.split()[0], "steps": steps}, f, indent=1)
    sys.exit(1 if any(s["status"] == "FAIL" for s in steps) else 0)


if __name__ == "__main__":
    main()
