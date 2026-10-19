"""
Build the release ZIP and VERIFY it. Refuses to write an archive that would contain secrets, caches, local databases, virtualenvs or
model artefacts, or that lacks a required file. After writing, re-opens the archive and re-checks it, then records size, file count
and SHA-256 (also written to <zip>.manifest.json and <zip>.sha256 next to the archive, NEVER inside it).

    python scripts/package_release.py --out /path/stock_prediction_python_v23_1_FINAL.zip
    python scripts/package_release.py --verify /path/stock_prediction_python_v23_1_FINAL.zip
"""
import argparse, fnmatch, hashlib, json, os, re, sys, zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TOP = "stock_prediction_python"
FIXED_DATETIME = (2026, 1, 1, 0, 0, 0)

EXCLUDE_DIRS = {
    "__pycache__", ".pytest_cache", ".mypy_cache", ".git", ".venv", "venv", "env",
    "node_modules", "outputs", "checkpoints", ".ipynb_checkpoints", "htmlcov"
}

EXCLUDE_GLOBS = [
    "*.pyc", "*.pyo", "*.zip", "*.log", ".env", ".env.*", "*.sqlite", "*.sqlite3", "*.db",
    "*.keras", "*.h5", "*.pkl", ".upstox_instruments_cache_*.json", ".paper_portfolio.json",
    ".coverage", "*.swp", ".DS_Store", "Thumbs.db",
    "*.manifest.json", "*.sha256", "*.sha512", "*.md5", "*.sig", "*.tar", "*.tar.gz", "*.tgz"
]

KEEP = {".env.example"}                     # the template is REQUIRED and holds placeholders only

REQUIRED = [
    ".env.example", ".gitignore", "README.md", "requirements.txt", "requirements-lock.txt", "requirements-dev-lock.txt", "config.py",
    "pipeline.py", "evaluation.py", "market_data.py", "webapp/app.py", "webapp/static/ws.js", "webapp/templates/workstation.html",
    "docs/API.md", "docs/V20_ENGINEERING_AUDIT.md", "docs/V20_RELEASE_NOTES.md", "docs/V20_FINAL_VERIFICATION_REPORT.md",
    "docs/V20_1_MODEL_INVESTIGATION.md",
    "docs/V21_BASELINE_AUDIT.md", "docs/V21_CHANGELOG.md", "docs/V21_RUNTIME_QA_REPORT.md", "docs/V21_UI_GUIDE.md",
    "docs/V22_UI_REDESIGN.md", "docs/V22_RUNTIME_QA_REPORT.md", "docs/V22_CHANGELOG.md", "tests/test_v22_ui.py",
    "reports/V23_BASELINE_AUDIT.md", "reports/V23_DEFECT_REGISTER.md", "reports/V23_TEST_REPORT.md",
    "reports/V23_RUNTIME_ACCEPTANCE.md", "reports/V23_RELEASE_REPORT.md", "reports/V23_1_RELEASE_REPORT.md",
    "tests/test_v23_audit_repairs.py",
    "scripts/release_gate.py", "scripts/package_release.py"
]

SECRET_PATTERNS = [
    r"AIza[0-9A-Za-z_\-]{30,}", r"sk-[A-Za-z0-9]{30,}", r"sk-ant-[A-Za-z0-9_\-]{20,}",
    r"gsk_[A-Za-z0-9]{30,}", r"xai-[A-Za-z0-9]{30,}", r"ghp_[A-Za-z0-9]{30,}",
    r"AKIA[0-9A-Z]{16}", r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----",
    r"(?i)(api[_-]?key|secret|token|password)\s*=\s*['\"]?[A-Za-z0-9_\-]{24,}['\"]?\s*$"
]
TEXT_EXT = {".py", ".md", ".txt", ".json", ".js", ".css", ".html", ".example", ".yml", ".yaml", ".cfg", ".ini", ".toml", ""}
SECRET_SCAN_SKIP = ("tests/",)             # test fixtures contain deliberately fake keys to prove redaction works


def excluded(rel: str) -> bool:
    parts = rel.split("/")
    if any(p in EXCLUDE_DIRS for p in parts[:-1]):
        return True
    name = parts[-1]
    if name in KEEP:
        return False
    if name.endswith(".manifest.json") or name.endswith(".sha256") or name.endswith(".zip"):
        return True
    return any(fnmatch.fnmatch(name, g) for g in EXCLUDE_GLOBS)


def collect(root):
    out = []
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in EXCLUDE_DIRS)
        for f in sorted(files):
            rel = os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/")
            if not excluded(rel):
                out.append(rel)
    return out


def scan_text(rel: str, data: bytes):
    if rel.startswith(SECRET_SCAN_SKIP) or os.path.splitext(rel)[1] not in TEXT_EXT:
        return []
    text = data.decode("utf-8", "ignore")
    hits = []
    for pat in SECRET_PATTERNS:
        for m in re.finditer(pat, text, re.M):
            line = text.count("\n", 0, m.start()) + 1
            if rel == ".env.example" and not re.search(r"[A-Za-z0-9_\-]{24,}", m.group(0).split("=")[-1] if "=" in m.group(0) else ""):
                continue
            hits.append(f"{rel}:{line}: matches secret pattern {pat[:24]}...")
    return hits


def problems_for(files, read):
    probs = [f"required file missing: {r}" for r in REQUIRED if r not in files]
    probs += [f"forbidden file present: {f}" for f in files if excluded(f) and f not in KEEP]
    for f in files:
        if f.endswith(".manifest.json") or f.endswith(".sha256") or f.endswith(".zip") or "manifest" in f.lower():
            probs.append(f"manifest, checksum, or archive file must NEVER be packaged inside archive: {f}")
        probs += scan_text(f, read(f))
    return probs


def verify_archive(archive_path):
    archive_path = os.path.abspath(archive_path)
    if not os.path.exists(archive_path):
        return [f"archive file not found: {archive_path}"]
    probs = []
    actual_bytes = os.path.getsize(archive_path)
    actual_sha = hashlib.sha256(open(archive_path, "rb").read()).hexdigest()
    basename = os.path.basename(archive_path)

    # 1. Check companion manifest
    manifest_path = archive_path + ".manifest.json"
    if not os.path.exists(manifest_path):
        probs.append(f"companion manifest missing: {manifest_path}")
    else:
        try:
            m = json.load(open(manifest_path, "r", encoding="utf-8"))
            if m.get("archive") != basename:
                probs.append(f"manifest 'archive' field mismatch: expected {basename!r}, got {m.get('archive')!r}")
            if m.get("bytes") != actual_bytes:
                probs.append(f"manifest 'bytes' mismatch: expected {actual_bytes}, got {m.get('bytes')}")
            if m.get("sha256") != actual_sha:
                probs.append(f"manifest 'sha256' mismatch: expected {actual_sha}, got {m.get('sha256')}")
        except Exception as e:
            probs.append(f"malformed manifest JSON: {e}")

    # 2. Check companion sha256 file
    sha_path = archive_path + ".sha256"
    if not os.path.exists(sha_path):
        probs.append(f"companion sha256 file missing: {sha_path}")
    else:
        try:
            line = open(sha_path, "r", encoding="utf-8").read().strip()
            parts = line.split(None, 1)
            if not parts or parts[0].lower() != actual_sha.lower():
                probs.append(f"sha256 file checksum mismatch: expected {actual_sha}, got {parts[0] if parts else 'empty'}")
        except Exception as e:
            probs.append(f"malformed sha256 file: {e}")

    # 3. Inspect ZIP structure & members
    try:
        with zipfile.ZipFile(archive_path, "r") as z:
            bad = z.testzip()
            if bad:
                probs.append(f"corrupt member: {bad}")
            names = [n for n in z.namelist() if not n.endswith("/")]
            if len(names) != len(set(names)):
                probs.append("duplicate archive member paths detected")
            inner = sorted(n[len(TOP) + 1:] for n in names if n.startswith(TOP + "/"))
            if os.path.exists(manifest_path):
                m = json.load(open(manifest_path, "r", encoding="utf-8"))
                if m.get("files") != len(names):
                    probs.append(f"manifest 'files' count mismatch: archive has {len(names)}, manifest reports {m.get('files')}")
            probs += problems_for(inner, lambda f: z.read(f"{TOP}/{f}"))
    except Exception as e:
        probs.append(f"cannot open or read zip archive: {e}")

    return probs


def package(root, out_path):
    root = os.path.abspath(root)
    out_path = os.path.abspath(out_path)
    files = collect(root)
    probs = problems_for(files, lambda f: open(os.path.join(root, f), "rb").read())
    if probs:
        print("REFUSING to package:\n  " + "\n  ".join(probs))
        sys.exit(2)

    # Build deterministic ZIP archive
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in sorted(files):
            data = open(os.path.join(root, f), "rb").read()
            zinfo = zipfile.ZipInfo(f"{TOP}/{f}", date_time=FIXED_DATETIME)
            zinfo.compress_type = zipfile.ZIP_DEFLATED
            zinfo.external_attr = 0o644 << 16
            z.writestr(zinfo, data, compresslevel=9)

    # Calculate archive digest & write external metadata
    actual_bytes = os.path.getsize(out_path)
    actual_sha = hashlib.sha256(open(out_path, "rb").read()).hexdigest()
    basename = os.path.basename(out_path)

    manifest_info = {
        "archive": basename,
        "bytes": actual_bytes,
        "files": len(files),
        "sha256": actual_sha,
        "required_present": len(REQUIRED),
        "convention": "manifest_outside_archive",
        "description": "Archive file members strictly; manifest and checksum files sit outside and are excluded from packaging."
    }
    with open(out_path + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_info, f, indent=1)

    with open(out_path + ".sha256", "w", encoding="utf-8") as f:
        f.write(f"{actual_sha} *{basename}\n")

    # Verify what was written
    verify_probs = verify_archive(out_path)
    if verify_probs:
        if os.path.exists(out_path):
            os.remove(out_path)
        if os.path.exists(out_path + ".manifest.json"):
            os.remove(out_path + ".manifest.json")
        if os.path.exists(out_path + ".sha256"):
            os.remove(out_path + ".sha256")
        print("ARCHIVE FAILED VERIFICATION (removed):\n  " + "\n  ".join(verify_probs))
        sys.exit(3)

    print(json.dumps(manifest_info, indent=1))


def main():
    ap = argparse.ArgumentParser(description="Package or verify Quant Workstation release archives.")
    ap.add_argument("--out", help="Path to write the release zip archive")
    ap.add_argument("--root", default=ROOT, help="Root directory of the repository")
    ap.add_argument("--verify", help="Verify an existing archive, its manifest, and its checksum")
    a = ap.parse_args()

    if a.verify:
        probs = verify_archive(a.verify)
        if probs:
            print(f"VERIFICATION FAILED for {a.verify}:\n  " + "\n  ".join(probs))
            sys.exit(1)
        print(f"VERIFICATION PASSED: {a.verify} matches manifest and checksum with zero corrupt members.")
        sys.exit(0)

    if not a.out:
        ap.error("Must specify either --out <path> to package or --verify <path> to verify.")

    package(a.root, a.out)


if __name__ == "__main__":
    main()
