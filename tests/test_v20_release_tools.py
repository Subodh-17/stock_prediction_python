"""The release tools must refuse bad input, not just accept good input."""
import hashlib
import json
import os
import subprocess
import sys
import zipfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import package_release as pk
import release_gate as rg


def make_tree(base, extra=None):
    for rel in pk.REQUIRED:
        p = base / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("placeholder\n")
    (base / ".env.example").write_text("GOOGLE_API_KEY=\nGROQ_API_KEY=your-key-here\n")
    for rel, content in (extra or {}).items():
        p = base / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(content)
    return base


def package(root, out):
    return subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "package_release.py"), "--root", str(root), "--out", str(out)],
                          capture_output=True, text=True)


def test_a_clean_tree_packages_verifies_and_records_a_matching_checksum(tmp_path):
    root = make_tree(tmp_path / "src", {"pipeline_extra.py": "x = 1\n", "tests/test_x.py": 'token = "MOCK_SYNTHETIC_TEST_TOKEN_FOR_PACKAGER_VERIFICATION_000000"\n'})
    out = tmp_path / "rel.zip"
    r = package(root, out)
    assert r.returncode == 0, r.stdout + r.stderr
    info = json.loads((tmp_path / "rel.zip.manifest.json").read_text())
    assert info["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest() and info["bytes"] == out.stat().st_size
    names = zipfile.ZipFile(out).namelist()
    assert info["files"] == len([n for n in names if not n.endswith("/")]) and "stock_prediction_python/.env.example" in names
    assert "stock_prediction_python/tests/test_x.py" in names                  # a fake fixture key under tests/ is allowed


@pytest.mark.parametrize("junk", ["__pycache__/m.cpython-312.pyc", "outputs/benchmark.json", ".env", "model.keras", "data.sqlite", "venv/bin/python",
                                  ".upstox_instruments_cache_NSE.json", ".paper_portfolio.json", "old.zip", "run.log", "checkpoints/a/metadata.json",
                                  ".pytest_cache/x", "node_modules/x/y.js", ".git/config", "old.zip.manifest.json", "old.sha256"])
def test_caches_secrets_databases_models_and_environments_are_never_packaged(tmp_path, junk):
    root = make_tree(tmp_path / "src", {junk: "junk"})
    out = tmp_path / "rel.zip"
    assert package(root, out).returncode == 0
    names = zipfile.ZipFile(out).namelist()
    assert not any(n.endswith(os.path.basename(junk)) for n in names if os.path.basename(junk) != ".env.example"), junk


def test_verify_archive_detects_all_anomalies(tmp_path):
    root = make_tree(tmp_path / "src")
    out = tmp_path / "rel.zip"
    r = package(root, out)
    assert r.returncode == 0
    assert pk.verify_archive(str(out)) == []

    # 1. Tampered byte in zip fails
    tampered = tmp_path / "tampered.zip"
    data = bytearray(out.read_bytes())
    data[-10] ^= 0xFF
    tampered.write_bytes(data)
    (tmp_path / "tampered.zip.manifest.json").write_text((tmp_path / "rel.zip.manifest.json").read_text())
    (tmp_path / "tampered.zip.sha256").write_text((tmp_path / "rel.zip.sha256").read_text())
    probs = pk.verify_archive(str(tampered))
    assert any("manifest 'sha256' mismatch" in p or "sha256 file checksum mismatch" in p for p in probs)

    # 2. Missing manifest fails
    no_manifest = tmp_path / "no_manifest.zip"
    no_manifest.write_bytes(out.read_bytes())
    (tmp_path / "no_manifest.zip.sha256").write_text((tmp_path / "rel.zip.sha256").read_text())
    assert any("companion manifest missing" in p for p in pk.verify_archive(str(no_manifest)))

    # 3. Stale manifest file count fails
    bad_count = tmp_path / "bad_count.zip"
    bad_count.write_bytes(out.read_bytes())
    m = json.loads((tmp_path / "rel.zip.manifest.json").read_text())
    m["files"] = m["files"] + 99
    (tmp_path / "bad_count.zip.manifest.json").write_text(json.dumps(m))
    (tmp_path / "bad_count.zip.sha256").write_text(f"{m['sha256']} *bad_count.zip\n")
    assert any("manifest 'files' count mismatch" in p for p in pk.verify_archive(str(bad_count)))


@pytest.mark.parametrize("secret", ['KEY = "AIzaSyREALLOOKINGKEY0123456789abcdefghijk"', "token = sk-" + "a" * 40, "GROQ_API_KEY=gsk_" + "b" * 40,
                                    "-----BEGIN PRIVATE KEY-----", "api_key = 'abcdefghijklmnopqrstuvwxyz0123456789'"])
def test_a_secret_in_source_or_docs_makes_the_packager_refuse(tmp_path, secret):
    root = make_tree(tmp_path / "src", {"module.py": secret + "\n"})
    out = tmp_path / "rel.zip"
    r = package(root, out)
    assert r.returncode == 2 and "REFUSING" in r.stdout and "module.py" in r.stdout and not out.exists()


def test_a_real_looking_value_in_the_env_template_is_refused(tmp_path):
    root = make_tree(tmp_path / "src")
    (root / ".env.example").write_text("GOOGLE_API_KEY=AIzaSyREALLOOKINGKEY0123456789abcdefghijk\n")
    assert package(root, tmp_path / "rel.zip").returncode == 2


def test_a_missing_required_file_makes_the_packager_refuse(tmp_path):
    root = make_tree(tmp_path / "src")
    (root / "webapp" / "app.py").unlink()
    r = package(root, tmp_path / "rel.zip")
    assert r.returncode == 2 and "required file missing: webapp/app.py" in r.stdout


def test_the_repository_itself_contains_no_secret_like_strings():
    hits = []
    for rel in pk.collect(ROOT):
        hits += pk.scan_text(rel, open(os.path.join(ROOT, rel), "rb").read())
    assert hits == []


# ---- gate helpers
def test_lock_parsing_follows_includes_and_comparison_reports_all_three_kinds_of_drift(tmp_path):
    (tmp_path / "a.txt").write_text("# c\nFlask==3.1.3\nmy_pkg==1.0\n")
    (tmp_path / "b.txt").write_text("-r a.txt\npytest==9.1.1\n")
    lock = rg.read_lock(str(tmp_path / "b.txt"))
    assert lock == {"flask": "3.1.3", "my-pkg": "1.0", "pytest": "9.1.1"}
    d = rg.compare_sets({"flask": "3.1.3", "pytest": "9.0.0", "extra": "1", "pip": "26"}, lock)
    assert d == {"missing_from_env": ["my-pkg"], "extra_in_env": ["extra"], "version_mismatch": ["pytest: env 9.0.0 != lock 9.1.1"]}
    assert rg.compare_sets(lock, lock) == {"missing_from_env": [], "extra_in_env": [], "version_mismatch": []}


def test_pytest_summary_parsing():
    assert rg.parse_pytest_summary("442 passed, 499 warnings in 262.16s (0:04:22)")["passed"] == 442
    s = rg.parse_pytest_summary("1 failed, 5 passed, 2 skipped in 1.0s")
    assert (s["failed"], s["passed"], s["skipped"]) == (1, 5, 2)
    assert rg.parse_pytest_summary("12 errors in 2.30s")["error"] == 12 and rg.parse_pytest_summary("")["pytest_reported_seconds"] is None


def test_no_document_in_the_release_contains_an_unresolved_placeholder():
    """V19 shipped docs/V19_FINAL_AUDIT.md with @@SUITE@@ / @@INSTALL@@ / @@TESTS@@ still in it."""
    import re
    bad = []
    pat = re.compile(r"@@[A-Za-z_]+@@|\bTBD\b|\bTODO\b|\bFIXME\b|<fill[ _-]?in>|\{\{\s*[A-Za-z_]+\s*\}\}(?![^<]*</)")
    for base in ("docs", "."):
        d = os.path.join(ROOT, base)
        for name in sorted(os.listdir(d)):
            if name.endswith(".md") and os.path.isfile(os.path.join(d, name)):
                text = open(os.path.join(d, name), encoding="utf-8").read()
                for m in pat.finditer(text):
                    bad.append(f"{base}/{name}: {m.group(0)!r} at line {text.count(chr(10), 0, m.start()) + 1}")
    assert bad == [], "\n".join(bad)


def test_gate_reports_the_number_of_packages_it_actually_compared():
    src = open(os.path.join(ROOT, "scripts", "release_gate.py")).read()
    assert "packages compared; pip/setuptools/wheel ignored" in src and 'l.startswith("tensorflow ")' in src
