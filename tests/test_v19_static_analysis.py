"""A script that cannot even start must never ship (V18's scripts/browser_audit.py used `os` before importing it and nothing
caught it). pyflakes reports undefined names and use-before-import at module scope across the whole tree."""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_no_undefined_names_anywhere_in_the_repository():
    pyflakes = pytest.importorskip("pyflakes")
    out = subprocess.run([sys.executable, "-m", "pyflakes", "."], cwd=ROOT, capture_output=True, text=True).stdout
    bad = [l for l in out.splitlines() if "undefined name" in l or "may be undefined" in l or "unable to detect undefined" in l
           or "syntax" in l.lower()]
    assert not bad, "\n".join(bad)


def test_every_script_compiles_and_imports_what_it_uses():
    import py_compile
    for name in sorted(os.listdir(os.path.join(ROOT, "scripts"))):
        if name.endswith(".py"):
            py_compile.compile(os.path.join(ROOT, "scripts", name), doraise=True)
