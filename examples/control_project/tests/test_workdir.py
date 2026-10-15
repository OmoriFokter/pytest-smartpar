"""OD pair (os.chdir): the victim is the only test using a cwd-relative path."""

import os
from pathlib import Path

from ctrl.paths import PROJECT_ROOT


def test_relative_ini_path():
    assert Path("pytest.ini").exists()


def test_enter_package_dir():
    os.chdir(PROJECT_ROOT / "src" / "ctrl")  # never restored


def test_chdir_with_monkeypatch(monkeypatch):
    monkeypatch.chdir(PROJECT_ROOT / "src")
    assert Path("ctrl").is_dir()
