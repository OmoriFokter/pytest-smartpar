"""Absolute paths inside the control project (tests never rely on the cwd)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT_ROOT / "output"
