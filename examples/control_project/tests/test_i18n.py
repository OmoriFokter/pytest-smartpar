"""OD pair (os.environ) and a brittle test needing an environment variable."""

import os

from ctrl.i18n import current_locale


def test_default_locale():
    assert current_locale() == "uk_UA"


def test_set_english_locale():
    os.environ["APP_LOCALE"] = "en_US"  # never restored
    assert current_locale() == "en_US"


def test_locale_with_monkeypatch(monkeypatch):
    monkeypatch.setenv("APP_LOCALE", "pl_PL")
    assert current_locale() == "pl_PL"


def test_set_api_token():
    os.environ["CTRL_TOKEN"] = "secret"  # prepares state for test_token_present


def test_token_present():
    assert os.environ.get("CTRL_TOKEN") == "secret"
