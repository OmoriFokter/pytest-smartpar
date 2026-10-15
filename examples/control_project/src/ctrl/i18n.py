"""Locale from the environment."""

import os


def current_locale() -> str:
    return os.environ.get("APP_LOCALE", "uk_UA")
