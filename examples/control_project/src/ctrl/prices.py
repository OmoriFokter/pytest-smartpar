"""Prices: formatting depends on the global settings, the rest is pure."""

from ctrl.config import SETTINGS


def format_price(amount: float) -> str:
    return f"{amount:.{SETTINGS['precision']}f} {SETTINGS['currency']}"


def apply_discount(amount: float, percent: float) -> float:
    if not 0 <= percent <= 100:
        raise ValueError("percent must be in [0, 100]")
    return round(amount * (100 - percent) / 100, 2)


def parse_price(text: str) -> float:
    return float(text.replace(",", ".").replace(" ", ""))


def add_vat(amount: float, rate: float = 0.2) -> float:
    return round(amount * (1 + rate), 2)
