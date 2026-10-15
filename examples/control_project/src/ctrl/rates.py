"""Exchange rates with an lru_cache in front of a mutable table."""

import functools

RATES = {"USD": 41.0, "EUR": 45.0, "PLN": 10.5}


@functools.cache
def exchange_rate(currency: str) -> float:
    return RATES[currency]


def set_rate(currency: str, value: float) -> None:
    RATES[currency] = value
