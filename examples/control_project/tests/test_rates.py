"""OD pair (lru_cache): the polluter restores the table but leaves a stale cache entry."""

import pytest
from ctrl import rates


def test_usd_rate():
    assert rates.exchange_rate("USD") == 41.0


def test_override_usd_rate():
    rates.set_rate("USD", 99.0)
    assert rates.exchange_rate("USD") in (41.0, 99.0)  # cached either way
    rates.set_rate("USD", 41.0)  # table restored; the cache may keep 99.0


@pytest.mark.parametrize("amount", [1, 10, 100, 1000])
def test_vat(amount):
    from ctrl.prices import add_vat

    assert add_vat(amount) > amount
