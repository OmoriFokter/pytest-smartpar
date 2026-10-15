"""OD pair (module-level dict): test_switch_to_usd pollutes, test_format_uah is the victim."""

import pytest
from ctrl import prices
from ctrl.config import SETTINGS


def test_format_uah():
    assert prices.format_price(10) == "10.00 UAH"


def test_switch_to_usd():
    SETTINGS["currency"] = "USD"  # never restored
    assert prices.format_price(1).endswith("USD")


def test_temporary_currency(monkeypatch):
    # Changes state but restores it: must not be reported.
    monkeypatch.setitem(SETTINGS, "currency", "EUR")
    assert prices.format_price(1) == "1.00 EUR"


@pytest.mark.parametrize("amount", [0, 1, 9.99, 100, 250.5, 1000, 12345.67, 0.01])
@pytest.mark.parametrize("percent", [0, 10, 50, 100])
def test_apply_discount(amount, percent):
    assert prices.apply_discount(amount, percent) <= amount


@pytest.mark.parametrize("text", ["1", "2,5", "10.75", "1 000", "0,01", "99"])
def test_parse_price(text):
    assert prices.parse_price(text) >= 0
