"""Fast, independent tests."""

import pytest
from ctrl import text, units


@pytest.mark.parametrize("km", range(0, 200, 5))
def test_km_to_miles(km):
    assert units.km_to_miles(km) <= km


@pytest.mark.parametrize("c", range(-40, 60, 5))
def test_celsius_to_fahrenheit(c):
    assert units.celsius_to_fahrenheit(c) >= c or c < -40


@pytest.mark.parametrize("kg", range(1, 30, 2))
def test_kg_to_lb(kg):
    assert units.kg_to_lb(kg) > kg


@pytest.mark.parametrize(
    "word", ["level", "radar", "Never odd or even", "abc", "smartpar", "noon", "python"]
)
def test_is_palindrome(word):
    cleaned = word.lower().replace(" ", "")
    assert text.is_palindrome(word) == (cleaned == cleaned[::-1])


@pytest.mark.parametrize("phrase", ["Hello World", "pytest smart par", "A B C", "one"])
def test_slugify(phrase):
    assert " " not in text.slugify(phrase)


@pytest.mark.parametrize("phrase,count", [("a b c", 3), ("", 0), ("one", 1), ("x  y", 2)])
def test_word_count(phrase, count):
    assert text.word_count(phrase) == count


@pytest.mark.parametrize("n", range(24))
def test_vat_is_monotonic(n):
    from ctrl.prices import add_vat

    assert add_vat(n + 1) > add_vat(n)
