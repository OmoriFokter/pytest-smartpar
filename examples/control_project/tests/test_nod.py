"""Two NOD tests: fail with ~20% probability regardless of order.

random.SystemRandom is used on purpose: pytest-randomly re-seeds the `random`
module before every test, but not the OS entropy source.
"""

import random

_rng = random.SystemRandom()


def test_flaky_network_simulation():
    assert _rng.random() >= 0.2


def test_flaky_timing_simulation():
    assert _rng.random() >= 0.2
