"""Ten long tests (0.3-1.5 s): LPT must spread them over the workers."""

import time

import pytest

DURATIONS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]


@pytest.mark.parametrize("seconds", DURATIONS)
def test_long(seconds):
    time.sleep(seconds)
