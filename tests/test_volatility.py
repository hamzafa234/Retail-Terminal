import numpy as np
import pytest

from merton_dd.models.volatility import annualized_volatility


def test_constant_growth_has_zero_volatility():
    prices = [100 * 1.01 ** i for i in range(50)]
    assert annualized_volatility(prices) == pytest.approx(0.0, abs=1e-9)


def test_known_value():
    # Log returns alternate +x, -x -> std of daily returns is exactly x.
    x = 0.01
    prices = [100.0]
    for i in range(100):
        prices.append(prices[-1] * np.exp(x if i % 2 == 0 else -x))
    assert annualized_volatility(prices) == pytest.approx(x * np.sqrt(252), rel=1e-5)


def test_returns_plain_float():
    assert isinstance(annualized_volatility([100, 101, 99, 102]), float)
