import numpy as np
import pytest
from scipy.stats import norm

from merton_dd.models.merton import default_probabilities, solve_merton


def equity_from_assets(V_A, sigma_A, D, r, T):
    """Forward Merton model: given the asset side, what equity value and volatility follow?"""
    d1 = (np.log(V_A / D) + (r + 0.5 * sigma_A ** 2) * T) / (sigma_A * np.sqrt(T))
    d2 = d1 - sigma_A * np.sqrt(T)
    E = V_A * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2)
    sigma_E = V_A / E * norm.cdf(d1) * sigma_A
    return E, sigma_E


@pytest.mark.parametrize("V_A, sigma_A, D", [
    (200.0, 0.25, 100.0),     # healthy firm
    (150.0, 0.40, 120.0),     # highly levered
    (5e9, 0.15, 1e9),         # realistic magnitudes
])
def test_solver_recovers_known_assets(V_A, sigma_A, D):
    E, sigma_E = equity_from_assets(V_A, sigma_A, D, r=0.04, T=1.0)
    solved_V, solved_sigma = solve_merton(E, sigma_E, D, 0.04, 1.0)
    assert solved_V == pytest.approx(V_A, rel=1e-6)
    assert solved_sigma == pytest.approx(sigma_A, rel=1e-6)


def test_single_input_returns_plain_floats():
    V, s = solve_merton(100.0, 0.3, 50.0, 0.04, 1.0)
    assert isinstance(V, float) and isinstance(s, float)


def test_integer_inputs_are_not_truncated():
    E, sigma_E = equity_from_assets(200.0, 0.25, 100.0, 0.04, 1.0)
    V, s = solve_merton([int(E), int(E)], [sigma_E, sigma_E], [100, 100], 0.04, 1)
    assert 0 < s[0] < 1   # an integer array would have rounded this to 0


def test_lists_are_solved_element_by_element_in_order():
    cases = [(200.0, 0.25, 100.0), (150.0, 0.40, 120.0)]
    equity = [equity_from_assets(V, s, D, 0.04, 1.0) for V, s, D in cases]
    V, s = solve_merton([e[0] for e in equity], [e[1] for e in equity], [c[2] for c in cases], 0.04, 1)
    assert list(V) == pytest.approx([200.0, 150.0], rel=1e-6)
    assert list(s) == pytest.approx([0.25, 0.40], rel=1e-6)


def test_default_probabilities_known_values():
    assert default_probabilities([0.0, 2.1, -1.0]) == [0.5, 0.0179, 0.8413]


def test_default_probability_falls_as_distance_rises():
    probs = default_probabilities([0.5, 1.0, 2.0, 3.0])
    assert probs == sorted(probs, reverse=True)
