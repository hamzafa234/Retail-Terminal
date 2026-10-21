"""Merton structural credit model."""
from typing import List

import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm


def _solve_one(E, sigma_E, D, r, T):
    def equations(x):
        V_A, sigma_A = x
        if V_A <= 0 or sigma_A <= 0:
            return [1e10, 1e10]
        d1 = (np.log(V_A / D) + (r + 0.5 * sigma_A ** 2) * T) / (sigma_A * np.sqrt(T))
        d2 = d1 - sigma_A * np.sqrt(T)
        eq_equity = V_A * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2) - E   # equity = call on assets
        eq_vol = (V_A / E) * norm.cdf(d1) * sigma_A - sigma_E                     # volatility linkage
        return [eq_equity, eq_vol]

    guess = [E + D, sigma_E * E / (E + D)]
    return fsolve(equations, guess)


def solve_merton(E, sigma_E, D, r, T):
    """
    Solve for asset value V_A and asset volatility sigma_A.

    E       : market value of equity (market cap)
    sigma_E : annualized equity volatility
    D       : default point / face value of debt
    r       : risk-free rate
    T       : horizon in years
    All accept scalars or array-likes (broadcast together).
    Returns (V_A, sigma_A) as scalars or arrays matching the input shape.
    """
    arrays = [np.atleast_1d(x) for x in (E, sigma_E, D, r, T)]
    shape = np.broadcast_shapes(*(a.shape for a in arrays))
    flat = [np.broadcast_to(a, shape).flatten() for a in arrays]

    V_A = np.zeros_like(flat[0])
    sigma_A = np.zeros_like(flat[0])
    for i, args in enumerate(zip(*flat)):
        V_A[i], sigma_A[i] = _solve_one(*args)

    V_A, sigma_A = V_A.reshape(shape), sigma_A.reshape(shape)
    if V_A.size == 1:
        return float(V_A), float(sigma_A)
    return V_A, sigma_A


def default_probabilities(distances_to_default) -> List[float]:
    """PD = N(-DD), rounded to 4 dp."""
    return [round(x, 4) for x in norm.cdf([-d for d in distances_to_default]).tolist()]
