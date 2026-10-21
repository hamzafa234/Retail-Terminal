import numpy as np

TRADING_DAYS_PER_YEAR = 252


def annualized_volatility(prices) -> float:
    """Annualized std-dev of daily log returns (sigma_E)."""
    p = np.array(prices)
    daily_returns = np.log(p[1:] / p[:-1])
    return float(round(np.std(daily_returns) * np.sqrt(TRADING_DAYS_PER_YEAR), 7))
