"""End-to-end run for one ticker: fetch statements, build model inputs, solve Merton, store results.

Values are always passed to the database as (calc.id, value) pairs, never by
list position, so rows that are skipped or deleted can't shift data onto the
wrong date.
"""
from datetime import date
from typing import List, Tuple

import numpy as np

from merton_dd.data.market import get_betas, get_closing_prices, get_treasury_yields
from merton_dd.data.sec import get_company_financials
from merton_dd.data.trading_calendar import get_last_n_trading_days, get_last_nyse_close_date
from merton_dd.db import calc as calc_db
from merton_dd.db import statements as statements_db
from merton_dd.models.merton import default_probabilities, solve_merton
from merton_dd.models.volatility import annualized_volatility

HORIZON_YEARS = 1


def load_statements(ticker: str, years: int) -> None:
    statements_db.clear_database()
    for statement_type in ("income", "cashflow", "balance"):
        rows = get_company_financials(ticker, statement_type, years=years)
        statements_db.insert_statements(rows, statement_type)
    statements_db.sync_calc_dates()


def equity_volatilities(ticker: str, dates: List[date]) -> List[float]:
    """Trailing one-year equity volatility as of each date."""
    vols = []
    for d in dates:
        window = get_last_n_trading_days(d)
        vols.append(annualized_volatility(get_closing_prices(ticker, window)))
    return vols


def write_market_inputs(ticker: str, ids: List[int], dates: List[date]) -> None:
    calc_db.write_column("volatility", zip(ids, equity_volatilities(ticker, dates)))
    calc_db.write_column("beta", zip(ids, get_betas(ticker, dates)))
    calc_db.write_column("share_price", zip(ids, get_closing_prices(ticker, dates)))
    calc_db.write_column("riskfreerate", zip(ids, get_treasury_yields(dates)))


def solve_and_store_assets() -> None:
    inputs = calc_db.load_merton_inputs()
    if not inputs:
        raise RuntimeError("No rows have complete inputs (market cap, volatility, default point, rate).")

    ids, caps, vols, default_points, rates = (list(col) for col in zip(*inputs))
    asset_values, asset_vols = solve_merton(caps, vols, default_points, rates, HORIZON_YEARS)

    # solve_merton returns plain floats when there is a single row.
    asset_values = np.atleast_1d(asset_values)
    asset_vols = np.atleast_1d(asset_vols)

    calc_db.write_column("market_val_assets", zip(ids, (int(v) for v in asset_values)))
    calc_db.write_column("asset_vol", zip(ids, (float(s) for s in asset_vols)))


def store_default_probabilities() -> None:
    distances = [(row_id, dd) for row_id, dd in calc_db.compute_distance_to_default() if dd is not None]
    print(distances)
    if not distances:
        return
    ids, dds = zip(*distances)
    calc_db.write_column("dtd_value", zip(ids, default_probabilities(dds)))


def run(ticker: str, years: int) -> Tuple[List[date], List[float]]:
    """Returns (dates, default probabilities), newest first, for every row that has a result."""
    load_statements(ticker, years)

    calc_rows = statements_db.prepare_dates(get_last_nyse_close_date())
    ids = [row_id for row_id, _ in calc_rows]
    dates = [d for _, d in calc_rows]
    print(dates)

    write_market_inputs(ticker, ids, dates)
    calc_db.compute_default_points()
    calc_db.compute_market_caps()

    # Drop unusable rows *before* solving, so the solver only sees real data.
    calc_db.remove_rows_without_default_point()
    solve_and_store_assets()
    store_default_probabilities()

    results = calc_db.load_results()
    return [d for d, _ in results], [p for _, p in results]
