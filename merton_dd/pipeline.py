"""End-to-end run for one ticker: fetch statements, build model inputs, solve Merton, store results."""
from datetime import date
from typing import List, Tuple

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


def run(ticker: str, years: int) -> Tuple[List[date], List[float]]:
    """Returns (dates, default probabilities)."""
    load_statements(ticker, years)

    dates = statements_db.prepare_dates(get_last_nyse_close_date())
    print(dates)

    # Market inputs
    vols = equity_volatilities(ticker, dates)
    yields = get_treasury_yields(dates)
    calc_db.write_column(vols, "volatility")
    calc_db.write_column(get_betas(ticker, dates), "beta")
    calc_db.write_column(get_closing_prices(ticker, dates), "share_price")
    calc_db.write_column(yields, "riskfreerate")

    # Balance-sheet inputs
    default_points = calc_db.compute_default_points()
    market_caps = [float(c) for c in calc_db.compute_market_caps()]

    # Solve
    asset_values, asset_vols = solve_merton(market_caps, vols, default_points, yields, HORIZON_YEARS)
    calc_db.remove_rows_without_default_point()
    calc_db.write_column([int(v) for v in asset_values], "market_val_assets")
    calc_db.write_column([float(s) for s in asset_vols], "asset_vol")

    distances = calc_db.compute_distance_to_default()
    print(distances)
    probabilities = default_probabilities(distances)
    calc_db.write_column(probabilities, "dtd_value")

    return dates, probabilities
