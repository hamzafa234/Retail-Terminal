"""Market data from Yahoo Finance: prices, betas, Treasury yields."""
from datetime import timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf


def get_closing_prices(ticker_symbol: str, target_dates: list) -> List[Optional[float]]:
    """Close on each target date, or the next trading day if the market was shut."""
    if not target_dates:
        return []

    start = min(target_dates)
    end = max(target_dates) + timedelta(days=8)
    history = yf.Ticker(ticker_symbol).history(
        start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d")
    )
    if history.empty:
        return [None] * len(target_dates)

    history.index = history.index.tz_localize(None)
    history = history[["Close"]].reset_index().sort_values("Date")
    history["Date"] = history["Date"].dt.as_unit("ns")

    requests_df = pd.DataFrame({
        "requested_date": pd.to_datetime(target_dates),
        "original_order": range(len(target_dates)),
    })
    requests_df["requested_date"] = requests_df["requested_date"].dt.as_unit("ns")
    requests_df = requests_df.sort_values("requested_date")

    merged = pd.merge_asof(
        requests_df, history,
        left_on="requested_date", right_on="Date", direction="forward",
    )
    return merged.sort_values("original_order")["Close"].round(3).tolist()


def get_betas(ticker: str, dates: list, benchmark: str = "^GSPC", years: int = 3) -> List[Optional[float]]:
    """Trailing `years`-year beta vs `benchmark`, as of each date."""
    if not dates:
        return []

    data = yf.download(
        [ticker, benchmark],
        start=min(dates) - timedelta(days=years * 365 + 15),
        end=max(dates) + timedelta(days=2),
        progress=False,
        auto_adjust=False,
    )
    if data.empty:
        return [None] * len(dates)

    close = data["Adj Close"] if "Adj Close" in data else data["Close"]
    returns = close.pct_change().dropna()

    betas: List[Optional[float]] = []
    for d in dates:
        end = pd.Timestamp(d)
        window = returns.loc[end - timedelta(days=years * 365):end]
        if len(window) < years * 150:
            betas.append(None)
            continue
        try:
            cov = np.cov(window[ticker], window[benchmark])
            betas.append(float(round(cov[0, 1] / cov[1, 1], 4)))  # plain float for psycopg2
        except (ZeroDivisionError, KeyError, ValueError):
            betas.append(None)
    return betas


def get_treasury_yields(target_dates: list) -> List[Optional[float]]:
    """10-year Treasury yield (^TNX) as a decimal, back-filled over non-trading days."""
    if not target_dates:
        return []

    ordered = sorted(target_dates)
    history = yf.Ticker("^TNX").history(
        start=ordered[0].strftime("%Y-%m-%d"),
        end=(ordered[-1] + pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
    )
    if history.empty:
        return [None] * len(target_dates)

    history.index = history.index.tz_localize(None)
    series = history["Close"].reindex(target_dates, method="bfill")
    return [round(float(v) / 100, 5) if pd.notnull(v) else None for v in series]
