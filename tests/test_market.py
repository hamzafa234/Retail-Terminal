"""Market-data helpers, with yfinance replaced by fake price histories."""
from datetime import date

import pandas as pd
import pytest

from merton_dd.data import market


def fake_history(closes):
    """closes: {date: close} -> a DataFrame shaped like yfinance's Ticker.history()."""
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in closes], name="Date").tz_localize("America/New_York")
    return pd.DataFrame({"Close": list(closes.values())}, index=idx)


@pytest.fixture
def fake_ticker(monkeypatch):
    def install(closes):
        class FakeTicker:
            def __init__(self, symbol):
                pass

            def history(self, start, end):
                return fake_history(closes)
        monkeypatch.setattr(market.yf, "Ticker", FakeTicker)
    return install


def test_closing_prices_use_next_trading_day_and_keep_input_order(fake_ticker):
    # Fri 2025-10-03, Mon 2025-10-06, Tue 2025-10-07
    fake_ticker({date(2025, 10, 3): 10.0, date(2025, 10, 6): 11.0, date(2025, 10, 7): 12.0})
    requested = [date(2025, 10, 7), date(2025, 10, 4), date(2025, 10, 3)]   # Sat -> Mon
    assert market.get_closing_prices("TEST", requested) == [12.0, 11.0, 10.0]


def test_closing_prices_empty_history_gives_nones(fake_ticker):
    fake_ticker({})
    assert market.get_closing_prices("TEST", [date(2025, 10, 3)]) == [None]


def test_closing_prices_no_dates():
    assert market.get_closing_prices("TEST", []) == []


def test_treasury_yields_are_decimals_and_backfilled(fake_ticker):
    fake_ticker({date(2025, 10, 3): 4.25, date(2025, 10, 6): 4.30})
    # Pass pandas Timestamps, as the pipeline's dates are matched against the history index.
    requested = [pd.Timestamp(2025, 10, 3), pd.Timestamp(2025, 10, 5)]      # Sun -> Mon
    assert market.get_treasury_yields(requested) == [0.0425, 0.043]
