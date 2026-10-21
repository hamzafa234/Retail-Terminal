"""NYSE trading-day helpers."""
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import pandas_market_calendars as mcal

_NYSE = mcal.get_calendar("NYSE")


def get_nyse_trading_dates(start_date, end_date) -> list:
    return _NYSE.schedule(start_date=start_date, end_date=end_date).index.tolist()


def get_last_n_trading_days(end_date=None, n: int = 252) -> list:
    """Exactly the last n NYSE trading days up to end_date."""
    if not isinstance(n, int):
        raise TypeError(f"Expected n to be an int, got {type(n).__name__}")
    if end_date is None:
        end_date = date.today()

    start_date = end_date - timedelta(days=int(n * 1.6))
    days = get_nyse_trading_dates(start_date, end_date)
    while len(days) < n:
        start_date -= timedelta(days=20)
        days = get_nyse_trading_dates(start_date, end_date)
    return days[-n:]


def get_last_nyse_close_date() -> Optional[date]:
    """Date of the most recent NYSE session that has already closed."""
    today = datetime.now().date()
    schedule = _NYSE.schedule(start_date=today - timedelta(days=10), end_date=today)
    closed = schedule[schedule["market_close"] <= pd.Timestamp.now(tz="UTC")]
    return closed.index[-1].date() if not closed.empty else None
