"""Parsing of SEC companyfacts data, with the network calls replaced by fake facts."""
from datetime import date

import pytest

from merton_dd.data import sec


def fact(*points, unit="USD"):
    """points: (end_date, value, form)"""
    return {"units": {unit: [{"end": e, "val": v, "form": f} for e, v, f in points]}}


@pytest.fixture
def fake_sec(monkeypatch):
    """Call with a facts dict; get_company_financials will then read it instead of EDGAR."""
    def install(facts):
        monkeypatch.setattr(sec, "lookup_cik", lambda ticker: "0000000001")
        monkeypatch.setattr(sec, "fetch_us_gaap_facts", lambda cik: facts)
    return install


def test_first_reported_tag_wins_and_non_primary_forms_are_ignored(fake_sec):
    fake_sec({
        "Revenue": fact(("2025-12-31", 100, "10-K"), ("2025-09-30", 999, "8-K")),
        "SalesRevenueNet": fact(("2025-12-31", 555, "10-K")),   # lower-priority fallback
    })
    rows = sec.get_company_financials("TEST", "income", years=5)
    assert len(rows) == 1                                     # the 8-K point is dropped
    assert rows[0]["statement_date"] == date(2025, 12, 31)
    assert rows[0]["revenue"] == 100


def test_every_field_is_present_even_when_unreported(fake_sec):
    fake_sec({"Revenue": fact(("2025-12-31", 100, "10-K"))})
    row = sec.get_company_financials("TEST", "income", years=5)[0]
    assert set(row) == {"statement_date", *sec.TAG_MAP["income"]}
    assert row["net_income"] is None


def test_rows_are_newest_first_and_old_years_cut_off(fake_sec):
    fake_sec({"Revenue": fact(("2024-06-30", 1, "10-Q"), ("2025-12-31", 2, "10-K"), ("2010-12-31", 3, "10-K"))})
    rows = sec.get_company_financials("TEST", "income", years=5)
    assert [r["statement_date"] for r in rows] == [date(2025, 12, 31), date(2024, 6, 30)]


def test_current_liabilities_filled_by_keyword_scan(fake_sec):
    fake_sec({
        "Assets": fact(("2025-12-31", 1000, "10-K"), ("2025-09-30", 900, "10-Q")),
        "LiabilitiesCurrent": fact(("2025-12-31", 300, "10-K")),
        "OtherLiabilitiesCurrent": fact(("2025-09-30", 250, "10-Q")),
    })
    rows = {r["statement_date"]: r for r in sec.get_company_financials("TEST", "balance", years=5)}
    assert rows[date(2025, 12, 31)]["total_current_liabilities"] == 300
    assert rows[date(2025, 9, 30)]["total_current_liabilities"] == 250


def test_missing_long_term_debt_defaults_to_zero(fake_sec):
    fake_sec({"Assets": fact(("2025-12-31", 1000, "10-K"))})
    row = sec.get_company_financials("TEST", "balance", years=5)[0]
    assert row["long_term_debt"] == 0


def test_unknown_ticker_returns_none(monkeypatch):
    monkeypatch.setattr(sec, "lookup_cik", lambda ticker: None)
    assert sec.get_company_financials("NOPE", "income") is None
