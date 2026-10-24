"""
End-to-end run against a real (test) Postgres database, with every network call
replaced by fake data that has known answers.

The fake data deliberately reproduces the situations that used to put values
on the wrong rows:
  * one quarter has no balance sheet -> dropped by the date sync -> gap in calc ids
  * one quarter has zero liabilities and debt -> default_point 0 -> row deleted
"""
from datetime import date

import pytest

from merton_dd import pipeline
from merton_dd.data.sec_tags import TAG_MAP
from merton_dd.db.connection import connect

Q1, Q2, Q3, Q4 = date(2026, 6, 30), date(2026, 3, 31), date(2025, 12, 31), date(2025, 9, 30)
LATEST = date(2026, 10, 19)

SHARES = {Q1: 1000, Q2: 1100, Q3: 1200, Q4: 1300}
LIABILITIES = {Q1: 50_000, Q2: 51_000, Q3: 0, Q4: 53_000}   # Q3 -> default point 0
DEBT = {Q1: 20_000, Q2: 21_000, Q3: 0, Q4: 23_000}
PRICE = {LATEST: 150.0, Q1: 140.0, Q2: 130.0, Q3: 120.0, Q4: 110.0}
VOL = {LATEST: 0.30, Q1: 0.31, Q2: 0.32, Q3: 0.33, Q4: 0.34}


def fake_financials(ticker, kind, years=5):
    rows = []
    for d in (Q1, Q2, Q3, Q4):
        if kind == "balance" and d == Q2:
            continue   # missing balance sheet
        row = {"statement_date": d, **{f: None for f in TAG_MAP[kind]}}
        if kind == "income":
            row["shares_outstanding"] = SHARES[d]
        if kind == "balance":
            row["total_current_liabilities"] = LIABILITIES[d]
            row["long_term_debt"] = DEBT[d]
        rows.append(row)
    return rows


@pytest.fixture
def run_result(test_db, monkeypatch):
    monkeypatch.setattr(pipeline, "get_company_financials", fake_financials)
    monkeypatch.setattr(pipeline, "get_last_nyse_close_date", lambda: LATEST)
    monkeypatch.setattr(pipeline, "equity_volatilities", lambda t, ds: [VOL[d] for d in ds])
    monkeypatch.setattr(pipeline, "get_closing_prices", lambda t, ds: [PRICE[d] for d in ds])
    monkeypatch.setattr(pipeline, "get_betas", lambda t, ds: [1.0] * len(ds))
    monkeypatch.setattr(pipeline, "get_treasury_yields", lambda ds: [0.04] * len(ds))

    dates, probs = pipeline.run("TEST", 5)

    conn = connect()
    cur = conn.cursor()
    cur.execute("""SELECT id, statement_date, share_price, volatility, market_cap,
                          default_point, market_val_assets, dtd_value
                   FROM calc ORDER BY id""")
    rows = {r[1]: r for r in cur.fetchall()}
    conn.close()
    return dates, probs, rows


def test_unusable_quarters_are_removed(run_result):
    _, _, rows = run_result
    assert set(rows) == {LATEST, Q1, Q4}   # Q2: no balance sheet, Q3: default point 0


def test_ids_have_gaps(run_result):
    """Sanity check that this test actually exercises the gap case."""
    _, _, rows = run_result
    ids = sorted(r[0] for r in rows.values())
    assert ids != list(range(1, len(ids) + 1))


def test_every_value_lands_on_its_own_date(run_result):
    _, _, rows = run_result
    for d, (_, _, price, vol, cap, *_rest) in rows.items():
        shares = SHARES[Q1] if d == LATEST else SHARES[d]   # latest reuses the last quarter
        assert float(price) == PRICE[d]
        assert float(vol) == VOL[d]
        assert float(cap) == PRICE[d] * shares


def test_default_points(run_result):
    _, _, rows = run_result
    assert float(rows[Q1][5]) == LIABILITIES[Q1] + 0.5 * DEBT[Q1]
    assert float(rows[Q4][5]) == LIABILITIES[Q4] + 0.5 * DEBT[Q4]
    assert rows[LATEST][5] == rows[Q1][5]   # latest copies the last quarter


def test_every_remaining_row_is_solved(run_result):
    _, _, rows = run_result
    for r in rows.values():
        assert r[6] is not None and float(r[6]) > 0    # market value of assets
        assert r[7] is not None                         # default probability


def test_returned_results_match_the_table(run_result):
    dates, probs, rows = run_result
    assert dates == [LATEST, Q1, Q4]     # newest first
    assert len(probs) == len(dates)
    assert probs == [float(rows[d][7]) for d in dates]
