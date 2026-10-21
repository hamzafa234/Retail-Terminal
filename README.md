# merton_dd

Merton distance-to-default for a single ticker, using SEC XBRL statements, Yahoo Finance prices and a local PostgreSQL database.

## Run

```bash
pip install -r requirements.txt
python -m merton_dd        # or: python main.py
```

Database settings come from environment variables (`DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `SEC_USER_AGENT`) with the old values as defaults — see `merton_dd/config.py`.

## Layout

```
merton_dd/
├── config.py              DB + SEC settings (env-overridable)
├── cli.py                 prompts for ticker/years, runs the pipeline, plots
├── __main__.py            enables `python -m merton_dd`
├── pipeline.py            the end-to-end run, step by step
├── plotting.py            terminal plot of default probabilities
├── data/
│   ├── sec.py             EDGAR fetch + parsing into per-period rows
│   ├── sec_tags.py        XBRL tag fallbacks (edit this to add tags)
│   ├── market.py          prices, betas, Treasury yields (yfinance)
│   └── trading_calendar.py NYSE trading-day helpers
├── db/
│   ├── connection.py      one connect() + a transaction() context manager
│   ├── statements.py      statement tables: clear, insert, date alignment
│   └── calc.py            calc table: write a column, default point, market cap, DD
└── models/
    ├── merton.py          solve_merton, default_probabilities
    └── volatility.py      annualized equity volatility
```

## Renamed functions

| Old (`main.py`) | New |
|---|---|
| `get_comp_fin` | `data.sec.get_company_financials` |
| `get_closing_prices_list` | `data.market.get_closing_prices` |
| `get_betas_for_dates` | `data.market.get_betas` |
| `get_treasury_yield_list_fast` | `data.market.get_treasury_yields` |
| `get_last_nyse_open_date` | `data.trading_calendar.get_last_nyse_close_date` |
| `calculate_equity_volatility` | `models.volatility.annualized_volatility` |
| `get_probabilities` | `models.merton.default_probabilities` |
| `cleardatabase` | `db.statements.clear_database` |
| `insert_multiple_statements` | `db.statements.insert_statements` |
| `copy_dates` | `db.statements.prepare_dates` |
| `insert_into_db` | `db.calc.write_column` |
| `find_default_point` | `db.calc.compute_default_points` |
| `copy_cap` | `db.calc.compute_market_caps` |
| `removeBadData` | `db.calc.remove_rows_without_default_point` |
| `calc_dd` | `db.calc.compute_distance_to_default` |

Removed as unused: `tracked_get` / `total_bytes` (nothing called it, so the "Total downloaded" line always printed 0), and the `sec_api`, `os`, `re`, `ast` imports.

## How values stay on the right row

`calc.id` can have gaps: the date sync drops quarters missing from any statement, and rows with `default_point = 0` are deleted. So nothing is ever matched up by list position. `prepare_dates()` returns `(id, date)` pairs, `write_column()` takes `(id, value)` pairs, and the Merton solver reads its inputs straight from `calc` (with their ids) after bad rows are removed. Rows missing an input are skipped instead of crashing the solver.

## Fixed bugs (from the original `main.py`)

1. Market caps were read with `SELECT DISTINCT` and no `ORDER BY`, so they could come back in any order and duplicates were dropped.
2. Values were written to `calc` by list position (1, 2, 3…) rather than by `id`. Once ids had gaps, values landed on the wrong dates or were lost. Bad rows were also deleted *after* solving, so the solver ran on them.

## Worth knowing

Row `id = 1` (the latest trading day) reuses the previous quarter's statements and default point by design.
