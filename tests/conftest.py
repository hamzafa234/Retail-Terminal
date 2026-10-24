"""
Shared test setup.

Unit tests need nothing. The database tests need a running Postgres; they use a
SEPARATE database (default: fin_data_test) so your real fin_data is never touched,
and are skipped automatically if Postgres can't be reached.

Override the test database name with TEST_DB_NAME; host/user/port/password come
from the usual DB_* environment variables.
"""
import os

# Must happen before merton_dd.config is imported anywhere.
os.environ["DB_NAME"] = os.environ.get("TEST_DB_NAME", "fin_data_test")

import psycopg2  # noqa: E402
import pytest  # noqa: E402

from merton_dd import config  # noqa: E402
from merton_dd.data.sec_tags import TAG_MAP  # noqa: E402
from merton_dd.db.connection import connect  # noqa: E402

STATEMENT_TABLES = {"income": "income_statement", "balance": "balance_sheet", "cashflow": "cashflow_statement"}


def _create_database_if_missing():
    try:
        connect().close()
        return
    except psycopg2.OperationalError as e:
        if "does not exist" not in str(e):
            raise
    admin = psycopg2.connect(dbname="postgres", user=config.DB_USER, password=config.DB_PASSWORD,
                             host=config.DB_HOST, port=config.DB_PORT)
    admin.autocommit = True
    admin.cursor().execute(f'CREATE DATABASE "{config.DB_NAME}"')
    admin.close()


def _create_schema():
    """Tables matching what the code reads and writes."""
    conn = connect()
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS calc, income_statement, balance_sheet, cashflow_statement")
    for kind, table in STATEMENT_TABLES.items():
        cols = ", ".join(f"{c} NUMERIC" for c in TAG_MAP[kind])
        cur.execute(f"CREATE TABLE {table} (id SERIAL PRIMARY KEY, statement_date DATE, {cols})")
    cur.execute("""
        CREATE TABLE calc (
            id INT PRIMARY KEY, statement_date DATE,
            volatility NUMERIC, beta NUMERIC, share_price NUMERIC, riskfreerate NUMERIC,
            default_point NUMERIC, market_cap NUMERIC, market_val_assets NUMERIC,
            asset_vol NUMERIC, distance_to_default NUMERIC, dtd_value NUMERIC
        )
    """)
    conn.commit()
    conn.close()


@pytest.fixture
def test_db():
    """A fresh, empty test database. Skips the test if Postgres isn't available."""
    if config.DB_NAME == "fin_data":
        pytest.fail("Refusing to run database tests against the real fin_data database.")
    try:
        _create_database_if_missing()
    except psycopg2.OperationalError as e:
        pytest.skip(f"Postgres not available: {e}")
    _create_schema()
    yield
