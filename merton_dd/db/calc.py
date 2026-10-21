"""Reads/writes for the calc table (model inputs and outputs)."""
from typing import List, Sequence

from psycopg2 import extras, sql

from merton_dd.db.connection import transaction


def write_column(values: Sequence, column: str) -> None:
    """
    Write `values` into calc.<column>, pairing values[0] with id 1, values[1] with id 2, ...
    """
    rows = [(val, i) for i, val in enumerate(values, start=1)]
    query = sql.SQL("""
        UPDATE calc AS c
        SET {col} = data.new_val
        FROM (VALUES %s) AS data (new_val, id)
        WHERE c.id = data.id;
    """).format(col=sql.Identifier(column))
    with transaction() as cur:
        extras.execute_values(cur, query, rows)


def compute_default_points() -> List:
    """default_point = current liabilities + 0.5 * long-term debt; id 1 reuses id 2's value."""
    results: List = []
    with transaction() as cur:
        cur.execute("""
            UPDATE calc c
            SET default_point = COALESCE(bs.total_current_liabilities, 0)
                              + 0.5 * COALESCE(bs.long_term_debt, 0)
            FROM balance_sheet bs
            WHERE c.statement_date = bs.statement_date;
        """)
        cur.execute("""
            UPDATE calc
            SET default_point = (SELECT default_point FROM calc WHERE id = 2)
            WHERE id = 1;
        """)
        cur.execute("SELECT default_point FROM calc ORDER BY id;")
        results = [row[0] for row in cur.fetchall()]
    return results


def compute_market_caps() -> List:
    """market_cap = share_price * shares_outstanding; returns caps ordered by calc.id."""
    caps: List = []
    with transaction("Error while processing") as cur:
        cur.execute("""
            UPDATE calc
            SET market_cap = calc.share_price * income_statement.shares_outstanding
            FROM income_statement
            WHERE calc.id = income_statement.id
              AND calc.statement_date = income_statement.statement_date;
        """)
        cur.execute("SELECT market_cap FROM calc ORDER BY id;")
        caps = [row[0] for row in cur.fetchall()]
    return caps


def remove_rows_without_default_point() -> None:
    with transaction() as cur:
        cur.execute("DELETE FROM calc WHERE default_point = 0;")
        print(f"Successfully deleted {cur.rowcount} rows with default_point = 0")


def compute_distance_to_default() -> List[float]:
    """Computes DD in SQL from market_val_assets, default_point, asset_vol and riskfreerate (T = 1)."""
    rows: List = []
    with transaction() as cur:
        cur.execute("""
            UPDATE calc
            SET
                statement_date = src.statement_date,
                distance_to_default = CASE
                    WHEN calc.market_val_assets <= 0 OR calc.default_point <= 0 OR calc.asset_vol <= 0 THEN NULL
                    WHEN calc.market_val_assets / calc.default_point < 1e-300 THEN -700.0
                    ELSE (LN(calc.market_val_assets / calc.default_point)
                          + (calc.riskfreerate - 0.5 * POWER(calc.asset_vol, 2)) * 1.0)
                         / (calc.asset_vol * SQRT(1.0))
                END
            FROM income_statement AS src
            WHERE calc.id = src.id
            RETURNING calc.distance_to_default;
        """)
        rows = cur.fetchall()
    return [float(r[0]) for r in rows]
