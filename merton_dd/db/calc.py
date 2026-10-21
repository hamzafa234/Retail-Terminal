"""Reads/writes for the calc table (model inputs and outputs).

Every value is written and read together with its calc.id, so deleting rows
(or gaps in the ids) can never shift values onto the wrong row.
"""
from datetime import date
from typing import Iterable, List, Optional, Tuple

from psycopg2 import extras, sql

from merton_dd.db.connection import transaction


def write_column(column: str, values_by_id: Iterable[Tuple[int, object]]) -> None:
    """Write (id, value) pairs into calc.<column>."""
    rows = [(val, row_id) for row_id, val in values_by_id]
    if not rows:
        return
    query = sql.SQL("""
        UPDATE calc AS c
        SET {col} = data.new_val
        FROM (VALUES %s) AS data (new_val, id)
        WHERE c.id = data.id;
    """).format(col=sql.Identifier(column))
    with transaction() as cur:
        extras.execute_values(cur, query, rows)


def compute_default_points() -> None:
    """default_point = current liabilities + 0.5 * long-term debt; id 1 reuses id 2's value."""
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


def compute_market_caps() -> None:
    """market_cap = share_price * shares_outstanding."""
    with transaction("Error while processing") as cur:
        cur.execute("""
            UPDATE calc
            SET market_cap = calc.share_price * income_statement.shares_outstanding
            FROM income_statement
            WHERE calc.id = income_statement.id
              AND calc.statement_date = income_statement.statement_date;
        """)


def remove_rows_without_default_point() -> None:
    with transaction() as cur:
        cur.execute("DELETE FROM calc WHERE default_point = 0;")
        print(f"Successfully deleted {cur.rowcount} rows with default_point = 0")


def load_merton_inputs() -> List[Tuple[int, float, float, float, float]]:
    """
    (id, market_cap, volatility, default_point, riskfreerate) for every row that
    has all four inputs, ordered by id. Rows missing an input are skipped rather
    than crashing the solver.
    """
    rows: List = []
    with transaction() as cur:
        cur.execute("""
            SELECT id, market_cap, volatility, default_point, riskfreerate
            FROM calc
            WHERE market_cap IS NOT NULL
              AND volatility IS NOT NULL
              AND default_point IS NOT NULL
              AND riskfreerate IS NOT NULL
            ORDER BY id;
        """)
        rows = cur.fetchall()
    # Postgres NUMERIC comes back as Decimal; the solver needs floats.
    return [(r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4])) for r in rows]


def compute_distance_to_default() -> List[Tuple[int, Optional[float]]]:
    """Computes DD in SQL (T = 1) and returns (id, dd) pairs ordered by id; dd is None where undefined."""
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
            RETURNING calc.id, calc.distance_to_default;
        """)
        rows = cur.fetchall()
    # RETURNING has no guaranteed order, so sort here.
    return sorted((r[0], None if r[1] is None else float(r[1])) for r in rows)


def load_results() -> List[Tuple[date, float]]:
    """(statement_date, default probability) for every row that has one, newest first."""
    rows: List = []
    with transaction() as cur:
        cur.execute("""
            SELECT statement_date, dtd_value
            FROM calc
            WHERE dtd_value IS NOT NULL
            ORDER BY id;
        """)
        rows = cur.fetchall()
    return [(r[0], float(r[1])) for r in rows]
