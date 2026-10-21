"""Reads/writes for the income_statement, balance_sheet and cashflow_statement tables."""
from datetime import date
from typing import Any, Dict, List, Tuple

from psycopg2 import extras, sql

from merton_dd.db.connection import transaction

STATEMENT_TABLES = {
    "income": "income_statement",
    "cashflow": "cashflow_statement",
    "balance": "balance_sheet",
}

ALL_TABLES = ["calc", "income_statement", "cashflow_statement", "balance_sheet"]

INCOME_COLUMNS = [
    "revenue", "cost_of_revenue", "gross_profit", "operating_expenses",
    "research_development", "selling_general_administrative", "operating_income",
    "interest_expense", "interest_income", "other_income_expense", "income_before_tax",
    "income_tax_expense", "net_income", "eps", "diluted_eps", "shares_outstanding",
    "diluted_shares_outstanding",
]

_COMMON_DATES_CTE = """
    WITH common_dates AS (
        SELECT statement_date FROM income_statement
        INTERSECT
        SELECT statement_date FROM balance_sheet
        INTERSECT
        SELECT statement_date FROM cashflow_statement
    )
"""

# Keep only dates present in all three statements.
SYNC_DATES_SQL = ";\n".join(
    f"{_COMMON_DATES_CTE} DELETE FROM {table} "
    f"WHERE statement_date NOT IN (SELECT statement_date FROM common_dates)"
    for table in ("income_statement", "balance_sheet", "cashflow_statement")
) + ";"

SHIFT_IDS_SQL = """
    UPDATE income_statement SET id = -id - 1;
    UPDATE income_statement SET id = -id;
"""

INSERT_LATEST_SQL = "INSERT INTO income_statement (id, statement_date) VALUES (1, %s);"

COPY_ROW_2_TO_1_SQL = (
    "UPDATE income_statement AS target SET ({cols}) = "
    "(SELECT {coalesced} FROM income_statement WHERE id = 2) "
    "WHERE target.id = 1;"
).format(
    cols=", ".join(INCOME_COLUMNS),
    coalesced=", ".join(f"COALESCE({c}, 0)" for c in INCOME_COLUMNS),
)

RESEED_CALC_SQL = """
    DELETE FROM calc;
    INSERT INTO calc (id, statement_date)
    SELECT id, statement_date FROM income_statement ORDER BY id;
"""

FETCH_CALC_ROWS_SQL = "SELECT id, statement_date FROM calc ORDER BY id;"


def clear_database() -> None:
    with transaction() as cur:
        for table in ALL_TABLES:
            cur.execute(
                sql.SQL("TRUNCATE TABLE {} RESTART IDENTITY CASCADE;").format(sql.Identifier(table))
            )


def insert_statements(data_list: List[Dict[str, Any]], statement_type: str) -> None:
    """Bulk-insert rows from get_company_financials() into the matching statement table."""
    if not data_list:
        print("Input list is empty. No data to insert.")
        return
    if statement_type not in STATEMENT_TABLES:
        raise ValueError(f"Unknown statement type: {statement_type!r}")

    columns = list(data_list[0].keys())
    rows = [tuple(row[c] for c in columns) for row in data_list]
    query = sql.SQL("INSERT INTO {} ({}) VALUES %s").format(
        sql.Identifier(STATEMENT_TABLES[statement_type]),
        sql.SQL(", ").join(map(sql.Identifier, columns)),
    )

    print(f"Attempting to insert {len(rows)} records...")
    with transaction("Database Error") as cur:
        extras.execute_values(cur, query, rows, page_size=1000)


def sync_calc_dates() -> None:
    """Fill calc.statement_date from income_statement where it's missing."""
    with transaction() as cur:
        cur.execute("""
            UPDATE calc
            SET statement_date = inc.statement_date
            FROM income_statement AS inc
            WHERE calc.id = inc.id AND calc.statement_date IS NULL;
        """)


def prepare_dates(latest: date) -> List[Tuple[int, date]]:
    """
    Drops dates not present in all three statements, inserts `latest` as id 1 in
    income_statement (copying id 2's figures into it), re-seeds calc with the
    income_statement ids/dates, and returns the calc rows as (id, date), newest first.

    The ids are returned because they can have gaps (rows dropped by the date
    sync above), so a value's list position is not its id.
    """
    rows: List[Tuple[int, date]] = []
    with transaction() as cur:
        cur.execute(SYNC_DATES_SQL)
        cur.execute(SHIFT_IDS_SQL)
        cur.execute(INSERT_LATEST_SQL, (latest,))
        cur.execute(COPY_ROW_2_TO_1_SQL)
        cur.execute(RESEED_CALC_SQL)
        cur.execute(FETCH_CALC_ROWS_SQL)
        rows = [(row[0], row[1]) for row in cur.fetchall()]
    return rows
