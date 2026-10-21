"""Single place that knows how to open a database connection."""
from contextlib import contextmanager

import psycopg2

from merton_dd import config


def connect():
    return psycopg2.connect(
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
        host=config.DB_HOST,
        port=config.DB_PORT,
    )


@contextmanager
def transaction(label: str = "Error"):
    """
    Yields a cursor. Commits on success; on failure rolls back and prints the
    error (the original code's behaviour), then closes everything.
    """
    conn = connect()
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception as e:
        print(f"{label}: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()
