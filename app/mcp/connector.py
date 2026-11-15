"""
The ONLY file that opens a database connection.

Every query is read-only: the session is read-only, statements time out after
10 seconds, and results are capped. Values are always passed as parameters
(%s), never glued into the SQL text.
"""

import psycopg2
from psycopg2.extras import RealDictCursor  # rows come back as dictionaries

from app import config

MAX_ROWS = 200  # no tool may return more than this many rows


def _connect():
    config.validate_db_config()
    conn = psycopg2.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
        sslmode=config.DB_SSLMODE,
        connect_timeout=15,
        options="-c statement_timeout=10000",  # kill any query over 10 seconds
    )
    conn.set_session(readonly=True)  # the database itself refuses writes
    return conn


def run_query(sql_text: str, params: tuple = (), limit: int = MAX_ROWS) -> list[dict]:
    """Run one SELECT and return rows as a list of dicts (empty list if none)."""
    first_word = sql_text.lstrip().split(None, 1)[0].lower() if sql_text.strip() else ""
    if first_word not in ("select", "with"):  # second safety layer
        raise ValueError("Only SELECT queries are allowed.")
    conn = _connect()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql_text, params)
            rows = cur.fetchmany(min(limit, MAX_ROWS))
            return [dict(r) for r in rows]
    finally:
        conn.close()  # always close, even after an error
