"""
Database test: connect to PostgreSQL (read-only), list tables, columns,
row counts and one sample row per table.

Run from the project root:
    python -m tests.test_db
"""

import psycopg2  # PostgreSQL driver
from psycopg2 import sql  # safe way to put table names inside a query

from app import config  # our central settings

# Row counts promised in the project brief (names have no spaces or underscores).
EXPECTED = {
    "patients": 25000,
    "admissions": 50000,
    "appointments": 200000,
    "billing": 100000,
    "prescriptions": 250000,
    "labresults": 500000,
    "doctors": 250,
    "departments": 25,
    "insurancepolicies": 500,
    "hospitalbranches": 20,
}


def get_connection():  # open one read-only connection
    config.validate_db_config()  # stop early if a DB setting is missing
    conn = psycopg2.connect(  # connect using the values from .env
        host=config.DB_HOST,
        port=config.DB_PORT,
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
        sslmode=config.DB_SSLMODE,  # Aiven requires SSL
        connect_timeout=15,  # give up after 15 seconds instead of hanging
    )
    conn.set_session(readonly=True)  # the database itself will refuse any write
    return conn


def normalize(name: str) -> str:  # "Lab_Results" -> "labresults"
    return name.lower().replace("_", "").replace(" ", "")


def main() -> None:
    try:
        conn = get_connection()  # connect
    except Exception as exc:  # wrong password, no network, SSL problem...
        print(f"CONNECTION FAILED: {type(exc).__name__}")
        print(str(exc).strip())
        return

    cur = conn.cursor()  # a cursor runs queries
    try:
        cur.execute("SELECT version();")  # simple "are you alive" query
        print("Connected OK ->", cur.fetchone()[0][:60])

        cur.execute(  # all normal tables in the public schema
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            ORDER BY table_name;
            """
        )
        tables = [row[0] for row in cur.fetchall()]  # list of table names
        print(f"Tables found: {len(tables)}\n")

        summary = []  # one line per table for the final table

        for table in tables:
            print("=" * 70)
            print(f"TABLE: {table}")
            print("=" * 70)

            cur.execute(  # column names and types for this table
                """
                SELECT column_name, data_type FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
                ORDER BY ordinal_position;
                """,
                (table,),  # %s is filled safely with the table name
            )
            columns = cur.fetchall()
            for col_name, col_type in columns:
                print(f"  {col_name:<28}{col_type}")

            cur.execute(  # exact row count
                sql.SQL("SELECT COUNT(*) FROM {};").format(sql.Identifier(table))
            )
            count = cur.fetchone()[0]

            cur.execute(  # one sample row, to see how the values look
                sql.SQL("SELECT * FROM {} LIMIT 1;").format(sql.Identifier(table))
            )
            sample = cur.fetchone()
            if sample:
                col_names = [d[0] for d in cur.description]  # column names of the sample
                short = {c: str(v)[:30] for c, v in zip(col_names, sample)}  # cut long values
                print(f"  SAMPLE ROW: {short}")

            print(f"  ROWS: {count:,}\n")
            summary.append((table, len(columns), count))

        # ---- Summary table ----
        print("=" * 70)
        print(f"{'TABLE':<24}{'COLUMNS':>8}{'ROWS':>14}{'EXPECTED':>12}  MATCH")
        print("-" * 70)
        for table, n_cols, count in summary:
            expected = EXPECTED.get(normalize(table))  # None if the name is not in the brief
            exp_text = f"{expected:,}" if expected else "-"
            match = "-" if not expected else ("yes" if count == expected else "close" if abs(count - expected) <= expected * 0.05 else "NO")
            print(f"{table:<24}{n_cols:>8}{count:>14,}{exp_text:>12}  {match}")

    except Exception as exc:  # any query problem
        print(f"QUERY FAILED: {type(exc).__name__}: {exc}")
    finally:
        cur.close()  # always close, even after an error
        conn.close()


if __name__ == "__main__":  # run main() only when started directly
    main()