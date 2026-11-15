"""Print every table and its columns on one line each (read-only)."""

from app.mcp.connector import run_query

tables = run_query(
    "SELECT table_name FROM information_schema.tables "
    "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' ORDER BY table_name"
)
for t in tables:
    cols = run_query(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = %s ORDER BY ordinal_position",
        (t["table_name"],),
    )
    print(f"{t['table_name']}: " + ", ".join(c["column_name"] for c in cols))