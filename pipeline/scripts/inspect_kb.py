#!/usr/bin/env python3
"""Inspect the existing fraud_kb database."""
import sqlite3
from pathlib import Path

db_path = Path("models/decompressed/fraud_kb/fraud_kb.db")
if not db_path.exists():
    print("DB not found:", db_path)
    exit(1)

conn = sqlite3.connect(str(db_path))
tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("TABLES:", [t[0] for t in tables])
print()

for (table,) in tables:
    count = conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]
    print(f"TABLE: {table} ({count} rows)")
    sample = conn.execute(f"SELECT * FROM [{table}] LIMIT 10").fetchall()
    for row in sample:
        print(" ", row)
    print()

# Check for source_metadata table
has_meta = conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name='source_metadata'"
).fetchone()
print("Has source_metadata table:", bool(has_meta))

conn.close()
