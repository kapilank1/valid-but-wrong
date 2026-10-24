"""Apply infra/sql/*.sql (runs table, frames/verdicts hypertables, verdicts_1s continuous aggregate, policies)
to the Tiger Data / TimescaleDB database in DATABASE_URL. Idempotent.

    DATABASE_URL=postgres://tsdbadmin:...@xxxx.tsdb.cloud.timescale.com:5432/tsdb?sslmode=require \\
        uv run python scripts/db_init.py
    uv run python scripts/db_init.py --check      # also print table/aggregate/policy status
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vbw.store.timescale import SQL_DIR, apply_schema  # noqa: E402

CHECKS = {
    "hypertables": "SELECT hypertable_name, num_chunks, compression_enabled FROM timescaledb_information.hypertables",
    "continuous aggregates": "SELECT view_name, materialized_only FROM timescaledb_information.continuous_aggregates",
    "jobs": "SELECT job_id, proc_name, hypertable_name, schedule_interval FROM timescaledb_information.jobs "
            "WHERE proc_name <> 'policy_telemetry' ORDER BY job_id",
}


def load_dotenv(path: Path) -> None:
    """Minimal .env reader (KEY=VALUE lines) so the script works without exporting variables."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def check(dsn: str) -> None:
    import psycopg

    with psycopg.connect(dsn) as conn:
        for title, sql in CHECKS.items():
            print(f"\n{title}:")
            for row in conn.execute(sql).fetchall():
                print("  ", *row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dsn", help="defaults to $DATABASE_URL (or .env)")
    parser.add_argument("--sql-dir", type=Path, default=SQL_DIR)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    dsn = args.dsn or os.environ.get("DATABASE_URL", "")
    if not dsn:
        print("DATABASE_URL is not set (and no --dsn given)", file=sys.stderr)
        return 2
    count = apply_schema(dsn, args.sql_dir)
    print(f"applied {count} statements from {args.sql_dir}")
    if args.check:
        check(dsn)
    return 0


if __name__ == "__main__":
    sys.exit(main())
