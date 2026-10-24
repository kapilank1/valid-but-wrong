"""Apply infra/sql/*.sql (runs table, frames/verdicts hypertables, verdicts_1s continuous aggregate, policies)
to the Tiger Data / TimescaleDB database in DATABASE_URL. Idempotent.

    # DATABASE_URL in the repo-root .env (gitignored), e.g. from Tiger Cloud:
    #   DATABASE_URL=postgres://tsdbadmin:<password>@<id>.<project>.tsdb.cloud.timescale.com:<port>/tsdb?sslmode=require
    uv run python scripts/db_init.py              # apply the schema
    uv run python scripts/db_init.py --check      # apply, then an end-to-end check that cleans up after itself

--check: connects, verifies the hypertables / continuous aggregate / policy jobs exist, writes a synthetic run
(frames + verdicts) through the same store code the server uses, refreshes the continuous aggregate, reads it back
through the dashboard queries (runs list, run timeline, analytics), then deletes every row it wrote. The synthetic
run is stamped 2000-01-01 so it can never mix with real runs.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vbw.store.timescale import SQL_DIR, apply_schema  # noqa: E402

EXPECTED_HYPERTABLES = {"frames", "verdicts"}
EXPECTED_CAGG = "verdicts_1s"
CHECK_T0 = datetime(2000, 1, 1, tzinfo=timezone.utc)
# Constant literals (not bind parameters): CALL refresh_continuous_aggregate must run outside any transaction block.
REFRESH_CHECK_WINDOW = ("CALL refresh_continuous_aggregate('verdicts_1s', '2000-01-01 00:00:00+00'::timestamptz, "
                        "'2000-01-01 01:00:00+00'::timestamptz)")
DROP_CHECK_CHUNKS = ("SELECT drop_chunks('{table}', older_than => '2000-01-02 00:00:00+00'::timestamptz, "
                     "newer_than => '1999-12-31 00:00:00+00'::timestamptz)")


def load_dotenv(path: Path) -> None:
    """Minimal .env reader (KEY=VALUE lines) so the script works without exporting variables."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def redact(dsn: str) -> str:
    """postgres://user:secret@host... -> postgres://user:***@host..."""
    if "@" not in dsn or "://" not in dsn:
        return dsn
    scheme, rest = dsn.split("://", 1)
    creds, host = rest.rsplit("@", 1)
    user = creds.split(":", 1)[0]
    return f"{scheme}://{user}:***@{host}"


def step(ok: bool, what: str, detail: str = "") -> bool:
    print(f"  [{'ok' if ok else 'FAIL'}] {what}{(' - ' + detail) if detail else ''}")
    return ok


def check_catalog(dsn: str) -> bool:
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as conn:
        version = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'").fetchone()
        ok = step(version is not None, "timescaledb extension", f"version {version[0]}" if version else "missing")
        hypertables = {r[0] for r in conn.execute(
            "SELECT hypertable_name FROM timescaledb_information.hypertables").fetchall()}
        ok &= step(EXPECTED_HYPERTABLES <= hypertables, "hypertables", ", ".join(sorted(hypertables)) or "none")
        caggs = {r[0]: r[1] for r in conn.execute(
            "SELECT view_name, materialized_only FROM timescaledb_information.continuous_aggregates").fetchall()}
        ok &= step(EXPECTED_CAGG in caggs, "continuous aggregate verdicts_1s",
                   f"materialized_only={caggs.get(EXPECTED_CAGG)}")
        jobs = conn.execute(
            "SELECT proc_name, hypertable_name, schedule_interval FROM timescaledb_information.jobs "
            "WHERE proc_name NOT IN ('policy_telemetry', 'policy_job_stat_history_retention') ORDER BY job_id").fetchall()
        procs = {r[0] for r in jobs}
        ok &= step("policy_refresh_continuous_aggregate" in procs, "continuous aggregate refresh policy")
        ok &= step(bool(procs & {"policy_compression", "policy_columnstore"}), "frames compression policy")
        for proc, table, every in jobs:
            print(f"         job {proc} on {table} every {every}")
    return ok


async def check_roundtrip(dsn: str) -> bool:
    """Write a synthetic run through the server's store code, read it back via the dashboard queries, clean up."""
    import psycopg

    from vbw.schema import FALCON9, Frame, L0Result, Verdict
    from vbw.store.base import RunRecord, TickRecord
    from vbw.store.timescale import Batch, RunEnd, TimescaleStore, frame_rows, make_database, verdict_row

    run_id = str(uuid.uuid4())
    run = RunRecord(run_id, CHECK_T0, "db_init:check", "check", "falcon9", "bias", {"start_t": 1.0}, "base", 1.0, 0)
    batch = Batch(meta=[run])
    frames, alerts = 50, 0
    for seq in range(frames):                       # 5 s at 10 Hz; the attack starts at t = 1 s, alert from 1.5 s
        t = seq / 10
        truth = Frame("check", FALCON9.schema_id, seq, t, {c: float(seq) for c in FALCON9.channels})
        reported, active = truth.copy(), t >= 1.0
        if active:
            reported.channels["altitude"] += 3000.0
        alert = t >= 1.5
        alerts += alert
        verdict = Verdict("check", seq, t, L0Result(), l1_score=0.9 if active else 0.05,
                          score=0.9 if active else 0.05, alert=alert, detector="base")
        tick = TickRecord(run, truth, reported, verdict, "bias", active)
        batch.frames.extend(frame_rows(tick))
        batch.verdicts.append(verdict_row(tick))
    batch.meta.append(RunEnd(run_id, CHECK_T0, frames, alerts))

    db = make_database(dsn)
    store = TimescaleStore(db)
    ok = True
    try:
        await asyncio.wait_for(db.write(batch), 30)
        ok &= step(True, "write a synthetic run", f"{batch.rows} rows via COPY ({type(db).__name__})")
        with psycopg.connect(dsn, autocommit=True) as conn:   # CALL refresh_... can't run in a transaction
            conn.execute(REFRESH_CHECK_WINDOW)
        step(True, "refresh verdicts_1s for the synthetic window")

        timeline = await store.timeline(run_id)
        summary, buckets = timeline.get("run") or {}, timeline.get("buckets") or []
        ok &= step(timeline.get("available") is True and len(buckets) == 5, "run timeline (from verdicts_1s)",
                   f"{len(buckets)} one-second buckets")
        latency = summary.get("detection_latency_s")
        ok &= step(latency is not None and abs(latency - 0.5) < 1e-6, "detection latency",
                   f"{latency} s (expected 0.5 s)")
        runs = await store.list_runs(500)
        mine = next((r for r in runs.get("runs", []) if r["run_id"] == run_id), None)
        ok &= step(mine is not None and mine["alerts"] == alerts and mine["alert_frames"] == alerts,
                   "runs list rollup", f"alerts={mine and mine['alerts']} alert_frames={mine and mine['alert_frames']}")
        analytics = await store.analytics()
        ok &= step(analytics.get("available") is True, "analytics (latency + scorecard)",
                   f"{len(analytics.get('scorecard', []))} scorecard rows")
    except Exception as exc:  # noqa: BLE001
        ok = step(False, "round trip", f"{type(exc).__name__}: {exc}")
    finally:
        await db.close()
        try:
            with psycopg.connect(dsn, autocommit=True) as conn:
                conn.execute("DELETE FROM verdicts WHERE run_id = %s::uuid", (run_id,))
                conn.execute("DELETE FROM frames WHERE run_id = %s::uuid", (run_id,))
                conn.execute("DELETE FROM runs WHERE run_id = %s::uuid", (run_id,))
                conn.execute(REFRESH_CHECK_WINDOW)
                for table in ("verdicts", "frames"):     # drop the now-empty year-2000 chunks
                    conn.execute(DROP_CHECK_CHUNKS.format(table=table))
            step(True, "clean up the synthetic run")
        except Exception as exc:  # noqa: BLE001
            ok = step(False, "clean up", f"{type(exc).__name__}: {exc} (run_id {run_id})")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dsn", help="defaults to $DATABASE_URL (or the repo-root .env)")
    parser.add_argument("--sql-dir", type=Path, default=SQL_DIR)
    parser.add_argument("--check", action="store_true", help="end-to-end write/read check after applying the schema")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    dsn = args.dsn or os.environ.get("DATABASE_URL", "")
    if not dsn:
        print("DATABASE_URL is not set (put it in the repo-root .env or pass --dsn)", file=sys.stderr)
        return 2
    print(f"database: {redact(dsn)}")
    try:
        count = apply_schema(dsn, args.sql_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] apply schema - {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"  [ok] applied {count} statements from {args.sql_dir}")
    if not args.check:
        return 0
    ok = check_catalog(dsn)
    ok &= asyncio.run(check_roundtrip(dsn))
    print("\nTiger Data check passed." if ok else "\nTiger Data check FAILED (see above).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
